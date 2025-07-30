import os
import os.path as osp
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml
import matplotlib.pyplot as plt

sys.path.append("./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice


def plot_and_save_confidence_distribution(confidence_data, iteration, output_dir, protect_threshold,
                                          candidate_threshold):
    """
    【修正版】生成并保存高斯点置信度的分布直方图。
    移除了固定的range，让matplotlib自动适应数据的完整范围。
    """
    if confidence_data.size == 0:
        print("[WARNING] Confidence data is empty, skipping plot generation.")
        return

    plt.figure(figsize=(14, 8))

    # 【核心修改】移除了 range=(-2, 2) 参数，让绘图范围自动适应所有数据
    plt.hist(confidence_data, bins=150, color='deepskyblue', edgecolor='black', alpha=0.7, log=True)

    plt.title(f'ROI Confidence Distribution at Iteration {iteration} (Log Scale)', fontsize=16)
    plt.xlabel('Confidence Value', fontsize=12)
    plt.ylabel('Number of Gaussian Points (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # 标记我们的关键阈值
    plt.axvline(x=protect_threshold, color='green', linestyle='--', linewidth=2,
                label=f'Protect Threshold ({protect_threshold})')
    plt.axvline(x=candidate_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Candidate Threshold ({candidate_threshold})')

    # 在图上显示统计数据
    mean_conf = np.mean(confidence_data)
    median_conf = np.median(confidence_data)
    std_conf = np.std(confidence_data)
    min_conf = np.min(confidence_data)
    max_conf = np.max(confidence_data)
    stats_text = (f'Total Points: {len(confidence_data)}\n'
                  f'Mean: {mean_conf:.3f}\n'
                  f'Median: {median_conf:.3f}\n'
                  f'Std Dev: {std_conf:.3f}\n'
                  f'Min: {min_conf:.3f}\n'
                  f'Max: {max_conf:.3f}')
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.legend(fontsize=12)
    plt.tight_layout()

    # 保存图像
    save_path = osp.join(output_dir, f"confidence_distribution_iter_{iteration}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    tqdm.write(f"[INFO] Confidence distribution plot saved to: {save_path}")

def training(args, dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, tb_writer, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    scene = Scene(dataset, shuffle=False)
    roi_management_active = scene.use_roi_masks
    if roi_management_active:
        print("ROI置信度管理已在训练流程中激活。")
        confidence_thresholds = {"prune": opt.roi_prune_threshold, "protect": opt.roi_protect_threshold}
        rewards = {"penalty": opt.roi_background_reward, "standard": opt.roi_standard_reward, "core_bonus": opt.roi_core_bonus_reward}
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = opt.densify_scale_threshold * volume_to_world if opt.densify_scale_threshold else None
    scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world if dataset.scale_min and dataset.scale_max else None
    queryfunc = lambda x: query(x, scanner_cfg["offOrigin"], scanner_cfg["nVoxel"], scanner_cfg["sVoxel"], pipe)
    gaussians = GaussianModel(scale_bound)

    initialize_gaussian(gaussians, dataset, None)

    if opt.intelligent_confidence_threshold > 0:
        print(f"\n--- [智能置信度初始化] ---")
        print(f"正在为初始点云赋予ROI置信度奖励，FDK密度阈值: {opt.intelligent_confidence_threshold}")
        with torch.no_grad():
            # 获取所有初始点的世界坐标
            initial_xyz = gaussians.get_xyz

            # 从scene对象获取FDK体积和坐标转换信息
            fdk_volume = scene.vol_gt.to(initial_xyz.device)
            grid_center = scene.grid_center.to(initial_xyz.device)
            voxel_size = scene.voxel_size.to(initial_xyz.device)
            grid_dims = torch.tensor(fdk_volume.shape, device=initial_xyz.device)

            # 将世界坐标转换为FDK体积的网格索引
            relative_coords = initial_xyz - grid_center
            grid_indices = torch.round((relative_coords / voxel_size) + (grid_dims / 2.0)).long()

            # 筛选出在FDK体积内的点
            valid_mask = (grid_indices[:, 2] >= 0) & (grid_indices[:, 2] < grid_dims[0]) & \
                         (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < grid_dims[1]) & \
                         (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < grid_dims[2])

            absolute_indices_of_valid_points = torch.where(valid_mask)[0]
            valid_grid_indices = grid_indices[valid_mask]

            if absolute_indices_of_valid_points.numel() > 0:
                # 查询这些点在FDK体积中的密度值
                point_densities = fdk_volume[
                    valid_grid_indices[:, 2],  # Z
                    valid_grid_indices[:, 1],  # Y
                    valid_grid_indices[:, 0]  # X
                ]

                # 找到密度高于我们设定阈值的点
                core_points_mask = point_densities > opt.intelligent_confidence_threshold
                target_indices = absolute_indices_of_valid_points[core_points_mask]

                # 为这些“核心点”赋予初始奖励
                # 我们使用 roi_core_bonus_reward 作为这个奖励值
                initial_reward = opt.roi_core_bonus_reward
                gaussians._roi_confidence[target_indices] = initial_reward

                print(
                    f"操作完成: {target_indices.numel()} / {initial_xyz.shape[0]} 个点被识别为核心点，并赋予了 {initial_reward:.2f} 的初始置信度。")
            else:
                print("警告: 没有一个初始点落在FDK体积内，无法进行智能置信度初始化。")
        print(f"--- [初始化结束] ---\n")

    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bbox = bbox.to(gaussians.get_xyz.device)
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        device = bbox.device
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size], device=device)
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"], device=device, dtype=torch.float32) * tv_vol_nVoxel
    use_shape_regularization = opt.lambda_shape > 0
    if use_shape_regularization: print(f"Use shape regularization loss with lambda = {opt.lambda_shape}")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations + 1), desc="Train", leave=False)
    for iteration in range(first_iter, opt.iterations + 1):
        if iteration == 0: continue
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if not viewpoint_stack: viewpoint_stack = scene.getTrainCameras().copy()
        if not viewpoint_stack: viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        active_sh_indices = None

        render_pkg = render(viewpoint_cam, gaussians, pipe)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        loss = {"total": 0.0}
        render_loss = l1_loss(image, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
        if use_tv:
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (bbox[1] - tv_vol_sVoxel - bbox[0]) * torch.rand(3, device=bbox.device)
            vol_pred = query(gaussians, tv_vol_center, tv_vol_nVoxel, tv_vol_sVoxel, pipe)["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv
        if use_shape_regularization:
            scales = gaussians.get_scaling
            if scales.shape[0] > 0:
                max_scales, _ = torch.max(scales, dim=1)
                min_scales, _ = torch.min(scales, dim=1)
                epsilon = 1e-8
                loss_shape = torch.mean(max_scales / (min_scales + epsilon) - 1.0)
                loss["shape"] = loss_shape
                loss["total"] = loss["total"] + opt.lambda_shape * loss_shape
        loss["total"].backward()
        iter_end.record()
        torch.cuda.synchronize()
        with torch.no_grad():
            if args.soft_mask_dir and args.core_mask_dir and iteration >= args.roi_update_start_iter and iteration % args.roi_management_interval == 0:
                # 定义我们最终的激励方案参数
                # 注意：我们不再使用旧的 rewards 字典
                background_reward = opt.roi_background_reward if hasattr(opt,
                                                                         'roi_background_reward') else -opt.roi_penalty

                gaussians.update_roi_confidence(
                    render_pkg,
                    viewpoint_cam,  # 直接传递当前视角的相机对象
                    opt.roi_standard_reward,
                    opt.roi_core_bonus_reward,
                    background_reward
                )

            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration < opt.densify_until_iter:
                if (iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0):
                    # 【修改】将所有需要的阈值参数完整地传递给函数
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                        roi_protect_threshold=opt.roi_protect_threshold if roi_management_active else 1e5,
                        roi_candidate_threshold=opt.roi_candidate_threshold if roi_management_active else -1e5
                    )
            if gaussians.get_density.shape[0] == 0: raise ValueError("没有剩余的高斯点。请调整自适应控制超参数！")
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] 保存高斯模型")
                scene.save(iteration, queryfunc)
            if iteration == opt.iterations and args.plot_confidence:
                tqdm.write("\n[INFO] 训练结束，开始生成最终的置信度分布图...")
                confidence_data = gaussians.get_roi_confidence.cpu().numpy()
                plot_and_save_confidence_distribution(
                    confidence_data,
                    iteration,
                    scene.model_path,  # 保存到主模型目录
                    opt.roi_protect_threshold,
                    opt.roi_candidate_threshold
                )
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] 保存检查点")
                torch.save((gaussians.capture(), iteration), ckpt_save_path + "/chkpnt" + str(iteration) + ".pth")
            if iteration % 10 == 0:
                progress_bar.set_postfix({"loss": f"{loss['total'].item():.1e}", "pts": f"{gaussians.get_density.shape[0]:2.1e}", "conf_avg": f"{gaussians.get_roi_confidence.mean().item():.2f}" if roi_management_active and gaussians.get_roi_confidence.numel() > 0 else "N/A"})
                progress_bar.update(10)
            if iteration == opt.iterations: progress_bar.close()
            metrics = {}
            for l in loss: metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups: metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            if roi_management_active and gaussians.get_roi_confidence.numel() > 0:
                metrics["confidence_avg"] = gaussians.get_roi_confidence.mean().item()
                metrics["confidence_max"] = gaussians.get_roi_confidence.max().item()
                metrics["confidence_min"] = gaussians.get_roi_confidence.min().item()
            training_report(tb_writer, iteration, metrics, iter_start.elapsed_time(iter_end), testing_iterations, scene, lambda x, y: render(x, y, pipe), queryfunc)

def training_report(tb_writer, iteration, metrics_train, elapsed, testing_iterations, scene: Scene, renderFunc, queryFunc):
    if tb_writer:
        for key in list(metrics_train.keys()): tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar("train/total_points", scene.gaussians.get_xyz.shape[0], iteration)
    if iteration in testing_iterations:
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()
        validation_configs = [{"name": "render_train", "cameras": scene.getTrainCameras()}, {"name": "render_test", "cameras": scene.getTestCameras()}]
        psnr_2d, ssim_2d, lpips_2d = None, None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images, gt_images, image_show_2d = [], [], []
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(viewpoint, scene.gaussians)["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx: image_show_2d.append(torch.from_numpy(show_two_slice(gt_image[0], image[0], f"{viewpoint.image_name} gt", f"{viewpoint.image_name} render", save=True)))
                images, gt_images = torch.concat(images, 0).permute(1, 2, 0), torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                # 【修复】现在直接传递 "lpips" 字符串
                lpips_2d, lpips_2d_projs = metric_proj(gt_images, images, "lpips")
                eval_dict_2d = {"psnr_2d": psnr_2d, "ssim_2d": ssim_2d, "lpips_2d": lpips_2d, "psnr_2d_projs": psnr_2d_projs, "ssim_2d_projs": ssim_2d_projs, "lpips_2d_projs": lpips_2d_projs}
                with open(osp.join(eval_save_path, f"eval2d_{config['name']}.yml"), "w") as f: yaml.dump(eval_dict_2d, f, default_flow_style=False, sort_keys=False)
                if tb_writer:
                    if image_show_2d:
                        image_show_2d_tensor = torch.from_numpy(np.concatenate(image_show_2d, axis=0))[None].permute([0, 3, 1, 2])
                        tb_writer.add_images(config["name"] + f"/{viewpoint.image_name}", image_show_2d_tensor, global_step=iteration)
                    tb_writer.add_scalar(config["name"] + "/psnr_2d", psnr_2d, iteration); tb_writer.add_scalar(config["name"] + "/ssim_2d", ssim_2d, iteration); tb_writer.add_scalar(config["name"] + "/lpips_2d", lpips_2d, iteration)
        vol_pred = queryFunc(scene.gaussians)["vol"]; vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr"); ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        # 【修复】现在直接传递 "lpips" 字符串
        lpips_3d, lpips_3d_axis = metric_vol(vol_gt, vol_pred, "lpips")
        eval_dict = {"psnr_3d": psnr_3d, "ssim_3d": ssim_3d, "lpips_3d": lpips_3d, "ssim_3d_x": ssim_3d_axis[0], "ssim_3d_y": ssim_3d_axis[1], "ssim_3d_z": ssim_3d_axis[2], "lpips_3d_x": lpips_3d_axis[0] if lpips_3d_axis else None, "lpips_3d_y": lpips_3d_axis[1] if lpips_3d_axis else None, "lpips_3d_z": lpips_3d_axis[2] if lpips_3d_axis else None}
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f: yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        if tb_writer:
            image_show_3d = np.concatenate([show_two_slice(vol_gt[..., i], vol_pred[..., i], f"slice {i} gt", f"slice {i} pred", vmin=vol_gt[..., i].min(), vmax=vol_gt[..., i].max(), save=True) for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]], axis=0)
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images("reconstruction/slice-gt_pred_diff", image_show_3d, global_step=iteration)
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration); tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration); tb_writer.add_scalar("reconstruction/lpips_3d", lpips_3d, iteration)
        tqdm.write(f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, lpips3d {lpips_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}, lpips2d {lpips_2d:.3f}")
        if tb_writer: tb_writer.add_histogram("scene/density_histogram", scene.gaussians.get_density, iteration)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0, help="指定要使用的GPU ID")
    parser.add_argument("--plot_confidence", action="store_true", help="在训练结束时生成并保存置信度分布直方图。")
    parser.add_argument("--roi_update_start_iter", type=int, default=5000,
                        help="Iteration to start ROI confidence updates.")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(1)

    safe_state(args.quiet)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"
    print(f"使用设备: {device}")
    if device.startswith("cuda"): torch.cuda.set_device(args.gpu_id)

    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()): args_dict[key] = cfg[key]

    tb_writer = prepare_output_and_logger(args)
    print("Optimizing " + args.model_path)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # 【修改】将args也传递给training函数
    training(
        args, dataset, opt, pipe, tb_writer,
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint
    )

    # 在点云可视化之前执行CT切片可视化
    print("训练完成，准备执行CT切片可视化...")

    # 获取最后一次迭代的模型结果
    if args.save_iterations:
        latest_iter = max(args.save_iterations)

        # 尝试加载体积数据
        vol_paths = []

        # 优先从eval目录查找
        eval_dir = os.path.join(args.model_path, "eval", f"iter_{latest_iter}")
        if os.path.exists(eval_dir):
            vol_pred_path = os.path.join(eval_dir, "vol_pred.npy")
            vol_gt_path = os.path.join(eval_dir, "vol_gt.npy")
            if os.path.exists(vol_pred_path) and os.path.exists(vol_gt_path):
                vol_paths = [vol_pred_path, vol_gt_path]

        # 如果eval目录没有找到，从point_cloud目录查找
        if not vol_paths:
            point_cloud_dir = os.path.join(args.model_path, "point_cloud", f"iteration_{latest_iter}")
            if os.path.exists(point_cloud_dir):
                vol_pred_path = os.path.join(point_cloud_dir, "vol_pred.npy")
                vol_gt_path = os.path.join(point_cloud_dir, "vol_gt.npy")
                if os.path.exists(vol_pred_path) and os.path.exists(vol_gt_path):
                    vol_paths = [vol_pred_path, vol_gt_path]

        # 如果还没找到，需要重新生成体积数据
        if not vol_paths:
            print("找不到体积数据，尝试重新生成...")

        # 如果成功找到或生成了体积数据，进行可视化
        if vol_paths:
            print(f"加载体积数据: {vol_paths}")
            vol_pred = np.load(vol_paths[0])
            vol_gt = np.load(vol_paths[1])

            print(f"体积数据形状 - 预测: {vol_pred.shape}, 真实: {vol_gt.shape}")

            # 创建保存目录
            ct_viz_dir = os.path.join(args.model_path, "ct_viz")
            os.makedirs(ct_viz_dir, exist_ok=True)

            # 创建三个轴向的目录
            axial_dir = os.path.join(ct_viz_dir, "axial")
            coronal_dir = os.path.join(ct_viz_dir, "coronal")
            sagittal_dir = os.path.join(ct_viz_dir, "sagittal")

            for d in [axial_dir, coronal_dir, sagittal_dir]:
                os.makedirs(d, exist_ok=True)

            # 归一化体积数据用于可视化
            vol_pred_norm = vol_pred.copy()
            vol_gt_norm = vol_gt.copy()

            if vol_pred_norm.max() > vol_pred_norm.min():
                vol_pred_norm = (vol_pred_norm - vol_pred_norm.min()) / (vol_pred_norm.max() - vol_pred_norm.min())
            if vol_gt_norm.max() > vol_gt_norm.min():
                vol_gt_norm = (vol_gt_norm - vol_gt_norm.min()) / (vol_gt_norm.max() - vol_gt_norm.min())

            print(
                f"归一化后 - 预测: {vol_pred_norm.min():.4f}-{vol_pred_norm.max():.4f}, 真实: {vol_gt_norm.min():.4f}-{vol_gt_norm.max():.4f}")

            # 生成CT切片可视化
            import matplotlib.pyplot as plt

            # 为每个方向选择5个切片
            show_slice = 5

            # 获取每个维度的大小
            depth_z = vol_gt_norm.shape[2]  # Z轴（轴状位）
            depth_y = vol_gt_norm.shape[1]  # Y轴（冠状位）
            depth_x = vol_gt_norm.shape[0]  # X轴（矢状位）

            # 计算每个方向的步长
            step_z = max(1, depth_z // show_slice)
            step_y = max(1, depth_y // show_slice)
            step_x = max(1, depth_x // show_slice)


            # 定义处理每个轴向的函数，以避免代码重复
            def process_slices(vol_gt, vol_pred, axis_name, axis_idx, step, output_dir):
                plt.figure(figsize=(15, 6))
                combined_slices = []

                for i in range(show_slice):
                    slice_idx = min((i + 1) * step - 1, axis_idx - 1)  # 确保不超出边界

                    # 根据轴向获取切片
                    if axis_name == "axial":
                        gt_slice = vol_gt[..., slice_idx]
                        pred_slice = vol_pred[..., slice_idx]
                    elif axis_name == "coronal":
                        gt_slice = vol_gt[:, slice_idx, :]
                        pred_slice = vol_pred[:, slice_idx, :]
                    else:  # sagittal
                        gt_slice = vol_gt[slice_idx, :, :]
                        pred_slice = vol_pred[slice_idx, :, :]

                    # 创建组合图像，GT在上，预测在下
                    plt.subplot(2, show_slice, i + 1)
                    plt.imshow(gt_slice, cmap='gray')
                    plt.title(f"GT {axis_name.capitalize()} {slice_idx}")
                    plt.axis('off')

                    plt.subplot(2, show_slice, i + 1 + show_slice)
                    plt.imshow(pred_slice, cmap='gray')
                    plt.title(f"Pred {axis_name.capitalize()} {slice_idx}")
                    plt.axis('off')

                    # 保存单独的切片图像
                    plt.figure(figsize=(10, 10))
                    plt.subplot(2, 1, 1)
                    plt.imshow(gt_slice, cmap='gray')
                    plt.title(f"GT {axis_name.capitalize()} {slice_idx}")
                    plt.axis('off')

                    plt.subplot(2, 1, 2)
                    plt.imshow(pred_slice, cmap='gray')
                    plt.title(f"Pred {axis_name.capitalize()} {slice_idx}")
                    plt.axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{axis_name}_slice_{slice_idx}.png"), dpi=200,
                                bbox_inches='tight')
                    plt.close()

                    # 创建与示例代码一致的可视化
                    combined_slice = np.vstack([gt_slice, pred_slice])
                    combined_slices.append(combined_slice)

                # 返回到主图形
                plt.figure(figsize=(15, 6))

                # 将所有切片水平连接
                all_slices = np.hstack(combined_slices)

                # 保存组合图像
                plt.figure(figsize=(15, 6))
                plt.imshow(all_slices, cmap='gray')
                plt.title(f"CT {axis_name.capitalize()} Slices (Top: GT, Bottom: Prediction)")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"all_{axis_name}_slices.png"), dpi=200, bbox_inches='tight')
                plt.close()

                return all_slices


            # 处理三个方向的切片
            print("生成轴状位(Axial)切片...")
            axial_slices = process_slices(vol_gt_norm, vol_pred_norm, "axial", depth_z, step_z, axial_dir)

            print("生成冠状位(Coronal)切片...")
            coronal_slices = process_slices(vol_gt_norm, vol_pred_norm, "coronal", depth_y, step_y, coronal_dir)

            print("生成矢状位(Sagittal)切片...")
            sagittal_slices = process_slices(vol_gt_norm, vol_pred_norm, "sagittal", depth_x, step_x, sagittal_dir)

            # 创建一个综合可视化，包含所有三个方向
            plt.figure(figsize=(15, 15))

            # 将三个方向的切片垂直堆叠
            all_directions = np.vstack([axial_slices, coronal_slices, sagittal_slices])

            plt.imshow(all_directions, cmap='gray')
            plt.title("CT Slices - All Directions\n(Top: Axial, Middle: Coronal, Bottom: Sagittal)")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(ct_viz_dir, "all_directions.png"), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"CT切片可视化已保存到: {ct_viz_dir}")

            # 添加到TensorBoard
            if tb_writer:
                # 单独添加每个方向
                for name, slices in [("axial", axial_slices), ("coronal", coronal_slices),
                                     ("sagittal", sagittal_slices)]:
                    slices_tensor = torch.from_numpy(slices)[None, ..., None]
                    slices_rgb = torch.cat([slices_tensor, slices_tensor, slices_tensor], dim=3)
                    tb_writer.add_image(f"ct_viz/{name}_slices", slices_rgb[0], global_step=latest_iter,
                                        dataformats="HWC")

                # 添加所有方向的综合图
                all_dir_tensor = torch.from_numpy(all_directions)[None, ..., None]
                all_dir_rgb = torch.cat([all_dir_tensor, all_dir_tensor, all_dir_tensor], dim=3)
                tb_writer.add_image("ct_viz/all_directions", all_dir_rgb[0], global_step=latest_iter, dataformats="HWC")

    # 运行点云可视化
    from point_cloud_visualizer import on_training_finish

    on_training_finish(args.model_path)

    # 完成
    print("所有处理步骤完成。")