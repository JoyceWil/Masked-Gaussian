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
from r2_gaussian.utils.mask_generator import check_and_generate_masks


def plot_and_save_confidence_distribution(confidence_data, iteration, output_dir, protect_threshold,
                                          candidate_threshold):
    if confidence_data.size == 0:
        print("[WARNING] Confidence data is empty, skipping plot generation.")
        return

    plt.figure(figsize=(14, 8))
    plt.hist(confidence_data, bins=150, color='deepskyblue', edgecolor='black', alpha=0.7, log=True)
    plt.title(f'ROI Confidence Distribution at Iteration {iteration} (Log Scale)', fontsize=16)
    plt.xlabel('Confidence Value', fontsize=12)
    plt.ylabel('Number of Gaussian Points (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.axvline(x=protect_threshold, color='green', linestyle='--', linewidth=2,
                label=f'Protect Threshold ({protect_threshold})')
    plt.axvline(x=candidate_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Candidate Threshold ({candidate_threshold})')

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
    save_path = osp.join(output_dir, f"confidence_distribution_iter_{iteration}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    tqdm.write(f"[INFO] Confidence distribution plot saved to: {save_path}")


def training(args, dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, tb_writer, testing_iterations,
             saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    scene = Scene(dataset, shuffle=False)
    roi_management_active = scene.use_roi_masks
    if roi_management_active:
        print("ROI置信度管理已在训练流程中激活。")

        confidence_thresholds = {"prune": opt.roi_prune_threshold, "protect": opt.roi_protect_threshold}
        rewards = {"penalty": opt.roi_background_reward, "standard": opt.roi_standard_reward,
                   "core_bonus": opt.roi_core_bonus_reward}

    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = opt.densify_scale_threshold * volume_to_world if opt.densify_scale_threshold else None
    scale_bound = np.array(
        [dataset.scale_min, dataset.scale_max]) * volume_to_world if dataset.scale_min and dataset.scale_max else None
    queryfunc = lambda x: query(x, scanner_cfg["offOrigin"], scanner_cfg["nVoxel"], scanner_cfg["sVoxel"], pipe)
    gaussians = GaussianModel(scale_bound)

    initialize_gaussian(gaussians, dataset, None)

    if opt.intelligent_confidence_mode != 'none':
        print(f"模式: {opt.intelligent_confidence_mode}")

        with torch.no_grad():
            initial_xyz = gaussians.get_xyz
            fdk_volume = scene.vol_gt.to(initial_xyz.device)
            grid_center = scene.grid_center.to(initial_xyz.device)
            voxel_size = scene.voxel_size.to(initial_xyz.device)
            grid_dims = torch.tensor(fdk_volume.shape, device=initial_xyz.device)

            relative_coords = initial_xyz - grid_center
            grid_indices = torch.round((relative_coords / voxel_size) + (grid_dims / 2.0)).long()

            valid_mask = (grid_indices[:, 2] >= 0) & (grid_indices[:, 2] < grid_dims[0]) & \
                         (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < grid_dims[1]) & \
                         (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < grid_dims[2])

            absolute_indices_of_valid_points = torch.where(valid_mask)[0]
            valid_grid_indices = grid_indices[valid_mask]

            if absolute_indices_of_valid_points.numel() > 0:
                point_densities = fdk_volume[
                    valid_grid_indices[:, 2],  # Z
                    valid_grid_indices[:, 1],  # Y
                    valid_grid_indices[:, 0]  # X
                ]

                if opt.intelligent_confidence_mode == 'percentile':
                    percentile_to_keep = opt.intelligent_confidence_percentile
                    q = 100.0 - percentile_to_keep
                    densities_np = point_densities.cpu().numpy()
                    if densities_np.size > 0 and densities_np.max() > densities_np.min():
                        calculated_threshold = np.percentile(densities_np, q)
                    else:
                        calculated_threshold = densities_np[0] if densities_np.size > 0 else 0.0
                    print(
                        f"筛选密度最高的 {percentile_to_keep}% 点, 自动计算出的FDK密度阈值为: {calculated_threshold:.4f}")
                    core_points_mask = point_densities >= calculated_threshold
                elif opt.intelligent_confidence_mode == 'fixed':
                    fixed_threshold = opt.intelligent_confidence_threshold
                    print(f"使用固定的FDK密度阈值: {fixed_threshold:.4f}")
                    core_points_mask = point_densities > fixed_threshold
                else:
                    core_points_mask = torch.zeros_like(point_densities, dtype=torch.bool)

                target_indices = absolute_indices_of_valid_points[core_points_mask]
                initial_reward = opt.roi_core_bonus_reward
                gaussians._roi_confidence[target_indices] = initial_reward
                print(
                    f"操作完成: {target_indices.numel()} / {initial_xyz.shape[0]} 个点被识别为核心点，并赋予了 {initial_reward:.2f} 的初始置信度。")
            else:
                print("警告: 没有一个初始点落在FDK体积内，无法进行智能置信度初始化。")

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
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        active_sh_indices = None
        render_pkg = render(viewpoint_cam, gaussians, pipe, active_sh_indices=active_sh_indices)
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
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (bbox[1] - tv_vol_sVoxel - bbox[0]) * torch.rand(3,
                                                                                                             device=bbox.device)
            vol_pred = query(gaussians, tv_vol_center, tv_vol_nVoxel, tv_vol_sVoxel, pipe)["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv

        loss["total"].backward()
        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            if roi_management_active and iteration >= args.roi_update_start_iter and iteration % opt.roi_management_interval == 0:
                gaussians.update_roi_confidence(
                    render_pkg,
                    viewpoint_cam,
                    opt.roi_standard_reward,
                    opt.roi_core_bonus_reward,
                    opt.roi_background_reward
                )

            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            viewspace_point_tensor = render_pkg["viewspace_points"]

            if active_sh_indices is not None:
                full_visibility_filter = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
                full_visibility_filter[active_sh_indices[visibility_filter]] = True
                gaussians.max_radii2D[full_visibility_filter] = torch.max(
                    gaussians.max_radii2D[full_visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, full_visibility_filter)
            else:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration < opt.densify_until_iter:
                if (iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0):
                    gaussians.densify_and_prune(
                        opt,
                        max_scale,
                        densify_scale_threshold,
                        scene.scene_scale
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
                    scene.model_path,
                    opt.roi_protect_threshold,
                    opt.roi_candidate_threshold
                )

            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] 保存检查点")
                torch.save((gaussians.capture(), iteration), ckpt_save_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"loss": f"{loss['total'].item():.1e}", "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                     "conf_avg": f"{gaussians.get_roi_confidence.mean().item():.2f}" if roi_management_active and gaussians.get_roi_confidence.numel() > 0 else "N/A"})
                progress_bar.update(10)

            if iteration == opt.iterations: progress_bar.close()

            metrics = {}
            for l in loss: metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups: metrics[f"lr_{param_group['name']}"] = param_group[
                "lr"]
            if roi_management_active and gaussians.get_roi_confidence.numel() > 0:
                metrics["confidence_avg"] = gaussians.get_roi_confidence.mean().item()
                metrics["confidence_max"] = gaussians.get_roi_confidence.max().item()
                metrics["confidence_min"] = gaussians.get_roi_confidence.min().item()
            training_report(tb_writer, iteration, metrics, iter_start.elapsed_time(iter_end), testing_iterations, scene,
                            lambda x, y: render(x, y, pipe), queryfunc)


def training_report(tb_writer, iteration, metrics_train, elapsed, testing_iterations, scene: Scene, renderFunc,
                    queryFunc):
    if tb_writer:
        for key in list(metrics_train.keys()): tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar("train/total_points", scene.gaussians.get_xyz.shape[0], iteration)

    if iteration in testing_iterations:
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()
        validation_configs = [{"name": "render_train", "cameras": scene.getTrainCameras()},
                              {"name": "render_test", "cameras": scene.getTestCameras()}]
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
                    if tb_writer and idx in show_idx: image_show_2d.append(torch.from_numpy(
                        show_two_slice(gt_image[0], image[0], f"{viewpoint.image_name} gt",
                                       f"{viewpoint.image_name} render", save=True)))

                images, gt_images = torch.concat(images, 0).permute(1, 2, 0), torch.concat(gt_images, 0).permute(1, 2,
                                                                                                                 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                lpips_2d, lpips_2d_projs = metric_proj(gt_images, images, "lpips")
                eval_dict_2d = {"psnr_2d": psnr_2d, "ssim_2d": ssim_2d, "lpips_2d": lpips_2d,
                                "psnr_2d_projs": psnr_2d_projs, "ssim_2d_projs": ssim_2d_projs,
                                "lpips_2d_projs": lpips_2d_projs}
                with open(osp.join(eval_save_path, f"eval2d_{config['name']}.yml"), "w") as f:
                    yaml.dump(eval_dict_2d, f, default_flow_style=False, sort_keys=False)

                if tb_writer:
                    if image_show_2d:
                        image_show_2d_tensor = torch.from_numpy(np.concatenate(image_show_2d, axis=0))[None].permute(
                            [0, 3, 1, 2])
                        tb_writer.add_images(config["name"] + f"/{viewpoint.image_name}", image_show_2d_tensor,
                                             global_step=iteration)
                    tb_writer.add_scalar(config["name"] + "/psnr_2d", psnr_2d, iteration);
                    tb_writer.add_scalar(config["name"] + "/ssim_2d", ssim_2d, iteration);
                    tb_writer.add_scalar(config["name"] + "/lpips_2d", lpips_2d, iteration)

        vol_pred = queryFunc(scene.gaussians)["vol"];
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr");
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        lpips_3d, lpips_3d_axis = metric_vol(vol_gt, vol_pred, "lpips")
        eval_dict = {"psnr_3d": psnr_3d, "ssim_3d": ssim_3d, "lpips_3d": lpips_3d, "ssim_3d_x": ssim_3d_axis[0],
                     "ssim_3d_y": ssim_3d_axis[1], "ssim_3d_z": ssim_3d_axis[2],
                     "lpips_3d_x": lpips_3d_axis[0] if lpips_3d_axis else None,
                     "lpips_3d_y": lpips_3d_axis[1] if lpips_3d_axis else None,
                     "lpips_3d_z": lpips_3d_axis[2] if lpips_3d_axis else None}
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

        if tb_writer:
            image_show_3d = np.concatenate([show_two_slice(vol_gt[..., i], vol_pred[..., i], f"slice {i} gt",
                                                           f"slice {i} pred", vmin=vol_gt[..., i].min(),
                                                           vmax=vol_gt[..., i].max(), save=True) for i in
                                            np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]], axis=0)
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images("reconstruction/slice-gt_pred_diff", image_show_3d, global_step=iteration)
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration);
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration);
            tb_writer.add_scalar("reconstruction/lpips_3d", lpips_3d, iteration)

        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, lpips3d {lpips_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}, lpips2d {lpips_2d:.3f}")
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
    parser.add_argument('--auto_mask', action='store_true', help="如果设置此项，将自动检查并生成ROI掩码。")
    parser.add_argument('--no_previews', dest='save_previews', action='store_false',
                        help="设置此项后，将不生成PNG格式的掩码预览图，只生成NPY文件。")
    parser.set_defaults(save_previews=True)
    parser.add_argument("--noisy_view_indices", type=int, nargs='+', default=None,
                        help="一个或多个训练视图的索引，只有这些视图会被添加噪声。如果未提供，则噪声会应用于所有视图。")

    parser.add_argument('--mask_bone_wl', type=int, default=380, help="核心骨架掩码的窗位 (Window Level)。")
    parser.add_argument('--mask_bone_ww', type=int, default=380, help="核心骨架掩码的窗宽 (Window Width)。")
    parser.add_argument('--mask_tissue_wl', type=int, default=40, help="软组织掩码的窗位 (Window Level)。")
    parser.add_argument('--mask_tissue_ww', type=int, default=400, help="软组织掩码的窗宽 (Window Width)。")

    op_group = parser.add_argument_group("Optimization", "Optimization parameters")

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

    if args.auto_mask:
        print("\n--- [自动化ROI掩码处理模块] ---")
        if not hasattr(args, 'source_path') or not args.source_path:
            print("错误: 必须在配置文件中定义 'source_path' 才能使用 --auto_mask。")
            sys.exit(1)

        # 1. 定义预期的掩码目录路径
        base_mask_dir = osp.join(args.source_path, "masks")
        expected_soft_mask_dir = osp.join(base_mask_dir, "tissue_masks_npy")
        expected_core_mask_dir = osp.join(base_mask_dir, "bone_masks_npy")

        # 2. 检查这些目录是否已经存在且非空
        if osp.exists(expected_soft_mask_dir) and osp.exists(expected_core_mask_dir) and \
                len(os.listdir(expected_soft_mask_dir)) > 0 and len(os.listdir(expected_core_mask_dir)) > 0:

            print("   - 状态: 已找到预先生成的软掩码目录，将直接使用。")
            args.soft_mask_dir = expected_soft_mask_dir
            args.core_mask_dir = expected_core_mask_dir

        else:
            # 3. 如果目录不存在或为空，则进入生成流程
            print("   - 状态: 未找到有效的软掩码目录，开始自动生成...")
            try:
                # 调用我们新的、基于窗宽窗位的生成函数
                generated_soft_dir, generated_core_dir = check_and_generate_masks(
                    source_path=args.source_path,
                    bone_wl=args.mask_bone_wl,
                    bone_ww=args.mask_bone_ww,
                    tissue_wl=args.mask_tissue_wl,
                    tissue_ww=args.mask_tissue_ww,
                    save_png_previews=args.save_previews
                )

                args.soft_mask_dir = generated_soft_dir
                args.core_mask_dir = generated_core_dir
                print("   - 生成成功！")

            except Exception as e:
                print(f"   - 错误：掩码生成失败: {e}")
                import traceback

                traceback.print_exc()
                print("   - 训练已终止。")
                sys.exit(1)

        # 4. 最终确认使用的目录
        print("   - 配置已确认，将使用以下NPY掩码目录:")
        print(f"     - 软组织 (Soft): {args.soft_mask_dir}")
        print(f"     - 核心骨架 (Core): {args.core_mask_dir}")

    tb_writer = prepare_output_and_logger(args)
    print("Optimizing " + args.model_path)
    final_model_params = lp.extract(args)
    print(f"  [INFO] Effective noise level for this run: {final_model_params.noise_level}")
    if final_model_params.noise_level is not None and final_model_params.noise_level > 0:
        print("  [STATUS] Noise injection is ACTIVE.")
        print("           Look for '[DEBUG] 噪声可视化已保存！' message during data loading.")
    else:
        print("  [STATUS] Noise injection is INACTIVE (level is 0 or None).")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = final_model_params
    opt = op.extract(args)
    pipe = pp.extract(args)

    training(
        args, dataset, opt, pipe, tb_writer,
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint
    )

    if args.save_iterations:
        # 获取最终迭代次数和TensorBoard writer
        latest_iter = max(args.save_iterations)

        # 运行点云可视化
        from point_cloud_visualizer import on_training_finish

        # 使用新的函数签名，传递所有需要的参数
        on_training_finish(
            model_path=args.model_path,
            latest_iter=latest_iter,
            tb_writer=tb_writer
        )

    # 完成
    print("\n所有处理步骤完成。")