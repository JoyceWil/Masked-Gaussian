import os
import os.path as osp
import torch
import torchvision
from random import randint
import sys
import logging
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import yaml
import datetime
import json

sys.path.append("./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss, gradient_difference_loss
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice
from r2_gaussian.utils.mask_generator import check_and_generate_masks
from r2_gaussian.utils.hu_utils import convert_mu_to_hu_torch, apply_windowing_torch


def normalize_to_4d_bchw(tensor):
    """将任意图像张量强制转换为标准的 (B, C, H, W) 格式"""
    if tensor is None: return None
    if not isinstance(tensor, torch.Tensor): tensor = torch.tensor(tensor, device="cuda")

    tensor = tensor.to("cuda")

    # 确保至少是4D
    while tensor.dim() < 4:
        tensor = tensor.unsqueeze(0)

    # (B, H, W, C) -> (B, C, H, W)
    if tensor.shape[1] != 1 and tensor.shape[3] == 1:
        tensor = tensor.permute(0, 3, 1, 2)

    # 确保通道数为1
    if tensor.shape[1] != 1:
        tensor = tensor[:, 0:1, :, :]

    return tensor


def _generate_and_save_single_mask(render_pkg, viewpoint_cam, output_dir, wl, ww, mask_type, iteration):
    """
    【内部辅助函数】严格根据传入的渲染结果(render_pkg)生成并保存掩码。
    绝不使用 viewpoint_cam 中预存的任何掩码数据。
    """
    if wl is None or ww is None:
        return

    save_dir = osp.join(output_dir, f"dynamic_{mask_type}_masks", f"iter_{iteration:07d}")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # 步骤4.1: 从渲染结果包中提取μ图像
        rendered_mu_image = normalize_to_4d_bchw(render_pkg["render"])

        # 步骤4.2: μ -> HU -> Windowing，生成动态软掩码
        rendered_hu = convert_mu_to_hu_torch(rendered_mu_image)
        dynamic_mask = apply_windowing_torch(rendered_hu, wl, ww)

        # 步骤5: 保存这张全新的、动态生成的掩码
        mask_np = dynamic_mask.squeeze().cpu().numpy()
        base_name = osp.splitext(viewpoint_cam.image_name)[0]
        npy_path = osp.join(save_dir, f"{base_name}.npy")
        png_path = osp.join(save_dir, f"{base_name}.png")

        np.save(npy_path, mask_np)
        torchvision.utils.save_image(dynamic_mask.squeeze(0), png_path, normalize=False)


### --- 【最终正确版 2/3】主函数：遍历、渲染、生成 --- ###
def generate_all_dynamic_masks(iteration, scene, gaussians, pipe, tissue_params, bone_params):
    """
    为【所有】训练视角，通过【实时渲染】来生成软组织和核心骨架的动态掩码。
    """
    tqdm.write(f"\n[实时诊断] Iter {iteration}: 开始为所有 {len(scene.getTrainCameras())} 个训练视角生成动态掩码...")

    if not tissue_params and not bone_params:
        tqdm.write("[警告] 无法生成动态掩码，因为软组织和核心骨架的窗参数都无效。")
        return

    # 步骤1: 获取当前的高斯模型 (通过参数 gaussians 传入)
    for viewpoint_cam in tqdm(scene.getTrainCameras(), desc=f"  - 实时渲染并生成掩码 (Iter {iteration})"):
        # 步骤2: 选择一个训练集里的相机视角 (viewpoint_cam)

        # 步骤3: 从此视角“拍摄”高斯模型，得到渲染包 render_pkg
        render_pkg = render(viewpoint_cam, gaussians, pipe)

        # 步骤4: 对渲染结果进行处理，生成并保存两种掩码
        if tissue_params:
            _generate_and_save_single_mask(
                render_pkg, viewpoint_cam, scene.model_path,
                tissue_params['wl'], tissue_params['ww'], 'tissue', iteration
            )
        if bone_params:
            _generate_and_save_single_mask(
                render_pkg, viewpoint_cam, scene.model_path,
                bone_params['wl'], bone_params['ww'], 'bone', iteration
            )

    tqdm.write(f"[实时诊断] Iter {iteration}: 所有动态掩码生成完毕。")


def plot_and_save_confidence_distribution(confidences_data, iteration, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure(figsize=(16, 9))
    confidences_np = confidences_data.flatten()
    total_points = len(confidences_np)
    if total_points == 0:
        logging.warning("无法绘制置信度分布图，因为模型中没有点。")
        plt.close()
        return
    mean_conf = np.mean(confidences_np)
    median_conf = np.median(confidences_np)
    std_conf = np.std(confidences_np)
    min_conf = np.min(confidences_np)
    max_conf = np.max(confidences_np)
    num_bins = 150
    stats_text = (
        f"Total Points: {total_points}\n"
        f"Mean: {mean_conf:.3f}\n"
        f"Median: {median_conf:.3f}\n"
        f"Std Dev: {std_conf:.3f}\n"
        f"Min: {min_conf:.3f}\n"
        f"Max: {max_conf:.3f}"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)
    plt.hist(confidences_np, bins=num_bins, range=(-5, 5), log=True, color='deepskyblue', alpha=0.8, edgecolor='black')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='deepskyblue', edgecolor='black', alpha=0.8, label=f'Confidence Bins')]
    plt.legend(handles=legend_elements, fontsize=12)
    plt.title(f"ROI Confidence Distribution at Iteration {iteration} (Log Scale)", fontsize=18)
    plt.xlabel("Confidence Value", fontsize=14)
    plt.ylabel("Number of Gaussian Points (Log Scale)", fontsize=14)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    save_path = os.path.join(output_dir, f"confidence_distribution_iter_{iteration}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Confidence distribution plot saved to: {save_path}")


def training(args, dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, tb_writer, testing_iterations,
             saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    scene = Scene(dataset, shuffle=False)

    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox.to("cuda")
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min and dataset.scale_max:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    gaussians = GaussianModel(scale_bound)
    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    tissue_params, bone_params = None, None
    if args.save_dynamic_mask_on_test:  # <--- 条件改变
        print("\n--- [窗参数加载模块] ---")
        mask_dir = os.path.join(dataset.source_path, 'masks')
        windows_json_path = os.path.join(mask_dir, 'optimal_windows.json')

        if os.path.exists(windows_json_path):
            with open(windows_json_path, 'r') as f:
                optimal_windows = json.load(f)

            if 'tissue' in optimal_windows and 'wl' in optimal_windows['tissue'] and 'ww' in optimal_windows['tissue']:
                tissue_params = optimal_windows['tissue']
                print(f"   - 成功加载软组织窗参数: WL={tissue_params['wl']}, WW={tissue_params['ww']}")
            else:
                print("   - 警告: 'optimal_windows.json' 中未找到有效的 'tissue' 窗参数。")

            if 'bone' in optimal_windows and 'wl' in optimal_windows['bone'] and 'ww' in optimal_windows['bone']:
                bone_params = optimal_windows['bone']
                print(f"   - 成功加载核心骨架窗参数: WL={bone_params['wl']}, WW={bone_params['ww']}")
            else:
                print("   - 警告: 'optimal_windows.json' 中未找到有效的 'bone' 窗参数。")
        else:
            print(f"   - 警告: 未找到 optimal_windows.json 文件 (路径: {windows_json_path})，无法生成动态掩码。")


    use_weighted_2d_loss = True
    use_charbonnier = False
    charbonnier_eps = 1e-3

    w_bg = 1.0
    w_soft = 1.3
    w_core_start = 1.6
    w_core_end = 2.0
    w_core_start_iter = max(1, args.roi_update_start_iter // 2)
    w_core_end_iter = max(w_core_start_iter + 1000, args.roi_update_start_iter + 3000)

    def get_core_weight(it):
        if it <= w_core_start_iter: return w_core_start
        if it >= w_core_end_iter: return w_core_end
        t = (it - w_core_start_iter) / float(w_core_end_iter - w_core_start_iter)
        return w_core_start + t * (w_core_end - w_core_start)

    def weighted_l1_loss(pred, gt, weights):
        diff = torch.abs(pred - gt)
        return torch.mean(diff * weights)

    def weighted_charbonnier_loss(pred, gt, weights, eps=charbonnier_eps):
        diff2 = (pred - gt) ** 2
        loss = torch.sqrt(diff2 + eps * eps)
        return torch.mean(loss * weights)

    if opt.intelligent_confidence_mode != 'none':
        print(f"\n--- [智能置信度初始化模块] 模式: {opt.intelligent_confidence_mode} ---")
        with torch.no_grad():
            initial_xyz = gaussians.get_xyz
            device = initial_xyz.device
            fdk_volume = scene.vol_gt.to(device)
            grid_center = scene.grid_center.to(device)
            voxel_size = scene.voxel_size.to(device)
            grid_dims = torch.tensor(fdk_volume.shape, device=device)
            grid_indices = torch.round((initial_xyz - grid_center) / voxel_size + (grid_dims.flip(0) / 2.0)).long()
            valid_mask = (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < grid_dims[2]) & \
                         (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < grid_dims[1]) & \
                         (grid_indices[:, 2] >= 0) & (grid_indices[:, 2] < grid_dims[0])
            absolute_indices_of_valid_points = torch.where(valid_mask)[0]
            valid_grid_indices = grid_indices[valid_mask]
            if absolute_indices_of_valid_points.numel() > 0:
                point_densities = fdk_volume[
                    valid_grid_indices[:, 2], valid_grid_indices[:, 1], valid_grid_indices[:, 0]]
                if opt.intelligent_confidence_mode == 'percentile':
                    q = 100.0 - opt.intelligent_confidence_percentile
                    calculated_threshold = np.percentile(point_densities.cpu().numpy(), q)
                    core_points_mask = point_densities >= calculated_threshold
                    print(
                        f"   - 百分位阈值 ({opt.intelligent_confidence_percentile}%): 计算得到FDK密度阈值为 {calculated_threshold:.4f}")
                elif opt.intelligent_confidence_mode == 'fixed':
                    core_points_mask = point_densities > opt.intelligent_confidence_threshold
                    print(f"   - 固定阈值: 使用FDK密度阈值 {opt.intelligent_confidence_threshold:.4f}")
                target_indices = absolute_indices_of_valid_points[core_points_mask]
                initial_reward = opt.roi_init_bonus
                gaussians._roi_confidence[target_indices] = initial_reward
                print(
                    f"   - 操作完成: {target_indices.numel()} / {initial_xyz.shape[0]} 个点被识别为核心点，并赋予了 {initial_reward:.2f} 的初始置信度。")
            else:
                print("   - 警告: 没有一个初始点落在FDK体积内，无法进行智能置信度初始化。")
        print("--- [模块结束] ---\n")

    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        device = bbox.device
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size], device=device)
        tv_vol_sVoxel = (torch.tensor(scanner_cfg["dVoxel"], device=device, dtype=torch.float32) * tv_vol_nVoxel)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )



        with torch.no_grad():
            if args.enable_confidence_update and iteration >= args.roi_update_start_iter and iteration % opt.roi_management_interval == 0:
                gaussians.update_roi_confidence(render_pkg, viewpoint_cam, opt)  ### --- 修改: 将args改为opt --- ###

        gt_image = normalize_to_4d_bchw(viewpoint_cam.original_image)
        image = normalize_to_4d_bchw(render_pkg["render"])

        base_weights = torch.ones_like(image)
        if use_weighted_2d_loss and hasattr(viewpoint_cam, 'soft_mask') and hasattr(viewpoint_cam, 'core_mask'):
            soft_m = normalize_to_4d_bchw(viewpoint_cam.soft_mask)
            core_m = normalize_to_4d_bchw(viewpoint_cam.core_mask)

            w_core_cur = get_core_weight(iteration)
            base_weights = (w_bg
                            + (w_soft - w_bg) * soft_m
                            + (w_core_cur - w_soft) * core_m)
            base_weights = torch.clamp(base_weights, min=0.5, max=3.0)

        final_weights = base_weights
        diff_map_for_vis = None
        if args.use_boost_loss and tissue_wl is not None:
            with torch.no_grad():
                try:
                    rendered_hu = convert_mu_to_hu_torch(image)
                    gt_hu = convert_mu_to_hu_torch(gt_image)
                    rendered_dist_map = apply_windowing_torch(rendered_hu, tissue_wl, tissue_ww)
                    gt_dist_map = apply_windowing_torch(gt_hu, tissue_wl, tissue_ww)
                    diff_map = torch.abs(rendered_dist_map - gt_dist_map)
                    diff_map_for_vis = diff_map.clone()
                    boost_map = torch.ones_like(diff_map)
                    boost_map[diff_map > args.boost_threshold] = args.boost_factor
                    final_weights = base_weights * boost_map
                except Exception as e:
                    if iteration % 500 == 0: print(f"\n[警告] 计算自适应权重时出错: {e}")
                    final_weights = base_weights

        if use_charbonnier:
            render_loss = weighted_charbonnier_loss(image, gt_image, final_weights)
        else:
            render_loss = weighted_l1_loss(image, gt_image, final_weights)

        loss = {"total": 0.0}
        loss["render"] = render_loss
        loss["total"] = (1.0 - opt.lambda_dssim) * render_loss

        if hasattr(args, 'lambda_proj_consistency') and args.lambda_proj_consistency > 0 and tissue_wl is not None:
            try:
                rendered_hu = convert_mu_to_hu_torch(image)
                gt_hu = convert_mu_to_hu_torch(gt_image)
                rendered_dist_map = apply_windowing_torch(rendered_hu, tissue_wl, tissue_ww)
                gt_dist_map = apply_windowing_torch(gt_hu, tissue_wl, tissue_ww)
                proj_consistency_loss = torch.nn.functional.mse_loss(rendered_dist_map, gt_dist_map)
                loss["proj_consistency"] = proj_consistency_loss
                loss["total"] = loss["total"] + args.lambda_proj_consistency * proj_consistency_loss
            except Exception as e:
                if iteration % 500 == 0: print(f"\n[警告] 计算投影一致性损失时出错: {e}")

        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
        if opt.lambda_gdl > 0:
            loss_gdl = gradient_difference_loss(image, gt_image, alpha=1.0)
            loss["gdl"] = loss_gdl
            loss["total"] = loss["total"] + opt.lambda_gdl * loss_gdl
        if use_tv:
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (bbox[1] - tv_vol_sVoxel - bbox[0]) * torch.rand(3,
                                                                                                             device=bbox.device)
            vol_pred = query(gaussians, tv_vol_center, tv_vol_nVoxel, tv_vol_sVoxel, pipe)["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv

        if opt.lambda_conf_punish > 0 and iteration >= opt.conf_punish_start_iter:
            # 1. 找出所有置信度低于阈值的“坏点”
            bad_points_mask = gaussians.get_roi_confidence.squeeze(-1) < opt.confidence_punish_threshold

            # 2. 仅在存在坏点时计算损失
            if bad_points_mask.any():
                # 3. 获取这些坏点激活后的不透明度
                opacities_of_bad_points = gaussians.get_density[bad_points_mask]

                # 4. 定义损失：我们希望这些不透明度都趋近于0。
                #    直接惩罚激活后的不透明度值，梯度会驱使它们下降。
                loss_conf_punish = torch.mean(opacities_of_bad_points)

                # 5. 加入总损失字典和总损失
                loss["conf_punish"] = loss_conf_punish
                loss["total"] = loss["total"] + opt.lambda_conf_punish * loss_conf_punish

        # 可视化模块的终极Bug修复
        if iteration % 200 == 0 and tb_writer is not None:
            try:
                # 终极修复：手动将两个 (1, 1, H, W) 的张量合并成一个 (2, 1, H, W) 的批次
                combined_images = torch.cat([gt_image, image], dim=0)

                comparison_grid = torchvision.utils.make_grid(
                    combined_images, nrow=2, normalize=True, scale_each=True
                )
                tb_writer.add_image("Images/1_GT_vs_Rendered", comparison_grid, global_step=iteration)

                if diff_map_for_vis is not None:
                    tb_writer.add_image("Images/2_Density_Diff_Map",
                                        torchvision.utils.make_grid(diff_map_for_vis, normalize=True, scale_each=True),
                                        global_step=iteration)
                if final_weights is not None and final_weights.numel() > 1:
                    tb_writer.add_image("Images/3_Final_Loss_Weights",
                                        torchvision.utils.make_grid(final_weights, normalize=True, scale_each=True),
                                        global_step=iteration)

            except Exception as e:
                import traceback
                if iteration % 500 == 0:
                    print(f"\n[警告] 在迭代 {iteration} 写入TensorBoard图像时出错: {e}")
                    traceback.print_exc()
        ### --- 修改结束 --- ###

        loss["total"].backward()
        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration < opt.densify_until_iter:
                if (iteration > opt.densify_from_iter) and (iteration % opt.densification_interval == 0):
                    if getattr(opt, "compat_mode", False):
                        gaussians.densify_and_prune_baseline(
                            opt.densify_grad_threshold,
                            opt.density_min_threshold,
                            opt.max_screen_size,
                            max_scale,
                            opt.max_num_gaussians,
                            densify_scale_threshold,
                            bbox,
                        )
                    else:
                        gaussians.densify_and_prune(
                            opt=opt,
                            max_scale=max_scale,
                            densify_scale_threshold=densify_scale_threshold,
                            scene_scale=scene.scene_scale,
                        )

            if gaussians.get_density.shape[0] == 0:
                raise ValueError("No Gaussian left. Change adaptive control hyperparameters!")

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), ckpt_save_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration % 10 == 0:
                avg_conf = gaussians._roi_confidence.mean().item() if gaussians._roi_confidence.numel() > 0 else 0.0
                progress_bar.set_postfix(
                    {"loss": f"{loss['total'].item():.2e}", "点数": f"{gaussians.get_density.shape[0]:,}",
                     "平均置信度": f"{avg_conf:.3f}"}
                )
                progress_bar.update(10)

            if iteration == opt.iterations and args.plot_confidence:
                tqdm.write("\n[INFO] 训练结束，开始生成最终的置信度分布图...")
                confidence_data = gaussians.get_roi_confidence.cpu().numpy()
                plot_and_save_confidence_distribution(confidence_data, iteration, scene.model_path)

            if iteration == opt.iterations:
                progress_bar.close()

            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            training_report(
                tb_writer, iteration, metrics, iter_start.elapsed_time(iter_end), testing_iterations, scene,
                lambda x, y: render(x, y, pipe), queryfunc,
                # 新增传递的参数
                gaussians=gaussians, pipe=pipe,
                tissue_params=tissue_params, bone_params=bone_params,
                save_dynamic_mask_on_test=args.save_dynamic_mask_on_test
            )


def training_report(tb_writer, iteration, metrics_train, elapsed, testing_iterations, scene: Scene, renderFunc,
                    queryFunc, gaussians=None, pipe=None, tissue_params=None, bone_params=None, save_dynamic_mask_on_test=False):
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar("train/total_points", scene.gaussians.get_xyz.shape[0], iteration)

    if iteration in testing_iterations:
        if save_dynamic_mask_on_test and gaussians and pipe:
            generate_all_dynamic_masks(iteration, scene, gaussians, pipe, tissue_params, bone_params)

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
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(torch.from_numpy(
                            show_two_slice(gt_image[0], image[0], f"{viewpoint.image_name} gt",
                                           f"{viewpoint.image_name} render", save=True)
                        ))
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                lpips_2d, lpips_2d_projs = metric_proj(gt_images, images, "lpips")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d, "ssim_2d": ssim_2d, "lpips_2d": lpips_2d,
                    "psnr_2d_projs": psnr_2d_projs, "ssim_2d_projs": ssim_2d_projs, "lpips_2d_projs": lpips_2d_projs,
                }
                with open(osp.join(eval_save_path, f"eval2d_{config['name']}.yml"), "w") as f:
                    yaml.dump(eval_dict_2d, f, default_flow_style=False, sort_keys=False)
                if tb_writer:
                    if image_show_2d:
                        image_show_2d_tensor = torch.from_numpy(np.concatenate(image_show_2d, axis=0))[None].permute(
                            [0, 3, 1, 2])
                        tb_writer.add_images(config["name"] + f"/{viewpoint.image_name}", image_show_2d_tensor,
                                             global_step=iteration)
                    tb_writer.add_scalar(config["name"] + "/psnr_2d", psnr_2d, iteration)
                    tb_writer.add_scalar(config["name"] + "/ssim_2d", ssim_2d, iteration)
                    tb_writer.add_scalar(config["name"] + "/lpips_2d", lpips_2d, iteration)

        vol_pred = queryFunc(scene.gaussians)["vol"]
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        lpips_3d, lpips_3d_axis = metric_vol(vol_gt, vol_pred, "lpips")
        eval_dict = {
            "psnr_3d": psnr_3d, "ssim_3d": ssim_3d, "lpips_3d": lpips_3d,
            "ssim_3d_x": ssim_3d_axis[0], "ssim_3d_y": ssim_3d_axis[1], "ssim_3d_z": ssim_3d_axis[2],
            "lpips_3d_x": lpips_3d_axis[0] if lpips_3d_axis else None,
            "lpips_3d_y": lpips_3d_axis[1] if lpips_3d_axis else None,
            "lpips_3d_z": lpips_3d_axis[2] if lpips_3d_axis else None,
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

        if tb_writer:
            image_show_3d = np.concatenate(
                [show_two_slice(vol_gt[..., i], vol_pred[..., i], f"slice {i} gt", f"slice {i} pred",
                                vmin=vol_gt[..., i].min(), vmax=vol_gt[..., i].max(), save=True)
                 for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]], axis=0,
            )
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images("reconstruction/slice-gt_pred_diff", image_show_3d, global_step=iteration)
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
            tb_writer.add_scalar("reconstruction/lpips_3d", lpips_3d, iteration)

        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, lpips3d {lpips_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}, lpips2d {lpips_2d:.3f}"
        )
        if tb_writer:
            tb_writer.add_histogram("scene/density_histogram", scene.gaussians.get_density, iteration)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000, 20_000, 30_000])
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
    parser.add_argument("--auto_mask_no_previews", action="store_true", help="禁用保存自动掩码的PNG预览图和HU直方图。")
    parser.add_argument('--no_previews', dest='save_previews', action='store_false',
                        help="设置此项后，将不生成PNG格式的掩码预览图，只生成NPY文件。")
    parser.add_argument("--noisy_view_indices", type=int, nargs='+', default=None,
                        help="一个或多个训练视图的索引，只有这些视图会被添加噪声。如果未提供，则噪声会应用于所有视图。")
    parser.add_argument("--save_dynamic_mask_on_test", action="store_true",
                        help="如果设置此项，将在每个测试迭代点 (test_iterations) 为所有训练视图【实时渲染并生成】动态掩码。")

    parser.set_defaults(save_previews=True)

    op_group = parser.add_argument_group("Optimization", "Optimization parameters")
    op_group.add_argument("--lambda_proj_consistency", type=float, default=0.0,
                          help="投影分布一致性损失的权重。设置为0以禁用。")
    op_group.add_argument("--use_boost_loss", action="store_true", help="启用基于密度分布差异的自适应强化损失。")
    op_group.add_argument("--boost_factor", type=float, default=3.0,
                          help="强化区域的权重提升因子。默认值3.0表示重点区域的L1损失权重是普通区域的3倍。")
    op_group.add_argument("--boost_threshold", type=float, default=0.1,
                          help="判定为需要强化的区域的差异阈值。默认值0.1表示当归一化密度差异超过10%时，该像素被视为重点区域。")

    ### --- 新增开始 --- ###
    # 添加新的命令行参数用于控制置信度惩罚损失
    op_group.add_argument("--lambda_conf_punish", type=float, default=1.0,
                          help="置信度惩罚损失的权重。设置为0以禁用。")
    op_group.add_argument("--confidence_punish_threshold", type=float, default=0.0,
                          help="判定为'坏点'的置信度阈值。")
    op_group.add_argument("--conf_punish_start_iter", type=int, default=5000,
                          help="启动置信度惩罚损失的迭代次数。")
    ### --- 新增结束 --- ###

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(1)

    safe_state(args.quiet)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"
    print(f"使用设备: {device}")
    if device.startswith("cuda"):
        torch.cuda.set_device(args.gpu_id)

    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            # 命令行参数的优先级高于配置文件
            if key not in sys.argv:
                args_dict[key] = cfg[key]

    final_model_params = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # 将args中所有不在opt里的参数也复制过去
    for k, v in vars(args).items():
        if not hasattr(opt, k):
            setattr(opt, k, v)

    print("\n--- [路径配置] ---")
    model_path_was_specified = args.model_path != parser.get_default('model_path')
    if model_path_was_specified:
        print(f"   - 已明确指定输出路径: '{args.model_path}'")
    else:
        print("   - 未指定输出路径，将自动生成...")
        if hasattr(args, 'source_path') and args.source_path:
            scene_name = osp.basename(osp.normpath(args.source_path))
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            new_model_path = osp.join("./output", scene_name, current_time)
            args.model_path = new_model_path
            opt.model_path = new_model_path
            final_model_params.model_path = new_model_path  # 确保所有参数对象都更新
        else:
            print("   - 警告: 未提供 source_path (-s)，将使用默认路径。")

    if args.auto_mask:
        print("\n--- [自动化ROI掩码处理模块] ---")
        if not hasattr(args, 'source_path') or not args.source_path:
            print("错误: 必须在配置文件中定义 'source_path' 才能使用 --auto_mask。")
            sys.exit(1)
        base_mask_dir = osp.join(args.source_path, "masks")
        expected_soft_mask_dir = osp.join(base_mask_dir, "tissue_masks_npy")
        expected_core_mask_dir = osp.join(base_mask_dir, "bone_masks_npy")
        if osp.exists(expected_soft_mask_dir) and osp.exists(expected_core_mask_dir) and \
                len(os.listdir(expected_soft_mask_dir)) > 0 and len(os.listdir(expected_core_mask_dir)) > 0:
            print("   - 状态: 已找到预先生成的软掩码目录，将直接使用。")
            args.soft_mask_dir = expected_soft_mask_dir
            args.core_mask_dir = expected_core_mask_dir
        else:
            print("   - 状态: 未找到有效的软掩码目录，开始自动生成...")
            try:
                generated_soft_dir, generated_core_dir = check_and_generate_masks(
                    source_path=args.source_path,
                    pre_threshold_ratio=args.auto_mask_pre_threshold_ratio,
                    save_previews=not args.auto_mask_no_previews
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
        print("   - 配置已确认，将使用以下NPY掩码目录:")
        print(f"     - 软组织 (Soft): {args.soft_mask_dir}")
        print(f"     - 核心骨架 (Core): {args.core_mask_dir}")

    ### --- 【核心修复补丁】 --- ###
    # 在调用 `lp.extract(args)` 之后，无论 `ParamGroup` 如何工作，
    # 我们都强行将 `args` 中由 --auto_mask 生成的路径，注入到 `final_model_params` 对象中。
    # 这确保了 `Scene` 类在初始化时一定能接收到正确的路径。
    if hasattr(args, 'soft_mask_dir') and args.soft_mask_dir:
        final_model_params.soft_mask_dir = args.soft_mask_dir
    if hasattr(args, 'core_mask_dir') and args.core_mask_dir:
        final_model_params.core_mask_dir = args.core_mask_dir
    ### --- 补丁结束 --- ###

    tb_writer = prepare_output_and_logger(args)
    print("Optimizing " + args.model_path)

    dataset = final_model_params

    training(
        args, dataset, opt, pipe, tb_writer,
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint
    )

    if args.save_iterations:
        latest_iter = max(args.save_iterations)
        try:
            from point_cloud_visualizer import on_training_finish

            on_training_finish(model_path=args.model_path, latest_iter=latest_iter, tb_writer=tb_writer)
        except ImportError:
            print("无法导入 point_cloud_visualizer，跳过训练后可视化。")

    print("\n所有处理步骤完成。")