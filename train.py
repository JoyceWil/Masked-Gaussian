import datetime
import os
import os.path as osp
import torch
import torch.nn.functional as F
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml
import uuid

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
from r2_gaussian.gaussian.structure_guardian import StructureGuardian


def weighted_l1_loss(pred_image, gt_image, s_map, beta):
    """
    计算空间加权的L1损失。
    权重 W = 1 + beta * s_map
    """
    weight_map = 1.0 + beta * s_map
    loss = torch.abs(pred_image - gt_image) * weight_map
    return loss.mean()


def _create_3d_sobel_kernels(device):
    """创建3x3x3的3D Sobel算子核"""
    kernel_x = torch.tensor([
        [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]],
        [[-3, 0, 3], [-6, 0, 6], [-3, 0, 3]],
        [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]
    ], dtype=torch.float32)
    kernel_y = kernel_x.permute(1, 0, 2)
    kernel_z = kernel_x.permute(2, 1, 0)
    # 返回 (out_channels=3, in_channels=1, D, H, W)
    return torch.stack([kernel_x, kernel_y, kernel_z], dim=0).unsqueeze(1).to(device)


def training(
        dataset: ModelParams,
        opt: OptimizationParams,
        pipe: PipelineParams,
        tb_writer,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        device: torch.device,
):
    first_iter = 0
    scene = Scene(dataset, shuffle=False)
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox

    # >>>>> 开始修改: 打印分阶段训练配置信息 <<<<<
    if opt.phase_switch_iter > 0:
        print("===============================================================")
        print("Phased Weighted Training (Scheme B) is ENABLED.")
        print(f"- Phase Switch Iteration: {opt.phase_switch_iter}")
        print(f"- Phase 1 Beta (0 -> {opt.phase_switch_iter}): {opt.beta_phase1}")
        print(f"- Phase 2 Beta ({opt.phase_switch_iter} -> end): {opt.beta_phase2}")
        if opt.use_beta_annealing:
            print("- Beta Transition: Smooth Annealing")
        else:
            print("- Beta Transition: Hard Switch")

        if opt.use_structure_protection or opt.use_structure_aware_densification:
            print(f"- Structure Guardian will be ACTIVE in Phase 1 and DISABLED in Phase 2.")
        print("===============================================================")
    else:
        # 非分阶段训练，使用 beta_phase2 作为恒定 beta 值
        if opt.beta_phase2 > 0.0:
            print(f"Spatially-Weighted Loss is ENABLED with constant beta = {opt.beta_phase2}.")
        else:
            print("Spatially-Weighted Loss is DISABLED.")
    # >>>>> 结束修改 <<<<<

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
        (model_params, first_iter) = torch.load(checkpoint, map_location=device)
        gaussians.restore(model_params, opt)

    bbox = bbox.to(gaussians.get_xyz.device)

    guardian = None
    if opt.use_structure_protection or opt.use_structure_aware_densification:
        print("Structure Guardian is ENABLED (using static P_vol).")
        if opt.use_structure_protection:
            print("- Structure-aware pruning is ACTIVE.")
        if opt.use_structure_aware_densification:
            print("- Structure-aware densification is ACTIVE.")
        p_vol_path = os.path.join(dataset.source_path, "P_vol.npy")
        if not os.path.exists(p_vol_path):
            raise FileNotFoundError(
                f"P_vol.npy not found at: {p_vol_path}\n"
                f"Please run 'prior_extraction.py' on the dataset first."
            )
        guardian = StructureGuardian(
            device=gaussians.get_xyz.device,
            scene_bbox=bbox,
            p_vol_path=p_vol_path
        )
    else:
        print("Structure Guardian is DISABLED.")

    # 确定是否在任何阶段使用空间加权损失
    use_s_map_loss = (opt.phase_switch_iter > 0) or (opt.beta_phase2 > 0.0)

    use_tv = opt.lambda_tv > 0
    tv_vol_size = 0
    tv_vol_nVoxel = None
    tv_vol_sVoxel = None

    if use_tv:
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size], device=gaussians.get_xyz.device)
        tv_vol_sVoxel = (torch.tensor(scanner_cfg["dVoxel"], device=gaussians.get_xyz.device) * tv_vol_nVoxel)
        print("Total variation loss is ENABLED.")

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

        # >>>>> 开始修改: 实现分阶段的 beta 值计算 <<<<<
        current_beta = 0.0
        if use_s_map_loss:
            if opt.phase_switch_iter > 0:  # 分阶段训练激活
                if iteration < opt.phase_switch_iter:
                    # 阶段一
                    if opt.use_beta_annealing:
                        # 平滑退火
                        progress = iteration / opt.phase_switch_iter
                        current_beta = opt.beta_phase1 + (opt.beta_phase2 - opt.beta_phase1) * progress
                    else:
                        # 硬切换，使用阶段一的 beta
                        current_beta = opt.beta_phase1
                else:
                    # 阶段二
                    current_beta = opt.beta_phase2
            else:  # 分阶段训练未激活，使用恒定的 beta_phase2
                current_beta = opt.beta_phase2
        # >>>>> 结束修改 <<<<<

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"],
            render_pkg["visibility_filter"], render_pkg["radii"],
        )
        gt_image = viewpoint_cam.original_image.to(device)
        loss = {"total": 0.0}

        # >>>>> 开始修改: 使用动态计算的 current_beta 计算损失 <<<<<
        s_map = viewpoint_cam.s_map
        # 检查是否启用加权损失、beta > 0 且当前相机有 s_map
        if use_s_map_loss and current_beta > 0 and s_map is not None:
            s_map = s_map.to(device)
            render_loss = weighted_l1_loss(image, gt_image, s_map, current_beta)
        else:
            render_loss = l1_loss(image, gt_image)
        # >>>>> 结束修改 <<<<<

        loss["render"] = render_loss
        loss["total"] += loss["render"]

        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim

        vol_pred_patch = None
        if use_tv:
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (bbox[1] - tv_vol_sVoxel - bbox[0]) * torch.rand(3,
                                                                                                             device=gaussians.get_xyz.device)
            vol_pred_patch = query(gaussians, tv_vol_center, tv_vol_nVoxel, tv_vol_sVoxel, pipe)["vol"]

        if use_tv and vol_pred_patch is not None:
            loss_tv = tv_3d_loss(vol_pred_patch, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv

        loss["total"].backward()
        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration < opt.densify_until_iter:
                # >>>>> 开始修改: 根据训练阶段控制 Structure Guardian <<<<<
                guardian_for_densify = guardian
                use_densify_aware = opt.use_structure_aware_densification

                # 如果分阶段训练激活且进入第二阶段，则禁用 Guardian
                if opt.phase_switch_iter > 0 and iteration >= opt.phase_switch_iter:
                    if guardian is not None and iteration == opt.phase_switch_iter:
                        tqdm.write(f"[ITER {iteration}] Entering Phase 2: Structure Guardian is now DISABLED.")
                    guardian_for_densify = None
                    use_densify_aware = False
                # >>>>> 结束修改 <<<<<

                if (iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                        guardian_for_densify,  # 传递条件性 Guardian 对象
                        opt.structure_protection_threshold,
                        use_densify_aware,  # 传递条件性增密标志
                        opt.structure_densification_threshold
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
                progress_bar.set_postfix(
                    {"loss": f"{loss['total'].item():.1e}", "pts": f"{gaussians.get_density.shape[0]:2.1e}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            metrics = {}
            for l in loss: metrics["loss_" + l] = loss[l] if isinstance(loss[l], float) else loss[l].item()
            for param_group in gaussians.optimizer.param_groups: metrics[f"lr_{param_group['name']}"] = param_group[
                "lr"]

            # >>>>> 开始修改: 将 current_beta 添加到监控指标 <<<<<
            metrics["beta_value"] = current_beta
            # >>>>> 结束修改 <<<<<

            training_report(tb_writer, iteration, metrics, iter_start.elapsed_time(iter_end), testing_iterations, scene,
                            lambda x, y: render(x, y, pipe), queryfunc, device)


def training_report(tb_writer, iteration, metrics_train, elapsed, testing_iterations, scene: Scene, renderFunc,
                    queryFunc, device: torch.device):
    if tb_writer:
        for key in list(metrics_train.keys()):
            # >>>>> 开始修改: 将 beta_value 单独处理或确保其存在 <<<<<
            if key == "beta_value":
                tb_writer.add_scalar("train/beta_value", metrics_train[key], iteration)
            else:
                tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
            # >>>>> 结束修改 <<<<<
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
                    gt_image = viewpoint.original_image.to(device)
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(torch.from_numpy(
                            show_two_slice(gt_image[0], image[0], f"{viewpoint.image_name} gt",
                                           f"{viewpoint.image_name} render", vmin=None, vmax=None, save=True)))
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, _ = metric_proj(gt_images, images, "psnr")
                ssim_2d, _ = metric_proj(gt_images, images, "ssim")
                lpips_2d, _ = metric_proj(gt_images, images, "lpips")
                eval_dict_2d = {"psnr_2d": psnr_2d, "ssim_2d": ssim_2d, "lpips_2d": lpips_2d}
                with open(osp.join(eval_save_path, f"eval2d_{config['name']}.yml"), "w") as f:
                    yaml.dump(eval_dict_2d, f, default_flow_style=False, sort_keys=False)
                if tb_writer:
                    image_show_2d = torch.from_numpy(np.concatenate(image_show_2d, axis=0))[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(config["name"] + f"/{viewpoint.image_name}", image_show_2d,
                                         global_step=iteration)
                    tb_writer.add_scalar(config["name"] + "/psnr_2d", psnr_2d, iteration)
                    tb_writer.add_scalar(config["name"] + "/ssim_2d", ssim_2d, iteration)
                    tb_writer.add_scalar(config["name"] + "/lpips_2d", lpips_2d, iteration)

        vol_pred = queryFunc(scene.gaussians)["vol"]
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
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
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
            tb_writer.add_scalar("reconstruction/lpips_3d", lpips_3d, iteration)
        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, lpips3d {lpips_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}, lpips2d {lpips_2d:.3f}")
        if tb_writer:
            tb_writer.add_histogram("scene/density_histogram", scene.gaussians.get_density, iteration)
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

    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU to use for training.")

    parser.add_argument("--use_structure_protection", action="store_true",
                        help="Enable P_vol-based structure pruning protection.")
    parser.add_argument("--structure_protection_threshold", type=float, default=0.1,
                        help="P_vol threshold to protect a gaussian from pruning.")

    parser.add_argument("--use_structure_aware_densification", action="store_true",
                        help="Enable P_vol-based structure-aware densification.")
    parser.add_argument("--structure_densification_threshold", type=float, default=0.0001,
                        help="P_vol threshold to allow densification in a region.")

    # >>>>> 开始修改: 添加分阶段加权训练 (方案B) 的参数 <<<<<
    parser.add_argument("--phase_switch_iter", type=int, default=0,
                        help="Iteration to switch training phases. Default 0 disables phased training.")
    parser.add_argument("--beta_phase1", type=float, default=0.5,
                        help="Beta for weighted L1 loss in phase 1. Also used as starting beta for annealing.")
    parser.add_argument("--beta_phase2", type=float, default=3.0,
                        help="Beta for weighted L1 loss in phase 2. Also used as constant beta if phased training is off.")
    parser.add_argument("--use_beta_annealing", action="store_true",
                        help="Enable smooth annealing of beta from beta_phase1 to beta_phase2.")
    # >>>>> 结束修改 <<<<<

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(1)

    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU. Exiting.")
        sys.exit(1)

    if args.gpu_id >= torch.cuda.device_count():
        print(f"Error: GPU ID {args.gpu_id} is invalid. Available GPUs: {torch.cuda.device_count()}.")
        sys.exit(1)

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    print(f"Using GPU: {args.gpu_id} ({torch.cuda.get_device_name(args.gpu_id)})")

    safe_state(args.quiet)

    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            if args_dict.get(key) is None or args_dict.get(key) == parser.get_default(key):
                args_dict[key] = cfg[key]

    final_model_params = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    for k, v in vars(args).items():
        if not hasattr(opt, k):
            setattr(opt, k, v)

    model_path_was_specified = args.model_path != parser.get_default('model_path')
    if not model_path_was_specified:
        if hasattr(args, 'source_path') and args.source_path:
            scene_name = osp.basename(osp.normpath(args.source_path))
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            new_model_path = osp.join("./output", scene_name, current_time)
            args.model_path = new_model_path
            opt.model_path = new_model_path
            final_model_params.model_path = new_model_path
        else:
            print("Warning: source_path (-s) not provided, using default model_path.")

    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        final_model_params,
        opt,
        pipe,
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        device,
    )

    if args.save_iterations:
        latest_iter = max(args.save_iterations)
        try:
            from point_cloud_visualizer import on_training_finish

            on_training_finish(model_path=args.model_path, latest_iter=latest_iter, tb_writer=tb_writer)
        except ImportError:
            print("point_cloud_visualizer not found, skipping final visualization.")

    print("Training complete.")