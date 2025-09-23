import os
import sys
import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement

sys.path.append("./")

from simple_knn._C import distCUDA2
from r2_gaussian.utils.system_utils import mkdir_p
from r2_gaussian.arguments import OptimizationParams
from r2_gaussian.utils.gaussian_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    inverse_softplus,
    strip_symmetric,
    build_scaling_rotation,
)


class GaussianModel:
    def setup_functions(self, scale_bound=None):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if scale_bound is not None:
            scale_min_bound, scale_max_bound = scale_bound
            self.scaling_activation = (
                lambda x: torch.sigmoid(x) * scale_max_bound + scale_min_bound
            )
            self.scaling_inverse_activation = lambda x: inverse_sigmoid(
                torch.relu((x - scale_min_bound) / scale_max_bound) + 1e-8
            )
        else:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.density_activation = torch.nn.Softplus()
        self.density_inverse_activation = inverse_softplus

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, scale_bound=None):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._density = torch.empty(0)
        self._roi_confidence = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.setup_functions(scale_bound)

    def capture(self):
        return (
            self._xyz, self._scaling, self._rotation, self._density, self._roi_confidence,
            self.max_radii2D, self.xyz_gradient_accum, self.denom, self.optimizer.state_dict(), self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        if len(model_args) == 9:
            (
                self._xyz, self._scaling, self._rotation, self._density,
                self.max_radii2D, xyz_gradient_accum, denom, opt_dict, self.spatial_lr_scale,
            ) = model_args
            self._roi_confidence = torch.zeros((self._xyz.shape[0], 1), device=self._xyz.device)
            print("\n[警告] 加载了不包含ROI置信度的旧版本Checkpoint。")
            print("所有点的ROI置信度已安全地初始化为0。\n")
        else:
            (
                self._xyz, self._scaling, self._rotation, self._density, self._roi_confidence,
                self.max_radii2D, xyz_gradient_accum, denom, opt_dict, self.spatial_lr_scale,
            ) = model_args
            print(f"\n成功从Checkpoint加载了 {self._xyz.shape[0]} 个点，包含ROI置信度信息。\n")

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_density(self):
        return self.density_activation(self._density)

    @property
    def get_roi_confidence(self):
        return self._roi_confidence

    def get_covariance(self, scaling_modifier=1):
        # 【修正】确保这里也使用正确的属性访问
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def create_from_pcd(self, xyz, density, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(xyz).float().cuda()
        print(f"从 {fused_point_cloud.shape[0]} 个估算点初始化高斯。")
        fused_density = self.density_inverse_activation(torch.tensor(density)).float().cuda()
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.001 ** 2)
        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._density = nn.Parameter(fused_density.requires_grad_(True))
        self._roi_confidence = nn.Parameter(
            torch.zeros((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._density], "lr": training_args.density_lr_init * self.spatial_lr_scale, "name": "density"},
            {"params": [self._scaling], "lr": training_args.scaling_lr_init * self.spatial_lr_scale, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr_init * self.spatial_lr_scale,
             "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    max_steps=training_args.position_lr_max_steps)
        self.density_scheduler_args = get_expon_lr_func(lr_init=training_args.density_lr_init * self.spatial_lr_scale,
                                                        lr_final=training_args.density_lr_final * self.spatial_lr_scale,
                                                        max_steps=training_args.density_lr_max_steps)
        self.scaling_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr_init * self.spatial_lr_scale,
                                                        lr_final=training_args.scaling_lr_final * self.spatial_lr_scale,
                                                        max_steps=training_args.scaling_lr_max_steps)
        self.rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr_init * self.spatial_lr_scale,
                                                         lr_final=training_args.rotation_lr_final * self.spatial_lr_scale,
                                                         max_steps=training_args.rotation_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz": lr = self.xyz_scheduler_args(iteration); param_group["lr"] = lr
            if param_group["name"] == "density": lr = self.density_scheduler_args(iteration); param_group["lr"] = lr
            if param_group["name"] == "scaling": lr = self.scaling_scheduler_args(iteration); param_group["lr"] = lr
            if param_group["name"] == "rotation": lr = self.rotation_scheduler_args(iteration); param_group["lr"] = lr

    def update_roi_confidence(self, render_pkg, viewpoint_cam, standard_reward, core_bonus_reward, background_reward):
        """
        【V14.1 - 比例化更新精简版】
        - 采用软掩码的连续值进行比例化置信度更新。
        - 移除了所有与 air_mask 相关的逻辑，仅使用 core_mask 和 soft_mask。
        """
        with torch.no_grad():
            # 检查视角是否有关联的掩码
            if not (hasattr(viewpoint_cam, 'soft_mask') and hasattr(viewpoint_cam, 'core_mask')):
                return

            # --- 步骤 1: 投影与采样 ---
            visibility_filter = render_pkg["visibility_filter"]
            visible_indices = visibility_filter.nonzero(as_tuple=True)[0]
            if visible_indices.numel() == 0:
                return

            visible_xyz_world = self._xyz[visible_indices]
            P = viewpoint_cam.full_proj_transform.cuda()
            H_img, W_img = viewpoint_cam.image_height, viewpoint_cam.image_width
            points_homogeneous = torch.cat(
                [visible_xyz_world, torch.ones(visible_xyz_world.shape[0], 1, device="cuda")], dim=1)
            points_clip = points_homogeneous @ P.T
            w = points_clip[..., 3:4]
            points_ndc = points_clip[..., :2] / (w + 1e-8)
            in_front_of_camera = w.squeeze() > 0.001
            if not in_front_of_camera.any():
                return

            reliable_xy_proj = torch.zeros_like(points_ndc)
            reliable_xy_proj[:, 0] = (points_ndc[:, 0] + 1.0) * W_img / 2.0
            reliable_xy_proj[:, 1] = (1.0 - points_ndc[:, 1]) * H_img / 2.0

            valid_coords_mask = (reliable_xy_proj[:, 0] >= 0) & (reliable_xy_proj[:, 0] < W_img) & \
                                (reliable_xy_proj[:, 1] >= 0) & (reliable_xy_proj[:, 1] < H_img) & \
                                in_front_of_camera

            in_image_indices_absolute = visible_indices[valid_coords_mask].contiguous()
            if in_image_indices_absolute.numel() == 0:
                return

            final_valid_xy = reliable_xy_proj[valid_coords_mask]

            soft_mask = viewpoint_cam.soft_mask.to(self.get_xyz.device)
            core_mask = viewpoint_cam.core_mask.to(self.get_xyz.device)

            soft_mask_for_sampling = soft_mask.unsqueeze(0).unsqueeze(0)
            core_mask_for_sampling = core_mask.unsqueeze(0).unsqueeze(0)

            normalized_xy = torch.zeros_like(final_valid_xy)
            normalized_xy[:, 0] = (final_valid_xy[:, 0] / (W_img - 1)) * 2 - 1
            normalized_xy[:, 1] = (final_valid_xy[:, 1] / (H_img - 1)) * 2 - 1
            grid = normalized_xy.unsqueeze(0).unsqueeze(0)

            sampled_soft_values = torch.nn.functional.grid_sample(soft_mask_for_sampling, grid, mode='bilinear',
                                                                  padding_mode='zeros', align_corners=True).squeeze()
            sampled_core_values = torch.nn.functional.grid_sample(core_mask_for_sampling, grid, mode='bilinear',
                                                                  padding_mode='zeros', align_corners=True).squeeze()

            # --- 步骤 2: 计算比例化的置信度更新量 ---
            delta_confidence = torch.zeros_like(sampled_soft_values)

            total_core_reward = standard_reward + core_bonus_reward
            delta_confidence += total_core_reward * sampled_core_values

            soft_only_values = torch.relu(sampled_soft_values - sampled_core_values)
            delta_confidence += standard_reward * soft_only_values

            prob_is_something = torch.max(sampled_soft_values, sampled_core_values)
            prob_is_background = 1.0 - prob_is_something
            delta_confidence += background_reward * prob_is_background

            # --- 步骤 3: 应用更新 ---
            self._roi_confidence[in_image_indices_absolute] += delta_confidence.unsqueeze(1)
            self._roi_confidence.clamp_(-5, 5)

    # 请将此函数替换掉您代码中所有的 densify_and_prune 函数

    def densify_and_prune(self, opt: OptimizationParams, max_scale, densify_scale_threshold, scene_scale):
        """
        【V23.0 - 三位一体控制系统】
        - 核心思想: 基于您的深刻洞见，将梯度、透明度和置信度进行深度耦合，实现确定性的、动态的演化控制。
        - 致密化: 置信度越高的点，其致密化梯度阈值越低，实现对重要区域的主动、精细优化。
        - 剪枝: 置信度越高的点，其透明度剪枝阈值越低（容忍度越高），保护重要区域的“潜力股”，同时严苛对待背景点。
        - 优点: 完全确定性，无随机性；逻辑优雅，符合直觉；最大化了置信度这一核心创新的价值。
        """

        # --- 准备阶段: 获取三大核心因素 ---
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        opacities = self.get_density.squeeze()
        confidences = self.get_roi_confidence.squeeze()

        # --- 1. 致密化阶段 (Densification) ---
        if opt.max_num_gaussians and self.get_xyz.shape[0] > opt.max_num_gaussians:
            prune_filter_split = torch.zeros(self.get_xyz.shape[0], device="cuda", dtype=torch.bool)
        else:
            # 【新致密化逻辑】动态梯度阈值
            # 使用sigmoid将(-inf, +inf)的置信度映射到(0, 1)的重要性因子
            importance_factor = torch.sigmoid(confidences / opt.confidence_densify_sensitivity)  # sensitivity控制敏感度

            # 动态阈值：重要性越高，阈值越低。我们用乘法实现：将梯度放大来比较
            # 重要性为1时，梯度放大2倍；重要性为0.5时，梯度放大1倍；重要性为0时，梯度放大0倍。
            effective_grads_norm = torch.norm(grads, dim=1) * (importance_factor * 2)

            densify_mask = (effective_grads_norm > opt.densify_grad_threshold)

            if densify_scale_threshold is not None:
                big_points_mask = torch.max(self.get_scaling, dim=1).values > densify_scale_threshold * scene_scale
                densify_mask = densify_mask & ~big_points_mask

            split_mask = densify_mask & (
                    torch.max(self.get_scaling, dim=1).values >= opt.densify_scale_threshold * scene_scale)
            clone_mask = densify_mask & ~split_mask

            # --- 数据收集与写入 (与之前版本类似，但逻辑更清晰) ---
            all_new_xyz, all_new_density, all_new_scaling, all_new_rotation = [], [], [], []
            all_new_roi_confidence, all_new_max_radii2D = [], []

            # 克隆点
            if torch.sum(clone_mask) > 0:
                new_confidence_clone = self._roi_confidence[clone_mask] * opt.clone_confidence_decay_factor
                # ... (此处省略append代码，与您之前的版本相同)
                all_new_xyz.append(self._xyz[clone_mask]);
                all_new_density.append(self._density[clone_mask]);
                all_new_scaling.append(self._scaling[clone_mask]);
                all_new_rotation.append(self._rotation[clone_mask]);
                all_new_roi_confidence.append(new_confidence_clone);
                all_new_max_radii2D.append(self.max_radii2D[clone_mask])

            # 分裂点
            if torch.sum(split_mask) > 0:
                new_confidence_split = (self._roi_confidence[split_mask] * opt.clone_confidence_decay_factor).repeat(2,
                                                                                                                     1)
                # ... (此处省略append代码，与您之前的版本相同)
                stds = self.get_scaling[split_mask].repeat(2, 1);
                stds[:, :2] *= 0.8
                all_new_xyz.append(self._xyz[split_mask].repeat(2, 1));
                all_new_density.append(self._density[split_mask].repeat(2, 1));
                all_new_scaling.append(self.scaling_inverse_activation(stds));
                all_new_rotation.append(self._rotation[split_mask].repeat(2, 1));
                all_new_roi_confidence.append(new_confidence_split);
                all_new_max_radii2D.append(torch.zeros(stds.shape[0], device="cuda"))

            # 统一写入
            if len(all_new_xyz) > 0:
                num_added = sum(p.shape[0] for p in all_new_xyz)
                self.densification_postfix(torch.cat(all_new_xyz, dim=0), torch.cat(all_new_density, dim=0),
                                           torch.cat(all_new_scaling, dim=0), torch.cat(all_new_rotation, dim=0),
                                           torch.cat(all_new_roi_confidence, dim=0),
                                           torch.cat(all_new_max_radii2D, dim=0))
                prune_filter_split = torch.cat((split_mask, torch.zeros(num_added, device="cuda", dtype=torch.bool)))
            else:
                prune_filter_split = torch.zeros(self.get_xyz.shape[0], device="cuda", dtype=torch.bool)

        # --- 2. 剪枝阶段 (Pruning) ---

        # 【新剪枝逻辑】动态透明度阈值 (动态容忍度)
        # 再次使用 sigmoid 映射的重要性因子
        importance_factor = torch.sigmoid(confidences)  # 这里可以直接用，或用不同的sensitivity

        # 动态阈值: 重要性越高，容忍度越高（阈值越低）。
        # 重要性为1 (核心ROI) -> 阈值 = 基础阈值 * (1-1) = 0 (极度容忍)
        # 重要性为0 (背景) -> 阈值 = 基础阈值 * (1-0) = 基础阈值 (标准严苛)
        # 重要性为0.5 (软组织) -> 阈值 = 基础阈值 * 0.5 (中等容忍)
        dynamic_opacity_threshold = opt.opacity_prune_threshold * (1.0 - importance_factor)

        opacity_prune_mask = (opacities < dynamic_opacity_threshold)

        # 其他剪枝规则保持不变
        scales = self.get_scaling
        scale_prune_mask = torch.zeros_like(opacity_prune_mask)
        if max_scale:
            scale_prune_mask = (torch.max(scales, dim=1).values > max_scale * scene_scale)

        screen_size_prune_mask = torch.zeros_like(opacity_prune_mask)
        if opt.max_screen_size:
            screen_size_prune_mask = (self.max_radii2D > opt.max_screen_size)

        # 合并所有剪枝条件
        final_prune_mask = opacity_prune_mask | scale_prune_mask | screen_size_prune_mask
        total_prune_mask = prune_filter_split | final_prune_mask

        if torch.sum(total_prune_mask) > 0:
            self.prune_points(total_prune_mask)

        # --- 3. 重置梯度 ---
        torch.cuda.empty_cache()
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz", "density"]
        for i in range(self._scaling.shape[1]): l.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]): l.append(f"rot_{i}")
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        densities = self.get_density.detach().cpu().numpy()  # 保存激活后的密度
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, densities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    # 【注意】文件中不应再有其他 densify_and_prune 函数

    def densify_and_clone_with_mask(self, mask):
        if not mask.any(): return
        new_xyz = self._xyz[mask]
        new_densities = self.density_inverse_activation(self.get_density[mask] * 0.5)
        new_scaling = self._scaling[mask]
        new_rotation = self._rotation[mask]
        new_max_radii2D = self.max_radii2D[mask]
        new_roi_confidence = self._roi_confidence[mask]
        self.densification_postfix(new_xyz, new_densities, new_scaling, new_rotation, new_roi_confidence,
                                   new_max_radii2D)

    def densify_and_split_with_mask(self, mask, initial_scales, initial_xyz, initial_rotation, initial_density,
                                    initial_max_radii2D, initial_roi_confidence, N=2):
        if not mask.any(): return
        stds = initial_scales[mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(initial_rotation[mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + initial_xyz[mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(initial_scales[mask].repeat(N, 1) / (0.8 * N))
        new_rotation = initial_rotation[mask].repeat(N, 1)
        new_density = self.density_inverse_activation(initial_density[mask].repeat(N, 1) * (1 / N))
        new_max_radii2D = initial_max_radii2D[mask].repeat(N)
        new_roi_confidence = initial_roi_confidence[mask].repeat(N, 1)
        self.densification_postfix(new_xyz, new_density, new_scaling, new_rotation, new_roi_confidence, new_max_radii2D)

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self._roi_confidence = self._roi_confidence[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_densities, new_scaling, new_rotation, new_roi_confidence,
                              new_max_radii2D):
        d = {"xyz": new_xyz, "density": new_densities, "scaling": new_scaling, "rotation": new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros_like(new_xyz[:, 0:1])])
        self.denom = torch.cat([self.denom, torch.zeros_like(new_xyz[:, 0:1])])
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=0)
        self._roi_confidence = torch.cat([self._roi_confidence, new_roi_confidence], dim=0)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1