import os
import sys
import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement

sys.path.append("./")

from simple_knn._C import distCUDA2
from r2_gaussian.utils.system_utils import mkdir_p
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
        # 【修改】增强了对旧版本checkpoint的兼容性提示
        if len(model_args) == 9:
            (
                self._xyz, self._scaling, self._rotation, self._density,
                self.max_radii2D, xyz_gradient_accum, denom, opt_dict, self.spatial_lr_scale,
            ) = model_args
            # 旧的checkpoint没有保存roi_confidence, 我们将其初始化为0
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

    def update_roi_confidence(self, render_pkg, viewpoint_cam, standard_reward, core_bonus_reward, background_reward,
                              air_penalty):
        """
        【V13.1 - 互斥修复版】
        修复了四级分类中存在的逻辑漏洞，确保每个点在一次更新中只被归入一个类别。
        分类严格遵循互斥优先级: 空气 -> 核心骨架 -> 软组织 -> 背景。
        """
        with torch.no_grad():
            use_roi = hasattr(viewpoint_cam, 'soft_mask') and hasattr(viewpoint_cam, 'core_mask')
            use_air_penalty = hasattr(viewpoint_cam, 'air_mask') and air_penalty < 0.0
            if not use_roi:
                return

            visibility_filter = render_pkg["visibility_filter"]
            visible_indices = visibility_filter.nonzero(as_tuple=True)[0]
            if visible_indices.numel() == 0:
                return

            # --- 坐标投影部分 (与之前版本相同) ---
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

            # --- 掩码采样部分 (与之前版本相同) ---
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

            # --- 【核心逻辑修正】严格互斥的四级分类 ---
            # 1. 首先，识别所有可能属于各类别的点的原始掩码
            is_in_air_mask = torch.zeros_like(sampled_soft_values, dtype=torch.bool)
            if use_air_penalty:
                air_mask = viewpoint_cam.air_mask.to(self.get_xyz.device)
                air_mask_for_sampling = air_mask.unsqueeze(0).unsqueeze(0)
                sampled_air_values = torch.nn.functional.grid_sample(air_mask_for_sampling, grid, mode='bilinear',
                                                                     padding_mode='zeros', align_corners=True).squeeze()
                is_in_air_mask = sampled_air_values > 0.5

            is_in_core_mask_raw = sampled_core_values > 0.08
            is_in_soft_mask_raw = sampled_soft_values > 0.07

            # 2. 应用互斥逻辑，确保一个点只属于一个类别
            # 类别1: 空气 (最高优先级)
            mask_air = is_in_air_mask

            # 类别2: 核心骨架 (必须不在空气中)
            mask_core = is_in_core_mask_raw & ~mask_air

            # 类别3: 软组织 (必须不在空气和核心中)
            mask_soft = is_in_soft_mask_raw & ~mask_core & ~mask_air

            # 类别4: 背景 (所有其他点)
            mask_background = ~mask_air & ~mask_core & ~mask_soft

            # 3. 根据最终的互斥掩码获取点索引
            indices_in_air = in_image_indices_absolute[mask_air]
            indices_in_core = in_image_indices_absolute[mask_core]
            indices_in_soft_only = in_image_indices_absolute[mask_soft]
            indices_background = in_image_indices_absolute[mask_background]

            # --- 更新置信度 ---
            if use_air_penalty and indices_in_air.numel() > 0:
                self._roi_confidence[indices_in_air] += air_penalty

            if indices_in_core.numel() > 0:
                self._roi_confidence[indices_in_core] += (standard_reward + core_bonus_reward)

            if indices_in_soft_only.numel() > 0:
                self._roi_confidence[indices_in_soft_only] += standard_reward

            if indices_background.numel() > 0:
                self._roi_confidence[indices_background] += background_reward

            self._roi_confidence.clamp_(-10, 10)

            print(f"\n[ROI Update] Air: {indices_in_air.numel()}, "
                  f"Core: {indices_in_core.numel()}, "
                  f"Soft: {indices_in_soft_only.numel()}, "
                  f"BG: {indices_background.numel()}, "
                  f"Total: {in_image_indices_absolute.numel()}")

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz", "density"]
        for i in range(self._scaling.shape[1]): l.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]): l.append(f"rot_{i}")
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        densities = self._density.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, densities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def densify_and_prune(self, opt, max_scale, densify_scale_threshold, extent):
        """
        [V16.0 最终版] - “创生”与“凋亡”分离
        - 使用 opt.density_min_threshold (Softplus) 来决定哪些点【有资格致密化】。
        - 使用新增的 opt.opacity_prune_threshold (Sigmoid) 来决定哪些点【应该被剪枝】。
        这从根本上解决了过度致密化的问题，同时保证了严格的噪声清除。
        """
        with torch.no_grad():
            # --- 0. 提取参数 ---
            densify_grad_threshold = opt.densify_grad_threshold
            # 【V16.0 核心】分离阈值
            densify_density_threshold = opt.density_min_threshold  # 用于控制致密化
            prune_opacity_threshold = opt.opacity_prune_threshold  # 用于控制剪枝
            max_screen_size = opt.max_screen_size

            # --- 1. 致密化逻辑 (Densification Logic) ---
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0

            # 【V16.0 核心】使用“致密化密度阈值”来筛选候选点
            # 只有物理密度足够高的点，才有资格参与后续的梯度判断和分裂
            candidates_mask = (self.get_density >= densify_density_threshold).squeeze()
            grad_norm = torch.norm(grads, dim=-1)

            if opt.use_confidence_modulation:
                multiplier = 1.0 + torch.tanh(self._roi_confidence / opt.confidence_densify_scale)
                effective_grad = grad_norm * multiplier.squeeze()
            else:
                effective_grad = grad_norm

            densify_mask = (effective_grad > densify_grad_threshold).squeeze()
            densify_mask &= candidates_mask

            clone_mask = densify_mask & (torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold)
            split_mask = densify_mask & (torch.max(self.get_scaling, dim=1).values > densify_scale_threshold)

            initial_scales = self.get_scaling
            initial_xyz = self.get_xyz
            initial_rotation = self._rotation
            initial_density = self.get_density
            initial_max_radii2D = self.max_radii2D
            initial_roi_confidence = self.get_roi_confidence

            self.densify_and_clone_with_mask(clone_mask)
            self.densify_and_split_with_mask(split_mask, initial_scales, initial_xyz, initial_rotation, initial_density,
                                             initial_max_radii2D, initial_roi_confidence)

            # --- 2. 剪枝逻辑 (Pruning Logic) ---
            num_points_before_prune = self.get_xyz.shape[0]

            # 【V16.0 核心】使用独立的、更严格的“不透明度剪枝阈值”
            prune_mask = (torch.sigmoid(self._density) < prune_opacity_threshold).squeeze()

            if max_screen_size:
                prune_mask |= (self.max_radii2D > max_screen_size)

            if max_scale:
                prune_mask |= (self.get_scaling.max(dim=1).values > max_scale)

            if opt.use_confidence_modulation:
                confidence_prune_mask = (self._roi_confidence < opt.confidence_prune_threshold).squeeze()
                num_conf_pruned = torch.sum(confidence_prune_mask).item()
                if num_conf_pruned > 0:
                    print(
                        f"\033[93m[PRUNING]\033[0m Pruned {num_conf_pruned} points due to low confidence (threshold < {opt.confidence_prune_threshold}).")
                prune_mask |= confidence_prune_mask

            num_new_points = num_points_before_prune - len(split_mask)
            padded_split_mask = torch.cat((split_mask, torch.zeros(num_new_points, dtype=torch.bool, device="cuda")),
                                          dim=0)
            prune_mask |= padded_split_mask

            if prune_mask.any():
                self.prune_points(prune_mask)

            torch.cuda.empty_cache()

    def densify_and_clone_with_mask(self, mask):
        if not mask.any(): return
        new_xyz = self._xyz[mask]
        new_densities = self.density_inverse_activation(self.get_density[mask] * 0.5)
        new_scaling = self._scaling[mask]
        new_rotation = self._rotation[mask]
        new_max_radii2D = self.max_radii2D[mask]

        # 【保留逻辑】继承父代的置信度
        new_roi_confidence = self._roi_confidence[mask]

        self._density[mask] = new_densities
        self.densification_postfix(new_xyz, new_densities, new_scaling, new_rotation, new_roi_confidence,
                                   new_max_radii2D)

    def densify_and_split_with_mask(self, mask, initial_scales, initial_xyz, initial_rotation, initial_density,
                                    initial_max_radii2D, initial_roi_confidence, N=2):
        if not mask.any(): return

        # 【修复】使用传入的初始状态进行计算，而不是self.get_...
        stds = initial_scales[mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(initial_rotation[mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + initial_xyz[mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(initial_scales[mask].repeat(N, 1) / (0.8 * N))
        new_rotation = initial_rotation[mask].repeat(N, 1)
        new_density = self.density_inverse_activation(initial_density[mask].repeat(N, 1) * (1 / N))
        new_max_radii2D = initial_max_radii2D[mask].repeat(N)

        # 【保留逻辑】继承父代的置信度
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
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=-1)
        self._roi_confidence = torch.cat([self._roi_confidence, new_roi_confidence], dim=0)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1