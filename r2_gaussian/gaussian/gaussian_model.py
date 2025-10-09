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
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self._roi_confidence,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        if len(model_args) == 9:  # 旧格式，不含置信度
            (self._xyz, self._scaling, self._rotation, self._density,
             self.max_radii2D, xyz_gradient_accum, denom, opt_dict, self.spatial_lr_scale
             ) = model_args
            self._roi_confidence = torch.zeros((self._xyz.shape[0], 1), device=self._xyz.device)
            print("\n[警告] 加载了不包含ROI置信度的旧版本Checkpoint。已将置信度初始化为0。\n")
        else:  # 新格式，包含置信度
            (self._xyz, self._scaling, self._rotation, self._density, self._roi_confidence,
             self.max_radii2D, xyz_gradient_accum, denom, opt_dict, self.spatial_lr_scale
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
        self._roi_confidence = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._density], "lr": training_args.density_lr_init * self.spatial_lr_scale, "name": "density"},
            {"params": [self._scaling], "lr": training_args.scaling_lr_init * self.spatial_lr_scale, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr_init * self.spatial_lr_scale, "name": "rotation"},
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
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "density":
                lr = self.density_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group["lr"] = lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        l.append("density")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
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

    def reset_density(self, reset_density=1.0):
        densities_new = self.density_inverse_activation(
            torch.min(
                self.get_density, torch.ones_like(self.get_density) * reset_density
            )
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
        self._density = optimizable_tensors["density"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        densities = np.asarray(plydata.elements[0]["density"])[..., np.newaxis]
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._density = nn.Parameter(torch.tensor(densities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

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
        if self._roi_confidence.numel() > 0:
            self._roi_confidence = self._roi_confidence[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_densities, new_scaling, new_rotation, new_roi_confidence, new_max_radii2D):
        d = {"xyz": new_xyz, "density": new_densities, "scaling": new_scaling, "rotation": new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # 同步 max_radii2D 与 置信度
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=0)
        if new_roi_confidence is not None and new_roi_confidence.numel() > 0:
            self._roi_confidence = torch.cat([self._roi_confidence, new_roi_confidence], dim=0)

    def update_roi_confidence(self, render_pkg, viewpoint_cam, opt: OptimizationParams):
        with torch.no_grad():
            if not (hasattr(viewpoint_cam, 'soft_mask') and hasattr(viewpoint_cam, 'core_mask')):
                return
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
            in_front_of_camera = w.squeeze() > 0.001
            if not in_front_of_camera.any():
                return
            points_ndc = points_clip[..., :2] / (w + 1e-8)
            pixel_coords = torch.zeros_like(points_ndc)
            pixel_coords[:, 0] = (points_ndc[:, 0] + 1.0) * W_img / 2.0
            pixel_coords[:, 1] = (1.0 - points_ndc[:, 1]) * H_img / 2.0
            valid_coords_mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W_img) & \
                                (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H_img) & \
                                in_front_of_camera
            if not valid_coords_mask.any():
                return
            final_valid_indices = visible_indices[valid_coords_mask]
            final_valid_xy = pixel_coords[valid_coords_mask]
            normalized_xy = torch.zeros_like(final_valid_xy)
            normalized_xy[:, 0] = (final_valid_xy[:, 0] / (W_img - 1)) * 2 - 1
            normalized_xy[:, 1] = (final_valid_xy[:, 1] / (H_img - 1)) * 2 - 1
            grid = normalized_xy.unsqueeze(0).unsqueeze(0)
            soft_mask = viewpoint_cam.soft_mask.to(self.get_xyz.device).unsqueeze(0).unsqueeze(0)
            core_mask = viewpoint_cam.core_mask.to(self.get_xyz.device).unsqueeze(0).unsqueeze(0)
            sampled_soft = torch.nn.functional.grid_sample(soft_mask, grid, mode='bilinear', padding_mode='zeros',
                                                           align_corners=True).squeeze()
            sampled_core = torch.nn.functional.grid_sample(core_mask, grid, mode='bilinear', padding_mode='zeros',
                                                           align_corners=True).squeeze()
            reward_delta = torch.zeros_like(sampled_soft)
            total_core_reward = opt.roi_standard_reward + opt.roi_core_bonus_reward
            reward_delta += total_core_reward * sampled_core
            soft_only_values = torch.relu(sampled_soft - sampled_core)
            reward_delta += opt.roi_standard_reward * soft_only_values
            if opt.use_reward_saturation:
                current_conf = self._roi_confidence[final_valid_indices].squeeze(-1)
                saturation_factor = 1.0 - torch.tanh(opt.confidence_saturation_sensitivity * torch.relu(current_conf))
                reward_delta *= saturation_factor
            prob_is_something = torch.max(sampled_soft, sampled_core)
            prob_is_background = 1.0 - prob_is_something
            penalty_delta = opt.roi_background_reward * prob_is_background
            total_delta = reward_delta + penalty_delta
            self._roi_confidence[final_valid_indices] += total_delta.unsqueeze(1)
            self._roi_confidence.clamp_(min=opt.confidence_min_val, max=opt.confidence_max_val)

    def densify_and_prune(self, opt: OptimizationParams, max_scale, densify_scale_threshold, scene_scale):
        # 防呆：compat_mode 下不应调用新策略
        if hasattr(opt, "compat_mode") and opt.compat_mode:
            raise RuntimeError("compat_mode=True 下请调用 densify_and_prune_baseline() 而非新策略。")

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        num_points_before_densify = self.get_xyz.shape[0]

        if not (opt.max_num_gaussians and num_points_before_densify > opt.max_num_gaussians):
            effective_grads_norm = torch.norm(grads, dim=1)
            if opt.use_confidence_for_densify:
                confidences = self.get_roi_confidence.squeeze(-1)
                if hasattr(opt, 'confidence_densify_mode') and opt.confidence_densify_mode == 'add':
                    confidence_bonus = opt.confidence_densify_bonus_scale * torch.relu(confidences)
                    densification_score = effective_grads_norm + confidence_bonus
                else:
                    confidence_multiplier = torch.sigmoid(
                        opt.confidence_densify_sensitivity * (confidences - opt.confidence_densify_center)
                    )
                    densification_score = effective_grads_norm * confidence_multiplier
                densify_mask = (densification_score > opt.densify_grad_threshold)
            else:
                densify_mask = (effective_grads_norm > opt.densify_grad_threshold)

            if densify_scale_threshold is not None:
                big_points_mask = torch.max(self.get_scaling, dim=1).values > densify_scale_threshold * scene_scale
                densify_mask = densify_mask & ~big_points_mask

            split_mask = densify_mask & (torch.max(self.get_scaling, dim=1).values >= opt.densify_scale_threshold * scene_scale)
            clone_mask = densify_mask & ~split_mask

            points_to_clone = self.densify_and_clone(clone_mask, opt)
            points_to_split, split_prune_mask = self.densify_and_split(split_mask, opt)

            all_new_points = []
            if points_to_clone is not None:
                all_new_points.append(points_to_clone)
            if points_to_split is not None:
                all_new_points.append(points_to_split)

            if all_new_points:
                combined_new_points = {}
                for key in all_new_points[0].keys():
                    combined_new_points[key] = torch.cat([d[key] for d in all_new_points], dim=0)
                num_new_points = combined_new_points["xyz"].shape[0]
                new_max_radii2D = torch.zeros(num_new_points, device="cuda")
                self.densification_postfix(
                    new_xyz=combined_new_points["xyz"],
                    new_densities=combined_new_points["densities"],
                    new_scaling=combined_new_points["scaling"],
                    new_rotation=combined_new_points["rotation"],
                    new_roi_confidence=combined_new_points["roi_confidence"],
                    new_max_radii2D=new_max_radii2D
                )
                padding = torch.zeros(num_new_points, device="cuda", dtype=torch.bool)
                split_prune_mask = torch.cat((split_prune_mask, padding))
        else:
            split_prune_mask = torch.zeros(num_points_before_densify, device="cuda", dtype=torch.bool)

        opacities = self.get_density
        scales = self.get_scaling
        opacity_prune_mask = (opacities < opt.opacity_prune_threshold).squeeze()
        scale_prune_mask = torch.zeros_like(opacity_prune_mask)
        if max_scale:
            scale_prune_mask = (torch.max(scales, dim=1).values > max_scale * scene_scale)
        screen_size_prune_mask = torch.zeros_like(opacity_prune_mask)
        if opt.max_screen_size:
            screen_size_prune_mask = (self.max_radii2D > opt.max_screen_size)

        base_prune_mask = opacity_prune_mask | scale_prune_mask | screen_size_prune_mask

        if opt.use_confidence_modulation:
            confidences = self.get_roi_confidence.squeeze(-1)
            pruning_prob = torch.sigmoid(-opt.confidence_prune_steepness * (confidences - opt.confidence_prune_center))
            random_values = torch.rand_like(pruning_prob)
            probabilistic_prune_mask = (random_values < pruning_prob)
            final_prune_mask = base_prune_mask | probabilistic_prune_mask
        else:
            final_prune_mask = base_prune_mask

        total_prune_mask = split_prune_mask | final_prune_mask
        if torch.sum(total_prune_mask) > 0:
            self.prune_points(total_prune_mask)

        # 注意：新策略中不要清零 max_radii2D，这会改变剪枝行为；保持与baseline一致
        torch.cuda.empty_cache()
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
        # self.max_radii2D.zero_()  # 不要清零，保持与baseline一致

    def densify_and_clone(self, clone_mask, opt: OptimizationParams):
        if not clone_mask.any():
            return None
        half_density = self.density_inverse_activation(self.get_density[clone_mask] * 0.5)
        self._density.data[clone_mask] = half_density.data
        new_confidence_clone = self._roi_confidence[clone_mask] * opt.clone_confidence_decay_factor
        new_points_dict = {
            "xyz": self._xyz[clone_mask].clone(),
            "densities": half_density,
            "scaling": self._scaling[clone_mask].clone(),
            "rotation": self._rotation[clone_mask].clone(),
            "roi_confidence": new_confidence_clone,
        }
        return new_points_dict

    def densify_and_split(self, split_mask, opt: OptimizationParams, N: int = 2):
        if not split_mask.any():
            return None, torch.zeros(self.get_xyz.shape[0], device="cuda", dtype=torch.bool)
        split_density = self.density_inverse_activation(self.get_density[split_mask].repeat(N, 1) * (1 / N))
        stds = self.get_scaling[split_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[split_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[split_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[split_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[split_mask].repeat(N, 1)
        new_confidence_split = (self._roi_confidence[split_mask] * opt.clone_confidence_decay_factor).repeat(N, 1)
        new_points_dict = {
            "xyz": new_xyz,
            "densities": split_density,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "roi_confidence": new_confidence_split,
        }
        return new_points_dict, split_mask

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify_and_split_baseline(self, grads, grad_threshold, densify_scale_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > densify_scale_threshold,
        )
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_density = self.density_inverse_activation(
            self.get_density[selected_pts_mask].repeat(N, 1) * (1 / N)
        )
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)
        # baseline densification_postfix
        self.densification_postfix(
            new_xyz,
            new_density,
            new_scaling,
            new_rotation,
            new_roi_confidence=self._roi_confidence[selected_pts_mask].repeat(N, 1) if self._roi_confidence.numel() > 0 else None,
            new_max_radii2D=new_max_radii2D,
        )
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool))
        )
        self.prune_points(prune_filter)

    def densify_and_clone_baseline(self, grads, grad_threshold, densify_scale_threshold):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold,
        )
        if not selected_pts_mask.any():
            return
        new_xyz = self._xyz[selected_pts_mask]
        new_densities = self.density_inverse_activation(self.get_density[selected_pts_mask] * 0.5)
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]
        # 同步原点密度折半
        self._density[selected_pts_mask] = new_densities
        self.densification_postfix(
            new_xyz,
            new_densities,
            new_scaling,
            new_rotation,
            new_roi_confidence=self._roi_confidence[selected_pts_mask] if self._roi_confidence.numel() > 0 else None,
            new_max_radii2D=new_max_radii2D,
        )

    def densify_and_prune_baseline(
        self,
        max_grad,
        min_density,
        max_screen_size,
        max_scale,
        max_num_gaussians,
        densify_scale_threshold,
        bbox=None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if densify_scale_threshold:
            if not max_num_gaussians or (max_num_gaussians and grads.shape[0] < max_num_gaussians):
                self.densify_and_clone_baseline(grads, max_grad, densify_scale_threshold)
                self.densify_and_split_baseline(grads, max_grad, densify_scale_threshold)

        prune_mask = (self.get_density < min_density).squeeze()

        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = (
                (xyz[:, 0] < bbox[0, 0]) | (xyz[:, 0] > bbox[1, 0]) |
                (xyz[:, 1] < bbox[0, 1]) | (xyz[:, 1] > bbox[1, 1]) |
                (xyz[:, 2] < bbox[0, 2]) | (xyz[:, 2] > bbox[1, 2])
            )
            prune_mask = prune_mask | prune_mask_xyz

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        if max_scale:
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale
            prune_mask = torch.logical_or(prune_mask, big_points_ws)

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
        return grads
