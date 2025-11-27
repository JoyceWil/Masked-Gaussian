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
from r2_gaussian.gaussian.structure_guardian import StructureGuardian


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
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
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

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def create_from_pcd(self, xyz, density, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(xyz).float().cuda()
        print(f"Initialize gaussians from {fused_point_cloud.shape[0]} estimated points")
        fused_density = self.density_inverse_activation(torch.tensor(density)).float().cuda()
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.001 ** 2)
        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._density = nn.Parameter(fused_density.requires_grad_(True))
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
            if param_group["name"] == "xyz": param_group["lr"] = self.xyz_scheduler_args(iteration)
            if param_group["name"] == "density": param_group["lr"] = self.density_scheduler_args(iteration)
            if param_group["name"] == "scaling": param_group["lr"] = self.scaling_scheduler_args(iteration)
            if param_group["name"] == "rotation": param_group["lr"] = self.rotation_scheduler_args(iteration)

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz", "density"]
        for i in range(self._scaling.shape[1]): l.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]): l.append(f"rot_{i}")
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz, normals, densities, scale, rotation = (
        self._xyz.detach().cpu().numpy(), np.zeros_like(self._xyz.detach().cpu().numpy()),
        self._density.detach().cpu().numpy(), self._scaling.detach().cpu().numpy(),
        self._rotation.detach().cpu().numpy())
        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, densities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_density(self, reset_density=1.0):
        densities_new = self.density_inverse_activation(
            torch.min(self.get_density, torch.ones_like(self.get_density) * reset_density))
        optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
        self._density = optimizable_tensors["density"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        densities = np.asarray(plydata.elements[0]["density"])[..., np.newaxis]
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names): scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names): rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
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

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
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

    def densification_postfix(self, new_xyz, new_densities, new_scaling, new_rotation, new_max_radii2D):
        d = {"xyz": new_xyz, "density": new_densities, "scaling": new_scaling, "rotation": new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=-1)

    def densify_and_split(self, grads, grad_threshold, densify_scale_threshold, N=2,
                          guardian: StructureGuardian = None,
                          use_structure_aware_densification: bool = False,
                          structure_densification_threshold: float = 0.5,
                          quota_add_global: int = None,
                          allow_split: bool = True,
                          guardian_for_quota: StructureGuardian = None):
        # 如果禁用 split，直接返回
        if not allow_split:
            return 0, None

        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask &= (torch.max(self.get_scaling, dim=1).values > densify_scale_threshold)

        # 结构感知“允许增密”过滤（你原逻辑）
        if guardian is not None and use_structure_aware_densification and torch.any(selected_pts_mask):
            candidate_xyz = self.get_xyz[selected_pts_mask]
            permission_mask = guardian.should_densify(candidate_xyz, structure_densification_threshold)
            full_permission_mask = torch.zeros_like(selected_pts_mask)
            full_permission_mask[selected_pts_mask] = permission_mask
            selected_pts_mask &= full_permission_mask

        if not torch.any(selected_pts_mask):
            return 0, selected_pts_mask

        # 若设置了全局配额，用 guardian_for_quota 的结构先验对候选排序，截断
        if quota_add_global is not None:
            if quota_add_global <= 0:
                return 0, selected_pts_mask
            candidate_idx = torch.nonzero(selected_pts_mask, as_tuple=False).squeeze(-1)
            # 基于结构显著性排序：优先结构强的点
            if guardian_for_quota is not None:
                scores, struct_mask = guardian_for_quota.get_pvol_score(self.get_xyz[candidate_idx])
                # 如果希望严格限制在结构区域，可启用以下筛选：
                # candidate_idx = candidate_idx[struct_mask]
                # scores = scores[struct_mask]
            else:
                scores = torch.ones(candidate_idx.numel(), dtype=torch.float, device="cuda")
            topk = min(candidate_idx.numel(), int(np.floor(quota_add_global / max(1, N))))
            if topk <= 0:
                return 0, selected_pts_mask
            _, order = torch.sort(scores, descending=True)
            keep_idx = candidate_idx[order[:topk]]
            mask_cut = torch.zeros_like(selected_pts_mask)
            mask_cut[keep_idx] = True
            selected_pts_mask = mask_cut

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_density = self.density_inverse_activation(self.get_density[selected_pts_mask].repeat(N, 1) * (1 / N))
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_density, new_scaling, new_rotation, new_max_radii2D)
        # prune 旧点
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        return int(N * selected_pts_mask.sum().item()), selected_pts_mask

    def densify_and_clone(self, grads, grad_threshold, densify_scale_threshold,
                          guardian: StructureGuardian = None,
                          use_structure_aware_densification: bool = False,
                          structure_densification_threshold: float = 0.5,
                          quota_add_global: int = None,
                          allow_clone: bool = True,
                          guardian_for_quota: StructureGuardian = None):
        if not allow_clone:
            return 0, None

        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask &= (torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold)

        if guardian is not None and use_structure_aware_densification and torch.any(selected_pts_mask):
            candidate_xyz = self.get_xyz[selected_pts_mask]
            permission_mask = guardian.should_densify(candidate_xyz, structure_densification_threshold)
            full_permission_mask = torch.zeros_like(selected_pts_mask)
            full_permission_mask[selected_pts_mask] = permission_mask
            selected_pts_mask &= full_permission_mask

        if not torch.any(selected_pts_mask):
            return 0, selected_pts_mask

        # 配额裁剪（每个 clone 新增1个点）
        if quota_add_global is not None:
            if quota_add_global <= 0:
                return 0, selected_pts_mask
            candidate_idx = torch.nonzero(selected_pts_mask, as_tuple=False).squeeze(-1)
            if guardian_for_quota is not None:
                scores, struct_mask = guardian_for_quota.get_pvol_score(self.get_xyz[candidate_idx])
                # 如需严格限制为结构区域内：
                # candidate_idx = candidate_idx[struct_mask]
                # scores = scores[struct_mask]
            else:
                scores = torch.ones(candidate_idx.numel(), dtype=torch.float, device="cuda")
            topk = min(candidate_idx.numel(), quota_add_global)
            if topk <= 0:
                return 0, selected_pts_mask
            _, order = torch.sort(scores, descending=True)
            keep_idx = candidate_idx[order[:topk]]
            mask_cut = torch.zeros_like(selected_pts_mask)
            mask_cut[keep_idx] = True
            selected_pts_mask = mask_cut

        new_xyz = self._xyz[selected_pts_mask]
        new_densities = self.density_inverse_activation(self.get_density[selected_pts_mask] * 0.5)
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]
        self._density[selected_pts_mask] = new_densities
        self.densification_postfix(new_xyz, new_densities, new_scaling, new_rotation, new_max_radii2D)
        return int(selected_pts_mask.sum().item()), selected_pts_mask

    def densify_and_prune(
            self,
            max_grad,
            min_density,
            max_screen_size,
            max_scale,
            max_num_gaussians,
            densify_scale_threshold,
            bbox=None,
            guardian: StructureGuardian = None,
            structure_protection_threshold: float = 0.5,
            use_structure_aware_densification: bool = False,
            structure_densification_threshold: float = 0.5,
            # === ADC new args ===
            quota_add_global: int = None,
            allow_split: bool = True,
            allow_clone: bool = True,
            guardian_for_quota: StructureGuardian = None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        added = 0
        if densify_scale_threshold:
            if not max_num_gaussians or (max_num_gaussians and grads.shape[0] < max_num_gaussians):
                # 先 clone（细节/小半径），再 split（大半径/结构），都受同一全局配额约束
                remaining_quota = quota_add_global if quota_add_global is not None else None

                # clone
                add_c, _ = self.densify_and_clone(
                    grads, max_grad, densify_scale_threshold,
                    guardian, use_structure_aware_densification, structure_densification_threshold,
                    quota_add_global=remaining_quota,
                    allow_clone=allow_clone,
                    guardian_for_quota=guardian_for_quota
                )
                added += add_c
                if remaining_quota is not None:
                    remaining_quota = max(0, remaining_quota - add_c)

                # split
                add_s, _ = self.densify_and_split(
                    grads, max_grad, densify_scale_threshold, N=2,
                    guardian=guardian,
                    use_structure_aware_densification=use_structure_aware_densification,
                    structure_densification_threshold=structure_densification_threshold,
                    quota_add_global=remaining_quota,
                    allow_split=allow_split,
                    guardian_for_quota=guardian_for_quota
                )
                added += add_s

        # ---- pruning (保持原逻辑，结构保护仍然有效) ----
        density_prune_candidates_mask = (self.get_density < min_density).squeeze()

        if guardian is not None and torch.any(density_prune_candidates_mask):
            candidate_xyz = self.get_xyz[density_prune_candidates_mask]
            protection_mask_for_candidates = guardian.should_protect(candidate_xyz, structure_protection_threshold)
            full_protection_mask = torch.zeros_like(density_prune_candidates_mask, dtype=torch.bool)
            full_protection_mask[density_prune_candidates_mask] = protection_mask_for_candidates
            density_prune_mask = density_prune_candidates_mask & (~full_protection_mask)
        else:
            density_prune_mask = density_prune_candidates_mask

        prune_mask = density_prune_mask

        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = ((xyz[:, 0] < bbox[0, 0]) | (xyz[:, 0] > bbox[1, 0]) |
                              (xyz[:, 1] < bbox[0, 1]) | (xyz[:, 1] > bbox[1, 1]) |
                              (xyz[:, 2] < bbox[0, 2]) | (xyz[:, 2] > bbox[1, 2]))
            prune_mask = prune_mask | prune_mask_xyz

        if max_screen_size:
            prune_mask = torch.logical_or(prune_mask, self.max_radii2D > max_screen_size)
        if max_scale:
            prune_mask = torch.logical_or(prune_mask, self.get_scaling.max(dim=1).values > max_scale)

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
        return grads

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1