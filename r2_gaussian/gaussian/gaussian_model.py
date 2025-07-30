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

    def update_roi_confidence(self, render_pkg, viewpoint_cam, standard_reward, core_bonus_reward, background_reward):
        """
        【V6.0 最终修复版】
        不再信任来自渲染管线的'viewspace_points'，而是在函数内部手动、可靠地重投影。
        这可以保证无论渲染器实现如何，我们的ROI分类总是基于正确的2D坐标。
        """
        with torch.no_grad():
            if not hasattr(viewpoint_cam, 'soft_mask') or not hasattr(viewpoint_cam, 'core_mask'):
                return

            # --- 步骤 1: 获取必要数据 ---
            H_img, W_img = viewpoint_cam.image_height, viewpoint_cam.image_width
            soft_mask = viewpoint_cam.soft_mask.to(self.get_xyz.device)
            core_mask = viewpoint_cam.core_mask.to(self.get_xyz.device)
            visibility_filter = render_pkg["visibility_filter"]

            # 获取所有可见点的绝对索引
            visible_indices = visibility_filter.nonzero(as_tuple=True)[0]
            if visible_indices.numel() == 0:
                return

            # 获取这些可见点的3D世界坐标
            visible_xyz_world = self._xyz[visible_indices]

            # --- 步骤 2: 可靠的3D到2D重投影 ---
            # 从相机获取完整的投影变换矩阵 (World -> NDC)
            P = viewpoint_cam.full_proj_transform.cuda()

            # 将3D世界坐标点转换为齐次坐标 (N, 4)
            points_homogeneous = torch.cat(
                [visible_xyz_world, torch.ones(visible_xyz_world.shape[0], 1, device="cuda")], dim=1)

            # 应用投影变换 (N, 4)
            points_clip = points_homogeneous @ P.T

            # 执行透视除法，得到归一化设备坐标 (NDC) (N, 3)
            # 我们只关心x, y。加上一个极小值避免除以0。
            points_ndc = points_clip[..., :2] / (points_clip[..., 3:4] + 1e-8)

            # 将NDC坐标 [-1, 1] 转换为像素坐标 [0, W-1] 和 [0, H-1]
            # x_pix = (x_ndc + 1) * W / 2
            # y_pix = (y_ndc + 1) * H / 2
            # 注意：在3DGS中，Y轴通常是向下的，所以是 (1 - y_ndc)
            reliable_xy_proj = torch.zeros_like(points_ndc)
            reliable_xy_proj[:, 0] = (points_ndc[:, 0] + 1.0) * W_img / 2.0
            reliable_xy_proj[:, 1] = (1.0 - points_ndc[:, 1]) * H_img / 2.0  # Y轴翻转

            # --- 步骤 3: 使用可靠的2D坐标进行后续操作 ---
            valid_xy_float = reliable_xy_proj

            # 诊断探针现在应该会显示正确的范围了
            if not hasattr(self, "final_probe_printed"):
                print("\n--- [决定性诊断探针 v6.0 - 可靠重投影] ---")
                min_x, max_x = valid_xy_float[:, 0].min(), valid_xy_float[:, 0].max()
                min_y, max_y = valid_xy_float[:, 1].min(), valid_xy_float[:, 1].max()
                print(f"  -> 探查 'reliable_xy_proj' 的坐标范围:")
                print(f"  -> X 范围: [{min_x:.4f}, {max_x:.4f}]")
                print(f"  -> Y 范围: [{min_y:.4f}, {max_y:.4f}]")
                self.final_probe_printed = True

            # 使用图像尺寸进行过滤
            valid_coords_mask = (valid_xy_float[:, 0] >= 0) & (valid_xy_float[:, 0] < W_img) & \
                                (valid_xy_float[:, 1] >= 0) & (valid_xy_float[:, 1] < H_img)

            in_image_indices_absolute = visible_indices[valid_coords_mask].contiguous()
            if in_image_indices_absolute.numel() == 0:
                return

            final_valid_xy = valid_xy_float[valid_coords_mask]

            # --- 步骤 4: 采样和分类 (这部分逻辑不变) ---
            soft_mask_for_sampling = soft_mask.unsqueeze(0).unsqueeze(0)
            core_mask_for_sampling = core_mask.unsqueeze(0).unsqueeze(0)

            # 归一化用于grid_sample
            normalized_xy = torch.zeros_like(final_valid_xy)
            normalized_xy[:, 0] = (final_valid_xy[:, 0] / (W_img - 1)) * 2 - 1
            normalized_xy[:, 1] = (final_valid_xy[:, 1] / (H_img - 1)) * 2 - 1
            grid = normalized_xy.unsqueeze(0).unsqueeze(0)

            # 采样
            sampled_soft_values = torch.nn.functional.grid_sample(soft_mask_for_sampling, grid, mode='bilinear',
                                                                  padding_mode='zeros', align_corners=True).squeeze()
            sampled_core_values = torch.nn.functional.grid_sample(core_mask_for_sampling, grid, mode='bilinear',
                                                                  padding_mode='zeros', align_corners=True).squeeze()

            # 使用您设定的阈值进行分类
            is_in_soft_mask = sampled_soft_values > 0.07  # 软区阈值
            is_in_core_mask = sampled_core_values > 0.08  # 核心区阈值

            indices_in_core = in_image_indices_absolute[is_in_core_mask]
            is_in_soft_only = is_in_soft_mask & ~is_in_core_mask
            indices_in_soft_only = in_image_indices_absolute[is_in_soft_only]
            is_background = ~is_in_soft_mask
            indices_background = in_image_indices_absolute[is_background]

            if hasattr(self, "final_probe_printed") and not hasattr(self, "final_count_printed"):
                print(
                    f"  -> [分类计数] 核心: {indices_in_core.numel()}, 软区: {indices_in_soft_only.numel()}, 背景: {indices_background.numel()}")
                print("--- [探针结束] ---\n")
                self.final_count_printed = True

            # 更新置信度
            self._roi_confidence[indices_in_core] += (standard_reward + core_bonus_reward)
            self._roi_confidence[indices_in_soft_only] += standard_reward
            self._roi_confidence[indices_background] += background_reward

            self._roi_confidence.clamp_(-10, 10)

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

    def densify_and_prune(
            self,
            max_grad,
            min_density,
            max_screen_size,
            max_scale,
            max_num_gaussians,
            densify_scale_threshold,
            bbox=None,
            roi_protect_threshold=1e5,
            roi_candidate_threshold=-1e5,
    ):
        """
        【最终实现】实现基于ROI置信度的三层管理策略。
        - 核心骨架 (roi_confidence > roi_protect_threshold): 优先分裂，受保护不被轻易剪枝。
        - 探索前沿 (roi_candidate_threshold < roi_confidence <= roi_protect_threshold): 标准密集化与剪枝。
        - 候选浮点 (roi_confidence <= roi_candidate_threshold): 禁止密集化，标准剪枝。
        """
        with torch.no_grad():
            num_points_before = self.get_xyz.shape[0]
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0

            # --- 保存所有需要的初始状态 ---
            initial_scales = self.get_scaling
            initial_xyz = self.get_xyz
            initial_rotation = self._rotation
            initial_density = self.get_density
            initial_max_radii2D = self.max_radii2D
            initial_roi_confidence = self.get_roi_confidence

            # --- 1. 定义置信度分层 ---
            # 如果 roi_protect_threshold > 1 (即ROI管理未激活), 则所有点都属于探索前沿
            is_roi_active = roi_protect_threshold <= 1.0

            core_mask = (
                        initial_roi_confidence > roi_protect_threshold).squeeze() if is_roi_active else torch.zeros_like(
                initial_roi_confidence, dtype=torch.bool).squeeze()
            candidate_mask = (
                        initial_roi_confidence <= roi_candidate_threshold).squeeze() if is_roi_active else torch.zeros_like(
                initial_roi_confidence, dtype=torch.bool).squeeze()
            # 探索前沿是既非核心也非候选的中间地带
            frontier_mask = ~(core_mask | candidate_mask)

            # --- 2. 密集化策略 ---
            # 计算标准密集化掩码 (基于梯度和点的大小)
            densify_mask = (torch.norm(grads, dim=-1) >= max_grad)

            # 【核心逻辑】禁止“候选浮点”阶层进行任何密集化
            # 无论梯度多大，低置信度的点都没有资格创造后代
            densify_mask[candidate_mask] = False

            # 正常分离克隆和分裂的掩码
            clone_mask = densify_mask & (torch.max(initial_scales, dim=1).values <= densify_scale_threshold)
            split_mask = densify_mask & (torch.max(initial_scales, dim=1).values > densify_scale_threshold)

            # 执行致密化
            self.densify_and_clone_with_mask(clone_mask)
            self.densify_and_split_with_mask(split_mask, initial_scales, initial_xyz, initial_rotation, initial_density,
                                             initial_max_radii2D, initial_roi_confidence)

            # --- 3. 剪枝策略 ---
            num_points_after = self.get_xyz.shape[0]

            # 计算标准剪枝掩码 (透明度低、出界、屏幕尺寸过大、物理尺寸过大)
            prune_mask = (self.get_density < min_density).squeeze()
            if bbox is not None:
                prune_mask |= ~((self._xyz > bbox[0]) & (self._xyz < bbox[1])).all(dim=1)
            if max_screen_size:
                prune_mask |= (self.max_radii2D > max_screen_size)
            if max_scale:
                prune_mask |= (self.get_scaling.max(dim=1).values > max_scale)

            # 【核心逻辑】应用分层剪枝规则
            if is_roi_active:
                # 获取当前所有点的置信度（因为密集化后有了新点）
                current_confidence = self.get_roi_confidence.squeeze()
                current_core_mask = (current_confidence > roi_protect_threshold)

                # 找到所有根据标准规则可能被剪枝的点
                prune_candidates = prune_mask.clone()

                # “核心骨架”阶层获得豁免权：如果一个点是核心骨架，就从剪枝候选中移除
                protected_points_mask = current_core_mask[prune_candidates]
                prune_candidates[prune_candidates.clone()] = ~protected_points_mask

                # 更新最终的剪枝掩码
                prune_mask = prune_candidates

            # 移除被成功分裂的父代高斯点 (这是必须的)
            num_new_points = num_points_after - num_points_before
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