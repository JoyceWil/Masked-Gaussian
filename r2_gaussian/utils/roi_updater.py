import torch
import torch.nn.functional as F


@torch.no_grad()
def update_roi_confidence_with_voting(
        gaussian_model,
        cameras,
        masks,
        confidence_update_rate=0.1,
        use_probabilistic_update=False
):
    """
    根据一组相机视角和对应的2D掩码，更新3D高斯核的ROI置信度。

    Args:
        gaussian_model (GaussianModel): 包含所有高斯核的模型实例。
        cameras (list): 数据集中的相机列表 (Camera对象)。
        masks (torch.Tensor): 一个形状为 (num_cameras, H, W) 的张量，包含所有视角的掩码。
        confidence_update_rate (float): 在概率更新模式下，每次更新的步长。
        use_probabilistic_update (bool): 是否使用概率累积更新法。False则使用多视角投票法。
    """

    # 获取所有高斯核的当前3D世界坐标
    xyz = gaussian_model.get_xyz
    num_points = xyz.shape[0]
    if num_points == 0:
        return

    device = xyz.device
    num_cameras = len(cameras)

    if use_probabilistic_update:
        # --- 方法B: 概率性累积更新法 ---
        # 随机选择一个视角进行更新，以减少计算量并增加随机性
        cam_idx = torch.randint(0, num_cameras, (1,)).item()
        camera = cameras[cam_idx]
        mask = masks[cam_idx]

        # 将3D点投影到该相机视角
        # world_view_transform将世界坐标转换为相机坐标
        # full_proj_transform将世界坐标转换为裁剪空间坐标
        xyz_homogeneous = F.pad(xyz, (0, 1), "constant", 1.0)
        clip_space_coords = xyz_homogeneous @ camera.full_proj_transform.T

        # 透视除法，得到归一化设备坐标 (NDC), 范围[-1, 1]
        # 我们只关心在视锥体内的点 (z > 0 in camera space, which is w > 0 in clip space)
        valid_depth_mask = clip_space_coords[:, 3] > 0.001

        ndc = clip_space_coords[valid_depth_mask, :2] / clip_space_coords[valid_depth_mask, 3, None]

        # 将NDC转换为像素坐标
        H, W = mask.shape
        pixel_coords_x = (ndc[:, 0] + 1.0) * W / 2.0
        pixel_coords_y = (ndc[:, 1] + 1.0) * H / 2.0
        pixel_coords = torch.stack([pixel_coords_x, pixel_coords_y], dim=1)

        # 检查哪些点投影在图像内
        in_frame_mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W) & \
                        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H)

        # 获取在图像内的点的整数坐标
        coords_in_frame = pixel_coords[in_frame_mask].long()

        # 采样掩码，判断这些点是否在ROI内
        is_in_roi_mask = mask[coords_in_frame[:, 1], coords_in_frame[:, 0]].bool()

        # 准备一个完整的更新掩码
        points_in_roi = torch.zeros(coords_in_frame.shape[0], dtype=torch.bool, device=device)
        points_in_roi[is_in_roi_mask] = True

        full_in_roi_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        full_in_roi_mask[valid_depth_mask] = in_frame_mask
        full_in_roi_mask[full_in_roi_mask.clone()] = points_in_roi

        full_out_roi_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        full_out_roi_mask[valid_depth_mask] = in_frame_mask
        full_out_roi_mask[full_out_roi_mask.clone()] = ~points_in_roi

        # 更新置信度
        current_confidence = gaussian_model._roi_confidence
        current_confidence[full_in_roi_mask] += confidence_update_rate
        current_confidence[full_out_roi_mask] -= confidence_update_rate

        # 将置信度限制在[0, 1]范围内
        gaussian_model._roi_confidence = torch.clamp(current_confidence, 0.0, 1.0)

    else:
        # --- 方法A: 多视角投票法 (更稳健) ---
        vote_count = torch.zeros((num_points, 1), device=device)

        for i, camera in enumerate(cameras):
            mask = masks[i]
            H, W = mask.shape

            # 将3D点投影到当前相机视角
            xyz_homogeneous = F.pad(xyz, (0, 1), "constant", 1.0)
            clip_space_coords = xyz_homogeneous @ camera.full_proj_transform.T

            # 过滤掉相机背后的点
            valid_depth_mask = clip_space_coords[:, 3] > 0.001

            # 对于有效点进行后续计算
            valid_xyz = xyz[valid_depth_mask]
            valid_clip_space = clip_space_coords[valid_depth_mask]

            if valid_xyz.shape[0] == 0:
                continue

            # 透视除法 -> NDC
            ndc = valid_clip_space[:, :2] / valid_clip_space[:, 3, None]

            # NDC -> 像素坐标
            pixel_coords_x = (ndc[:, 0] + 1.0) * W / 2.0
            pixel_coords_y = (ndc[:, 1] + 1.0) * H / 2.0
            pixel_coords = torch.stack([pixel_coords_x, pixel_coords_y], dim=1)

            # 找到投影在图像帧内的点
            in_frame_mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W) & \
                            (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H)

            if in_frame_mask.sum() == 0:
                continue

            # 获取这些点的整数坐标
            coords_in_frame = pixel_coords[in_frame_mask].long()

            # 采样掩码，判断这些点是否在ROI内
            is_in_mask = mask[coords_in_frame[:, 1], coords_in_frame[:, 0]].bool()

            # 找到那些既在帧内又在掩码内的点的原始索引
            original_indices = torch.where(valid_depth_mask)[0][in_frame_mask][is_in_mask]

            # 为这些点投上一票
            vote_count[original_indices] += 1

        # 计算最终置信度：得票数 / 总视角数
        new_confidence = vote_count / num_cameras

        # 更新模型中的置信度
        # 为了平滑更新，我们可以使用移动平均
        # new_confidence = old_confidence * (1 - alpha) + new_confidence * alpha
        alpha = 0.5
        gaussian_model._roi_confidence = gaussian_model._roi_confidence * (1 - alpha) + new_confidence * alpha