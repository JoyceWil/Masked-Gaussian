# r2_gaussian/dataset/cameras.py (已更新)
import os
import sys
import torch
from torch import nn
import numpy as np

sys.path.append("./")
from r2_gaussian.utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        scanner_cfg,
        R,
        T,
        angle,
        mode,
        FoVx,
        FoVy,
        image,
        image_name,
        uid,
        args,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.angle = angle
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.mode = mode
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.original_image = image.to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # 【修改2】动态加载所有掩码
        # 检查并加载 soft_mask
        if hasattr(args, 'soft_mask_dir') and args.soft_mask_dir:
            soft_mask_path = os.path.join(args.soft_mask_dir, self.image_name + ".npy")
            if os.path.exists(soft_mask_path):
                self.soft_mask = torch.from_numpy(np.load(soft_mask_path)).float()

        # 检查并加载 core_mask
        if hasattr(args, 'core_mask_dir') and args.core_mask_dir:
            core_mask_path = os.path.join(args.core_mask_dir, self.image_name + ".npy")
            if os.path.exists(core_mask_path):
                self.core_mask = torch.from_numpy(np.load(core_mask_path)).float()

        # 检查并加载 air_mask
        if hasattr(args, 'air_mask_dir') and args.air_mask_dir:
            air_mask_path = os.path.join(args.air_mask_dir, self.image_name + ".npy")
            if os.path.exists(air_mask_path):
                self.air_mask = torch.from_numpy(np.load(air_mask_path)).float()

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                fovX=self.FoVx,
                fovY=self.FoVy,
                mode=mode,
                scanner_cfg=scanner_cfg,
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    # ... (MiniCam 类保持不变) ...
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]