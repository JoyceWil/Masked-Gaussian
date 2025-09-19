# r2_gaussian/dataset/__init__.py (最终修复版)
import os
import sys
import random
import json
import numpy as np
import os.path as osp
import torch
import pickle

sys.path.append("./")
from r2_gaussian.gaussian import GaussianModel
from r2_gaussian.arguments import ModelParams
from r2_gaussian.dataset.dataset_readers import sceneLoadTypeCallbacks
from r2_gaussian.utils.camera_utils import cameraList_from_camInfos, Camera
from r2_gaussian.utils.general_utils import t2a


class Scene:
    gaussians: GaussianModel

    def __init__(
            self,
            args: ModelParams,
            shuffle=True,
    ):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = None

        # --- 【核心修改 1】 (这部分是您之前的修改，保持不变) ---
        self.soft_mask_dir = args.soft_mask_dir
        self.core_mask_dir = args.core_mask_dir

        self.use_roi_masks = self.soft_mask_dir and self.core_mask_dir and \
                             osp.exists(self.soft_mask_dir) and osp.exists(self.core_mask_dir)

        if self.use_roi_masks:
            print("2D动态掩码目录已找到，ROI管理将被激活。")
            print(f"  - 软组织掩码目录: {self.soft_mask_dir}")
            print(f"  - 核心骨架掩码目录: {self.core_mask_dir}")
        else:
            print("警告: 未提供有效的2D掩码目录，ROI管理将不会被激活。")

        # --- 【核心修复 1/2：接收 noise_level 参数】 ---
        # 从传入的 args 对象中安全地获取 noise_level，如果不存在则默认为 0.0
        self.noise_level = getattr(args, 'noise_level', 0.0)

        # Read scene info
        if osp.exists(osp.join(args.source_path, "meta_data.json")):
            # --- 【核心修复 2/2：传递 noise_level 参数】 ---
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.eval, self.noise_level)
        elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
            # --- 【核心修复 2/2：传递 noise_level 参数】 ---
            scene_info = sceneLoadTypeCallbacks["NAF"](args.source_path, args.eval, self.noise_level)
        else:
            assert False, f"Could not recognize scene type: {args.source_path}."

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)

        # --- 【核心修改 2】 (这部分是您之前的修改，保持不变) ---
        if self.use_roi_masks:
            print("正在为训练相机加载2D掩码...")
            for camera in self.train_cameras:
                base_name = osp.splitext(osp.basename(camera.image_name))[0] + '.npy'
                soft_mask_path = osp.join(self.soft_mask_dir, base_name)
                core_mask_path = osp.join(self.core_mask_dir, base_name)

                if osp.exists(soft_mask_path):
                    camera.soft_mask = torch.from_numpy(np.load(soft_mask_path)).float()
                else:
                    print(f"警告: 找不到视图 {base_name} 对应的软组织掩码，将使用空掩码。")
                    h, w = camera.image_height, camera.image_width
                    camera.soft_mask = torch.zeros((h, w), dtype=torch.float32)

                if osp.exists(core_mask_path):
                    camera.core_mask = torch.from_numpy(np.load(core_mask_path)).float()
                else:
                    print(f"警告: 找不到视图 {base_name} 对应的核心骨架掩码，将使用空掩码。")
                    h, w = camera.image_height, camera.image_width
                    camera.core_mask = torch.zeros((h, w), dtype=torch.float32)

        self.vol_gt = scene_info.vol
        self.scanner_cfg = scene_info.scanner_cfg
        self.scene_scale = scene_info.scene_scale
        self.bbox = torch.stack(
            [
                torch.tensor(self.scanner_cfg["offOrigin"])
                - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
                torch.tensor(self.scanner_cfg["offOrigin"])
                + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )
        self.grid_center = torch.tensor(self.scanner_cfg["offOrigin"], dtype=torch.float32)
        sVoxel = torch.tensor(self.scanner_cfg["sVoxel"], dtype=torch.float32)
        nVoxel = torch.tensor(self.scanner_cfg["nVoxel"], dtype=torch.float32)
        self.voxel_size = sVoxel / nVoxel
        print("FDK体积和坐标转换信息已准备好。")

    def save(self, iteration, queryfunc):
        point_cloud_path = osp.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(osp.join(point_cloud_path, "point_cloud.ply"))
        if queryfunc is not None:
            vol_pred = queryfunc(self.gaussians)["vol"]
            vol_gt = self.vol_gt
            np.save(osp.join(point_cloud_path, "vol_gt.npy"), t2a(vol_gt))
            np.save(
                osp.join(point_cloud_path, "vol_pred.npy"),
                t2a(vol_pred),
            )

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras