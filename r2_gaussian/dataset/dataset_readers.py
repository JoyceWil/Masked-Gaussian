# r2_gaussian/dataset/dataset_readers.py (包含可视化调试的最终版)
import os
import sys
from typing import NamedTuple
import numpy as np
import os.path as osp
import json
import torch
import pickle
import matplotlib.pyplot as plt

sys.path.append("./")
from r2_gaussian.utils.graphics_utils import BasicPointCloud, fetchPly

mode_id = {
    "parallel": 0,
    "cone": 1,
}

def add_noise(image, noise_level):
    if noise_level == 0:
        return image
    signal_max = image.max()
    if signal_max == 0:
        return image
    noise_std = signal_max * noise_level
    noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
    noisy_image = image + noise
    return np.maximum(0, noisy_image)

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    angle: float
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mode: int
    scanner_cfg: dict

class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    vol: torch.tensor
    scanner_cfg: dict
    scene_scale: float

def readBlenderInfo(path, eval, noise_level=0.0, noisy_view_indices=None):
    meta_data_path = osp.join(path, "meta_data.json")
    with open(meta_data_path, "r") as handle:
        meta_data = json.load(handle)
    meta_data["vol"] = osp.join(path, meta_data["vol"])
    if not "dVoxel" in meta_data["scanner"]:
        meta_data["scanner"]["dVoxel"] = list(np.array(meta_data["scanner"]["sVoxel"]) / np.array(meta_data["scanner"]["nVoxel"]))
    if not "dDetector" in meta_data["scanner"]:
        meta_data["scanner"]["dDetector"] = list(np.array(meta_data["scanner"]["sDetector"]) / np.array(meta_data["scanner"]["nDetector"]))
    scene_scale = 2 / max(meta_data["scanner"]["sVoxel"])
    for key_to_scale in ["dVoxel", "sVoxel", "sDetector", "dDetector", "offOrigin", "offDetector", "DSD", "DSO"]:
        meta_data["scanner"][key_to_scale] = (np.array(meta_data["scanner"][key_to_scale]) * scene_scale).tolist()
    cam_infos = readCTameras(meta_data, path, eval, scene_scale, noise_level, noisy_view_indices)
    train_cam_infos = cam_infos["train"]
    test_cam_infos = cam_infos["test"]
    vol_gt = torch.from_numpy(np.load(meta_data["vol"])).float().cuda()
    scene_info = SceneInfo(
        train_cameras=train_cam_infos, test_cameras=test_cam_infos,
        scanner_cfg=meta_data["scanner"], vol=vol_gt, scene_scale=scene_scale,
    )
    return scene_info

def readCTameras(meta_data, source_path, eval=False, scene_scale=1.0, noise_level=0.0, noisy_view_indices=None):
    cam_cfg = meta_data["scanner"]
    if eval: splits = ["train", "test"]
    else: splits = ["train"]
    cam_infos = {"train": [], "test": []}
    for split in splits:
        split_info = meta_data["proj_" + split]
        n_split = len(split_info)
        if split == "test": uid_offset = len(meta_data["proj_train"])
        else: uid_offset = 0
        for i_split in range(n_split):
            sys.stdout.write("\r")
            sys.stdout.write(f"Reading camera {i_split + 1}/{n_split} for {split}")
            sys.stdout.flush()
            frame_info = meta_data["proj_" + split][i_split]
            frame_angle = frame_info["angle"]
            c2w = angle2pose(cam_cfg["DSO"], frame_angle)
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            image_path = osp.join(source_path, frame_info["file_path"])
            image_name_with_ext = osp.basename(image_path)
            image_name = image_name_with_ext.split(".")[0]
            clean_image = np.load(image_path) * scene_scale
            image_to_use = clean_image

            should_add_noise = False
            if split == 'train' and noise_level is not None and noise_level > 0:
                # 场景1: 未指定索引，对所有视图加噪 (原始行为)
                if noisy_view_indices is None:
                    should_add_noise = True
                # 场景2: 已指定索引，仅当当前索引匹配时加噪
                elif i_split in noisy_view_indices:
                    should_add_noise = True

            if should_add_noise:
                noisy_image = add_noise(clean_image, noise_level)
                image_to_use = noisy_image

                # 可视化调试逻辑 (只对第一个加噪的图像进行可视化)
                # 我们用一个标志来确保只打印一次
                if not hasattr(readCTameras, 'viz_saved'):
                    viz_dir = "debug_noise_viz"
                    os.makedirs(viz_dir, exist_ok=True)
                    clean_path = osp.join(viz_dir, f"{image_name}_clean.png")
                    plt.imsave(clean_path, clean_image, cmap='gray')
                    noisy_path = osp.join(viz_dir, f"{image_name}_noisy.png")
                    plt.imsave(noisy_path, noisy_image, cmap='gray')
                    print(f"\n[DEBUG] 噪声可视化已保存 (视图索引: {i_split})！请检查以下文件：")
                    print(f"        - 干净图像: {clean_path}")
                    print(f"        - 加噪图像: {noisy_path}\n")
                    readCTameras.viz_saved = True  # 设置标志
            FovX = np.arctan2(cam_cfg["sDetector"][1] / 2, cam_cfg["DSD"]) * 2
            FovY = np.arctan2(cam_cfg["sDetector"][0] / 2, cam_cfg["DSD"]) * 2
            mode = mode_id[cam_cfg["mode"]]
            cam_info = CameraInfo(
                uid=i_split + uid_offset, R=R, T=T, angle=frame_angle, FovY=FovY, FovX=FovX,
                image=image_to_use,
                image_path=image_path, image_name=image_name,
                width=cam_cfg["nDetector"][1], height=cam_cfg["nDetector"][0],
                mode=mode, scanner_cfg=cam_cfg,
            )
            cam_infos[split].append(cam_info)
        sys.stdout.write("\n")
    return cam_infos

def angle2pose(DSO, angle):
    phi1 = -np.pi / 2
    R1 = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi1), -np.sin(phi1)], [0.0, np.sin(phi1), np.cos(phi1)]])
    phi2 = np.pi / 2
    R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0], [np.sin(phi2), np.cos(phi2), 0.0], [0.0, 0.0, 1.0]])
    R3 = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    rot = np.dot(np.dot(R3, R2), R1)
    trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans
    return transform

def readNAFInfo(path, eval, noise_level=0.0, noisy_view_indices=None):
    with open(path, "rb") as f:
        data = pickle.load(f)
    scanner_cfg = {
        "DSD": data["DSD"] / 1000, "DSO": data["DSO"] / 1000, "nVoxel": data["nVoxel"],
        "dVoxel": (np.array(data["dVoxel"]) / 1000).tolist(),
        "sVoxel": (np.array(data["nVoxel"]) * np.array(data["dVoxel"]) / 1000).tolist(),
        "nDetector": data["nDetector"], "dDetector": (np.array(data["dDetector"]) / 1000).tolist(),
        "sDetector": (np.array(data["nDetector"]) * np.array(data["dDetector"]) / 1000).tolist(),
        "offOrigin": (np.array(data["offOrigin"]) / 1000).tolist(),
        "offDetector": (np.array(data["offDetector"]) / 1000).tolist(),
        "totalAngle": data["totalAngle"], "startAngle": data["startAngle"],
        "accuracy": data["accuracy"], "mode": data["mode"], "filter": None,
    }
    scene_scale = 2 / max(scanner_cfg["sVoxel"])
    for key_to_scale in ["dVoxel", "sVoxel", "sDetector", "dDetector", "offOrigin", "offDetector", "DSD", "DSO"]:
        scanner_cfg[key_to_scale] = (np.array(scanner_cfg[key_to_scale]) * scene_scale).tolist()
    if eval: splits = ["train", "test"]
    else: splits = ["train"]
    cam_infos = {"train": [], "test": []}
    for split in splits:
        if split == "test":
            uid_offset = data["numTrain"]
            n_split = data["numVal"]
        else:
            uid_offset = 0
            n_split = data["numTrain"]
        if split == "test" and "val" in data: data_split = data["val"]
        else: data_split = data[split]
        angles = data_split["angles"]
        projs = data_split["projections"]
        for i_split in range(n_split):
            sys.stdout.write("\r")
            sys.stdout.write(f"Reading camera {i_split + 1}/{n_split} for {split}")
            sys.stdout.flush()
            frame_angle = angles[i_split]
            c2w = angle2pose(scanner_cfg["DSO"], frame_angle)
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            clean_image = projs[i_split] * scene_scale
            image_to_use = clean_image
            if split == 'train' and noise_level is not None and noise_level > 0:
                noisy_image = add_noise(clean_image, noise_level)
                image_to_use = clean_image

                # <<< 核心修改：更新加噪逻辑 >>>
                should_add_noise = False
                if split == 'train' and noise_level is not None and noise_level > 0:
                    # 场景1: 未指定索引，对所有视图加噪 (原始行为)
                    if noisy_view_indices is None:
                        should_add_noise = True
                    # 场景2: 已指定索引，仅当当前索引匹配时加噪
                    elif i_split in noisy_view_indices:
                        should_add_noise = True

                if should_add_noise:
                    noisy_image = add_noise(clean_image, noise_level)
                    image_to_use = noisy_image

                    # 可视化调试逻辑 (只对第一个加噪的图像进行可视化)
                    if not hasattr(readNAFInfo, 'viz_saved'):
                        viz_dir = "debug_noise_viz"
                        os.makedirs(viz_dir, exist_ok=True)
                        clean_path = osp.join(viz_dir, f"naf_{i_split + uid_offset:04d}_clean.png")
                        plt.imsave(clean_path, clean_image, cmap='gray')
                        noisy_path = osp.join(viz_dir, f"naf_{i_split + uid_offset:04d}_noisy.png")
                        plt.imsave(noisy_path, noisy_image, cmap='gray')
                        print(
                            f"\n[DEBUG] NAF噪声可视化已保存 (视图索引: {i_split})！请检查: {clean_path} 和 {noisy_path}\n")
                        readNAFInfo.viz_saved = True  # 设置标志
            FovX = np.arctan2(scanner_cfg["sDetector"][1] / 2, scanner_cfg["DSD"]) * 2
            FovY = np.arctan2(scanner_cfg["sDetector"][0] / 2, scanner_cfg["DSD"]) * 2
            mode = mode_id[scanner_cfg["mode"]]
            cam_info = CameraInfo(
                uid=i_split + uid_offset, R=R, T=T, angle=frame_angle, FovY=FovY, FovX=FovX,
                image=image_to_use, image_path=None, image_name=f"{i_split + uid_offset:04d}",
                width=scanner_cfg["nDetector"][1], height=scanner_cfg["nDetector"][0],
                mode=mode, scanner_cfg=scanner_cfg,
            )
            cam_infos[split].append(cam_info)
        sys.stdout.write("\n")
    train_cam_infos = cam_infos["train"]
    test_cam_infos = cam_infos["test"]
    vol_gt = torch.from_numpy(data["image"]).float().cuda()
    scene_info = SceneInfo(
        train_cameras=train_cam_infos, test_cameras=test_cam_infos,
        scanner_cfg=scanner_cfg, vol=vol_gt, scene_scale=scene_scale,
    )
    return scene_info

sceneLoadTypeCallbacks = {
    "Blender": readBlenderInfo,
    "NAF": readNAFInfo,
}