import os
import sys
from typing import NamedTuple
import numpy as np
import os.path as osp
import json
import torch
import pickle

sys.path.append("./")
from r2_gaussian.utils.graphics_utils import BasicPointCloud, fetchPly

mode_id = {
    "parallel": 0,
    "cone": 1,
}


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
    # >>>>> 开始修改: 添加 s_map 字段 <<<<<
    s_map: torch.Tensor
    # >>>>> 结束修改 <<<<<


class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    vol: torch.tensor
    scanner_cfg: dict
    scene_scale: float


def readBlenderInfo(path, eval):
    """Read blender format CT data."""
    # Read meta data
    meta_data_path = osp.join(path, "meta_data.json")
    with open(meta_data_path, "r") as handle:
        meta_data = json.load(handle)
    meta_data["vol"] = osp.join(path, meta_data["vol"])

    if not "dVoxel" in meta_data["scanner"]:
        meta_data["scanner"]["dVoxel"] = list(
            np.array(meta_data["scanner"]["sVoxel"])
            / np.array(meta_data["scanner"]["nVoxel"])
        )
    if not "dDetector" in meta_data["scanner"]:
        meta_data["scanner"]["dDetector"] = list(
            np.array(meta_data["scanner"]["sDetector"])
            / np.array(meta_data["scanner"]["nDetector"])
        )

    # ! We will scale the scene so that the volume of interest is in [-1, 1]^3 cube.
    scene_scale = 2 / max(meta_data["scanner"]["sVoxel"])
    for key_to_scale in [
        "dVoxel",
        "sVoxel",
        "sDetector",
        "dDetector",
        "offOrigin",
        "offDetector",
        "DSD",
        "DSO",
    ]:
        meta_data["scanner"][key_to_scale] = (
                np.array(meta_data["scanner"][key_to_scale]) * scene_scale
        ).tolist()

    cam_infos = readCTameras(meta_data, path, eval, scene_scale)
    train_cam_infos = cam_infos["train"]
    test_cam_infos = cam_infos["test"]

    vol_gt = torch.from_numpy(np.load(meta_data["vol"])).float().cuda()

    scene_info = SceneInfo(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        scanner_cfg=meta_data["scanner"],
        vol=vol_gt,
        scene_scale=scene_scale,
    )
    return scene_info


def readCTameras(meta_data, source_path, eval=False, scene_scale=1.0):
    """Read camera info."""
    cam_cfg = meta_data["scanner"]

    if eval:
        splits = ["train", "test"]
    else:
        splits = ["train"]

    cam_infos = {"train": [], "test": []}
    for split in splits:
        split_info = meta_data["proj_" + split]
        n_split = len(split_info)
        if split == "test":
            uid_offset = len(meta_data["proj_train"])
        else:
            uid_offset = 0

        # >>>>> 开始修改: 准备 S_map 目录 <<<<<
        s_map_dir = None
        use_s_map = False
        # 只为训练集加载 S_map
        if split == "train":
            s_map_dir = osp.join(source_path, "proj_train_s_map", "npy_data")
            use_s_map = osp.isdir(s_map_dir)
            if not use_s_map:
                print(f"\n警告: S_map 目录 '{s_map_dir}' 不存在。将不为训练集加载 S_map。")
        # >>>>> 结束修改 <<<<<

        for i_split in range(n_split):
            sys.stdout.write("\r")
            sys.stdout.write(f"Reading camera {i_split + 1}/{n_split} for {split}")
            sys.stdout.flush()

            frame_info = meta_data["proj_" + split][i_split]
            frame_angle = frame_info["angle"]

            # CT 'transform_matrix' is a camera-to-world transform
            c2w = angle2pose(cam_cfg["DSO"], frame_angle)  # c2w
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = osp.join(source_path, frame_info["file_path"])
            image = np.load(image_path) * scene_scale
            # Note, dDetector is [v, u] not [u, v]
            FovX = np.arctan2(cam_cfg["sDetector"][1] / 2, cam_cfg["DSD"]) * 2
            FovY = np.arctan2(cam_cfg["sDetector"][0] / 2, cam_cfg["DSD"]) * 2

            mode = mode_id[cam_cfg["mode"]]

            # >>>>> 开始修改: 加载对应的 S_map 文件 <<<<<
            s_map_tensor = None
            if use_s_map:
                # 假设 S_map 文件名与投影文件名完全相同
                s_map_filename = osp.basename(frame_info["file_path"])
                s_map_path = osp.join(s_map_dir, s_map_filename)
                if osp.exists(s_map_path):
                    s_map_np = np.load(s_map_path)
                    # 确保 s_map 是 float32 并有 channel 维度 (1, H, W)
                    s_map_tensor = torch.from_numpy(s_map_np.astype(np.float32)).unsqueeze(0)
                else:
                    # 仅在第一次未找到时警告，避免刷屏
                    if i_split == 0:
                        print(f"\n警告: 找不到与 {s_map_filename} 对应的 S_map 文件: {s_map_path}")
            # >>>>> 结束修改 <<<<<

            cam_info = CameraInfo(
                uid=i_split + uid_offset,
                R=R,
                T=T,
                angle=frame_angle,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=osp.basename(image_path).split(".")[0],
                width=cam_cfg["nDetector"][1],
                height=cam_cfg["nDetector"][0],
                mode=mode,
                scanner_cfg=cam_cfg,
                # >>>>> 开始修改: 传递 s_map <<<<<
                s_map=s_map_tensor
                # >>>>> 结束修改 <<<<<
            )
            cam_infos[split].append(cam_info)
        sys.stdout.write("\n")
    return cam_infos


def angle2pose(DSO, angle):
    """Transfer angle to pose (c2w) based on scanner geometry.
    1. rotate -90 degree around x-axis (fixed axis),
    2. rotate 90 degree around z-axis  (fixed axis),
    3. rotate angle degree around z axis  (fixed axis)"""

    phi1 = -np.pi / 2
    R1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi1), -np.sin(phi1)],
            [0.0, np.sin(phi1), np.cos(phi1)],
        ]
    )
    phi2 = np.pi / 2
    R2 = np.array(
        [
            [np.cos(phi2), -np.sin(phi2), 0.0],
            [np.sin(phi2), np.cos(phi2), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R3 = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot = np.dot(np.dot(R3, R2), R1)
    trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans

    return transform


def readNAFInfo(path, eval):
    """Read blender format CT data."""
    # Read data
    with open(path, "rb") as f:
        data = pickle.load(f)
    # ! NAF scanner are measured in mm, but projections are measured in m. Therefore we need to / 1000.
    scanner_cfg = {
        "DSD": data["DSD"] / 1000,
        "DSO": data["DSO"] / 1000,
        "nVoxel": data["nVoxel"],
        "dVoxel": (np.array(data["dVoxel"]) / 1000).tolist(),
        "sVoxel": (np.array(data["nVoxel"]) * np.array(data["dVoxel"]) / 1000).tolist(),
        "nDetector": data["nDetector"],
        "dDetector": (np.array(data["dDetector"]) / 1000).tolist(),
        "sDetector": (
                np.array(data["nDetector"]) * np.array(data["dDetector"]) / 1000
        ).tolist(),
        "offOrigin": (np.array(data["offOrigin"]) / 1000).tolist(),
        "offDetector": (np.array(data["offDetector"]) / 1000).tolist(),
        "totalAngle": data["totalAngle"],
        "startAngle": data["startAngle"],
        "accuracy": data["accuracy"],
        "mode": data["mode"],
        "filter": None,
    }

    # ! We will scale the scene so that the volume of interest is in [-1, 1]^3 cube.
    scene_scale = 2 / max(scanner_cfg["sVoxel"])
    for key_to_scale in [
        "dVoxel",
        "sVoxel",
        "sDetector",
        "dDetector",
        "offOrigin",
        "offDetector",
        "DSD",
        "DSO",
    ]:
        scanner_cfg[key_to_scale] = (
                np.array(scanner_cfg[key_to_scale]) * scene_scale
        ).tolist()

    # Generate camera infos
    if eval:
        splits = ["train", "test"]
    else:
        splits = ["train"]
    cam_infos = {"train": [], "test": []}

    # >>>>> 开始修改: 准备 S_map 目录 (NAF) <<<<<
    # NAF 格式的数据源是单个 pickle 文件，我们需要找到其所在目录
    source_dir = osp.dirname(path)
    s_map_dir = None
    use_s_map = False
    # >>>>> 结束修改 <<<<<

    for split in splits:
        if split == "test":
            uid_offset = data["numTrain"]
            n_split = data["numVal"]
        else:
            uid_offset = 0
            n_split = data["numTrain"]
        if split == "test" and "val" in data:
            data_split = data["val"]
        else:
            data_split = data[split]
        angles = data_split["angles"]
        projs = data_split["projections"]

        # >>>>> 开始修改: 检查 S_map 目录是否存在 (NAF) <<<<<
        if split == "train":
            s_map_dir = osp.join(source_dir, "proj_train_s_map", "npy_data")
            use_s_map = osp.isdir(s_map_dir)
            if not use_s_map:
                print(f"\n警告: S_map 目录 '{s_map_dir}' 不存在。将不为训练集加载 S_map。")
        # >>>>> 结束修改 <<<<<

        for i_split in range(n_split):
            sys.stdout.write("\r")
            sys.stdout.write(f"Reading camera {i_split + 1}/{n_split} for {split}")
            sys.stdout.flush()

            frame_angle = angles[i_split]
            c2w = angle2pose(scanner_cfg["DSO"], frame_angle)
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image = projs[i_split] * scene_scale

            # Note, dDetector is [v, u] not [u, v]
            FovX = np.arctan2(scanner_cfg["sDetector"][1] / 2, scanner_cfg["DSD"]) * 2
            FovY = np.arctan2(scanner_cfg["sDetector"][0] / 2, scanner_cfg["DSD"]) * 2

            mode = mode_id[scanner_cfg["mode"]]

            # >>>>> 开始修改: 加载对应的 S_map 文件 (NAF) <<<<<
            s_map_tensor = None
            if use_s_map:
                # NAF 格式没有文件名，我们根据索引来匹配
                s_map_filename = f"{i_split + uid_offset:04d}.npy"
                s_map_path = osp.join(s_map_dir, s_map_filename)
                if osp.exists(s_map_path):
                    s_map_np = np.load(s_map_path)
                    s_map_tensor = torch.from_numpy(s_map_np.astype(np.float32)).unsqueeze(0)
                else:
                    if i_split == 0:
                        print(f"\n警告: 找不到与索引 {i_split + uid_offset:04d} 对应的 S_map 文件: {s_map_path}")
            # >>>>> 结束修改 <<<<<

            cam_info = CameraInfo(
                uid=i_split + uid_offset,
                R=R,
                T=T,
                angle=frame_angle,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=None,
                image_name=f"{i_split + uid_offset:04d}",
                width=scanner_cfg["nDetector"][1],
                height=scanner_cfg["nDetector"][0],
                mode=mode,
                scanner_cfg=scanner_cfg,
                # >>>>> 开始修改: 传递 s_map (NAF) <<<<<
                s_map=s_map_tensor
                # >>>>> 结束修改 <<<<<
            )
            cam_infos[split].append(cam_info)
        sys.stdout.write("\n")

    # Store other data
    train_cam_infos = cam_infos["train"]
    test_cam_infos = cam_infos["test"]
    vol_gt = torch.from_numpy(data["image"]).float().cuda()
    scene_info = SceneInfo(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        scanner_cfg=scanner_cfg,
        vol=vol_gt,
        scene_scale=scene_scale,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Blender": readBlenderInfo,
    "NAF": readNAFInfo,
}