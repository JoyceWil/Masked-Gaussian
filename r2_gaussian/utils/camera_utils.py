import sys
import torch
import numpy as np

sys.path.append("./")
from r2_gaussian.dataset.cameras import Camera


def loadCam(args, id, cam_info):
    gt_image = torch.from_numpy(cam_info.image)[None]

    # <<< 修改: 增加对 s_map 类型的判断
    s_map_data = getattr(cam_info, 's_map', None)
    s_map_tensor = None
    if s_map_data is not None:
        # 检查 s_map_data 的类型
        if isinstance(s_map_data, np.ndarray):
            # 如果是 numpy array, 就从 numpy 转换
            s_map_tensor = torch.from_numpy(s_map_data)[None]
        elif isinstance(s_map_data, torch.Tensor):
            # 如果已经是一个 tensor, 就直接使用
            # 确保它有和 gt_image 一样的批次维度
            if len(s_map_data.shape) == 2: # 假设 s_map 是 HxW
                 s_map_tensor = s_map_data[None]
            else:
                 s_map_tensor = s_map_data
        else:
            print(f"[Warning] s_map has an unexpected type: {type(s_map_data)}")
    # >>>>> 结束修改

    return Camera(
        colmap_id=cam_info.uid,
        scanner_cfg=cam_info.scanner_cfg,
        R=cam_info.R,
        T=cam_info.T,
        angle=cam_info.angle,
        mode=cam_info.mode,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        image_name=cam_info.image_name,
        uid=id,
        s_map=s_map_tensor,
        data_device=args.data_device,
    )


def cameraList_from_camInfos(cam_infos, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.eye(4)
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T

    W2C = Rt
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "mode": camera.mode,
        "position_w2c": pos.tolist(),
        "rotation_w2c": serializable_array_2d,
        "FovY": camera.FovY,
        "FovX": camera.FovX,
    }
    return camera_entry