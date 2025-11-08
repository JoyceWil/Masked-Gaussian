import os
import json
import numpy as np
import tigre
import tigre.algorithms
from tqdm import tqdm
import matplotlib.pyplot as plt


def generate_fdk_and_importance(base_dir: str,
                                fdk_filename: str = "V_fdk.npy",
                                importance_filename: str = "V_importance.npy"):
    """
    执行FDK重建以获得 V_fdk (强度)。
    同时，加载 2D S_maps 并反投影它们，以获得 V_importance (结构)。
    """
    print("--- Starting FDK & Importance Volume Generation ---")

    # --- 1. 加载元数据 ---
    meta_data_path = os.path.join(base_dir, 'meta_data.json')
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    scanner_params = meta_data['scanner']
    proj_train_info = meta_data['proj_train']
    print("Successfully loaded metadata.")

    # --- 2. 加载投影和 S_maps ---
    projections = []
    s_maps = []
    angles = []

    s_map_dir = os.path.join(base_dir, 'proj_train_s_map', 'npy_data')
    if not os.path.isdir(s_map_dir):
        print(f"错误: S_map 目录未找到: {s_map_dir}")
        print("请先运行 'step_0_preprocess_s_map_2d.py'")
        return

    print("Loading projection data and S_map data...")
    for proj_info in tqdm(proj_train_info, desc="Loading data"):
        proj_path = os.path.join(base_dir, proj_info['file_path'])
        projections.append(np.load(proj_path))

        # 构造 s_map 路径
        s_map_filename = os.path.basename(proj_info['file_path'])
        s_map_path = os.path.join(s_map_dir, s_map_filename)

        if not os.path.exists(s_map_path):
            print(f"错误: 对应的 S_map 文件未找到: {s_map_path}")
            return

        s_maps.append(np.load(s_map_path))
        angles.append(proj_info['angle'])

    projections = np.stack(projections, axis=0).astype(np.float32)
    s_maps = np.stack(s_maps, axis=0).astype(np.float32)
    angles = np.array(angles, dtype=np.float32)

    print(f"Loaded {projections.shape[0]} projections and {s_maps.shape[0]} S_maps.")

    # --- 3. 配置 TIGRE 几何 ---
    geo = tigre.geometry(mode='cone')
    geo.DSD = scanner_params['DSD']
    geo.DSO = scanner_params['DSO']
    geo.nDetector = np.array(scanner_params['nDetector'], dtype=np.int32)
    geo.sDetector = np.array(scanner_params['sDetector'], dtype=np.float32)
    geo.offDetector = np.array(scanner_params['offDetector'], dtype=np.float32)
    geo.nVoxel = np.array(scanner_params['nVoxel'], dtype=np.int32)
    geo.sVoxel = np.array(scanner_params['sVoxel'], dtype=np.float32)
    geo.offOrigin = np.array(scanner_params['offOrigin'], dtype=np.float32)
    geo.rotDetector = np.array([0, 0, 0], dtype=np.float32)
    geo.dDetector = geo.sDetector / geo.nDetector
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.accuracy = 0.5
    print("TIGRE geometry configured.")

    # --- 4. 执行 FDK (强度体积) ---
    print("Running FDK algorithm (for Intensity)...")
    fdk_volume = tigre.algorithms.fdk(projections, geo, angles)
    print("FDK reconstruction complete.")

    output_path_fdk = os.path.join(base_dir, fdk_filename)
    np.save(output_path_fdk, fdk_volume)
    print(f"FDK (Intensity) volume saved to: {output_path_fdk}")

    # --- 5. 执行反投影 (结构体积) [!!! 修正点 !!!] ---
    print("Running Back-Projection (Atb) on S_maps (for Structure)...")
    # 'Atb' (Transpose of A) 是 TIGRE 的反投影操作
    importance_volume = tigre.Atb(s_maps, geo, angles)
    print("S_map back-projection complete.")

    output_path_importance = os.path.join(base_dir, importance_filename)
    np.save(output_path_importance, importance_volume)
    print(f"Importance volume saved to: {output_path_importance}")

    # --- 6. (可选) 可视化 ---
    if fdk_volume is not None and importance_volume is not None:
        slice_idx = fdk_volume.shape[0] // 2

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        im1 = axes[0].imshow(fdk_volume[slice_idx, :, :], cmap='gray')
        axes[0].set_title(f'V_fdk (Intensity) - Slice {slice_idx}')
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(importance_volume[slice_idx, :, :], cmap='hot')
        axes[1].set_title(f'V_importance (Structure) - Slice {slice_idx}')
        fig.colorbar(im2, ax=axes[1])

        plt.show()


if __name__ == '__main__':
    dataset_base_dir = r'data\synthetic_dataset\cone_ntrain_25_angle_360\0_chest_cone'

    if not os.path.isdir(dataset_base_dir):
        print(f"Error: Directory not found at '{dataset_base_dir}'")
    else:
        # 确保 S_map 预处理已运行
        s_map_dir_check = os.path.join(dataset_base_dir, 'proj_train_s_map', 'npy_data')
        if not os.path.isdir(s_map_dir_check):
            print(f"Error: S_map 目录未找到: {s_map_dir_check}")
            print("请在运行此脚本前，先运行 'step_0_preprocess_s_map_2d.py'")
        else:
            generate_fdk_and_importance(dataset_base_dir)