import os
import json
import numpy as np
import tigre
import tigre.algorithms
from tigre.utilities import filtering
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time


def run_reconstruction(args):
    """
    主函数，用于加载数据、配置几何、执行重建并保存结果。
    """
    base_dir = args.base_dir
    algorithm = args.algorithm
    n_iter = args.n_iter
    chunk_size = args.chunk_size

    print(f"--- Starting Reconstruction ---")
    print(f"Algorithm: {algorithm.upper()}")
    if algorithm != 'fdk':
        print(f"Iterations: {n_iter}")
    print(f"Data Path: {base_dir}")
    print(f"Chunk Size for Atb: {chunk_size}")
    print("---------------------------------")

    # --- 1. & 2. 加载数据 ---
    meta_data_path = os.path.join(base_dir, 'meta_data.json')
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    scanner_params = meta_data['scanner']
    proj_train_info = meta_data['proj_train']
    print("Step 1: Successfully loaded metadata.")

    projections, s_maps, angles_rad_from_file = [], [], []
    s_map_dir = os.path.join(base_dir, 'proj_train_s_map', 'npy_data')
    print("Step 2: Loading projection data and S_map data...")
    for proj_info in tqdm(proj_train_info, desc="Loading data"):
        proj_path = os.path.join(base_dir, proj_info['file_path'])
        projections.append(np.load(proj_path))
        s_map_filename = os.path.basename(proj_info['file_path'])
        s_map_path = os.path.join(s_map_dir, s_map_filename)
        s_maps.append(np.load(s_map_path))
        angles_rad_from_file.append(proj_info['angle'])

    projections = np.stack(projections, axis=0).astype(np.float32)
    s_maps = np.stack(s_maps, axis=0).astype(np.float32)
    angles_rad = np.array(angles_rad_from_file, dtype=np.float32)

    print(f"Loaded {projections.shape[0]} projections and {s_maps.shape[0]} S_maps.")

    # --- 3. 配置 TIGRE 几何 ---
    geo = tigre.geometry(mode='cone')

    # ### <<< 关键修复 4：等比缩放几何参数 >>> ###
    # 原始几何参数的绝对值过小 (e.g., DSD=7.0)，可能导致TIGRE CUDA内核的数值不稳定。
    # 我们将所有长度单位乘以一个缩放因子，以将它们带入一个更常规的数值范围。
    # 这不会改变重建的几何形状，只会改变其绝对尺度。
    scale_factor = 1.0
    print(f"Applying a scaling factor of {scale_factor} to all length parameters to improve numerical stability.")

    geo.DSD = scanner_params['DSD'] * scale_factor
    geo.DSO = scanner_params['DSO'] * scale_factor

    # 非长度单位参数，不需要缩放
    geo.nDetector = np.array(scanner_params['nDetector'], dtype=np.int32)
    geo.nVoxel = np.array(scanner_params['nVoxel'], dtype=np.int32)

    # 长度单位参数，需要缩放
    geo.sDetector = np.array(scanner_params['sDetector'], dtype=np.float32) * scale_factor
    geo.sVoxel = np.array(scanner_params['sVoxel'], dtype=np.float32) * scale_factor

    # 偏移量是长度单位，也需要缩放
    geo.offOrigin = np.array(scanner_params['offOrigin'], dtype=np.float32) * scale_factor
    geo.offDetector = np.array(scanner_params['offDetector'], dtype=np.float32) * scale_factor

    # 其他参数
    geo.rotDetector = np.array([0, 0, 0], dtype=np.float32)
    geo.dDetector = geo.sDetector / geo.nDetector
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.accuracy = 0.5

    # 确保DSD/DSO是数组形式
    if not isinstance(geo.DSD, np.ndarray):
        geo.DSD = np.array([geo.DSD], dtype=np.float32)
    if not isinstance(geo.DSO, np.ndarray):
        geo.DSO = np.array([geo.DSO], dtype=np.float32)

    print("Step 3: TIGRE geometry configured.")

    # --- 诊断信息输出 ---
    print("\n--- Pre-Reconstruction Diagnostics ---")
    print(f"Projections array shape: {projections.shape}, dtype: {projections.dtype}")
    print(f"Angles array shape: {angles_rad.shape}, dtype: {angles_rad.dtype}")
    print(f"First 5 angles (radians): {angles_rad[:5]}")
    print("Geometry object details (after scaling):")
    print(geo)
    print("------------------------------------")

    # --- 4. 执行所选算法以获得强度体积 (Intensity Volume) ---
    print(f"\nStep 4: Running {algorithm.upper()} algorithm (for Intensity)...")
    start_time = time.time()

    intensity_volume = None

    try:
        if algorithm == 'fdk':
            print("Step 4.1: Filtering projections...")
            filtered_projections = filtering.filtering(projections, geo, angles_rad, parker=False)
            print("Filtering complete.")

            print("Step 4.2: Back-projecting filtered projections in chunks...")
            num_projections_fdk = filtered_projections.shape[0]
            intensity_volume = np.zeros(geo.nVoxel.astype(int), dtype=np.float32)

            for i in tqdm(range(0, num_projections_fdk, chunk_size), desc="Back-projecting FDK chunks"):
                end_index = min(i + chunk_size, num_projections_fdk)
                current_filtered_chunk = filtered_projections[i:end_index]
                current_angles_chunk = angles_rad[i:end_index]
                partial_volume = tigre.Atb(current_filtered_chunk, geo, current_angles_chunk)
                intensity_volume += partial_volume

            print("FDK reconstruction complete.")

        else:  # SART, SIRT, CGLS 等迭代算法
            if algorithm == 'sart':
                intensity_volume = tigre.algorithms.sart(projections, geo, angles_rad, niter=n_iter, blocksize=1)
            elif algorithm == 'sirt':
                intensity_volume = tigre.algorithms.sirt(projections, geo, angles_rad, niter=n_iter)
            elif algorithm == 'cgls':
                intensity_volume = tigre.algorithms.cgls(projections, geo, angles_rad, niter=n_iter)

            print(f"{algorithm.upper()} reconstruction complete.")

    except Exception as e:
        print(f"\n--- {algorithm.upper()} reconstruction failed! Error: ---\n{e}\n")
        # 打印更详细的追溯信息
        import traceback
        traceback.print_exc()
        intensity_volume = None

    recon_time = time.time() - start_time
    print(f"Reconstruction finished in {recon_time:.2f} seconds.")

    if intensity_volume is not None:
        if algorithm == 'fdk':
            filename = f"V_{algorithm}.npy"
        else:
            filename = f"V_{algorithm}_{n_iter}iter.npy"
        output_path_intensity = os.path.join(base_dir, filename)
        np.save(output_path_intensity, intensity_volume)
        print(f"Intensity volume saved to: {output_path_intensity}")

    # --- 5. 执行反投影以获得结构体积 (Structure/Importance Volume) ---
    print("\nStep 5: Running Back-Projection (Atb) on S_maps (for Structure)...")

    num_projections_s_map = s_maps.shape[0]
    importance_volume = np.zeros(geo.nVoxel.astype(int), dtype=np.float32)

    print(f"Processing {num_projections_s_map} S_maps in chunks of {chunk_size}...")

    try:
        for i in tqdm(range(0, num_projections_s_map, chunk_size), desc="Back-projecting S-map chunks"):
            end_index = min(i + chunk_size, num_projections_s_map)
            current_s_maps_chunk = s_maps[i:end_index]
            current_angles_chunk = angles_rad[i:end_index]
            partial_volume = tigre.Atb(current_s_maps_chunk, geo, current_angles_chunk)
            importance_volume += partial_volume

        print("S_map back-projection complete.")
        output_path_importance = os.path.join(base_dir, "V_importance.npy")
        np.save(output_path_importance, importance_volume)
        print(f"Importance volume saved to: {output_path_importance}")

    except Exception as e:
        print(f"\n--- S_map back-projection failed! Error: ---\n{e}\n")
        import traceback
        traceback.print_exc()

    if args.visualize and intensity_volume is not None and importance_volume is not None:
        print("\nStep 6: Generating visualization...")
        slice_idx = intensity_volume.shape[0] // 2

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        im1 = axes[0].imshow(intensity_volume[slice_idx, :, :], cmap='gray')
        axes[0].set_title(f'V_intensity ({algorithm.upper()}) - Slice {slice_idx}')
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(importance_volume[slice_idx, :, :], cmap='hot')
        axes[1].set_title(f'V_importance (Structure) - Slice {slice_idx}')
        fig.colorbar(im2, ax=axes[1])

        plt.suptitle(f"Reconstruction Results for {os.path.basename(base_dir)}")
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Perform CT reconstruction using various algorithms (FDK, SART, SIRT, CGLS) 
                     and generate an importance volume from S-maps. This script is based on 
                     the provided run_fdk.py, retaining its chunking logic for memory efficiency."""
    )

    parser.add_argument('--base_dir', type=str, required=True,
                        help="Path to the base directory of the dataset (e.g., 'data/synthetic_dataset/.../0_chest_cone').")

    parser.add_argument('--algorithm', type=str, default='sart',
                        choices=['fdk', 'sart', 'sirt', 'cgls'],
                        help="The reconstruction algorithm to use for the intensity volume. Default is 'fdk'.")

    parser.add_argument('--n_iter', type=int, default=5,
                        help="Number of iterations for iterative algorithms (sart, sirt, cgls). Ignored for 'fdk'. Default is 20.")

    parser.add_argument('--chunk_size', type=int, default=25,
                        help="Number of projections to process in each chunk for back-projection (Atb) steps. "
                             "Adjust based on your GPU/CPU memory. Default is 25.")

    parser.add_argument('--visualize', action='store_true',
                        help="If set, display a plot of the central slice of the reconstructed volumes.")

    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        print(f"Error: Directory not found at '{args.base_dir}'")
    else:
        s_map_dir_check = os.path.join(args.base_dir, 'proj_train_s_map', 'npy_data')
        if not os.path.isdir(s_map_dir_check):
            print(f"Error: S_map directory not found: {s_map_dir_check}")
            print("Please ensure that the S-map preprocessing step has been run.")
        else:
            run_reconstruction(args)