import os
import json
import numpy as np
import tigre
# 我们可以选择在这里导入子模块，让代码更简洁
# from tigre import algorithms
from tqdm import tqdm
import matplotlib.pyplot as plt


def generate_fdk_volume(base_dir: str, output_filename: str = "V_fdk.npy"):
    """
    Performs FDK reconstruction using TIGRE based on the provided dataset.

    Args:
        base_dir (str): The path to the specific dataset directory.
        output_filename (str): The name of the output file for the FDK volume.
    """
    print("--- Starting FDK Reconstruction ---")

    # --- 1. Load Metadata ---
    meta_data_path = os.path.join(base_dir, 'meta_data.json')
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)

    scanner_params = meta_data['scanner']
    proj_train_info = meta_data['proj_train']

    print("Successfully loaded metadata.")

    # --- 2. Load Projections and Angles ---
    projections = []
    angles = []

    print("Loading projection data...")
    for proj_info in tqdm(proj_train_info, desc="Loading projections"):
        proj_path = os.path.join(base_dir, proj_info['file_path'])
        projections.append(np.load(proj_path))
        angles.append(proj_info['angle'])

    projections = np.stack(projections, axis=0).astype(np.float32)
    angles = np.array(angles, dtype=np.float32)

    print(f"Loaded {projections.shape[0]} projections.")
    print(f"Projection data shape: {projections.shape}")
    print(f"Angles shape: {angles.shape}")

    # --- 3. Configure TIGRE Geometry ---
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

    geo.dDetector = np.divide(geo.sDetector, geo.nDetector,
                              out=np.zeros_like(geo.sDetector),
                              where=geo.nDetector != 0)
    geo.dVoxel = np.divide(geo.sVoxel, geo.nVoxel,
                           out=np.zeros_like(geo.sVoxel),
                           where=geo.nVoxel != 0)

    geo.accuracy = 0.5

    print("TIGRE geometry configured correctly.")
    print(geo)

    # --- 4. Perform FDK Reconstruction ---
    print("Running FDK algorithm... (This may take a moment, especially on CPU)")

    # --- THIS IS THE CORRECTED LINE ---
    # FDK algorithm is located in the 'tigre.algorithms' submodule.
    fdk_volume = tigre.algorithms.fdk(projections, geo, angles)
    # --- END OF CORRECTION ---

    print("FDK reconstruction complete.")
    print(f"Reconstructed volume shape: {fdk_volume.shape}")

    # --- 5. Save the Result ---
    output_path = os.path.join(base_dir, output_filename)
    np.save(output_path, fdk_volume)
    print(f"FDK volume saved to: {output_path}")

    # --- 6. (Optional) Visualize a central slice ---
    if fdk_volume is not None and fdk_volume.size > 0:
        center_slice_idx = fdk_volume.shape[0] // 2
        center_slice = fdk_volume[center_slice_idx, :, :]

        plt.figure(figsize=(8, 8))
        plt.imshow(center_slice, cmap='gray')
        plt.title(f'Central Slice ({center_slice_idx}) of FDK Reconstruction')
        plt.colorbar(label='Reconstructed Value')
        plt.show()
    else:
        print("Warning: Reconstructed volume is empty or invalid. Skipping visualization.")


if __name__ == '__main__':
    dataset_base_dir = r'data\synthetic_dataset\cone_ntrain_25_angle_360\0_chest_cone'

    if not os.path.isdir(dataset_base_dir):
        print(f"Error: Directory not found at '{dataset_base_dir}'")
        print("Please update the 'dataset_base_dir' variable in the script.")
    else:
        generate_fdk_volume(dataset_base_dir, output_filename="V_fdk.npy")