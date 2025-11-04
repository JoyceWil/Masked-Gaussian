import os
import json
import numpy as np


def inspect_data_structure(base_dir: str):
    """
    Inspects the structure of meta_data.json and a sample projection .npy file.

    Args:
        base_dir (str): The path to the specific dataset directory, e.g., 
                        'C:/.../0_chest_cone'.
    """
    print(f"--- Starting data inspection for directory: {base_dir} ---\n")

    # --- 1. Inspect meta_data.json ---
    meta_data_path = os.path.join(base_dir, 'meta_data.json')

    if not os.path.exists(meta_data_path):
        print(f"Error: meta_data.json not found at '{meta_data_path}'")
        return

    print(f"--- Contents of {meta_data_path} ---")
    try:
        with open(meta_data_path, 'r') as f:
            meta_data = json.load(f)

        # Pretty-print the JSON content for readability
        print(json.dumps(meta_data, indent=4))

    except Exception as e:
        print(f"Error reading or parsing meta_data.json: {e}")
        return

    print("\n" + "=" * 50 + "\n")

    # --- 2. Inspect a sample projection file from proj_train ---
    proj_train_dir = os.path.join(base_dir, 'proj_train')

    if not os.path.isdir(proj_train_dir):
        print(f"Error: proj_train directory not found at '{proj_train_dir}'")
        return

    # Find the first .npy file in the directory
    try:
        first_proj_file = sorted([f for f in os.listdir(proj_train_dir) if f.endswith('.npy')])[0]
        proj_path = os.path.join(proj_train_dir, first_proj_file)
    except IndexError:
        print(f"Error: No .npy files found in '{proj_train_dir}'")
        return

    print(f"--- Analysis of a Sample Projection File: {proj_path} ---")

    if not os.path.exists(proj_path):
        print(f"Error: Sample projection file not found at '{proj_path}'")
        return

    try:
        proj_data = np.load(proj_path)

        print(f"Data Type (dtype): {proj_data.dtype}")
        print(f"Shape: {proj_data.shape}")
        print(f"Minimum Value: {np.min(proj_data)}")
        print(f"Maximum Value: {np.max(proj_data)}")
        print(f"Mean Value: {np.mean(proj_data)}")

    except Exception as e:
        print(f"Error loading or analyzing projection file: {e}")
        return

    print("\n--- Inspection Complete ---")
    print("Please provide the output of this script for the next step.")


if __name__ == '__main__':
    # --- IMPORTANT ---
    # Please modify this path to point to your specific dataset directory
    # For example: 'C:/Files/Workbench/PythonProjects/r2_gaussian-main/data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone'

    # Using a relative path might be easier if the script is in the project root
    # For example: './data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone'

    dataset_base_dir = r'data\synthetic_dataset\cone_ntrain_25_angle_360\0_chest_cone'

    inspect_data_structure(dataset_base_dir)