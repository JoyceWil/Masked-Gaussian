import numpy as np
import torch
import open3d as o3d
import os
import time

# tqdm 不再需要，因为没有慢速的旋转计算

# =================================================================================
# 1. 配置参数 (Configuration)
# =================================================================================
CONFIG = {
    "input_path": "data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/P_vol.npy",
    "output_ply_path": "data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/initialized_point_cloud.ply",
    "output_npy_path": "data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/initial_gaussians.npy",
    "num_points": 50000,

    # 采样权重 (p(x) ∝ α·G(x) + β·W(x) + γ)
    "alpha": 10.0,
    "beta": 2.0,
    "gamma": 0.1,

    # 差异化参数设置 (各向同性)
    "scale_config": {
        "g_thresh": 0.1,
        "w_thresh": 0.2,
        "s_small": 0.001,  # 边界
        "s_medium": 0.01,  # 内部
        "s_large": 0.05,  # 背景
    },

    # [还原] 不再需要 'opacity_config'，我们将使用原始的FDK强度

    "colors": {
        "edge": [1.0, 0.0, 0.0],  # 红色: 边界 (小尺寸)
        "interior": [0.0, 1.0, 0.0],  # 绿色: 组织内部 (中等尺寸)
        "background": [0.0, 0.0, 1.0]  # 蓝色: 背景 (大尺寸)
    }
}


# =================================================================================
# 2. 核心初始化函数 (Core Initialization Function)
# =================================================================================
def structure_aware_initialization(P_vol, config):
    """
    根据3通道 P_vol 执行各向同性初始化。
    """
    num_gaussians = config["num_points"]
    alpha = config["alpha"]
    beta = config["beta"]
    gamma = config["gamma"]
    scale_config = config["scale_config"]

    D, H, W, C = P_vol.shape
    if C != 3:
        raise ValueError(f"输入 P_vol 的通道数应为3，但得到了 {C}。请运行新版的 prior_extraction.py。")

    # --- 阶段一: 分层重要性采样 ---
    print("步骤 1/3: 构建采样概率密度函数 (PDF)...")
    I_vol = P_vol[..., 0]
    G_vol = P_vol[..., 1]  # 这是我们干净的 G 通道
    W_vol = P_vol[..., 2]

    pdf_unnormalized = alpha * G_vol + beta * W_vol + gamma
    np.maximum(pdf_unnormalized, 0, out=pdf_unnormalized)

    total_sum = np.sum(pdf_unnormalized)
    pdf_normalized = pdf_unnormalized / (total_sum + 1e-8)
    pdf_flat = pdf_normalized.flatten()

    print(f"步骤 2/3: 根据PDF采样 {num_gaussians} 个点...")
    sampled_indices_flat = np.random.choice(
        a=len(pdf_flat),
        size=num_gaussians,
        replace=True,
        p=pdf_flat
    )
    sampled_coords_3d = np.unravel_index(sampled_indices_flat, (D, H, W))
    sampled_coords_3d = np.stack(sampled_coords_3d, axis=-1)

    # --- 阶段二: 差异化参数设置 ---
    print("步骤 3/3: 为每个点设置差异化参数 (各向同性)...")

    # 2.1. 获取采样点处的特征值
    d_indices, h_indices, w_indices = sampled_coords_3d[:, 0], sampled_coords_3d[:, 1], sampled_coords_3d[:, 2]
    sampled_G = G_vol[d_indices, h_indices, w_indices]
    sampled_W = W_vol[d_indices, h_indices, w_indices]
    sampled_I = I_vol[d_indices, h_indices, w_indices]

    # 2.2. 计算点云位置 (means)
    means = np.zeros((num_gaussians, 3))
    means[:, 0] = (w_indices / (W - 1)) * 2 - 1  # X轴
    means[:, 1] = (h_indices / (H - 1)) * 2 - 1  # Y轴
    means[:, 2] = (d_indices / (D - 1)) * 2 - 1  # Z轴

    voxel_size = 2.0 / max(D, H, W)
    jitter = (np.random.rand(num_gaussians, 3) - 0.5) * voxel_size
    means += jitter

    # 2.3. 根据区域属性设置 Scales 和 Colors
    g_thresh = scale_config['g_thresh']
    w_thresh = scale_config['w_thresh']

    is_edge = sampled_G > g_thresh
    is_interior = (~is_edge) & (sampled_W > w_thresh)
    is_background = (~is_edge) & (~is_interior)

    scales_val = np.zeros((num_gaussians,))
    colors_for_viz = np.zeros((num_gaussians, 3))

    scales_val[is_edge] = scale_config['s_small']
    colors_for_viz[is_edge] = config['colors']['edge']

    scales_val[is_interior] = scale_config['s_medium']
    colors_for_viz[is_interior] = config['colors']['interior']

    scales_val[is_background] = scale_config['s_large']
    colors_for_viz[is_background] = config['colors']['background']

    # 将标量尺度复制到 (N, 3)
    scales = np.repeat(scales_val[:, np.newaxis], 3, axis=1)

    # 2.4. 设置其他高斯参数 (密度, SH, 旋转)

    # [还原] 密度(opacity)使用原始的FDK强度
    np.clip(sampled_I, 1e-7, 1.0 - 1e-7, out=sampled_I)  # 裁剪以避免log(0)
    opacities = np.log(sampled_I / (1 - sampled_I))

    # [还原] 球谐系数(SH)
    sh_c0 = np.repeat(sampled_I[:, np.newaxis], 3, axis=1)

    # [还原] 旋转 (四元数) 初始化为单位旋转
    quats = np.zeros((num_gaussians, 4))
    quats[:, 0] = 1.0

    # 2.5. 转换为Torch张量
    return {
        "means": torch.tensor(means, dtype=torch.float32),
        "scales": torch.tensor(scales, dtype=torch.float32),
        "opacities": torch.tensor(opacities, dtype=torch.float32),
        "sh_c0": torch.tensor(sh_c0, dtype=torch.float32),
        "quats": torch.tensor(quats, dtype=torch.float32),
        "colors_for_viz": colors_for_viz
    }


# =================================================================================
# 3. 保存与可视化函数 (与之前相同)
# =================================================================================
def save_point_cloud_ply(filepath, means_tensor, colors_array):
    print(f"正在保存PLY可视化文件到: {filepath}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means_tensor.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors_array)
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    o3d.io.write_point_cloud(filepath, pcd)
    print("PLY文件保存成功！")


def save_gaussian_npy(filepath, initialized_data):
    print(f"正在打包并保存NPY文件到: {filepath}")
    means = initialized_data["means"].cpu().numpy()
    scales = initialized_data["scales"].cpu().numpy()
    quats = initialized_data["quats"].cpu().numpy()
    opacities = initialized_data["opacities"].cpu().numpy()
    sh_c0 = initialized_data["sh_c0"].cpu().numpy()
    if opacities.ndim == 1:
        opacities = opacities[:, np.newaxis]
    combined_array = np.hstack((means, scales, quats, opacities, sh_c0))
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(filepath, combined_array)
    print(f"NPY文件保存成功！数组形状: {combined_array.shape}")


# =================================================================================
# 4. 主执行流程 (Main Execution Flow)
# =================================================================================
if __name__ == "__main__":
    start_time = time.time()
    input_file = CONFIG["input_path"]

    if not os.path.exists(input_file):
        print(f"错误: 输入文件未找到 '{input_file}'")
    else:
        print(f"正在从 '{input_file}' 加载结构先验体积...")
        P_vol = np.load(input_file)
        print(f"加载成功！体积形状: {P_vol.shape}")

        if P_vol.shape[-1] != 3:
            print(f"错误: P_vol 最后一个维度应为 3, 但检测到 {P_vol.shape[-1]}。")
            print("请运行新版的 3-channel prior_extraction.py。")
        else:
            print("\n开始执行结构感知初始化 (各向同性)...")
            initialized_data = structure_aware_initialization(P_vol, CONFIG)
            print("初始化完成！")

            print("\n准备保存输出文件...")
            save_point_cloud_ply(
                CONFIG["output_ply_path"],
                initialized_data["means"],
                initialized_data["colors_for_viz"]
            )
            save_gaussian_npy(
                CONFIG["output_npy_path"],
                initialized_data
            )

            # 统计
            num_total = len(initialized_data["means"])
            colors = initialized_data["colors_for_viz"]
            num_edge = np.sum(np.all(colors == CONFIG['colors']['edge'], axis=1))
            num_interior = np.sum(np.all(colors == CONFIG['colors']['interior'], axis=1))
            num_background = np.sum(np.all(colors == CONFIG['colors']['background'], axis=1))
            print("\n--- 初始化统计 ---")
            print(f"总点数: {num_total}")
            print(f"边界点 (红色):   {num_edge} ({num_edge / num_total:.2%})")
            print(f"组织内部点 (绿色): {num_interior} ({num_interior / num_total:.2%})")
            print(f"背景点 (蓝色):   {num_background} ({num_background / num_total:.2%})")

            end_time = time.time()
            print(f"\n总耗时: {end_time - start_time:.2f} 秒")