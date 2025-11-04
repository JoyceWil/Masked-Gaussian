import numpy as np
import torch
import open3d as o3d
import os
import time

# =================================================================================
# 1. 配置参数 (Configuration)
# 您可以在这里调整所有参数
# =================================================================================
CONFIG = {
    # --- 文件路径 ---
    "input_path": "data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/P_vol.npy",
    "output_ply_path": "data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/initialized_point_cloud.ply",
    "output_npy_path": "data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/initial_gaussians.npy",
    # 新增：NPY文件输出路径

    # --- 初始化参数 ---
    "num_points": 50000,

    # --- 分层重要性采样权重 (p(x) ∝ α·G(x) + β·W(x) + γ) ---
    "alpha": 10.0,  # 梯度(边界)的权重，设为最高
    "beta": 2.0,  # 组织窗口的权重，设为中等
    "gamma": 0.1,  # 基础采样率，确保全覆盖

    # --- 差异化参数设置 ---
    "scale_config": {
        # 用于区分区域的阈值
        "g_thresh": 0.1,  # 梯度值大于此阈值被认为是“边界”
        "w_thresh": 0.2,  # 窗口值大于此阈值被认为是“重要组织”

        # 初始高斯核的逻辑尺寸 (在3DGS优化中使用)
        "s_small": 0.001,  # 用于边界
        "s_medium": 0.01,  # 用于组织内部
        "s_large": 0.05,  # 用于背景/均匀区域
    },

    # --- 可视化颜色 ---
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
    根据结构先验体积 P_vol 执行结构感知的高斯初始化。

    Args:
        P_vol (np.ndarray): (D, H, W, 3) 形状的结构先验体积 [I, G, W]。
        config (dict): 包含所有参数的配置字典。

    Returns:
        dict: 包含初始化高斯参数的字典，包括用于可视化的颜色。
    """
    num_gaussians = config["num_points"]
    alpha = config["alpha"]
    beta = config["beta"]
    gamma = config["gamma"]
    scale_config = config["scale_config"]

    D, H, W, _ = P_vol.shape

    # --- 阶段一: 分层重要性采样 ---

    print("步骤 1/3: 构建采样概率密度函数 (PDF)...")
    I_vol = P_vol[..., 0]
    G_vol = P_vol[..., 1]
    W_vol = P_vol[..., 2]

    # 1.1. 构建非归一化的PDF
    pdf_unnormalized = alpha * G_vol + beta * W_vol + gamma

    # 1.2. 确保PDF非负
    np.maximum(pdf_unnormalized, 0, out=pdf_unnormalized)

    # 1.3. 归一化PDF，使其总和为1
    total_sum = np.sum(pdf_unnormalized)
    if total_sum == 0:
        print("警告: PDF总和为0，将回退到均匀采样。")
        pdf_normalized = np.ones_like(pdf_unnormalized) / (D * H * W)
    else:
        pdf_normalized = pdf_unnormalized / total_sum

    # 1.4. 将PDF展平以便于采样
    pdf_flat = pdf_normalized.flatten()

    print(f"步骤 2/3: 根据PDF采样 {num_gaussians} 个点...")
    # 1.5. 使用 np.random.choice 根据概率分布 p 来采样一维索引
    sampled_indices_flat = np.random.choice(
        a=len(pdf_flat),
        size=num_gaussians,
        replace=True,  # 允许在同一体素多次采样
        p=pdf_flat
    )

    # 1.6. 将一维索引转换回三维体素坐标
    sampled_coords_3d = np.unravel_index(sampled_indices_flat, (D, H, W))
    sampled_coords_3d = np.stack(sampled_coords_3d, axis=-1)

    # --- 阶段二: 差异化参数设置 ---

    print("步骤 3/3: 为每个点设置差异化参数...")

    # 2.1. 获取采样点处的特征值
    d_indices, h_indices, w_indices = sampled_coords_3d[:, 0], sampled_coords_3d[:, 1], sampled_coords_3d[:, 2]
    sampled_G = G_vol[d_indices, h_indices, w_indices]
    sampled_W = W_vol[d_indices, h_indices, w_indices]
    sampled_I = I_vol[d_indices, h_indices, w_indices]

    # 2.2. 计算点云位置 (means)
    # 将体素坐标映射到 [-1, 1] 的世界坐标系（这是一个通用约定）
    # 注意轴的映射：Z -> 0, Y -> 1, X -> 2
    means = np.zeros((num_gaussians, 3))
    means[:, 0] = (w_indices / (W - 1)) * 2 - 1  # X轴
    means[:, 1] = (h_indices / (H - 1)) * 2 - 1  # Y轴
    means[:, 2] = (d_indices / (D - 1)) * 2 - 1  # Z轴

    # 添加微小的随机抖动，以打破网格结构，使分布更自然
    voxel_size = 2.0 / max(D, H, W)
    jitter = (np.random.rand(num_gaussians, 3) - 0.5) * voxel_size
    means += jitter

    # 2.3. 根据区域属性设置逻辑尺寸 (scales) 和可视化颜色 (colors)
    g_thresh = scale_config['g_thresh']
    w_thresh = scale_config['w_thresh']

    # 使用矢量化操作，高效判断每个点所属的区域
    is_edge = sampled_G > g_thresh
    is_interior = (~is_edge) & (sampled_W > w_thresh)
    is_background = (~is_edge) & (~is_interior)

    # 初始化参数数组
    scales = np.zeros((num_gaussians, 3))
    colors_for_viz = np.zeros((num_gaussians, 3))

    # 根据区域设置尺寸和颜色
    scales[is_edge] = scale_config['s_small']
    colors_for_viz[is_edge] = config['colors']['edge']

    scales[is_interior] = scale_config['s_medium']
    colors_for_viz[is_interior] = config['colors']['interior']

    scales[is_background] = scale_config['s_large']
    colors_for_viz[is_background] = config['colors']['background']

    # 2.4. 设置其他高斯参数 (密度和球谐系数)
    # 密度(opacity)通常使用logit函数转换，以便于优化
    opacities = np.log(sampled_I / (1 - sampled_I + 1e-7))

    # 球谐系数(SH)的零阶系数(c0)代表基础颜色（灰度值）
    # 很多框架期望3个通道的颜色，我们将灰度值复制3次
    sh_c0 = np.repeat(sampled_I[:, np.newaxis], 3, axis=1)

    # 旋转（四元数）初始化为单位旋转 (w, x, y, z)
    quats = np.zeros((num_gaussians, 4))
    quats[:, 0] = 1.0

    # 2.5. 将所有参数转换为Torch张量，为后续训练做准备
    return {
        "means": torch.tensor(means, dtype=torch.float32),
        "scales": torch.tensor(scales, dtype=torch.float32),
        "opacities": torch.tensor(opacities, dtype=torch.float32),
        "sh_c0": torch.tensor(sh_c0, dtype=torch.float32),
        "quats": torch.tensor(quats, dtype=torch.float32),
        "colors_for_viz": colors_for_viz  # 用于可视化的颜色数组
    }


# =================================================================================
# 3. 保存与可视化函数 (Saving and Visualization Functions)
# =================================================================================
def save_point_cloud_ply(filepath, means_tensor, colors_array):
    """
    使用Open3D将点云保存为.ply文件以供可视化。

    Args:
        filepath (str): 输出文件路径。
        means_tensor (torch.Tensor): 点的位置坐标。
        colors_array (np.ndarray): 点的颜色。
    """
    print(f"正在保存PLY可视化文件到: {filepath}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means_tensor.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors_array)

    # 确保输出目录存在
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    o3d.io.write_point_cloud(filepath, pcd)
    print("PLY文件保存成功！")


def save_gaussian_npy(filepath, initialized_data):
    """
    将所有高斯参数打包成一个Numpy数组并保存为.npy文件。

    Args:
        filepath (str): 输出的.npy文件路径。
        initialized_data (dict): 包含所有高斯参数的字典。
    """
    print(f"正在打包并保存NPY文件到: {filepath}")

    # 从字典中提取Torch张量并转换为Numpy数组
    means = initialized_data["means"].cpu().numpy()
    scales = initialized_data["scales"].cpu().numpy()
    quats = initialized_data["quats"].cpu().numpy()
    opacities = initialized_data["opacities"].cpu().numpy()
    sh_c0 = initialized_data["sh_c0"].cpu().numpy()

    # 确保opacities是 (N, 1) 形状以便拼接
    if opacities.ndim == 1:
        opacities = opacities[:, np.newaxis]

    # 按照特定顺序将所有参数水平拼接起来
    # 顺序: 位置(3), 缩放(3), 旋转(4), 不透明度(1), 颜色(3) -> 总共 14 列
    combined_array = np.hstack((means, scales, quats, opacities, sh_c0))

    # 确保输出目录存在
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 保存为.npy文件
    np.save(filepath, combined_array)
    print(f"NPY文件保存成功！数组形状: {combined_array.shape}")
    print("NPY文件列结构: [X, Y, Z, ScaleX, ScaleY, ScaleZ, QuatW, QuatX, QuatY, QuatZ, Opacity, SH_R, SH_G, SH_B]")


# =================================================================================
# 4. 主执行流程 (Main Execution Flow)
# =================================================================================
if __name__ == "__main__":
    start_time = time.time()

    # --- 检查输入文件 ---
    input_file = CONFIG["input_path"]
    if not os.path.exists(input_file):
        print(f"错误: 输入文件未找到 '{input_file}'")
        print("请确保 P_vol.npy 文件位于正确的路径下，并且其维度是 (D, H, W, C)。")
    else:
        # --- 加载数据 ---
        print(f"正在从 '{input_file}' 加载结构先验体积...")
        P_vol = np.load(input_file)
        print(f"加载成功！体积形状: {P_vol.shape}")

        # --- 执行初始化 ---
        print("\n开始执行结构感知初始化...")
        initialized_data = structure_aware_initialization(P_vol, CONFIG)
        print("初始化完成！")

        # --- 保存结果 ---
        print("\n准备保存输出文件...")

        # 1. 保存用于可视化的PLY文件
        save_point_cloud_ply(
            CONFIG["output_ply_path"],
            initialized_data["means"],
            initialized_data["colors_for_viz"]
        )

        # 2. 保存用于训练的NPY文件
        save_gaussian_npy(
            CONFIG["output_npy_path"],
            initialized_data
        )

        # --- 打印统计信息 ---
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
        print(f"\n您现在可以使用 MeshLab 或 CloudCompare 打开 '{CONFIG['output_ply_path']}' 查看可视化结果。")
        print(f"同时，您可以使用 '{CONFIG['output_npy_path']}' 作为您训练流程的输入。")