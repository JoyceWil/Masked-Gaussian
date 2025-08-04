import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse


def generate_hierarchical_masks(projection_data: np.ndarray, high_density_percentile: int = 95):
    """
    从单张2D投影数据生成分层的二值掩码。

    参数:
    projection_data (np.ndarray): 原始的2D投影Numpy数组。
    high_density_percentile (int): 用于定义“高密度”的百分位数 (0-100)。
                                   95意味着取像素值最高的5%的区域。

    返回:
    tuple: 包含 (通用轮廓掩码, 高密度核心掩码) 的元组。
           两个掩码都是uint8类型，值为0或1。
    """
    # --- 步骤1: 生成通用轮廓掩码 (与之前的方法相同) ---
    p_min, p_max = projection_data.min(), projection_data.max()
    if p_max > p_min:
        normalized_data = 255 * (projection_data - p_min) / (p_max - p_min)
    else:
        normalized_data = np.zeros_like(projection_data)
    image_uint8 = normalized_data.astype(np.uint8)

    _, binary_mask_255 = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_opened = cv2.morphologyEx(binary_mask_255, cv2.MORPH_OPEN, kernel, iterations=2)
    clean_mask_255 = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    general_mask = (clean_mask_255 / 255).astype(np.uint8)

    # --- 步骤2: 基于百分位数生成高密度核心掩码 ---
    # 只在通用掩码定义的物体区域内分析像素值
    object_pixels = projection_data[general_mask == 1]

    if object_pixels.size == 0:
        # 如果通用掩码为空，则核心掩码也为空
        core_mask = np.zeros_like(projection_data, dtype=np.uint8)
        return general_mask, core_mask

    # 计算高密度阈值
    density_threshold = np.percentile(object_pixels, high_density_percentile)

    # 根据新阈值创建核心掩码
    # 只有原始数据中高于该阈值的像素才被认为是核心
    core_mask = (projection_data >= density_threshold).astype(np.uint8)

    # (可选但推荐) 对核心掩码也进行清理，去除小的孤立点
    core_mask_cleaned = cv2.morphologyEx(core_mask * 255, cv2.MORPH_OPEN, kernel, iterations=1)
    core_mask_final = (core_mask_cleaned / 255).astype(np.uint8)

    return general_mask, core_mask_final


def visualize_comparison(original_proj, general_mask, core_mask, file_path):
    """
    并排可视化原始投影、通用掩码和核心掩码。
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 原始投影
    ax = axes[0]
    im = ax.imshow(original_proj, cmap='gray')
    ax.set_title('原始投影数据', fontsize=16)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2. 通用轮廓掩码
    ax = axes[1]
    ax.imshow(general_mask, cmap='gray')
    ax.set_title('通用轮廓掩码 (旧)', fontsize=16)
    ax.axis('off')

    # 3. 高密度核心掩码
    ax = axes[2]
    ax.imshow(core_mask, cmap='gray')
    ax.set_title('高密度核心掩码 (新)', fontsize=16)
    ax.axis('off')

    fig.suptitle(f'掩码分层对比: {os.path.basename(file_path)}', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    # --- 请在这里修改为您原始 proj_train_*.npy 文件的路径 ---
    # 注意：不是掩码文件，而是生成掩码的原始数据文件！
    # 例如: "/media/data2/hezhipeng/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/proj_train/proj_train_0000.npy"

    # 使用命令行参数来接收文件路径
    parser = argparse.ArgumentParser(description="可视化分层掩码生成。")
    parser.add_argument('filepath', type=str, help="原始投影.npy文件的路径。")
    args = parser.parse_args()

    original_proj_path = args.filepath

    if not os.path.exists(original_proj_path):
        print(f"错误: 原始投影文件未找到于 '{original_proj_path}'")
        sys.exit(1)

    # 加载原始投影数据
    projection_data = np.load(original_proj_path)

    # 生成两种掩码
    # 您可以调整 high_density_percentile=95 这个值，90会包含更多组织，98会更严格
    general_silhouette_mask, high_density_core_mask = generate_hierarchical_masks(
        projection_data,
        high_density_percentile=30
    )

    # 可视化对比结果
    visualize_comparison(projection_data, general_silhouette_mask, high_density_core_mask, original_proj_path)