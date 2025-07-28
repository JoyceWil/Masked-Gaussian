import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm


def generate_dual_percentile_masks(
        projection_data: np.ndarray,
        soft_percentile: int,
        core_percentile: int
):
    """
    使用双百分位数法，从单张2D投影数据生成分层的二值掩码。

    参数:
    projection_data (np.ndarray): 原始的2D投影Numpy数组。
    soft_percentile (int): 用于定义“软组织掩码”的较低百分位数 (例如 30)。
    core_percentile (int): 用于定义“核心骨架掩码”的较高百分位数 (例如 85)。

    返回:
    tuple: (软组织掩码, 核心骨架掩码)，均为uint8类型，值为0或1。
    """
    # --- 步骤1: 生成“软组织掩码” (Soft Mask) ---
    # 首先，我们需要一个基础轮廓来确定分析范围，这里用一个极低阈值确保完整性
    base_mask_for_analysis = projection_data > 0.01
    object_pixels = projection_data[base_mask_for_analysis]

    if object_pixels.size == 0:
        # 如果图像几乎是全黑的，返回两个空掩码
        empty_mask = np.zeros_like(projection_data, dtype=np.uint8)
        return empty_mask, empty_mask

    # 计算软组织阈值
    soft_threshold = np.percentile(object_pixels, soft_percentile)
    soft_mask_raw = (projection_data >= soft_threshold).astype(np.uint8) * 255

    # 对软组织掩码进行清理，主要是填充可能存在的小洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    soft_mask_cleaned = cv2.morphologyEx(soft_mask_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    soft_mask = (soft_mask_cleaned / 255).astype(np.uint8)

    # --- 步骤2: 生成“核心骨架掩码” (Core Mask) ---
    # 在刚才确定的物体像素中，计算核心骨架的阈值
    core_threshold = np.percentile(object_pixels, core_percentile)
    core_mask_raw = (projection_data >= core_threshold).astype(np.uint8) * 255

    # 对核心掩码进行清理，主要是去除小的孤立点
    core_mask_cleaned = cv2.morphologyEx(core_mask_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    core_mask_final = (core_mask_cleaned / 255).astype(np.uint8)

    return soft_mask, core_mask_final


def process_directory_final(input_dir: str, output_base_dir: str, soft_p: int, core_p: int):
    """
    处理目录中的所有.npy文件，生成并保存两种由百分位数定义的掩码。
    """
    # 根据百分位数动态创建目录名，方便追溯
    output_dir_soft = os.path.join(output_base_dir, f'soft_masks_p{soft_p}')
    output_dir_core = os.path.join(output_base_dir, f'core_masks_p{core_p}')
    os.makedirs(output_dir_soft, exist_ok=True)
    os.makedirs(output_dir_core, exist_ok=True)

    print(f"软组织掩码 (p={soft_p}) 将保存至: '{output_dir_soft}'")
    print(f"核心骨架掩码 (p={core_p}) 将保存至: '{output_dir_core}'")

    try:
        file_list = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
        if not file_list:
            print(f"错误: 在 '{input_dir}' 中未找到.npy文件。")
            return
    except FileNotFoundError:
        print(f"错误: 输入目录 '{input_dir}' 不存在。")
        return

    for filename in tqdm(file_list, desc=f"生成 p{soft_p}/p{core_p} 掩码"):
        input_path = os.path.join(input_dir, filename)
        projection_data = np.load(input_path)

        # 生成两种掩码
        soft_mask, core_mask = generate_dual_percentile_masks(
            projection_data,
            soft_percentile=soft_p,
            core_percentile=core_p
        )

        # 保存软组织掩码
        output_path_soft = os.path.join(output_dir_soft, filename)
        np.save(output_path_soft, soft_mask)

        # 保存核心骨架掩码
        output_path_core = os.path.join(output_dir_core, filename)
        np.save(output_path_core, core_mask)

    print("\n所有分层掩码已成功生成并保存。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="批量为2D投影生成基于双百分位数的分层掩码。")
    parser.add_argument('--input_dir', type=str, required=True, help="包含原始投影.npy文件的输入目录。")
    parser.add_argument('--output_dir', type=str, required=True, help="用于保存掩码的根目录。")
    parser.add_argument('--soft_percentile', type=int, default=30, help="定义软组织掩码的百分位数。")
    parser.add_argument('--core_percentile', type=int, default=50, help="定义核心骨架掩码的百分位数。")
    args = parser.parse_args()

    if args.soft_percentile >= args.core_percentile:
        raise ValueError("核心掩码的百分位数 (`core_percentile`) 必须高于软组织掩码的百分位数 (`soft_percentile`)。")

    process_directory_final(args.input_dir, args.output_dir, args.soft_percentile, args.core_percentile)