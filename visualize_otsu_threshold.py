# 文件名: visualize_final_threshold.py (或者您使用的文件名)

import numpy as np
import cv2
import os
from pathlib import Path
from skimage.filters import threshold_multiotsu
import argparse

# --- 全局参数和常量设置 ---
POINT_CLOUD_FILE = "data/0_chest_cone/init_0_chest_cone.npy"
OUTPUT_DIR = Path('output/pointcloud_final_preview/')

# 可视化常量
IMG_SIZE = 512
ALL_POINTS_COLOR = (128, 128, 128)  # 灰色
CORE_POINTS_COLOR = (255, 255, 255)  # 白色
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_TITLE = 0.9
FONT_SCALE_SUB = 0.7
FONT_COLOR = (0, 255, 255)  # 青色
FONT_THICKNESS = 2


# --- 辅助函数 ---
def normalize_coordinates(xyz_coords, img_size):
    min_coords = xyz_coords.min(axis=0)
    max_coords = xyz_coords.max(axis=0)
    scale = max_coords - min_coords
    scale[scale == 0] = 1
    normalized = (xyz_coords - min_coords) / scale
    pixel_coords = (normalized * (img_size - 1)).astype(int)
    return pixel_coords


def project_and_draw(canvas, points, x_axis_idx, y_axis_idx, color):
    for point in points:
        x, y = point[x_axis_idx], point[y_axis_idx]
        cv2.circle(canvas, (x, y), 1, color, -1)
    return canvas


# --- 脚本主逻辑 ---
def main(args):
    print(f"正在加载点云数据: {POINT_CLOUD_FILE}")
    try:
        data = np.load(POINT_CLOUD_FILE)
        xyz = data[:, :3]
        densities = data[:, 3]
        print(f"加载成功! 点数: {data.shape[0]}")
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    # --- 1. 根据选择的模式计算阈值或筛选点 ---
    non_zero_densities = densities[densities > 0]

    if args.mode == 'multi_otsu':
        print("\n模式: 多级大津法")
        thresholds = threshold_multiotsu(non_zero_densities, classes=args.classes)
        core_threshold = thresholds[-1]
        print(f"使用 {args.classes} 类，计算出的核心阈值: {core_threshold:.4f}")
        core_mask = densities > core_threshold
        title_text = f"Multi-Otsu (classes={args.classes}, thresh={core_threshold:.4f})"
        filtered_label = "Multi-Otsu Filtered"

    elif args.mode == 'percentile':
        print("\n模式: 百分位法")
        percentile_q = 100 - args.percentile
        core_threshold = np.percentile(densities, percentile_q)
        print(f"筛选密度最高的 {args.percentile}% 点, 计算出的密度阈值: {core_threshold:.4f}")
        core_mask = densities >= core_threshold
        title_text = f"Percentile Method (top {args.percentile}%, thresh={core_threshold:.4f})"
        filtered_label = f"Top {args.percentile}% Filtered"

    else:
        print(f"错误: 未知模式 '{args.mode}'")
        return

    # --- 2. 准备可视化 ---
    pixel_coords_all = normalize_coordinates(xyz, IMG_SIZE)
    pixel_coords_core = pixel_coords_all[core_mask]

    num_total = data.shape[0]
    num_core = pixel_coords_core.shape[0]
    print(f"核心点数: {num_core} / {num_total} ({num_core / num_total * 100:.2f}%)")

    # --- 3. 创建视图 ---
    print("正在生成对比可视化图像...")
    views = {}
    for perspective in ["Axial", "Coronal", "Sagittal"]:
        views[f"all_{perspective.lower()}"] = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        views[f"core_{perspective.lower()}"] = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    project_and_draw(views['all_axial'], pixel_coords_all, 0, 1, ALL_POINTS_COLOR)
    project_and_draw(views['core_axial'], pixel_coords_all, 0, 1, ALL_POINTS_COLOR)
    project_and_draw(views['core_axial'], pixel_coords_core, 0, 1, CORE_POINTS_COLOR)

    project_and_draw(views['all_coronal'], pixel_coords_all, 0, 2, ALL_POINTS_COLOR)
    project_and_draw(views['core_coronal'], pixel_coords_all, 0, 2, ALL_POINTS_COLOR)
    project_and_draw(views['core_coronal'], pixel_coords_core, 0, 2, CORE_POINTS_COLOR)

    project_and_draw(views['all_sagittal'], pixel_coords_all, 1, 2, ALL_POINTS_COLOR)
    project_and_draw(views['core_sagittal'], pixel_coords_all, 1, 2, ALL_POINTS_COLOR)
    project_and_draw(views['core_sagittal'], pixel_coords_core, 1, 2, CORE_POINTS_COLOR)

    # --- 4. 拼接成一张大图 ---
    header_height = 60
    padding = 10
    composite_width = IMG_SIZE * 3 + padding * 4
    composite_height = IMG_SIZE * 2 + padding * 3 + header_height
    composite_img = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)

    cv2.putText(composite_img, title_text, (padding, 40), FONT, FONT_SCALE_TITLE, FONT_COLOR, FONT_THICKNESS)

    y_offset = header_height + padding
    for i, p_type in enumerate(["All Points", filtered_label]):
        x_offset = padding
        for j, view_name in enumerate(["Axial", "Coronal", "Sagittal"]):
            canvas_key = "all_" if i == 0 else "core_"
            canvas = views[f"{canvas_key}{view_name.lower()}"]

            cv2.putText(canvas, f"{p_type.split('(')[0].strip()} - {view_name}", (10, 30), FONT, FONT_SCALE_SUB,
                        FONT_COLOR, FONT_THICKNESS)

            composite_img[y_offset: y_offset + IMG_SIZE, x_offset: x_offset + IMG_SIZE] = canvas
            x_offset += IMG_SIZE + padding
        y_offset += IMG_SIZE + padding

    # --- 5. 保存结果 ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = f"preview_{args.mode}_{args.percentile if args.mode == 'percentile' else args.classes}.png"
    output_path = OUTPUT_DIR / output_filename
    cv2.imwrite(str(output_path), composite_img)

    print(f"\n\033[92m[SUCCESS]\033[0m 对比预览图已保存到: {output_path.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize point cloud thresholding with different methods.")
    parser.add_argument('--mode', type=str, default='percentile', choices=['percentile', 'multi_otsu'],
                        help="The thresholding method to use.")
    parser.add_argument('--percentile', type=float, default=40.0,
                        help="For 'percentile' mode, the percentage of top density points to keep.")
    parser.add_argument('--classes', type=int, default=4,
                        help="For 'multi_otsu' mode, the number of classes to segment into.")

    args = parser.parse_args()
    main(args)