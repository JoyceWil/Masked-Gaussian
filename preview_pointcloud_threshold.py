# 文件名: preview_pointcloud_threshold.py

import numpy as np
import cv2
import os
from pathlib import Path

# --- 参数设置 ---
# 请将这里替换为您的 .npy 文件路径
POINT_CLOUD_FILE = "data/0_chest_cone/init_0_chest_cone.npy"

# 输出目录
OUTPUT_DIR = Path('output/pointcloud_threshold_previews/')

# 要测试的密度阈值
THRESHOLDS = [0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.1]

# 生成的预览图大小 (像素)
IMG_SIZE = 512

# 点的颜色 (B, G, R) 和大小
POINT_COLOR = (255, 255, 255)  # 白色
POINT_RADIUS = 1


# --- 脚本主逻辑 ---

def normalize_coordinates(xyz_coords, img_size):
    """将点云坐标归一化到图像像素空间"""
    # 找到每个轴的边界
    min_coords = xyz_coords.min(axis=0)
    max_coords = xyz_coords.max(axis=0)

    # 计算范围
    scale = max_coords - min_coords

    # 防止除以零
    scale[scale == 0] = 1

    # 归一化到 [0, 1]
    normalized = (xyz_coords - min_coords) / scale

    # 缩放到图像尺寸
    pixel_coords = (normalized * (img_size - 1)).astype(int)

    return pixel_coords


def project_and_draw(canvas, points, x_axis_idx, y_axis_idx):
    """将点投影到指定的2D平面并绘制"""
    for point in points:
        x, y = point[x_axis_idx], point[y_axis_idx]
        cv2.circle(canvas, (x, y), POINT_RADIUS, POINT_COLOR, -1)
    return canvas


def main():
    print(f"正在加载点云数据: {POINT_CLOUD_FILE}")
    if not os.path.exists(POINT_CLOUD_FILE):
        print(f"错误: 文件未找到! '{POINT_CLOUD_FILE}'")
        return

    try:
        # 加载数据，假设格式为 (N, 4) -> [x, y, z, density]
        data = np.load(POINT_CLOUD_FILE)
        if data.ndim != 2 or data.shape[1] != 4:
            print(f"错误: 文件格式不正确。期望维度 (N, 4)，实际为 {data.shape}")
            return

        xyz = data[:, :3]
        densities = data[:, 3]
        print(f"加载成功! 点数: {data.shape[0]}, 维度: {data.shape[1]}")

    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    # 预先归一化所有点的坐标
    print("正在归一化坐标以适应图像...")
    pixel_coords_all = normalize_coordinates(xyz, IMG_SIZE)
    print("归一化完成。")

    for threshold in THRESHOLDS:
        print(f"\n--- 正在处理阈值: {threshold} ---")

        # 创建输出目录
        thresh_dir = OUTPUT_DIR / f'threshold_{threshold:.2f}'
        thresh_dir.mkdir(parents=True, exist_ok=True)

        # 1. 根据阈值筛选点
        mask = densities > threshold
        surviving_points = pixel_coords_all[mask]
        num_survived = surviving_points.shape[0]

        if num_survived == 0:
            print("在此阈值下没有点存活，跳过绘图。")
            continue

        print(f"存活点数: {num_survived} / {data.shape[0]} ({num_survived / data.shape[0] * 100:.2f}%)")

        # 2. 创建三个空白画布
        axial_canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        coronal_canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        sagittal_canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # 3. 投影并绘制点
        # 轴状位 (Axial): X-Y 平面
        project_and_draw(axial_canvas, surviving_points, 0, 1)
        # 冠状位 (Coronal): X-Z 平面
        project_and_draw(coronal_canvas, surviving_points, 0, 2)
        # 矢状位 (Sagittal): Y-Z 平面
        project_and_draw(sagittal_canvas, surviving_points, 1, 2)

        # 4. 保存图像
        cv2.imwrite(str(thresh_dir / 'preview_axial.png'), axial_canvas)
        cv2.imwrite(str(thresh_dir / 'preview_coronal.png'), coronal_canvas)
        cv2.imwrite(str(thresh_dir / 'preview_sagittal.png'), sagittal_canvas)

    print(f"\n所有预览图已保存到: {OUTPUT_DIR.resolve()}")


if __name__ == '__main__':
    main()