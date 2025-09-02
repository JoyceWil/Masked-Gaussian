import numpy as np
import cv2
import os

# --- 1. 配置与常量 ---

# 输入的NPY投影文件名 (请根据需要修改)
PROJECTION_FILE = "/media/data2/synthetic_dataset/cone_ntrain_50_angle_360/0_head_cone/proj_train/proj_train_0000.npy"
# 输出可视化图像的文件夹 (请根据需要修改)
OUTPUT_DIR = '/media/data2/synthetic_dataset/cone_ntrain_50_angle_360/0_head_cone/hu_visualization'

# 人体组织CT值表 (HU范围)
HU_RANGES = {
    'bone': (400, 3000),  # 骨组织
    'calcium': (150, 400),  # 钙化/高密度软组织
    'soft_tissue': (20, 80),  # 大部分软组织/器官
    'fat': (-100, -20),  # 脂肪
    'air': (-1024, -950)  # 空气
}

# 标定所用的锚点值
# 我们假设数据的最小值对应空气，最大值对应骨骼
CALIBRATION_POINTS = {
    'air': {'npy_value': 0.0, 'hu_value': -1000.0},
    'bone': {'hu_value': 1000.0}  # 我们将使用数据中的最大值作为骨骼的npy_value
}


# --- 2. 全新的核心功能函数 ---

def calculate_hu_mapping_from_stats(data: np.ndarray) -> tuple[float, float]:
    """
    根据数据的统计极值（最小值和最大值）来计算HU映射参数 a 和 b。
    这是一个更鲁棒的、为归一化数据设计的方法。

    参数:
        data: 形状为 (H, W) 的单个2D投影Numpy数组。

    返回:
        一个元组 (a, b)，其中 a 是斜率，b 是截距。
    """
    print("正在基于数据统计信息计算HU映射参数 (a, b)...")

    # 获取数据的最小值和最大值
    i_min = np.min(data)
    i_max = np.max(data)

    print(f"检测到NPY数据范围: Min={i_min:.4f}, Max={i_max:.4f}")

    # 设置我们的两个锚点
    # 点1：空气 (使用数据中的最小值)
    p1_npy = i_min
    p1_hu = CALIBRATION_POINTS['air']['hu_value']

    # 点2：骨骼 (使用数据中的最大值)
    p2_npy = i_max
    p2_hu = CALIBRATION_POINTS['bone']['hu_value']

    # 检查两个点是否相同，避免除以零
    if np.isclose(p1_npy, p2_npy):
        raise ValueError("数据中的最小值和最大值几乎相同，无法进行标定。")

    # 求解线性方程组:
    # p1_hu = a * p1_npy + b
    # p2_hu = a * p2_npy + b
    # 解得 a = (p2_hu - p1_hu) / (p2_npy - p1_npy)
    #       b = p1_hu - a * p1_npy

    a = (p2_hu - p1_hu) / (p2_npy - p1_npy)
    b = p1_hu - a * p1_npy

    print(f"计算完成: a = {a:.4f}, b = {b:.4f}")
    print(f"映射公式: HU = {a:.4f} * I_npy + {b:.4f}")
    return a, b


# --- 3. 主函数 ---
def process_and_visualize(projection_file: str, output_dir: str):
    """
    主函数：加载单个2D NPY文件，计算HU值，并保存可视化分割图像。
    """
    # 检查输入文件是否存在
    if not os.path.exists(projection_file):
        raise FileNotFoundError(f"错误: 输入文件 '{projection_file}' 不存在。")

    # 加载数据
    print(f"正在加载投影文件: {projection_file}")
    view = np.load(projection_file)
    print(f"加载成功，数据形状: {view.shape}")

    if view.ndim != 2:
        raise ValueError(f"此脚本仅支持2D数组，但输入数据为 {view.ndim}D。")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"可视化结果将保存在: {output_dir}")

    try:
        # 1. 计算HU映射参数
        a, b = calculate_hu_mapping_from_stats(view)

        # 2. 转换并保存可视化结果
        print("正在处理视角并生成可视化图像...")

        # 将整个视图转换为HU值图
        hu_view = a * view + b

        # 为每个组织类别生成并保存一个二值图像
        for class_name, (hu_min, hu_max) in HU_RANGES.items():
            mask = (hu_view >= hu_min) & (hu_view <= hu_max)
            vis_image = (mask * 255).astype(np.uint8)

            # 从输入文件名派生输出文件名
            base_filename = os.path.splitext(os.path.basename(projection_file))[0]
            output_filename = os.path.join(output_dir, f"{base_filename}_{class_name}.png")
            cv2.imwrite(output_filename, vis_image)

        print(f"\n处理完成！所有可视化图像已保存在 '{output_dir}' 文件夹中。")

    except (ValueError, FileNotFoundError) as e:
        print(f"程序终止，原因: {e}")


# --- 4. 运行主程序 ---
if __name__ == '__main__':
    process_and_visualize(PROJECTION_FILE, OUTPUT_DIR)