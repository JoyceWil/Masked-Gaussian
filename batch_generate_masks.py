import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm


def generate_mask_from_projection(projection_data: np.ndarray) -> np.ndarray:
    """
    从单张2D投影数据生成一个干净的二值掩码。

    该函数执行以下步骤：
    1. 将输入数据归一化到0-255范围，并转换为uint8类型。
    2. 使用Otsu's方法自动计算阈值并进行二值化。
    3. 应用形态学开运算和闭运算来移除噪声并填充空洞。
    4. 返回一个与输入同尺寸的uint8类型的二值掩码，其中物体为1，背景为0。

    参数:
    projection_data (np.ndarray): 一个2D的Numpy数组，代表投影图像。

    返回:
    np.ndarray: 一个uint8类型的二值掩码 (值为0或1)。
    """
    # 1. 数据预处理：归一化到 0-255 并转为 uint8
    p_min, p_max = projection_data.min(), projection_data.max()
    if p_max > p_min:
        normalized_data = 255 * (projection_data - p_min) / (p_max - p_min)
    else:
        normalized_data = np.zeros_like(projection_data)

    image_uint8 = normalized_data.astype(np.uint8)

    # 2. 使用Otsu's方法进行自适应二值化
    # cv2.THRESH_OTSU 会忽略我们设置的阈值(0)，并自动计算最佳阈值
    _, binary_mask_255 = cv2.threshold(
        image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 3. 形态学操作清理掩码
    # 定义一个5x5的椭圆形结构元素，这在大多数情况下效果良好
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 开运算: 腐蚀->膨胀 (有效去除掩码外部的孤立噪声点)
    mask_opened = cv2.morphologyEx(binary_mask_255, cv2.MORPH_OPEN, kernel, iterations=2)

    # 闭运算: 膨胀->腐蚀 (有效填充物体内部的小空洞)
    clean_mask_255 = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. 将掩码从 (0, 255) 转换为 (0, 1) 并返回
    final_mask = (clean_mask_255 / 255).astype(np.uint8)

    return final_mask


def process_directory(input_dir: str, output_dir: str):
    """
    处理指定目录中的所有.npy文件，生成并保存掩码。
    """
    # 确保输出目录存在，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: '{output_dir}'")

    # 获取所有.npy文件列表
    try:
        file_list = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
        if not file_list:
            print(f"错误: 在目录 '{input_dir}' 中没有找到.npy文件。")
            return
        print(f"在 '{input_dir}' 中找到 {len(file_list)} 个 .npy 文件。")
    except FileNotFoundError:
        print(f"错误: 输入目录 '{input_dir}' 不存在。")
        return

    # 使用tqdm创建进度条并处理文件
    for filename in tqdm(file_list, desc="正在生成掩码"):
        input_path = os.path.join(input_dir, filename)

        # 构建输出文件名
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_mask.npy"
        output_path = os.path.join(output_dir, output_filename)

        # 加载投影数据
        projection_data = np.load(input_path)

        # 生成掩码
        mask = generate_mask_from_projection(projection_data)

        # 保存掩码
        np.save(output_path, mask)

    print("\n所有掩码已成功生成并保存。")


if __name__ == '__main__':
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="批量为2D投影NPY文件生成二值掩码。")

    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="包含原始投影.npy文件的输入目录的路径。"
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="用于保存生成的掩码.npy文件的输出目录的路径。"
    )

    args = parser.parse_args()

    # --- 执行主函数 ---
    process_directory(args.input_dir, args.output_dir)