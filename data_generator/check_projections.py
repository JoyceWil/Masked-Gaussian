import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm


def normalize_to_uint8(data):
    """
    将输入的numpy数组从任意浮点数范围归一化到0-255的8位无符号整数。

    Args:
        data (np.ndarray): 输入的2D numpy数组。

    Returns:
        np.ndarray: 归一化后的uint8类型的2D数组。
    """
    # 检查数据是否有效
    if data.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)

    # 获取数据的最大值和最小值
    min_val = np.min(data)
    max_val = np.max(data)

    # 如果最大值和最小值相等（例如，图像是纯色），避免除以零
    if max_val == min_val:
        # 可以返回全零或一个中间值，这里返回全零
        return np.zeros(data.shape, dtype=np.uint8)

    # 线性归一化到 0-1 范围
    normalized_data = (data - min_val) / (max_val - min_val)

    # 缩放到 0-255 并转换为 uint8 类型
    uint8_data = (normalized_data * 255).astype(np.uint8)

    return uint8_data


def process_npy_files(input_dir, output_dir):
    """
    处理指定目录下的所有.npy文件，并将它们转换为.png图像。

    Args:
        input_dir (str): 包含.npy文件的输入目录路径。
        output_dir (str): 保存.png图像的输出目录路径。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录 '{output_dir}' 已准备好。")

    # 获取所有.npy文件列表
    try:
        npy_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.npy')]
        if not npy_files:
            print(f"警告：在目录 '{input_dir}' 中没有找到.npy文件。")
            return
    except FileNotFoundError:
        print(f"错误：输入目录 '{input_dir}' 不存在。")
        return

    print(f"在 '{input_dir}' 中找到 {len(npy_files)} 个.npy文件，开始处理...")

    # 使用tqdm创建进度条
    for filename in tqdm(npy_files, desc="转换进度"):
        input_path = os.path.join(input_dir, filename)

        # 构建输出文件名，将.npy后缀替换为.png
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(output_dir, output_filename)

        try:
            # 加载.npy文件
            projection_data = np.load(input_path)

            # 确保数据是2D的
            if projection_data.ndim != 2:
                print(f"\n警告：文件 '{filename}' 不是一个2D数组（维度为{projection_data.ndim}），已跳过。")
                continue

            # 归一化数据
            image_data = normalize_to_uint8(projection_data)

            # 使用Pillow从numpy数组创建图像
            # 'L' 模式表示8位像素的灰度图
            image = Image.fromarray(image_data, mode='L')

            # 保存图像
            image.save(output_path)

        except Exception as e:
            print(f"\n处理文件 '{filename}' 时发生错误: {e}")

    print(f"\n处理完成！所有转换后的图像已保存到 '{output_dir}' 目录。")


def main():
    """
    主函数，用于解析命令行参数并启动处理流程。
    """
    parser = argparse.ArgumentParser(
        description="将目录中的.npy投影文件转换为.png图像以进行可视化检查。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        required=True,
        help="包含.npy文件的输入目录的路径。"
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default=None,
        help="保存.png图像的输出目录的路径。\n(默认: 在输入目录旁创建一个名为 'input_dir_pngs' 的新目录)"
    )

    args = parser.parse_args()

    # 如果没有指定输出目录，则在输入目录旁边创建一个
    output_dir = args.output_dir
    if output_dir is None:
        # 获取输入目录的绝对路径，并移除末尾的斜杠（如果有）
        abs_input_dir = os.path.abspath(args.input_dir)
        base_name = os.path.basename(abs_input_dir)
        parent_dir = os.path.dirname(abs_input_dir)
        output_dir = os.path.join(parent_dir, f"{base_name}_pngs")

    process_npy_files(args.input_dir, output_dir)


if __name__ == "__main__":
    main()