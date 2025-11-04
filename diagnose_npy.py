import numpy as np
import matplotlib.pyplot as plt
import sys

# --- 请修改为您自己的 .npy 文件路径 ---
# npy_file_path = 'your_projection_file.npy' 
# 为了能直接运行，我们先从命令行获取文件名
if len(sys.argv) < 2:
    print("请提供.npy文件路径作为参数运行脚本。")
    print("用法: python diagnose_npy.py your_file.npy")
    sys.exit(1)
npy_file_path = sys.argv[1]
# -----------------------------------------

print(f"--- 正在分析文件: {npy_file_path} ---")

try:
    data = np.load(npy_file_path)
    data = np.nan_to_num(data)  # 替换nan和inf以防出错

    print("\n--- 1. 基本信息 ---")
    print(f"数据类型 (dtype): {data.dtype}")
    print(f"数据形状 (shape): {data.shape}")

    print("\n--- 2. 数值统计 ---")
    print(f"最小值: {np.min(data):.6f}")
    print(f"最大值: {np.max(data):.6f}")
    print(f"平均值: {np.mean(data):.6f}")
    print(f"标准差: {np.std(data):.6f}")

    # 检查是否有负值，这对于判断是否是log转换后的数据很重要
    if np.min(data) < 0:
        print("提示: 数据中包含负值。")
    else:
        print("提示: 数据中所有值均为非负数。")

    print("\n--- 3. 绘制直方图 ---")
    # 将数据展平以便绘制直方图
    flat_data = data.flatten()

    # 移除极端离群值以便更好地可视化主要分布
    # 我们只看99.8%的数据的分布情况
    p_low = np.percentile(flat_data, 0.1)
    p_high = np.percentile(flat_data, 99.9)
    vis_data = flat_data[(flat_data > p_low) & (flat_data < p_high)]

    plt.figure(figsize=(12, 6))
    plt.hist(vis_data, bins=256, color='blue', alpha=0.7)
    plt.title(f'数据值分布直方图 (Histogram of Pixel Values)\n(File: {npy_file_path})')
    plt.xlabel('像素值 (Pixel Value)')
    plt.ylabel('频数 (Frequency)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')  # 使用对数刻度Y轴，以便看清低频数的分布
    plt.show()

    print("\n--- 分析完成 ---")
    print("请观察弹出的直方图和控制台输出的统计信息。")

except FileNotFoundError:
    print(f"错误: 文件未找到 {npy_file_path}")
except Exception as e:
    print(f"分析时发生错误: {e}")