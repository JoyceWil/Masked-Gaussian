import numpy as np
import os

# --- 1. 请在这里配置您的NPY文件路径 ---
FDK_CLOUD_PATH = '/media/data2/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone/init_0_chest_cone.npy'

# --- 2. 主程序：读取并分析NPY文件 ---
def analyze_npy_file(file_path):
    """加载一个NPY文件并打印其详细的元信息和统计数据。"""
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到，请检查路径是否正确。\n路径: {file_path}")
        return

    print(f"--- 正在分析文件: {os.path.basename(file_path)} ---")
    try:
        data = np.load(file_path)
        print("文件加载成功！")
        print("\n--- 核心属性 ---")
        print(f"1. 数据类型 (dtype): {data.dtype}")
        print(f"2. 数据形状 (shape): {data.shape}")
        print(f"3. 数据维度 (ndim): {data.ndim}")
        print("\n--- 统计信息 ---")
        print(f"4. 最小值 (Min Value): {np.min(data)}")
        print(f"5. 最大值 (Max Value): {np.max(data)}")
    except Exception as e:
        print(f"\n在读取或分析文件时发生了一个意料之外的错误: {e}")

# --- 3. 运行分析 ---
if __name__ == '__main__':
    analyze_npy_file(FDK_CLOUD_PATH)