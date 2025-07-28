# =================================================================================
# check_data_range.py
# 功能: 检查 .npy 文件的数值范围，以确认是否被归一化。
# =================================================================================
import numpy as np
import os

# --- 配置区 ---
# 请将此路径修改为您的 vol_gt.npy 文件的实际路径
GT_VOLUME_PATH = "/media/data2/hezhipeng/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/vol_gt.npy"

if __name__ == "__main__":
    if not os.path.exists(GT_VOLUME_PATH):
        print(f"错误: 找不到文件 '{GT_VOLUME_PATH}'")
    else:
        print("正在加载文件并分析数值范围...")
        volume = np.load(GT_VOLUME_PATH)

        print("\n--- 数据范围分析报告 ---")
        print(f"文件路径: {GT_VOLUME_PATH}")
        print(f"数据维度 (Shape): {volume.shape}")
        print(f"数据类型 (Dtype): {volume.dtype}")
        print(f"最小值 (Min): {np.min(volume):.4f}")
        print(f"最大值 (Max): {np.max(volume):.4f}")
        print(f"平均值 (Mean): {np.mean(volume):.4f}")
        print("--------------------------\n")

        print("分析结论:")
        if np.max(volume) <= 1.0 and np.min(volume) >= 0.0:
            print("数据很可能已被归一化到 [0, 1] 范围。")
        elif np.max(volume) <= 1.0 and np.min(volume) >= -1.0:
            print("数据很可能已被归一化到 [-1, 1] 范围。")
        elif np.max(volume) > 100:
            print("数据看起来像是原始的HU值，请检查阈值设置。")
        else:
            print("数据范围未知，请仔细检查数据来源。")