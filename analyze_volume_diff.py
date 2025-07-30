import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim

# --- 1. 配置您的文件路径 ---
# Baseline模型的预测结果
# PRED_PATH = "/home/hezhipeng/Workbench/r2_gaussian-original/output/teapot/2025-07-22_04-30-34/point_cloud/iteration_30000/vol_pred.npy"
PRED_PATH = "/home/hezhipeng/Workbench/r2_gaussian-main/output/teapot/2025-07-29_02-05-31/point_cloud/iteration_30000/vol_pred.npy"
# Ground Truth 体数据
# GT_PATH = "/home/hezhipeng/Workbench/r2_gaussian-original/output/teapot/2025-07-22_04-30-34/point_cloud/iteration_30000/vol_gt.npy"
GT_PATH = "/home/hezhipeng/Workbench/r2_gaussian-main/output/teapot/2025-07-29_02-05-31/point_cloud/iteration_30000/vol_gt.npy"
# 结果图片保存的目录
OUTPUT_DIR = "/home/hezhipeng/Workbench/r2_gaussian-main/output/volume_analysis_output"


# --- 2. 主分析函数 ---
def analyze_and_visualize(pred_vol, gt_vol, output_dir):
    """
    对预测体数据和GT体数据进行全面的定量和可视化分析。
    """
    print("--- 开始分析 ---")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"分析结果将保存到: {os.path.abspath(output_dir)}")

    # 检查维度是否匹配
    if pred_vol.shape != gt_vol.shape:
        print(f"错误: 预测体 {pred_vol.shape} 和 GT体 {gt_vol.shape} 的维度不匹配!")
        return

    print(f"体数据维度: {gt_vol.shape} (Z, Y, X)")

    # --- 定量分析 ---
    print("\n--- 1. 定量指标 ---")

    # 计算误差
    absolute_diff = np.abs(pred_vol - gt_vol)
    squared_diff = (pred_vol - gt_vol) ** 2

    # MAE (Mean Absolute Error)
    mae = np.mean(absolute_diff)
    print(f"平均绝对误差 (MAE): {mae:.6f}")

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean(squared_diff))
    print(f"均方根误差 (RMSE): {rmse:.6f}")

    # SSIM (Structural Similarity Index)
    # data_range是数据值的范围 (max - min)
    data_range = gt_vol.max() - gt_vol.min()
    ssim_score = ssim(gt_vol, pred_vol, data_range=data_range)
    print(f"结构相似性 (SSIM): {ssim_score:.6f} (越接近1越好)")

    # --- 误差分布直方图 ---
    print("\n--- 2. 生成误差分布直方图 ---")
    raw_diff = (pred_vol - gt_vol).flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(raw_diff, bins=100, color='skyblue', alpha=0.8)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零误差')
    plt.title("误差 (Prediction - Ground Truth) 分布直方图")
    plt.xlabel("误差值")
    plt.ylabel("体素数量")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "error_distribution_histogram.png"))
    plt.close()
    print("直方图已保存。")

    # --- 可视化分析 (切片对比) ---
    print("\n--- 3. 生成正交平面切片对比图 ---")

    # 选择中心切片进行可视化
    z_slice_idx = gt_vol.shape[0] // 2
    y_slice_idx = gt_vol.shape[1] // 2
    x_slice_idx = gt_vol.shape[2] // 2

    views = {
        "axial": {"axis": 0, "slice_idx": z_slice_idx},
        "sagittal": {"axis": 2, "slice_idx": x_slice_idx},
        "coronal": {"axis": 1, "slice_idx": y_slice_idx},
    }

    for view_name, view_params in views.items():
        axis = view_params["axis"]
        slice_idx = view_params["slice_idx"]

        if axis == 0:  # Axial (Z)
            gt_slice = gt_vol[slice_idx, :, :]
            pred_slice = pred_vol[slice_idx, :, :]
        elif axis == 1:  # Coronal (Y)
            gt_slice = gt_vol[:, slice_idx, :]
            pred_slice = pred_vol[:, slice_idx, :]
        else:  # Sagittal (X)
            gt_slice = gt_vol[:, :, slice_idx]
            pred_slice = pred_vol[:, :, slice_idx]

        diff_slice = np.abs(pred_slice - gt_slice)

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))

        # Ground Truth
        im1 = axes[0].imshow(gt_slice, cmap='gray')
        axes[0].set_title(f"Ground Truth ({view_name.capitalize()} View)")
        axes[0].axis('off')

        # Prediction
        im2 = axes[1].imshow(pred_slice, cmap='gray')
        axes[1].set_title(f"Prediction ({view_name.capitalize()} View)")
        axes[1].axis('off')

        # Absolute Difference
        im3 = axes[2].imshow(diff_slice, cmap='viridis')  # 使用'viridis'或'plasma'热图
        axes[2].set_title("Absolute Difference (Error)")
        axes[2].axis('off')
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(f"Comparison at {view_name.capitalize()} Slice {slice_idx}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_path = os.path.join(output_dir, f"{view_name}_view_comparison.png")
        plt.savefig(output_path)
        plt.close(fig)
        print(f"{view_name.capitalize()} 视图对比图已保存。")

    print("\n--- 分析完成 ---")


# --- 3. 运行脚本 ---
if __name__ == "__main__":
    # 加载数据
    try:
        print("正在加载体数据...")
        pred_volume = np.load(PRED_PATH)
        gt_volume = np.load(GT_PATH)
        print("数据加载成功！")
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
        exit()

    # 执行分析
    analyze_and_visualize(pred_volume, gt_volume, OUTPUT_DIR)