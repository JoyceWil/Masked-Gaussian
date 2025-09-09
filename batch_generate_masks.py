import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm


# --- 核心函数区 ---

def convert_mu_to_hu(mu_image: np.ndarray) -> np.ndarray:
    """
    将原始的mu值（衰减系数）图像通过线性映射转换为HU（亨氏单位）图像。
    这个逻辑完全来自于您提供的脚本。

    Args:
        mu_image (np.ndarray): 包含原始mu值的输入图像。

    Returns:
        np.ndarray: 转换后的HU值图像。
    """
    # 定义用于线性映射的HU参考点
    HU_AIR = -1000.0
    HU_HIGH_DENSITY = 1000.0

    # 清理数据中的NaN值
    mu_image = np.nan_to_num(mu_image)

    mu_min = np.min(mu_image)
    mu_max = np.max(mu_image)

    # 避免除以零的边缘情况
    if mu_max - mu_min < 1e-6:
        # 如果图像是纯色的，则所有值都映射到空气值
        return np.full_like(mu_image, HU_AIR, dtype=np.float32)
    else:
        a = (HU_HIGH_DENSITY - HU_AIR) / (mu_max - mu_min)
        b = HU_AIR - a * mu_min
        hu_image = a * mu_image + b
        return hu_image.astype(np.float32)


def apply_windowing(hu_image: np.ndarray, window_level: int, window_width: int) -> np.ndarray:
    """
    根据给定的窗位(WL)和窗宽(WW)将HU值图像转换为归一化的软掩码。
    """
    min_hu = window_level - (window_width / 2.0)
    max_hu = window_level + (window_width / 2.0)

    soft_mask = hu_image.astype(np.float32)
    np.clip(soft_mask, min_hu, max_hu, out=soft_mask)

    if window_width == 0:
        return np.zeros_like(soft_mask)

    soft_mask = (soft_mask - min_hu) / window_width
    return soft_mask


# ==============================================================================
# 主执行块
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 用户配置 ---
    INPUT_DIR = 'data/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone/proj_train'
    # 主输出文件夹：将在这里创建所有子文件夹和输出文件
    MASK_OUTPUT_DIR = 'data/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone/masks'

    # 我们之前确定的窗宽窗位参数
    BONE_MASK_PARAMS = {'window_level': 380, 'window_width': 380}
    TISSUE_MASK_PARAMS = {'window_level': 40, 'window_width': 400}

    # --- 2. 准备目录结构 ---
    bone_mask_npy_dir = os.path.join(MASK_OUTPUT_DIR, 'bone_masks_npy')
    tissue_mask_npy_dir = os.path.join(MASK_OUTPUT_DIR, 'tissue_masks_npy')
    bone_mask_png_dir = os.path.join(MASK_OUTPUT_DIR, 'bone_masks_png')
    tissue_mask_png_dir = os.path.join(MASK_OUTPUT_DIR, 'tissue_masks_png')

    output_dirs = [bone_mask_npy_dir, tissue_mask_npy_dir, bone_mask_png_dir, tissue_mask_png_dir]

    print("正在准备输出目录结构...")
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)
    print(" -> 所有输出目录已就绪。")

    # --- 3. 检查输入并获取文件列表 ---
    if not os.path.isdir(INPUT_DIR):
        print(f"\n错误: 输入目录 '{INPUT_DIR}' 不存在！请修改第69行 'INPUT_DIR' 的路径。")
        exit()

    file_list = sorted(glob.glob(os.path.join(INPUT_DIR, '*.npy')))

    if not file_list:
        print(f"\n警告: 在目录 '{INPUT_DIR}' 中没有找到任何 .npy 文件。")
        exit()

    print(f"\n在 '{INPUT_DIR}' 中找到 {len(file_list)} 个 .npy 文件。开始生成软掩码...")

    # --- 4. 批处理循环 (已集成mu->HU转换) ---
    for file_path in tqdm(file_list, desc="生成软掩码"):
        base_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]

        # 步骤 1: 加载原始mu值图像
        original_mu_image = np.load(file_path)

        # 步骤 2: 【关键步骤】将mu值转换为HU值
        hu_image = convert_mu_to_hu(original_mu_image)

        # 步骤 3: 对转换后的HU图像应用窗宽窗位来生成软掩码
        bone_soft_mask = apply_windowing(
            hu_image,
            BONE_MASK_PARAMS['window_level'],
            BONE_MASK_PARAMS['window_width']
        )

        soft_tissue_soft_mask = apply_windowing(
            hu_image,
            TISSUE_MASK_PARAMS['window_level'],
            TISSUE_MASK_PARAMS['window_width']
        )

        # 步骤 4: 保存.npy数据和.png可视化文件
        # 保存.npy
        np.save(os.path.join(bone_mask_npy_dir, f"{file_name_without_ext}.npy"), bone_soft_mask)
        np.save(os.path.join(tissue_mask_npy_dir, f"{file_name_without_ext}.npy"), soft_tissue_soft_mask)

        # 保存.png
        plt.imsave(os.path.join(bone_mask_png_dir, f"{file_name_without_ext}.png"), bone_soft_mask,
                   cmap='gray', vmin=0, vmax=1)
        plt.imsave(os.path.join(tissue_mask_png_dir, f"{file_name_without_ext}.png"), soft_tissue_soft_mask,
                   cmap='gray', vmin=0, vmax=1)

    print(f"\n--- 处理完成！---")
    print(f"所有 {len(file_list)} 个文件的软掩码数据和可视化图像已成功生成。")
    print("输出文件已保存至:")
    for directory in output_dirs:
        print(f"  - {directory}")