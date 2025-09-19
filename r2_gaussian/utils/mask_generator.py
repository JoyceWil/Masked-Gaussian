# r2_gaussian/utils/mask_generator.py (最终集成版)
import os
import os.path as osp
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# 将您脚本中的核心函数直接移入此处，并添加文档字符串
def convert_mu_to_hu(mu_image: np.ndarray) -> np.ndarray:
    """
    将原始的mu值（衰减系数）图像通过线性映射转换为HU（亨氏单位）图像。
    这个映射是基于图像本身的最大和最小值。

    Args:
        mu_image (np.ndarray): 包含原始mu值的输入图像。

    Returns:
        np.ndarray: 转换后的HU值图像。
    """
    HU_AIR = -1000.0
    HU_HIGH_DENSITY = 1000.0
    mu_image = np.nan_to_num(mu_image)
    mu_min = np.min(mu_image)
    mu_max = np.max(mu_image)

    if mu_max - mu_min < 1e-6:
        return np.full_like(mu_image, HU_AIR, dtype=np.float32)
    else:
        a = (HU_HIGH_DENSITY - HU_AIR) / (mu_max - mu_min)
        b = HU_AIR - a * mu_min
        hu_image = a * mu_image + b
        return hu_image.astype(np.float32)


def apply_windowing(hu_image: np.ndarray, window_level: int, window_width: int) -> np.ndarray:
    """
    根据给定的窗位(WL)和窗宽(WW)将HU值图像转换为归一化的[0, 1]软掩码。

    Args:
        hu_image (np.ndarray): HU值图像。
        window_level (int): 窗位 (Window Level)。
        window_width (int): 窗宽 (Window Width)。

    Returns:
        np.ndarray: 归一化后的软掩码。
    """
    min_hu = window_level - (window_width / 2.0)
    max_hu = window_level + (window_width / 2.0)

    soft_mask = hu_image.astype(np.float32)
    np.clip(soft_mask, min_hu, max_hu, out=soft_mask)

    if window_width == 0:
        return np.zeros_like(soft_mask)

    soft_mask = (soft_mask - min_hu) / window_width
    return soft_mask


# 这是 train.py 将要调用的主函数
def check_and_generate_masks(
        source_path: str,
        bone_wl: int,
        bone_ww: int,
        tissue_wl: int,
        tissue_ww: int,
        save_png_previews: bool = True
) -> (str, str):
    """
    检查掩码是否存在，如果不存在则使用窗宽窗位法从原始投影生成软掩码。

    Args:
        source_path (str): 场景的根目录 (例如 'data/.../0_chest_cone')。
        bone_wl (int): 骨骼掩码的窗位。
        bone_ww (int): 骨骼掩码的窗宽。
        tissue_wl (int): 软组织掩码的窗位。
        tissue_ww (int): 软组织掩码的窗宽。
        save_png_previews (bool): 是否保存PNG格式的预览图。

    Returns:
        tuple[str, str]: 返回 (软组织掩码目录, 核心骨架掩码目录) 的路径。
    """
    print("   - 正在执行窗宽窗位软掩码生成流程...")

    # 1. 定义所有需要的路径
    proj_train_dir = osp.join(source_path, 'proj_train')
    base_mask_dir = osp.join(source_path, 'masks')

    # 核心骨架掩码 (之前叫 bone_mask)
    core_mask_npy_dir = osp.join(base_mask_dir, 'bone_masks_npy')
    core_mask_png_dir = osp.join(base_mask_dir, 'bone_masks_png')

    # 软组织掩码 (之前叫 tissue_mask)
    soft_mask_npy_dir = osp.join(base_mask_dir, 'tissue_masks_npy')
    soft_mask_png_dir = osp.join(base_mask_dir, 'tissue_masks_png')

    output_dirs = [core_mask_npy_dir, soft_mask_npy_dir]
    if save_png_previews:
        output_dirs.extend([core_mask_png_dir, soft_mask_png_dir])

    # 2. 准备目录结构
    print("   - 正在准备输出目录...")
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)

    # 3. 检查输入并获取文件列表
    if not osp.isdir(proj_train_dir):
        raise FileNotFoundError(f"输入目录 '{proj_train_dir}' 不存在！无法生成掩码。")

    file_list = sorted(glob.glob(osp.join(proj_train_dir, '*.npy')))
    if not file_list:
        raise FileNotFoundError(f"在目录 '{proj_train_dir}' 中没有找到任何 .npy 投影文件。")

    print(f"   - 在 '{proj_train_dir}' 中找到 {len(file_list)} 个投影文件。")

    # 4. 批处理循环
    for file_path in tqdm(file_list, desc="   - 生成软掩码", leave=False):
        base_name = osp.basename(file_path)
        file_name_without_ext = osp.splitext(base_name)[0]

        original_mu_image = np.load(file_path)
        hu_image = convert_mu_to_hu(original_mu_image)

        core_mask = apply_windowing(hu_image, bone_wl, bone_ww)
        soft_mask = apply_windowing(hu_image, tissue_wl, tissue_ww)

        # 保存.npy数据
        np.save(osp.join(core_mask_npy_dir, f"{file_name_without_ext}.npy"), core_mask)
        np.save(osp.join(soft_mask_npy_dir, f"{file_name_without_ext}.npy"), soft_mask)

        # 如果需要，保存.png可视化文件
        if save_png_previews:
            plt.imsave(osp.join(core_mask_png_dir, f"{file_name_without_ext}.png"), core_mask, cmap='gray', vmin=0,
                       vmax=1)
            plt.imsave(osp.join(soft_mask_png_dir, f"{file_name_without_ext}.png"), soft_mask, cmap='gray', vmin=0,
                       vmax=1)

    print(f"   - 所有 {len(file_list)} 个文件的软掩码已成功生成。")

    # 5. 返回生成的NPY目录路径，供 train.py 使用
    return soft_mask_npy_dir, core_mask_npy_dir