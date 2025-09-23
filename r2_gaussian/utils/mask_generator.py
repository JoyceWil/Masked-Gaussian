# r2_gaussian/utils/mask_generator.py (最终零参数自动化版)
import os
import os.path as osp
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks
from skimage.measure import shannon_entropy


# ==============================================================================
# 核心工具函数 (无变化)
# ==============================================================================

def convert_mu_to_hu(mu_image: np.ndarray) -> np.ndarray:
    """将原始的mu值（衰减系数）图像通过线性映射转换为HU（亨氏单位）图像。"""
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
    """根据给定的窗位(WL)和窗宽(WW)将HU值图像转换为归一化的[0, 1]软掩码。"""
    min_hu = window_level - (window_width / 2.0)
    max_hu = window_level + (window_width / 2.0)
    soft_mask = hu_image.astype(np.float32)
    np.clip(soft_mask, min_hu, max_hu, out=soft_mask)
    if window_width == 0:
        return np.zeros_like(soft_mask)
    soft_mask = (soft_mask - min_hu) / window_width
    return soft_mask


# ==============================================================================
# 自动化优化函数 (现在直接使用所有HU图像)
# ==============================================================================

def _find_optimal_window(
        all_hu_images: list,
        peak_map: dict,
        target_class: str,
        pre_threshold_ratio: float
) -> dict:
    """
    【内部函数】针对给定的目标类别（'soft'或'hard'），自动搜索并返回最优的窗宽窗位。
    """
    print(f"   - [Auto] 开始为 '{target_class}' 类别搜索最优窗...")

    target_hu = peak_map[target_class]
    print(f"   - [Auto] 分析目标: '{target_class}' -> 动态设定目标HU为: {target_hu}")

    pre_threshold = None
    if target_class == 'hard':
        threshold_point = peak_map['soft'] + (peak_map['hard'] - peak_map['soft']) * pre_threshold_ratio
        pre_threshold = int(threshold_point)
        print(f"   - [Auto] 将使用约束熵优化，只考虑HU值 > {pre_threshold} 的区域。")

    all_hu_values = np.concatenate([img.flatten() for img in all_hu_images])
    foreground_hu = all_hu_values[(all_hu_values > -500) & (all_hu_values < 2500)]

    std_hu = np.std(foreground_hu[(foreground_hu > target_hu - 150) & (foreground_hu < target_hu + 150)])
    wl_start = int(target_hu - 1.5 * std_hu)
    wl_stop = int(target_hu + 1.5 * std_hu)
    wl_step = max(5, int(std_hu / 10))
    ww_start = max(20, int(std_hu * 0.5))
    ww_stop = int(std_hu * 4)
    ww_step = max(10, int(std_hu / 5))

    wls_to_search = range(wl_start, wl_stop, wl_step)
    wws_to_search = range(ww_start, ww_stop, ww_step)
    if not wls_to_search or not wws_to_search:
        print(f"   - [Auto] 错误: 为'{target_class}'自动计算的搜索范围为空。跳过。")
        return None

    best_entropy = -1
    optimal_wl, optimal_ww = -1, -1
    hu_stack = np.stack(all_hu_images, axis=0)
    total_iterations = len(wls_to_search) * len(wws_to_search)

    with tqdm(total=total_iterations, desc=f"   - [Auto] 搜索'{target_class}'窗", leave=False) as pbar:
        for wl in wls_to_search:
            for ww in wws_to_search:
                all_soft_masks = apply_windowing(hu_stack, wl, ww)
                all_masks_uint8 = (all_soft_masks * 255).astype(np.uint8)

                # 使用更鲁棒的约束熵计算
                current_entropies = []
                for i in range(len(all_hu_images)):
                    mask_uint8 = all_masks_uint8[i]
                    if pre_threshold is not None:
                        roi_mask = all_hu_images[i] > pre_threshold
                        entropy = shannon_entropy(mask_uint8[roi_mask]) if np.any(roi_mask) else 0
                    else:
                        entropy = shannon_entropy(mask_uint8)
                    current_entropies.append(entropy)

                avg_entropy = np.mean(current_entropies)

                if avg_entropy > best_entropy:
                    best_entropy = avg_entropy
                    optimal_wl, optimal_ww = wl, ww
                pbar.update(1)

    print(f"   - [Auto] '{target_class}' 优化完成! 最优WL: {optimal_wl}, 最优WW: {optimal_ww}")
    return {'wl': optimal_wl, 'ww': optimal_ww}


# ==============================================================================
# train.py 将要调用的主函数 (最终零参数自动化版)
# ==============================================================================

def check_and_generate_masks(
        source_path: str,
        pre_threshold_ratio: float = 0.5,
        save_previews: bool = True
) -> (str, str):
    """
    【最终全自动版】检查掩码是否存在，如果不存在，则通过分析【所有】投影找到最优窗，并生成软掩码。

    Args:
        source_path (str): 场景的根目录 (例如 'data/.../0_chest_cone')。
        pre_threshold_ratio (float): 优化硬质窗时，用于约束熵计算的阈值比例。
        save_previews (bool): 是否保存PNG格式的预览图和HU直方图。

    Returns:
        tuple[str, str]: 返回 (软组织掩码目录, 核心骨架掩码目录) 的路径。
    """
    print("   - 正在执行【最终全自动】窗宽窗位软掩码生成流程...")

    # 1. 定义路径
    proj_train_dir = osp.join(source_path, 'proj_train')
    base_mask_dir = osp.join(source_path, 'masks')
    tissue_mask_npy_dir = osp.join(base_mask_dir, 'tissue_masks_npy')
    bone_mask_npy_dir = osp.join(base_mask_dir, 'bone_masks_npy')

    # 2. 检查是否已存在
    if osp.exists(tissue_mask_npy_dir) and osp.exists(bone_mask_npy_dir):
        print("   - 掩码目录已存在，跳过生成。")
        return tissue_mask_npy_dir, bone_mask_npy_dir

    # --- 自动化分析流程开始 ---
    # 3. 加载【所有】文件用于分析
    print("   - [Auto] 未找到掩码，开始对所有投影进行自动化分析...")
    file_list = sorted(glob.glob(osp.join(proj_train_dir, '*.npy')))
    if not file_list:
        raise FileNotFoundError(f"在目录 '{proj_train_dir}' 中没有找到任何 .npy 投影文件。")

    print(f"   - [Auto] 找到 {len(file_list)} 个投影文件，将全部用于分析。")
    all_hu_images = [convert_mu_to_hu(np.load(file)) for file in
                     tqdm(file_list, desc="   - [Auto] 加载并转换所有投影", leave=False)]

    # 4. 自主发现物质峰值
    print("   - [Auto] 正在分析图像的物质构成...")
    all_hu_values = np.concatenate([img.flatten() for img in all_hu_images])
    foreground_hu = all_hu_values[(all_hu_values > -500) & (all_hu_values < 2500)]
    hist, bin_edges = np.histogram(foreground_hu, bins=1000, range=(-500, 2500))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peaks, _ = find_peaks(hist, prominence=np.max(hist) * 0.005, distance=50)

    if len(peaks) < 2:
        raise RuntimeError("未能检测到至少两个显著的物质峰值。无法自动区隔软/硬质材料。")

    peak_hus = sorted([int(bin_centers[p]) for p in peaks])
    peak_map = {'soft': peak_hus[0], 'hard': peak_hus[-1]}
    print(f"   - [Auto] 物质分析报告: '软质'峰(Soft) @ {peak_map['soft']} HU, '硬质'峰(Hard) @ {peak_map['hard']} HU")

    if save_previews:
        hist_path = osp.join(base_mask_dir, 'auto_hu_histogram.png')
        os.makedirs(base_mask_dir, exist_ok=True)
        plt.figure(figsize=(12, 6));
        plt.plot(bin_centers, hist);
        plt.scatter(bin_centers[peaks], hist[peaks], color='red', zorder=5)
        plt.title('Automated HU Peak Detection');
        plt.xlabel('HU');
        plt.ylabel('Frequency');
        plt.grid(True, alpha=0.3);
        plt.savefig(hist_path)
        print(f"   - [Auto] HU分布图已保存至: {hist_path}")

    # 5. 分别为 'soft' (tissue) 和 'hard' (bone) 找到最优窗
    optimal_tissue_window = _find_optimal_window(all_hu_images, peak_map, 'soft', pre_threshold_ratio)
    optimal_bone_window = _find_optimal_window(all_hu_images, peak_map, 'hard', pre_threshold_ratio)

    if not optimal_tissue_window or not optimal_bone_window:
        raise RuntimeError("自动寻找最优窗失败。")

    # --- 自动化分析流程结束，开始批量生成 ---
    # 6. 准备目录并批量生成所有掩码
    print("   - [Generate] 使用自动找到的最优窗批量生成所有掩码...")
    tissue_mask_png_dir = osp.join(base_mask_dir, 'tissue_masks_png')
    bone_mask_png_dir = osp.join(base_mask_dir, 'bone_masks_png')
    output_dirs = [tissue_mask_npy_dir, bone_mask_npy_dir]
    if save_previews:
        output_dirs.extend([tissue_mask_png_dir, bone_mask_png_dir])
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)

    # 注意：这里我们已经加载了所有HU图像，可以直接复用，无需重新加载文件
    for i, file_path in enumerate(tqdm(file_list, desc="   - [Generate] 生成掩码", leave=False)):
        base_name = osp.basename(file_path)
        file_name_without_ext = osp.splitext(base_name)[0]

        hu_image = all_hu_images[i]  # 直接复用已加载的HU图像

        tissue_mask = apply_windowing(hu_image, optimal_tissue_window['wl'], optimal_tissue_window['ww'])
        bone_mask = apply_windowing(hu_image, optimal_bone_window['wl'], optimal_bone_window['ww'])

        np.save(osp.join(tissue_mask_npy_dir, f"{file_name_without_ext}.npy"), tissue_mask)
        np.save(osp.join(bone_mask_npy_dir, f"{file_name_without_ext}.npy"), bone_mask)

        if save_previews:
            plt.imsave(osp.join(tissue_mask_png_dir, f"{file_name_without_ext}.png"), tissue_mask, cmap='gray', vmin=0,
                       vmax=1)
            plt.imsave(osp.join(bone_mask_png_dir, f"{file_name_without_ext}.png"), bone_mask, cmap='gray', vmin=0,
                       vmax=1)

    print(f"   - 所有 {len(file_list)} 个文件的软掩码已成功生成。")

    # 7. 返回生成的NPY目录路径
    return tissue_mask_npy_dir, bone_mask_npy_dir