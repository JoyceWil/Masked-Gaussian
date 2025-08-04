import cv2
import numpy as np
import os
from tqdm import tqdm


def _generate_single_mask_set(projection_data: np.ndarray, soft_p: int, core_p: int):
    """
    【内部函数】为单张投影生成软组织和核心骨架掩码。(逻辑保持不变)
    """
    # 步骤1: 使用Otsu阈值法获取精确的初始轮廓
    p_min, p_max = projection_data.min(), projection_data.max()
    if p_max <= p_min:
        empty_mask = np.zeros_like(projection_data, dtype=np.uint8)
        return empty_mask, empty_mask

    normalized_data = (255 * (projection_data - p_min) / (p_max - p_min)).astype(np.uint8)
    _, binary_mask_255 = cv2.threshold(normalized_data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    base_mask_for_analysis = cv2.morphologyEx(binary_mask_255, cv2.MORPH_CLOSE, kernel, iterations=3)
    base_mask_for_analysis = (base_mask_for_analysis / 255).astype(np.uint8)

    # 步骤2: 在精确轮廓内计算百分位数
    object_pixels = projection_data[base_mask_for_analysis == 1]
    if object_pixels.size == 0:
        empty_mask = np.zeros_like(projection_data, dtype=np.uint8)
        return empty_mask, empty_mask

    # 步骤3: 生成掩码
    soft_threshold = np.percentile(object_pixels, soft_p)
    soft_mask = np.where(projection_data >= soft_threshold, 1, 0).astype(np.uint8)
    soft_mask = cv2.bitwise_and(soft_mask, soft_mask, mask=base_mask_for_analysis)

    core_threshold = np.percentile(object_pixels, core_p)
    core_mask = np.where(projection_data >= core_threshold, 1, 0).astype(np.uint8)
    core_mask = cv2.bitwise_and(core_mask, core_mask, mask=base_mask_for_analysis)

    core_mask_cleaned = cv2.morphologyEx(core_mask * 255, cv2.MORPH_OPEN, kernel, iterations=1)
    core_mask_final = (core_mask_cleaned / 255).astype(np.uint8)

    return soft_mask, core_mask_final


def check_and_generate_masks(
        source_path: str,
        soft_p: int,
        core_p: int,
        save_png_previews: bool = True  # <-- 新增的控制参数
):
    """
    【主调用函数】检查并按需生成掩码，并返回用于训练的NPY掩码目录路径。

    参数:
        source_path (str): 数据集的根目录。
        soft_p (int): 软组织掩码的百分位数。
        core_p (int): 核心骨架掩码的百分位数。
        save_png_previews (bool): 是否同时保存一份PNG格式的预览图。默认为True。

    返回:
        tuple: (soft_mask_dir_npy, core_mask_dir_npy) 包含NPY掩码的目录路径。
    """
    proj_train_dir = os.path.join(source_path, 'proj_train')
    mask_base_dir = os.path.join(source_path, 'masks')

    # --- 路径定义 (NPY用于训练, PNG用于预览) ---
    output_dir_soft_npy = os.path.join(mask_base_dir, f'soft_masks_p{soft_p}')
    output_dir_core_npy = os.path.join(mask_base_dir, f'core_masks_p{core_p}')

    if save_png_previews:
        output_dir_soft_png = os.path.join(mask_base_dir, f'soft_masks_p{soft_p}_previews')
        output_dir_core_png = os.path.join(mask_base_dir, f'core_masks_p{core_p}_previews')

    # 检查输入目录
    if not os.path.isdir(proj_train_dir):
        raise FileNotFoundError(f"错误: 找不到投影目录 '{proj_train_dir}'。")

    # --- 检查逻辑 (只关心NPY文件是否存在) ---
    try:
        proj_files = [f for f in os.listdir(proj_train_dir) if f.endswith('.npy')]
        if (os.path.isdir(output_dir_soft_npy) and os.path.isdir(output_dir_core_npy) and
                len(os.listdir(output_dir_soft_npy)) == len(proj_files) and
                len(os.listdir(output_dir_core_npy)) == len(proj_files)):
            print(f"✅ NPY掩码已存在且数量匹配，跳过生成过程。")
            return output_dir_soft_npy, output_dir_core_npy
    except FileNotFoundError:
        pass

    # --- 生成逻辑 ---
    print(f"⚠️  检测到NPY掩码不存在或不完整，开始自动生成...")
    os.makedirs(output_dir_soft_npy, exist_ok=True)
    os.makedirs(output_dir_core_npy, exist_ok=True)
    print(f"   - NPY软组织掩码 (p={soft_p}) 将保存至: '{output_dir_soft_npy}'")
    print(f"   - NPY核心骨架掩码 (p={core_p}) 将保存至: '{output_dir_core_npy}'")

    if save_png_previews:
        os.makedirs(output_dir_soft_png, exist_ok=True)
        os.makedirs(output_dir_core_png, exist_ok=True)
        print(f"   - PNG预览图将同步保存。")

    for filename in tqdm(proj_files, desc=f"生成 p{soft_p}/p{core_p} 掩码"):
        input_path = os.path.join(proj_train_dir, filename)
        projection_data = np.load(input_path)

        soft_mask, core_mask = _generate_single_mask_set(projection_data, soft_p, core_p)

        # 1. 保存NPY格式 (用于训练)
        np.save(os.path.join(output_dir_soft_npy, filename), soft_mask)
        np.save(os.path.join(output_dir_core_npy, filename), core_mask)

        # 2. 如果需要，保存PNG格式 (用于预览)
        if save_png_previews:
            png_filename = filename.replace('.npy', '.png')
            cv2.imwrite(os.path.join(output_dir_soft_png, png_filename), soft_mask * 255)
            cv2.imwrite(os.path.join(output_dir_core_png, png_filename), core_mask * 255)

    print("\n✅ 所有掩码已成功生成并保存。")
    return output_dir_soft_npy, output_dir_core_npy