import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image  # 我们将使用PIL来保存为PNG，因为这样更直观

# --- 配置区 ---
# 请根据您的实际路径进行修改
BASE_DIR = 'data/0_chest_cone'
META_FILE = os.path.join(BASE_DIR, 'meta_data.json')

# 【新】定义三个输出目录
OUTPUT_AIR_DIR = os.path.join(BASE_DIR, 'masks', 'air_masks')
OUTPUT_CORE_DIR = os.path.join(BASE_DIR, 'masks', 'core_masks')  # core = bone + calcium
OUTPUT_SOFT_DIR = os.path.join(BASE_DIR, 'masks', 'soft_masks')  # soft = fat + soft_tissue

# 定义HU值范围 (Hounsfield Units) - 保持不变
HU_RANGES = {
    "bone": (400, 3000),
    "calcium": (150, 400),
    "soft_tissue": (20, 80),
    "fat": (-100, -20),
    "air": (-1024, -950),
}

# 定义用于线性映射的HU参考点 - 保持不变
HU_AIR = -1000.0
HU_HIGH_DENSITY = 1000.0


# --- 配置区结束 ---

def generate_specific_masks():
    """
    主函数，根据您的要求，从原始2D投影生成三个独立的二值掩码：
    1. air_mask.npy
    2. core_mask.npy (bone + calcium)
    3. soft_mask.npy (fat + soft_tissue)
    """
    print("--- 开始从原始2D投影生成指定的ROI掩码 ---")

    # 创建所有输出目录
    for path in [OUTPUT_AIR_DIR, OUTPUT_CORE_DIR, OUTPUT_SOFT_DIR]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"已创建输出目录: {path}")

    try:
        with open(META_FILE, 'r') as f:
            meta_data = json.load(f)
        proj_train_views = meta_data['proj_train']
        print(f"成功加载元数据，找到 {len(proj_train_views)} 个训练投影视图。")
    except Exception as e:
        print(f"错误：无法加载或解析元数据文件 {META_FILE}。请检查路径和文件内容。")
        print(e)
        return

    print("\n--- 正在处理每个投影文件 ---")
    for view_data in tqdm(proj_train_views, desc="生成ROI掩码"):
        relative_path = view_data['file_path']
        input_path = os.path.join(BASE_DIR, relative_path)

        # 从相对路径中提取基础文件名，用于命名输出文件
        # 例如 'proj_train/proj_train_0000.npy' -> 'proj_train_0000'
        base_name = os.path.splitext(os.path.basename(relative_path))[0]

        if not os.path.exists(input_path):
            print(f"警告：找不到文件 {input_path}，已跳过。")
            continue

        proj_data = np.load(input_path)
        proj_data = np.nan_to_num(proj_data)

        # --- 将投影数据从mu值转换为HU值 (逻辑保持不变) ---
        mu_min = np.min(proj_data)
        mu_max = np.max(proj_data)

        if mu_max - mu_min < 1e-6:
            a = 0
            b = HU_AIR
        else:
            a = (HU_HIGH_DENSITY - HU_AIR) / (mu_max - mu_min)
            b = HU_AIR - a * mu_min
        hu_image = a * proj_data + b

        # --- 【新逻辑】生成并保存三个独立的二值掩码 ---

        # 1. 生成 Air 掩码
        air_min, air_max = HU_RANGES["air"]
        air_mask = ((hu_image >= air_min) & (hu_image < air_max)).astype(np.float32)
        np.save(os.path.join(OUTPUT_AIR_DIR, f"{base_name}.npy"), air_mask)

        # 2. 生成 Core 掩码 (Bone + Calcium)
        bone_min, bone_max = HU_RANGES["bone"]
        calc_min, calc_max = HU_RANGES["calcium"]
        core_mask = (
                ((hu_image >= bone_min) & (hu_image < bone_max)) |
                ((hu_image >= calc_min) & (hu_image < calc_max))
        ).astype(np.float32)
        np.save(os.path.join(OUTPUT_CORE_DIR, f"{base_name}.npy"), core_mask)

        # 3. 生成 Soft 掩码 (Fat + Soft Tissue)
        fat_min, fat_max = HU_RANGES["fat"]
        soft_min, soft_max = HU_RANGES["soft_tissue"]
        soft_mask = (
                ((hu_image >= fat_min) & (hu_image < fat_max)) |
                ((hu_image >= soft_min) & (hu_image < soft_max))
        ).astype(np.float32)
        np.save(os.path.join(OUTPUT_SOFT_DIR, f"{base_name}.npy"), soft_mask)

    print("\n--- 所有指定的ROI掩码生成完毕！---")
    print(f"Air掩码已保存至: {OUTPUT_AIR_DIR}")
    print(f"Core掩码已保存至: {OUTPUT_CORE_DIR}")
    print(f"Soft掩码已保存至: {OUTPUT_SOFT_DIR}")


if __name__ == '__main__':
    generate_specific_masks()