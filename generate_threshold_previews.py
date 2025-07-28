# =================================================================================
# generate_threshold_previews.py
#
# 功能: 为一个3D .npy 体数据生成一系列应用了不同阈值的2D预览图。
#       专为服务器环境设计，无需GUI。
# =================================================================================
import numpy as np
import matplotlib

# !!! 关键：在导入pyplot之前，设置matplotlib使用非GUI后端 !!!
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- 1. 配置区 ---

# 请将此路径修改为您的 vol_gt.npy 文件的实际路径
GT_VOLUME_PATH = "/media/data2/hezhipeng/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/init_0_chest_cone.npy"

# 请设置一个文件夹用于存放生成的预览图
OUTPUT_DIR = "/home/hezhipeng/Workbench/r2_gaussian-main/output/threshold1/"

# 定义您想要测试的阈值列表。
# 格式: ('类型', 值1, [值2])
# 'gt': 大于值1 (value > value1)
# 'lt': 小于值1 (value < value1)
# 'range': 在值1和值2之间 (value1 < value < value2)
THRESHOLDS_TO_TEST = [
    # --- 测试高密度区域 (可能对应骨骼) ---
    ('gt', 0.5),
    ('gt', 0.6),
    ('gt', 0.7),
    ('gt', 0.8),
    ('gt', 0.9),

    # --- 测试中等密度区域 (可能对应软组织/病灶) ---
    ('range', 0.3, 0.5),
    ('range', 0.4, 0.6),
    ('range', 0.5, 0.7),

    # --- 测试低密度区域 (可能对应脂肪/坏死) ---
    ('range', 0.1, 0.3),

    # --- 测试非常低的密度区域 (可能对应空气/肺) ---
    ('lt', 0.2),
    ('lt', 0.1),
]


# --- 2. 核心功能区 ---

def apply_threshold(image, thresh_config):
    """根据配置对图像应用阈值，返回一个布尔掩码"""
    thresh_type = thresh_config[0]
    if thresh_type == 'gt':
        return image > thresh_config[1]
    elif thresh_type == 'lt':
        return image < thresh_config[1]
    elif thresh_type == 'range':
        return (image > thresh_config[1]) & (image < thresh_config[2])
    return np.zeros_like(image, dtype=bool)


def create_preview_image(slice_data, mask, title, output_path):
    """创建并保存一张预览图"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. 绘制底层的灰度CT图
    ax.imshow(slice_data, cmap='gray')

    # 2. 将为True的掩码区域用半透明红色覆盖
    # 创建一个RGBA图像用于覆盖，只在mask为True的地方有颜色和不透明度
    overlay = np.zeros((*slice_data.shape, 4))  # (H, W, 4)
    overlay[mask] = [1, 0, 0, 0.5]  # R=1, G=0, B=0, Alpha=0.5
    ax.imshow(overlay)

    ax.set_title(title, fontsize=10)
    ax.axis('off')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # 关键：关闭图像以释放内存


# --- 3. 主程序 ---

if __name__ == "__main__":
    # 检查输入文件是否存在
    if not os.path.exists(GT_VOLUME_PATH):
        print(f"错误: 找不到真值体数据 '{GT_VOLUME_PATH}'")
        exit()

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"预览图将保存到: {OUTPUT_DIR}")

    # 加载体数据
    print("正在加载真值体数据...")
    volume = np.load(GT_VOLUME_PATH)
    print(f"加载成功! 体数据维度: {volume.shape}")

    # 定义要查看的三个轴向和切片位置
    # (轴, 轴名称, 尺寸)
    axes_to_process = [
        (0, 'coronal', volume.shape[0]),  # 冠状位 (从前到后)
        (1, 'sagittal', volume.shape[1]),  # 矢状位 (从左到右)
        (2, 'axial', volume.shape[2]),  # 轴状位 (从上到下)
    ]

    total_images = 0

    # 使用tqdm创建进度条
    with tqdm(total=len(axes_to_process) * 3 * len(THRESHOLDS_TO_TEST), desc="生成预览图") as pbar:
        for axis_idx, axis_name, axis_size in axes_to_process:
            # 为每个轴选择3个切片：25%, 50%, 75%
            slice_indices = [axis_size // 4, axis_size // 2, axis_size * 3 // 4]

            for slice_idx in slice_indices:
                # 根据轴向提取2D切片
                if axis_idx == 0:
                    current_slice = volume[slice_idx, :, :]
                elif axis_idx == 1:
                    current_slice = volume[:, slice_idx, :]
                else:  # axis_idx == 2
                    current_slice = volume[:, :, slice_idx]

                # Matplotlib imshow希望(H, W)，所以如果需要可以转置
                # 在这里，我们假设切片已经是正确的方向

                for thresh_config in THRESHOLDS_TO_TEST:
                    # 生成掩码
                    mask = apply_threshold(current_slice, thresh_config)

                    # 构建文件名和标题
                    thresh_str = f"{thresh_config[0]}_{thresh_config[1]}"
                    if thresh_config[0] == 'range':
                        thresh_str += f"_to_{thresh_config[2]}"

                    filename = f"{axis_name}_slice_{slice_idx}_{thresh_str}.png"
                    title = f"View: {axis_name.capitalize()}, Slice: {slice_idx}\nThreshold: {thresh_str.replace('_', ' ')}"
                    output_path = os.path.join(OUTPUT_DIR, filename)

                    # 创建并保存图像
                    create_preview_image(current_slice, mask, title, output_path)
                    total_images += 1
                    pbar.update(1)

    print(f"\n处理完成! 总共生成了 {total_images} 张预览图。")
    print(f"请下载 '{OUTPUT_DIR}' 文件夹并查看其中的图片。")