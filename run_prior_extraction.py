import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os # 确保导入os模块

# --- Step 0: 加载数据并进行基本预处理 ---
# ==============================================================================
print("Step 0: Loading FDK reconstruction volume...")
# 请将这里的路径替换为您 V_fdk.npy 文件的实际路径
fdk_volume_path = 'data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/V_fdk.npy'
V_fdk = np.load(fdk_volume_path)

print(f"Loaded volume shape: {V_fdk.shape}")
# 确保数据类型为float32，这在后续处理中能获得更好的性能和兼容性
V_fdk = V_fdk.astype(np.float32)

# --- Step 1: 生成强度通道 I(x) ---
# ==============================================================================
# I(x) 本质上是归一化后的 V_fdk。归一化到 [0, 1] 区间是良好实践，
# 便于后续与其他通道结合，也便于神经网络处理。
print("Step 1: Generating Intensity Channel I(x)...")

min_val = np.min(V_fdk)
max_val = np.max(V_fdk)
I_channel = (V_fdk - min_val) / (max_val - min_val)

print(f"Intensity channel created. Shape: {I_channel.shape}, Range: [{I_channel.min()}, {I_channel.max()}]")

# --- Step 2: 生成梯度通道 G(x) ---
# ==============================================================================
# 我们使用高斯梯度幅值来计算3D梯度。相比简单的Sobel算子，
# 高斯梯度对噪声更鲁棒。sigma参数控制了高斯平滑的程度。
print("Step 2: Generating Gradient Channel G(x)...")

# sigma=1 是一个比较通用的起始值，可以根据效果调整
# ndimage.gaussian_gradient_magnitude 会在所有三个维度（Z, Y, X）上计算梯度
G_channel_raw = ndimage.gaussian_gradient_magnitude(V_fdk, sigma=1)

# 同样，将梯度通道也归一化到 [0, 1]
G_channel = (G_channel_raw - np.min(G_channel_raw)) / (np.max(G_channel_raw) - np.min(G_channel_raw))

print(f"Gradient channel created. Shape: {G_channel.shape}, Range: [{G_channel.min()}, {G_channel.max()}]")

# --- Step 3: 生成窗函数通道 W(x) ---
# ==============================================================================
# 这是最关键的一步，模拟CT中的窗宽(WW)和窗位(WL)技术，以突出不同组织。
# 注意：您的V_fdk值范围不是标准的Hounsfield Units (HU)，所以我们不能用
# 标准的HU窗位/窗宽。我们将根据归一化后的I_channel的[0, 1]范围来定义窗口。
print("Step 3: Generating Windowing Channel W(x)...")


def apply_window(volume, level, width):
    """
    在[0, 1]范围的体积上应用窗函数。
    - volume: 输入的3D体积，值在[0, 1]之间。
    - level: 窗位 (中心)，在[0, 1]之间。
    - width: 窗宽，在[0, 1]之间。
    """
    lower_bound = level - width / 2
    upper_bound = level + width / 2

    windowed_vol = np.clip(volume, lower_bound, upper_bound)

    # 将窗口内的值重新映射到[0, 1]
    # 添加一个小的epsilon防止除以零
    windowed_vol = (windowed_vol - lower_bound) / (width + 1e-8)
    return windowed_vol


# 根据I_channel ([0, 1]范围) 定义三个窗口
# 1. 软组织窗: 关注中间的密度范围
soft_tissue_window = apply_window(I_channel, level=0.5, width=0.4)

# 2. 肺窗: 关注较低的密度范围 (肺部在CT中是暗的)
lung_window = apply_window(I_channel, level=0.25, width=0.3)

# 3. 骨窗: 关注较高的密度范围 (骨骼和高密度组织是亮的)
bone_window = apply_window(I_channel, level=0.75, width=0.3)

# 融合三个窗口：取每个体素在三个窗口中的最大值。
# 这能确保无论是骨骼、肺部还是软组织的特征都能被高亮显示。
print("Fusing multiple windows...")
W_channel = np.maximum(np.maximum(soft_tissue_window, lung_window), bone_window)

print(f"Windowing channel created. Shape: {W_channel.shape}, Range: [{W_channel.min()}, {W_channel.max()}]")

# --- Step 4: 组合通道并保存 ---
# ==============================================================================
print("Step 4: Combining channels into a multi-channel volume...")

# 我们将三个通道堆叠成一个4D数组。
# 格式为 (channels, depth, height, width)，这是深度学习中常见的格式。
P_vol_channel_first = np.stack([I_channel, G_channel, W_channel], axis=0)

# ==============================================================================
# --- 这是关键的修复步骤 ---
# 将数组从 (C, D, H, W) 格式转置为 (D, H, W, C) 格式
# `np.moveaxis(array, source_axis, destination_axis)`
# 我们将源轴0（通道轴）移动到目标轴-1（最后一个轴）
print(f"Original shape (C, D, H, W): {P_vol_channel_first.shape}")
P_vol = np.moveaxis(P_vol_channel_first, 0, -1)
print(f"Transposed to correct shape (D, H, W, C): {P_vol.shape}")
# ==============================================================================

# 保存这个结构先验体积
output_path = 'data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/P_vol.npy'
# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.save(output_path, P_vol)

print(f"Structural prior volume P_vol created with shape: {P_vol.shape}")
print(f"Saved to: {output_path}")

# --- Step 5: 可视化验证 ---
# ==============================================================================
# 从3D体积的中心抽取一个2D切片进行可视化，以验证每一步的效果。
print("Step 5: Visualizing a central slice for verification...")

# 注意：现在P_vol的形状是(D, H, W, C)，所以I, G, W通道在最后一个维度
slice_idx = V_fdk.shape[0] // 2  # 取最中间的Z轴切片

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(f'3D Structural Prior Generation - Verification (Slice {slice_idx})', fontsize=16)

# 1. 原始FDK切片
im1 = axes[0].imshow(V_fdk[slice_idx, :, :], cmap='gray')
axes[0].set_title('Original FDK Slice')
fig.colorbar(im1, ax=axes[0])

# 2. 强度通道 I(x) - 从P_vol中提取
im2 = axes[1].imshow(P_vol[slice_idx, :, :, 0], cmap='gray')
axes[1].set_title('Intensity Channel I(x)')
fig.colorbar(im2, ax=axes[1])

# 3. 梯度通道 G(x) - 从P_vol中提取
im3 = axes[2].imshow(P_vol[slice_idx, :, :, 1], cmap='hot')
axes[2].set_title('Gradient Channel G(x)')
fig.colorbar(im3, ax=axes[2])

# 4. 窗函数通道 W(x) - 从P_vol中提取
im4 = axes[3].imshow(P_vol[slice_idx, :, :, 2], cmap='hot')
axes[3].set_title('Windowing Channel W(x)')
fig.colorbar(im4, ax=axes[3])

for ax in axes:
    ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()