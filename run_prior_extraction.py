import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os

print("--- 3-Channel Prior Extraction (Clean G-Channel Version) ---")

# --- Step 0: 定义路径 ---
base_dir = 'data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone'
fdk_volume_path = os.path.join(base_dir, 'V_fdk.npy')
importance_volume_path = os.path.join(base_dir, 'V_importance.npy')
output_path = os.path.join(base_dir, 'P_vol.npy')

if not os.path.exists(fdk_volume_path) or not os.path.exists(importance_volume_path):
    print(f"Error: 找不到输入文件。")
    print(f"V_fdk.npy 存在吗? {os.path.exists(fdk_volume_path)}")
    print(f"V_importance.npy 存在吗? {os.path.exists(importance_volume_path)}")
    print("请先运行 'generate_fdk_and_importance.py'")
    exit()

# --- Step 1: 加载数据 ---
print("Step 1: Loading V_fdk (for I/W) and V_importance (for G)...")
V_fdk = np.load(fdk_volume_path).astype(np.float32)
V_importance = np.load(importance_volume_path).astype(np.float32)

if V_fdk.shape != V_importance.shape:
    print(f"Error: 体积形状不匹配! {V_fdk.shape} vs {V_importance.shape}")
    exit()
print(f"Loaded volumes shape: {V_fdk.shape}")

# --- Step 2: 生成强度通道 I(x) ---
print("Step 2: Generating Intensity Channel I(x)...")
min_val = np.min(V_fdk)
max_val = np.max(V_fdk)
I_channel = (V_fdk - min_val) / (max_val - min_val + 1e-8)

# --- Step 3: 生成梯度通道 G(x) [核心修改] ---
print("Step 3: Generating *Clean* Gradient Channel G(x) from V_importance...")
# G_channel 不再是 V_fdk 的 3D 梯度，而是 V_importance 的归一化
min_val_g = np.min(V_importance)
max_val_g = np.max(V_importance)
G_channel = (V_importance - min_val_g) / (max_val_g - min_val_g + 1e-8)

# (可选) 对反投影的结果进行一次轻微平滑，以去除高频伪影
G_channel = ndimage.gaussian_filter(G_channel, sigma=0.5)
# 再次归一化
G_channel = (G_channel - np.min(G_channel)) / (np.max(G_channel) - np.min(G_channel) + 1e-8)
print(f"Clean Gradient channel created. Range: [{G_channel.min()}, {G_channel.max()}]")

# --- Step 4: 生成窗函数通道 W(x) ---
print("Step 4: Generating Windowing Channel W(x) from I_channel...")

def apply_window(volume, level, width):
    lower_bound = level - width / 2
    upper_bound = level + width / 2
    windowed_vol = np.clip(volume, lower_bound, upper_bound)
    windowed_vol = (windowed_vol - lower_bound) / (width + 1e-8)
    return windowed_vol

soft_tissue_window = apply_window(I_channel, level=0.5, width=0.4)
lung_window = apply_window(I_channel, level=0.25, width=0.3)
bone_window = apply_window(I_channel, level=0.75, width=0.3)

W_channel = np.maximum(np.maximum(soft_tissue_window, lung_window), bone_window)

# --- Step 5: 组合通道 (3通道) 并保存 ---
print("Step 5: Combining channels into a 3-channel volume (D, H, W, C)...")
# 我们只保留 I, G, W 三个通道。
# 形状: (D, H, W, 3)
P_vol = np.stack([I_channel, G_channel, W_channel], axis=-1)

np.save(output_path, P_vol)
print(f"Structural prior volume P_vol (3-channel) created with shape: {P_vol.shape}")
print(f"Saved to: {output_path}")

# --- Step 6: 可视化验证 ---
print("Step 6: Visualizing a central slice for verification...")
slice_idx = V_fdk.shape[0] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'3D Structural Prior (Clean G-Channel) - Verification (Slice {slice_idx})', fontsize=16)

im1 = axes[0].imshow(P_vol[slice_idx, :, :, 0], cmap='gray')
axes[0].set_title('Channel 0: Intensity I(x)')
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(P_vol[slice_idx, :, :, 1], cmap='hot')
axes[1].set_title('Channel 1: *Clean* Gradient G(x)')
fig.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(P_vol[slice_idx, :, :, 2], cmap='hot')
axes[2].set_title('Channel 2: Windowing W(x)')
fig.colorbar(im3, ax=axes[2])

for ax in axes:
    ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()