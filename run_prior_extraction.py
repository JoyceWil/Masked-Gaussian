import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os

print("--- 3-Channel Prior Extraction (Clean G-Channel Version) ---")

# --- Step 0: 定义路径 ---
# 使用您之前成功运行的路径
base_dir = 'data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone'
# <<< MODIFIED: 变量名修正，使其与文件名匹配，更清晰
sart_volume_path = os.path.join(base_dir, 'V_sart_5iter.npy')
importance_volume_path = os.path.join(base_dir, 'V_importance.npy')
output_path = os.path.join(base_dir, 'P_vol.npy')
# <<< MODIFIED: 新增保存图像的路径
output_image_path = os.path.join(base_dir, 'P_vol_slice_verification.png')

# 检查输入文件是否存在
if not os.path.exists(sart_volume_path) or not os.path.exists(importance_volume_path):
    print(f"Error: 找不到输入文件。")
    # <<< MODIFIED: 打印正确的变量名
    print(f"V_sart_30iter.npy 存在吗? {os.path.exists(sart_volume_path)}")
    print(f"V_importance.npy 存在吗? {os.path.exists(importance_volume_path)}")
    print("请先运行 'run_start.py' (或类似的脚本) 生成初始重建卷。")
    exit()

# --- Step 1: 加载数据 ---
# <<< MODIFIED: 打印更清晰的日志
print(f"Step 1: Loading V_sart (for I/W) from '{sart_volume_path}' and V_importance (for G) from '{importance_volume_path}'...")
V_sart = np.load(sart_volume_path).astype(np.float32)
V_importance = np.load(importance_volume_path).astype(np.float32)

if V_sart.shape != V_importance.shape:
    print(f"Error: 体积形状不匹配! {V_sart.shape} vs {V_importance.shape}")
    exit()
print(f"Loaded volumes shape: {V_sart.shape}")
# <<< MODIFIED: 增加原始数据范围的打印，用于诊断
print(f"  - V_sart original range: [{np.min(V_sart)}, {np.max(V_sart)}]")
print(f"  - V_importance original range: [{np.min(V_importance)}, {np.max(V_importance)}]")


# --- Step 2: 生成强度通道 I(x) ---
print("\nStep 2: Generating Intensity Channel I(x)...")
min_val = np.min(V_sart)
max_val = np.max(V_sart)
I_channel = (V_sart - min_val) / (max_val - min_val + 1e-8)
# <<< MODIFIED: 打印归一化后的范围，确认操作正确
print(f"  - Intensity channel I(x) created. Range: [{I_channel.min():.4f}, {I_channel.max():.4f}]")

# --- Step 3: 生成梯度通道 G(x) [核心修改] ---
print("\nStep 3: Generating *Clean* Gradient Channel G(x) from V_importance...")
# G_channel 不再是 V_sart 的 3D 梯度，而是 V_importance 的归一化
min_val_g = np.min(V_importance)
max_val_g = np.max(V_importance)
G_channel = (V_importance - min_val_g) / (max_val_g - min_val_g + 1e-8)
print(f"  - Initial G_channel from V_importance. Range: [{G_channel.min():.4f}, {G_channel.max():.4f}]")

# (可选) 对反投影的结果进行一次轻微平滑，以去除高频伪影
print("  - Applying Gaussian filter (sigma=0.5) to G_channel...")
G_channel = ndimage.gaussian_filter(G_channel, sigma=0.5)
# 再次归一化
G_channel = (G_channel - np.min(G_channel)) / (np.max(G_channel) - np.min(G_channel) + 1e-8)
print(f"  - Final *Clean* Gradient channel G(x) created. Range: [{G_channel.min():.4f}, {G_channel.max():.4f}]")

# --- Step 4: 生成窗函数通道 W(x) ---
print("\nStep 4: Generating Windowing Channel W(x) from I_channel...")

def apply_window(volume, level, width):
    lower_bound = level - width / 2
    upper_bound = level + width / 2
    windowed_vol = np.clip(volume, lower_bound, upper_bound)
    # 归一化到 [0, 1]
    windowed_vol = (windowed_vol - lower_bound) / (width + 1e-8)
    return windowed_vol

# 使用基于 I_channel [0, 1] 范围的窗位窗宽
soft_tissue_window = apply_window(I_channel, level=0.5, width=0.4)
lung_window = apply_window(I_channel, level=0.25, width=0.3)
bone_window = apply_window(I_channel, level=0.75, width=0.3)

W_channel = np.maximum(np.maximum(soft_tissue_window, lung_window), bone_window)
# <<< MODIFIED: 打印归一化后的范围
print(f"  - Windowing channel W(x) created. Range: [{W_channel.min():.4f}, {W_channel.max():.4f}]")

# --- Step 5: 组合通道 (3通道) 并保存 ---
print("\nStep 5: Combining channels into a 3-channel volume (D, H, W, C)...")
# 我们只保留 I, G, W 三个通道。
# 形状: (D, H, W, 3)
P_vol = np.stack([I_channel, G_channel, W_channel], axis=-1).astype(np.float32)

np.save(output_path, P_vol)
print(f"Structural prior volume P_vol (3-channel) created with shape: {P_vol.shape}")
print(f"Saved to: {output_path}")

# --- Step 6: 可视化验证 (保存到文件) ---
# <<< MODIFIED: 整个可视化部分被修改为保存文件而不是显示
print("\nStep 6: Generating and saving a verification image...")
slice_idx = V_sart.shape[0] // 2

fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # 稍微调整尺寸以容纳 colorbar
fig.suptitle(f'3D Structural Prior (Clean G-Channel) - Verification (Slice {slice_idx})', fontsize=16)

# Channel 0: Intensity
im1 = axes[0].imshow(P_vol[slice_idx, :, :, 0], cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Channel 0: Intensity I(x)')
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# Channel 1: Clean Gradient
im2 = axes[1].imshow(P_vol[slice_idx, :, :, 1], cmap='hot', vmin=0, vmax=1)
axes[1].set_title('Channel 1: *Clean* Gradient G(x)')
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# Channel 2: Windowing
im3 = axes[2].imshow(P_vol[slice_idx, :, :, 2], cmap='hot', vmin=0, vmax=1)
axes[2].set_title('Channel 2: Windowing W(x)')
fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

for ax in axes:
    ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 将图像保存到文件，而不是显示它
plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
plt.close(fig) # 关闭图像，释放内存

print(f"Verification image saved to: {output_image_path}")
print("\n--- Script Finished Successfully ---")