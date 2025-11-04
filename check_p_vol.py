import numpy as np

# --- 配置 ---
P_VOL_PATH = "data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/P_vol.npy"
GRADIENT_THRESHOLD = 0.01 # 使用一个较低的阈值来捕捉任何非零梯度

# --- 加载数据 ---
try:
    P_vol = np.load(P_VOL_PATH)
    print(f"成功加载 '{P_VOL_PATH}', 形状为: {P_vol.shape}")
except FileNotFoundError:
    print(f"错误: 文件 '{P_VOL_PATH}' 未找到。")
    exit()

# 提取梯度通道 G_vol
# P_vol 的形状是 (D, H, W, 3)，其中最后一个维度是 [I, G, W]
G_vol = P_vol[..., 1]

# --- 查找高梯度值的深度切片 ---
depth_dim, height_dim, width_dim = G_vol.shape
active_depth_indices = set()

# 遍历每个深度切片
for d in range(depth_dim):
    depth_slice = G_vol[d, :, :]
    # 如果这个切片上的最大梯度值大于阈值，就记录下它的索引
    if np.max(depth_slice) > GRADIENT_THRESHOLD:
        active_depth_indices.add(d)

# --- 打印结果 ---
print("\n--- 梯度分布分析 ---")
if not active_depth_indices:
    print("在所有深度切片中都没有找到显著的梯度值。")
else:
    print(f"在总共 {depth_dim} 个深度切片中，仅在以下索引处发现了显著的梯度值:")
    # 排序后打印，更清晰
    sorted_indices = sorted(list(active_depth_indices))
    print(sorted_indices)
    print(f"\n结论：这解释了为什么点云只生成在 {len(sorted_indices)} 个离散的平面上。")