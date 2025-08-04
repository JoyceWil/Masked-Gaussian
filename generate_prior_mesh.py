import numpy as np
import os
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go


def save_obj(vertices, faces, output_path):
    """将顶点和面数据保存为 .obj 文件。"""
    with open(output_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def visualize_mesh_as_images(vertices, faces, output_prefix):
    """使用 Plotly 的离屏渲染功能，为网格生成预览图。"""
    print("\n--- 开始生成预览图 (使用 Plotly 后台) ---")
    try:
        # 定义相机视角
        camera_views = {
            'front': dict(eye=dict(x=0, y=0, z=2.5)),  # 正前方
            'top': dict(eye=dict(x=0, y=2.5, z=0)),  # 正上方
            'side': dict(eye=dict(x=2.5, y=0, z=0))  # 正侧方
        }

        for view_name, camera_view in camera_views.items():
            # 创建 Mesh3d 图形对象
            fig = go.Figure(data=[go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='white',
                opacity=1.0,
                lighting=dict(ambient=0.4, diffuse=1, specular=0.5, roughness=0.5),
                lightposition=dict(x=100, y=200, z=2000)
            )])

            # 更新布局和相机
            fig.update_layout(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    bgcolor='black'
                ),
                scene_camera=camera_view
            )

            # 保存为图片
            output_image_path = f"{output_prefix}_{view_name}.png"
            fig.write_image(output_image_path, width=800, height=800)
            print(f"已保存预览图: {output_image_path}")

        print("--- 预览图生成完毕 ---")
    except Exception as e:
        print(f"\n错误：生成 Plotly 预览图时发生错误: {e}")
        traceback.print_exc()


def point_cloud_to_volume(points, weights, grid_resolution, smoothing_sigma):
    """
    将点云体素化为三维体数据。

    参数:
    - points (np.array): Nx3 的点云坐标。
    - weights (np.array): Nx1 的点权重（例如密度或不透明度）。
    - grid_resolution (tuple): 体数据网格的分辨率 (D, H, W)。
    - smoothing_sigma (float): 高斯平滑的强度。
    """
    print("开始体素化点云...")
    # 1. 确定点云边界并增加一点填充
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    padding = (max_bound - min_bound) * 0.05  # 5% padding
    min_bound -= padding
    max_bound += padding

    # 2. 创建一个空的3D网格
    grid = np.zeros(grid_resolution, dtype=np.float32)

    # 3. 计算每个维度上的体素大小
    voxel_size = (max_bound - min_bound) / (np.array(grid_resolution) - 1)

    # 4. 将点坐标映射到网格索引
    # (points - min_bound) 将坐标原点移到网格的起始点
    # / voxel_size 将世界坐标转换为网格索引坐标
    grid_indices = np.floor((points - min_bound) / voxel_size).astype(int)

    # 5. 过滤掉超出边界的点
    valid_mask = np.all((grid_indices >= 0) & (grid_indices < grid_resolution), axis=1)
    grid_indices = grid_indices[valid_mask]
    valid_weights = weights[valid_mask]

    # 6. 将点的权重累加到对应的体素中
    # 使用 np.add.at 实现高效、无冲突的累加
    np.add.at(grid, tuple(grid_indices.T), valid_weights)

    print(f"体素化完成。网格数据范围: [{grid.min():.4f}, {grid.max():.4f}]")

    # 7. 应用高斯平滑
    if smoothing_sigma > 0:
        print(f"正在应用高斯平滑 (sigma={smoothing_sigma})...")
        grid = gaussian_filter(grid, sigma=smoothing_sigma)
        print(f"平滑后网格数据范围: [{grid.min():.4f}, {grid.max():.4f}]")

    return grid, min_bound, voxel_size


if __name__ == '__main__':
    # --- 参数配置 ---
    INPUT_FILE = "/media/data2/hezhipeng/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/init_0_chest_cone.npy"
    OUTPUT_DIR = "/media/data2/hezhipeng/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/prior_generation"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 关键可调参数 ---
    # 1. 体素化网格的分辨率 (D, H, W)。值越高，细节越丰富，但内存和计算开销越大。
    #    可以先从较低的分辨率开始，例如 (128, 128, 128)。
    GRID_RESOLUTION = (256, 256, 256)

    # 2. 高斯平滑的强度。值越大，表面越平滑，但可能丢失细节。
    #    建议从 1.0 或 2.0 开始。
    SMOOTHING_SIGMA = 6.0

    # 3. Marching Cubes 的等值面阈值。现在它作用于平滑后的密度值。
    #    你需要根据打印出的 "平滑后网格数据范围" 来试验并选择一个合适的值。
    ISO_LEVEL = 0.00025

    # --- 执行流程 ---
    print(f"--- 开始处理点云文件: {os.path.basename(INPUT_FILE)} ---")

    # 1. 加载点云数据
    try:
        point_cloud_data = np.load(INPUT_FILE)
        if point_cloud_data.ndim != 2 or point_cloud_data.shape[1] != 4:
            raise ValueError(f"期望的数组形状是 (N, 4)，但得到 {point_cloud_data.shape}")

        points_xyz = point_cloud_data[:, :3]
        # 假设第4列是某种形式的权重/密度/不透明度
        point_weights = point_cloud_data[:, 3]
        print(f"点云加载成功。{len(points_xyz)} 个点。")

    except Exception as e:
        print(f"错误：加载或解析点云文件失败: {e}")
        exit()

    # 2. 将点云转换为体数据
    volume, volume_origin, voxel_size = point_cloud_to_volume(
        points_xyz, point_weights, GRID_RESOLUTION, SMOOTHING_SIGMA
    )

    # 3. 在生成的体数据上运行 Marching Cubes
    try:
        import skimage.measure

        print(f"正在运行 Marching Cubes 算法... (iso-level: {ISO_LEVEL})")
        verts, faces, _, _ = skimage.measure.marching_cubes(volume, level=ISO_LEVEL)
    except Exception as e:
        print(f"Marching Cubes 算法执行出错: {e}")
        exit()

    if len(verts) == 0:
        print("警告：未生成任何顶点。这通常意味着 'ISO_LEVEL' 设置得太高或太低。")
        print(f"请尝试一个在平滑后的数据范围 [{volume.min():.4f}, {volume.max():.4f}] 内的值。")
        exit()

    # 4. 将顶点坐标从网格空间转换回世界空间
    verts = verts * voxel_size + volume_origin
    print(f"网格生成成功: {len(verts)} 个顶点, {len(faces)} 个面。")

    # 5. 保存和可视化
    base_name = f"prior_mesh_res_{GRID_RESOLUTION[0]}_smooth_{SMOOTHING_SIGMA}_iso_{ISO_LEVEL}"
    output_obj_file = os.path.join(OUTPUT_DIR, f"{base_name}.obj")

    save_obj(verts, faces, output_obj_file)
    print(f"网格已成功保存到 {output_obj_file}")

    visualize_mesh_as_images(verts, faces, os.path.join(OUTPUT_DIR, base_name))

    print("--- 处理完成 ---")