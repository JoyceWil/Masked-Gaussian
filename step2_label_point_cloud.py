import numpy as np
import json
import os
from tqdm import tqdm

# --- 1. 配置与常量 (根据您的信息更新) ---

# --- 输入文件 ---
# FDK重建的点云文件
FDK_CLOUD_FILE = '/media/data2/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone/init_0_chest_cone.npy'
# 元数据JSON文件
META_DATA_FILE = '/media/data2/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone/meta_data.json'
# 包含所有文件的根目录，用于拼接相对路径
BASE_DIR = '/media/data2/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone/'

# --- 输出文件 ---
# 保存带有语义标签的点云的文件名
OUTPUT_LABELED_CLOUD_FILE = os.path.join(BASE_DIR, 'labeled_fdk_cloud.npy')

# --- 参数与定义 (与上一步保持一致) ---
HU_RANGES = {
    'air': (-1024, -950), 'fat': (-100, -20), 'soft_tissue': (20, 80),
    'calcium': (150, 400), 'bone': (400, 3000)
}
CLASS_TO_ID = {name: i + 1 for i, name in enumerate(HU_RANGES.keys())}
ID_TO_CLASS = {i: name for name, i in CLASS_TO_ID.items()}
ID_TO_CLASS[0] = 'background'

CALIBRATION_POINTS = {'air': {'hu_value': -1000.0}, 'bone': {'hu_value': 1000.0}}


# --- 2. 核心功能函数 ---

def get_projection_matrices_from_meta(meta: dict) -> np.ndarray:
    """
    根据meta_data.json中的CT几何参数计算所有训练视角的投影矩阵。
    这是本脚本的关键新增部分。
    """
    print("正在根据JSON元数据计算相机投影矩阵...")
    scanner = meta['scanner']
    DSD = scanner['DSD']
    DSO = scanner['DSO']
    n_detector = np.array(scanner['nDetector'])
    s_detector = np.array(scanner['sDetector'])

    # 计算每个探测器像素的物理尺寸
    pixel_size = s_detector / n_detector

    # 构建内参矩阵 K
    # fx = fy = DSD / pixel_size
    # cx = cy = n_detector / 2
    K = np.array([
        [DSD / pixel_size[0], 0, n_detector[0] / 2],
        [0, DSD / pixel_size[1], n_detector[1] / 2],
        [0, 0, 1]
    ])

    matrices = []
    # 我们只关心训练集
    for view_info in meta['proj_train']:
        angle = view_info['angle']

        # 构建外参矩阵 [R|t]
        # 1. 旋转矩阵 R: 相机绕Z轴旋转，然后调整坐标系使其看向-X方向
        rot_z = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        # CT相机标准朝向：x->down, y->right, z->from patient to source
        # 我们需要从世界坐标系（Z轴向上）转换到相机坐标系
        # 世界到相机的旋转
        R = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]) @ rot_z.T

        # 2. 平移向量 t: -R * C, 其中C是相机中心在世界坐标系的位置
        cam_center_world = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
        t = -R @ cam_center_world

        # 3. 组合成外参矩阵
        Rt = np.hstack((R, t.reshape(-1, 1)))

        # 4. 最终的投影矩阵 P = K @ [R|t]
        P = K @ Rt
        matrices.append(P)

    print(f"成功计算了 {len(matrices)} 个投影矩阵。")
    return np.array(matrices)


def calculate_hu_mapping_from_stats(data: np.ndarray) -> tuple[float, float]:
    """根据数据的统计极值计算HU映射参数 a 和 b。"""
    i_min, i_max = np.min(data), np.max(data)
    if np.isclose(i_min, i_max): return 1.0, 0.0
    a = (CALIBRATION_POINTS['bone']['hu_value'] - CALIBRATION_POINTS['air']['hu_value']) / (i_max - i_min)
    b = CALIBRATION_POINTS['air']['hu_value'] - a * i_min
    return a, b


def assign_label_to_point(hu_value: float) -> int:
    """根据HU值返回对应的类别ID。"""
    for name, (hu_min, hu_max) in HU_RANGES.items():
        if hu_min <= hu_value <= hu_max:
            return CLASS_TO_ID[name]
    return 0


# --- 3. 主函数 ---

def label_3d_point_cloud():
    """主函数：加载所有数据，为每个3D点赋予语义标签。"""
    # --- 加载输入数据 ---
    print("--- 开始加载所有必需数据 ---")
    try:
        # 加载元数据JSON
        with open(META_DATA_FILE, 'r') as f:
            meta = json.load(f)

        # 加载FDK点云 (N, 3 or 4 or 7)
        point_cloud = np.load(FDK_CLOUD_FILE)
        points_3d = point_cloud[:, :3]  # 我们只关心XYZ坐标进行投影

        # 从元数据中获取训练投影图的相对路径
        proj_train_info = meta['proj_train']
        # 拼接成绝对路径
        proj_files = [os.path.join(BASE_DIR, info['file_path']) for info in proj_train_info]
        projections = [np.load(f) for f in tqdm(proj_files, desc="加载投影图")]
        img_h, img_w = projections[0].shape

        # 动态计算相机矩阵
        cameras = get_projection_matrices_from_meta(meta)
        num_views = len(cameras)

    except (FileNotFoundError, KeyError) as e:
        print(f"\n错误: 加载数据失败 - {e}")
        print("请确保所有文件路径和JSON结构都正确无误。")
        return

    print(f"\n数据加载完成: {len(points_3d)}个3D点, {num_views}个训练视角。")

    # --- 预计算所有视角的HU映射参数 ---
    print("正在预计算所有视角的HU映射参数...")
    hu_mappers = [calculate_hu_mapping_from_stats(proj) for proj in tqdm(projections, desc="计算HU映射")]

    # --- 主循环：为每个3D点分配标签 ---
    labeled_points = []
    for i in tqdm(range(len(points_3d)), desc="为3D点分配标签"):
        p_3d_homo = np.append(points_3d[i], 1)
        observed_hus = []

        for j in range(num_views):
            P, (a, b) = cameras[j], hu_mappers[j]
            p_2d_homo = P @ p_3d_homo

            w = p_2d_homo[2]
            if w > 1e-6:  # 确保点在相机前方
                u, v = int(p_2d_homo[0] / w), int(p_2d_homo[1] / w)
                if 0 <= u < img_w and 0 <= v < img_h:
                    observed_hus.append(a * projections[j][v, u] + b)

        label_id = assign_label_to_point(np.median(observed_hus)) if observed_hus else 0
        labeled_points.append(label_id)

    # --- 保存结果 ---
    print("\n--- 处理完成，正在保存结果 ---")
    final_labeled_cloud = np.hstack((point_cloud, np.array(labeled_points).reshape(-1, 1)))
    np.save(OUTPUT_LABELED_CLOUD_FILE, final_labeled_cloud)

    print(f"带标签的点云已保存到: '{OUTPUT_LABELED_CLOUD_FILE}'")
    print(f"原始点云形状: {point_cloud.shape}")
    print(f"新点云形状: {final_labeled_cloud.shape}")
    print("\n最终点云各类别统计:")
    for i in range(len(CLASS_TO_ID) + 2):  # Check up to one more ID just in case
        if i in ID_TO_CLASS:
            count = np.sum(final_labeled_cloud[:, -1] == i)
            if count > 0:
                print(f"  - ID {i} ({ID_TO_CLASS[i]}): {count} 个点")


# --- 4. 运行主程序 ---
if __name__ == '__main__':
    label_3d_point_cloud()