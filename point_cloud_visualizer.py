# point_cloud_visualizer.py (最终重构版)

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from plyfile import PlyData
from PIL import Image
from tqdm import tqdm
import argparse
import time


def generate_comprehensive_ct_visualization(model_path: str, latest_iter: int, tb_writer=None):
    """
    从 train.py 迁移过来的、功能更全面的CT切片可视化函数。
    它会生成三个方向的独立切片图、组合图，并上传到TensorBoard。

    Args:
        model_path (str): 模型根目录。
        latest_iter (int): 训练的最终迭代次数。
        tb_writer: TensorBoard writer 实例，如果提供，则会将图像上传。
    """
    print("\n--- [综合CT切片可视化模块] ---")

    # 1. 查找体积数据
    vol_pred_path = find_volume_file(model_path, "vol_pred.npy")
    vol_gt_path = find_volume_file(model_path, "vol_gt.npy")

    if not vol_pred_path:
        print("   - 错误: 找不到 vol_pred.npy，无法生成CT切片可视化。")
        return

    print(f"   - 加载体积数据: {os.path.basename(vol_pred_path)}")
    vol_pred = np.load(vol_pred_path)
    vol_gt = np.load(vol_gt_path) if vol_gt_path else None

    if vol_gt is not None:
        print(f"   - 体积数据形状 - 预测: {vol_pred.shape}, 真实: {vol_gt.shape}")
    else:
        print(f"   - 体积数据形状 - 预测: {vol_pred.shape} (未找到真实GT体积)")
        # 如果没有GT，创建一个空的替代，以简化后续代码
        vol_gt = np.zeros_like(vol_pred)

    # 2. 创建保存目录
    ct_viz_dir = os.path.join(model_path, "ct_viz_comprehensive")
    os.makedirs(ct_viz_dir, exist_ok=True)
    axial_dir = os.path.join(ct_viz_dir, "axial")
    coronal_dir = os.path.join(ct_viz_dir, "coronal")
    sagittal_dir = os.path.join(ct_viz_dir, "sagittal")
    for d in [axial_dir, coronal_dir, sagittal_dir]:
        os.makedirs(d, exist_ok=True)

    # 3. 归一化数据用于可视化
    vol_pred_norm = (vol_pred - vol_pred.min()) / (vol_pred.max() - vol_pred.min())
    vol_gt_norm = (vol_gt - vol_gt.min()) / (vol_gt.max() - vol_gt.min())

    # 4. 定义切片处理的内部函数
    def process_slices(vol_gt, vol_pred, axis_name, axis_dim, step, output_dir):
        combined_slices = []
        for i in range(5):  # 固定生成5个切片
            slice_idx = min((i + 1) * step - 1, axis_dim - 1)

            if axis_name == "axial":
                gt_slice, pred_slice = vol_gt[..., slice_idx], vol_pred[..., slice_idx]
            elif axis_name == "coronal":
                gt_slice, pred_slice = vol_gt[:, slice_idx, :], vol_pred[:, slice_idx, :]
            else:  # sagittal
                gt_slice, pred_slice = vol_gt[slice_idx, :, :], vol_pred[slice_idx, :, :]

            # 保存单独的对比图
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1);
            plt.imshow(gt_slice, cmap='gray');
            plt.title(f"GT {axis_name.capitalize()} {slice_idx}");
            plt.axis('off')
            plt.subplot(1, 2, 2);
            plt.imshow(pred_slice, cmap='gray');
            plt.title(f"Pred {axis_name.capitalize()} {slice_idx}");
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{axis_name}_slice_{slice_idx}.png"), dpi=200, bbox_inches='tight')
            plt.close()

            combined_slice = np.vstack([gt_slice, pred_slice])
            combined_slices.append(combined_slice)

        all_slices_combined = np.hstack(combined_slices)
        plt.figure(figsize=(15, 6))
        plt.imshow(all_slices_combined, cmap='gray')
        plt.title(f"CT {axis_name.capitalize()} Slices (Top: GT, Bottom: Prediction)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"all_{axis_name}_slices.png"), dpi=200, bbox_inches='tight')
        plt.close()
        return all_slices_combined

    # 5. 生成三个方向的切片
    depth_z, depth_y, depth_x = vol_gt_norm.shape[2], vol_gt_norm.shape[1], vol_gt_norm.shape[0]
    step_z, step_y, step_x = max(1, depth_z // 5), max(1, depth_y // 5), max(1, depth_x // 5)

    print("   - 生成轴状位(Axial)切片...")
    axial_slices = process_slices(vol_gt_norm, vol_pred_norm, "axial", depth_z, step_z, axial_dir)
    print("   - 生成冠状位(Coronal)切片...")
    coronal_slices = process_slices(vol_gt_norm, vol_pred_norm, "coronal", depth_y, step_y, coronal_dir)
    print("   - 生成矢状位(Sagittal)切片...")
    sagittal_slices = process_slices(vol_gt_norm, vol_pred_norm, "sagittal", depth_x, step_x, sagittal_dir)

    # 6. 上传到 TensorBoard
    if tb_writer:
        print("   - 正在上传可视化结果到 TensorBoard...")
        for name, slices in [("axial", axial_slices), ("coronal", coronal_slices), ("sagittal", sagittal_slices)]:
            slices_tensor = torch.from_numpy(slices)[None, ..., None]
            slices_rgb = torch.cat([slices_tensor, slices_tensor, slices_tensor], dim=3)
            tb_writer.add_image(f"ct_viz_final/{name}_slices", slices_rgb[0], global_step=latest_iter,
                                dataformats="HWC")

    print(f"--- [综合CT切片可视化完成] 结果保存在: {ct_viz_dir} ---")


def visualize_point_cloud(model_path):
    # ... (这部分代码保持不变) ...
    print("\n--- [点云GIF动画生成模块] ---")
    # ... (从 "创建可视化结果保存路径" 到 "plt.close(fig)" 的所有代码) ...
    # 只是需要删除对 generate_ct_slices 的调用
    # ...
    # 创建可视化结果保存路径
    vis_path = os.path.join(model_path, "point_cloud_visualization")
    os.makedirs(vis_path, exist_ok=True)

    # 查找最新的点云文件
    point_cloud_dir = os.path.join(model_path, "point_cloud")
    if not os.path.exists(point_cloud_dir):
        print(f"   - 找不到点云目录: {point_cloud_dir}")
        return

    # 找到所有包含point_cloud.ply的子文件夹
    ply_folders = glob.glob(os.path.join(point_cloud_dir, "iteration_*"))
    if not ply_folders:
        print(f"   - 找不到点云文件: {point_cloud_dir}")
        return

    # 按迭代次数排序，取最大的
    latest_folder = sorted(ply_folders, key=lambda x: int(x.split("_")[-1]))[-1]
    ply_path = os.path.join(latest_folder, "point_cloud.ply")

    if not os.path.exists(ply_path):
        print(f"   - 找不到PLY文件: {ply_path}")
        return

    print(f"   - 正在处理点云文件: {ply_path}")

    # 获取场景名称
    scene_name = os.path.basename(model_path.rstrip("/"))
    if not scene_name:
        scene_name = "scene"

    # 创建场景特定的可视化路径
    scene_vis_path = os.path.join(vis_path, f"{scene_name}")
    os.makedirs(scene_vis_path, exist_ok=True)

    # 加载PLY文件
    try:
        plydata = PlyData.read(ply_path)
    except Exception as e:
        print(f"   - 读取PLY文件出错: {e}")
        return

    # 提取xyz坐标
    xyz = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ), axis=1)

    # 尝试提取密度信息
    try:
        densities = np.asarray(plydata.elements[0]["density"])
        density_threshold = np.percentile(densities, 10)
        valid_indices = densities > density_threshold
        xyz = xyz[valid_indices]
        print(f"   - 根据密度过滤后剩余点数: {xyz.shape[0]} / {len(valid_indices)}")
    except (ValueError, KeyError):
        print("   - 找不到密度信息，使用所有点")

    # 尝试提取颜色信息
    try:
        colors = np.stack((
            np.asarray(plydata.elements[0]["red"]),
            np.asarray(plydata.elements[0]["green"]),
            np.asarray(plydata.elements[0]["blue"])
        ), axis=1) / 255.0
        if 'valid_indices' in locals():
            colors = colors[valid_indices]
    except (ValueError, KeyError):
        colors = 'gray'

    # 创建图形对象
    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    center = np.mean(xyz, axis=0)
    max_range = np.max(np.ptp(xyz, axis=0)) * 0.55
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    point_size = max(0.1, min(1.0, 1000.0 / np.sqrt(xyz.shape[0])))

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=point_size, alpha=0.5, c=colors, edgecolors='none')
    ax.axis('off')
    ax.grid(False)

    # 生成旋转GIF
    proj_num = 36
    angle_interval = 360 / proj_num
    image_files = []
    print("   - 生成旋转视图序列...")
    for i in tqdm(range(proj_num), leave=False):
        angle = i * angle_interval
        ax.view_init(elev=20, azim=angle)
        frame_path = os.path.join(scene_vis_path, f'frame_{i:03d}.png')
        plt.savefig(frame_path, dpi=200, bbox_inches='tight', pad_inches=0)
        image_files.append(frame_path)

    gif_filename = os.path.join(scene_vis_path, f'{scene_name}_rotation.gif')
    print(f"   - 正在创建GIF动画: {gif_filename}")
    gif_frames = [Image.open(filename) for filename in image_files]
    gif_frames[0].save(gif_filename, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)

    for file in image_files:
        os.remove(file)

    plt.close(fig)
    print(f"--- [点云GIF动画完成] 结果保存在: {scene_vis_path} ---")


def find_volume_file(base_path, filename):
    for root, dirs, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def on_training_finish(model_path, latest_iter, tb_writer):
    """
    训练完成后执行的总回调函数。
    它会依次调用点云GIF生成和综合CT切片可视化。
    """

    # 1. 智能等待文件生成
    print("   - 正在确认最终体积数据是否已保存...")
    vol_pred_path = None
    for _ in range(15):  # 等待最多15秒
        vol_pred_path = find_volume_file(model_path, "vol_pred.npy")
        if vol_pred_path:
            print("   - 确认成功！")
            break
        time.sleep(1)
    if not vol_pred_path:
        print("   - 警告: 未找到 vol_pred.npy，部分可视化可能失败。")

    # 2. 执行点云GIF可视化
    visualize_point_cloud(model_path)

    # 3. 执行综合CT切片可视化
    generate_comprehensive_ct_visualization(model_path, latest_iter, tb_writer)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练后可视化工具')
    parser.add_argument('--model_path', type=str, default='', help='模型输出路径，为空则使用最新的输出')
    args = parser.parse_args()

    if not args.model_path:
        output_dir = "./output/"
        if os.path.exists(output_dir):
            folders = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if
                       os.path.isdir(os.path.join(output_dir, d))]
            if folders:
                latest_folder = max(folders, key=os.path.getmtime)
                args.model_path = latest_folder
                print(f"   - 未指定路径，自动使用最新模型: {args.model_path}")
            else:
                print(f"错误: 在 {output_dir} 中找不到任何模型文件夹。")
                exit(1)
        else:
            print(f"错误: 输出目录 {output_dir} 不存在。")
            exit(1)

    # 独立运行时，我们无法获取 tb_writer，所以传 None
    # 尝试从文件夹名称或内容中推断 latest_iter
    latest_iter = 30000  # 使用一个合理的默认值
    try:
        point_cloud_dir = os.path.join(args.model_path, "point_cloud")
        if os.path.exists(point_cloud_dir):
            iter_folders = glob.glob(os.path.join(point_cloud_dir, "iteration_*"))
            if iter_folders:
                latest_iter = max([int(f.split("_")[-1]) for f in iter_folders])
    except Exception:
        print(f"   - 无法自动推断最终迭代次数，将使用默认值 {latest_iter}。")

    on_training_finish(args.model_path, latest_iter=latest_iter, tb_writer=None)