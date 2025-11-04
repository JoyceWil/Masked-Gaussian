import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm


class StructureImportanceMap2D:
    """
    计算静态结构重要性图 (S_map_2D) 的独立模块。

    该模块接收一个 2D 真实训练图像（如 CT 投影），并生成一个 2D 的重要性图。
    此版本专为预处理设计，仅依赖于输入图像本身。
    """

    def __init__(self, alpha: float = 0.5, device: str = 'cpu'):
        """
        初始化 2D S_map 计算器。

        Args:
            alpha (float): 梯度通道 G(x) 和窗宽窗位通道 W(x) 之间的融合权重。
            device (str): 计算设备 ('cpu' or 'cuda').
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha 必须在 [0, 1] 范围内。")
        self.alpha = alpha
        self.device = device
        print(f"S_map 2D preprocessor initialized with alpha={alpha} on device='{device}'")
        self._sobel_kernels = None

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量归一化到 [0, 1] 范围。"""
        min_val = tensor.min()
        max_val = tensor.max()
        # 增加一个小的 epsilon 防止除以零
        return (tensor - min_val) / (max_val - min_val + 1e-8)

    def _get_sobel_kernels(self):
        """创建并缓存 2D Sobel 算子核。"""
        if self._sobel_kernels is None:
            sobel_y_2d = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
            sobel_x_2d = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

            # 转换为 PyTorch 张量并增加 batch 和 channel 维度
            # 目标形状: (out_channels, in_channels, H, W) -> (1, 1, 3, 3)
            sobel_kernel_y = torch.from_numpy(sobel_y_2d).unsqueeze(0).unsqueeze(0).to(self.device)
            sobel_kernel_x = torch.from_numpy(sobel_x_2d).unsqueeze(0).unsqueeze(0).to(self.device)

            self._sobel_kernels = (sobel_kernel_y, sobel_kernel_x)

        return self._sobel_kernels

    def _calculate_gradient_channel(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        计算梯度通道 G(x)。使用 2D Sobel 算子。
        """
        if image_tensor.dim() != 4:
            raise ValueError("输入 image_tensor 必须是 4D 张量 (B, C, H, W)。")

        sobel_kernel_y, sobel_kernel_x = self._get_sobel_kernels()

        padding = 1  # for 3x3 kernel
        grad_y = F.conv2d(image_tensor, sobel_kernel_y, padding=padding)
        grad_x = F.conv2d(image_tensor, sobel_kernel_x, padding=padding)

        gradient_magnitude = torch.sqrt(grad_y ** 2 + grad_x ** 2)
        return self._normalize_tensor(gradient_magnitude)

    def _calculate_windowing_channel(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        计算窗宽窗位通道 W_max(x)。
        对于值范围不固定的投影数据，此方法会根据数据本身动态生成窗。
        """
        min_val, max_val = image_tensor.min(), image_tensor.max()
        data_range = max_val - min_val

        # 如果图像是全黑或全白的，直接返回0
        if data_range < 1e-6:
            return torch.zeros_like(image_tensor)

        # 动态生成三个有代表性的窗
        window_settings = [
            (data_range, min_val + data_range / 2),  # 覆盖整个范围的宽窗
            (data_range / 2, min_val + data_range / 4),  # 关注低值区的窄窗
            (data_range / 2, max_val - data_range / 4),  # 关注高值区的窄窗
        ]

        windowed_maps = []
        for width, level in window_settings:
            lower_bound = level - width / 2
            upper_bound = level + width / 2

            windowed_map = (image_tensor - lower_bound) / (width + 1e-8)
            windowed_map = torch.clamp(windowed_map, 0.0, 1.0)
            windowed_maps.append(windowed_map)

        # 取所有窗结果中的最大值，以聚合不同关注区域的重要性
        stacked_maps = torch.stack(windowed_maps, dim=0)
        max_windowed_map, _ = torch.max(stacked_maps, dim=0)

        return max_windowed_map

    def compute(self, image_2d: torch.Tensor) -> torch.Tensor:
        """
        执行完整的 2D S_map 计算。

        Args:
            image_2d (torch.Tensor): 原始 2D 图像，形状为 (H, W)。

        Returns:
            torch.Tensor: 最终计算出的 2D S_map，形状为 (H, W)。
        """
        if image_2d.dim() != 2:
            raise ValueError("输入图像必须是 2D 张量 (H, W)。")

        # 增加 batch 和 channel 维度以进行卷积操作
        image_4d = image_2d.unsqueeze(0).unsqueeze(0).to(self.device)

        # 1. 计算梯度通道 G(x)
        gradient_channel = self._calculate_gradient_channel(image_4d)

        # 2. 计算窗宽窗位通道 W(x)
        windowing_channel = self._calculate_windowing_channel(image_4d)

        # 3. 融合通道
        s_map = self.alpha * gradient_channel + (1 - self.alpha) * windowing_channel

        # 再次归一化确保最终结果在 [0, 1] 范围内
        s_map_normalized = self._normalize_tensor(s_map)

        # 移除 batch 和 channel 维度并移回 CPU
        s_map_2d = s_map_normalized.squeeze(0).squeeze(0).cpu()

        return s_map_2d


def preprocess_dataset(dataset_root: str, alpha: float):
    """
    遍历数据集目录，为 proj_train 中的每个 .npy 文件生成 S_map。
    """
    print("--- Starting S_map Preprocessing ---")

    # 1. 定义和验证路径
    input_dir = os.path.join(dataset_root, 'proj_train')
    output_dir_base = os.path.join(dataset_root, 'proj_train_s_map')
    output_dir_npy = os.path.join(output_dir_base, 'npy_data')
    output_dir_png = os.path.join(output_dir_base, 'png_preview')

    if not os.path.isdir(input_dir):
        print(f"错误：输入目录 '{input_dir}' 不存在。请检查您的数据集根目录。")
        return

    print(f"Input directory:  '{input_dir}'")
    print(f"Output for .npy:  '{output_dir_npy}'")
    print(f"Output for .png:  '{output_dir_png}'")

    # 2. 创建输出目录
    os.makedirs(output_dir_npy, exist_ok=True)
    os.makedirs(output_dir_png, exist_ok=True)

    # 3. 初始化 S_map 计算器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    s_map_calculator = StructureImportanceMap2D(alpha=alpha, device=device)

    # 4. 查找并处理所有 .npy 文件
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    if not npy_files:
        print(f"警告：在 '{input_dir}' 中没有找到任何 .npy 文件。")
        return

    print(f"Found {len(npy_files)} .npy files to process.")

    for filename in tqdm(npy_files, desc="Processing Images"):
        try:
            # 加载数据
            input_path = os.path.join(input_dir, filename)
            image_np = np.load(input_path)

            if image_np.ndim != 2:
                print(f"\nSkipping {filename}: Expected a 2D array, but got shape {image_np.shape}")
                continue

            image_tensor = torch.from_numpy(image_np.astype(np.float32))

            # 计算 S_map
            s_map_tensor = s_map_calculator.compute(image_tensor)
            s_map_np = s_map_tensor.numpy()

            # 保存为 .npy 文件 (高精度，用于训练)
            npy_output_path = os.path.join(output_dir_npy, filename)
            np.save(npy_output_path, s_map_np)

            # 保存为 .png 文件 (用于可视化预览)
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_output_path = os.path.join(output_dir_png, png_filename)
            plt.imsave(png_output_path, s_map_np, cmap='hot')

        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    print("--- Preprocessing Complete ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Standalone script to preprocess a dataset and generate 2D Structure Importance Maps (S_map_2D).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help="Path to the root directory of your dataset.\n"
             "The script will look for a subdirectory named 'proj_train' inside this path."
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help="Weight for combining gradient and windowing channels. (default: 0.5)\n"
             "alpha * gradient + (1 - alpha) * windowing"
    )
    args = parser.parse_args()

    preprocess_dataset(dataset_root=args.dataset_root, alpha=args.alpha)