import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


class StructureImportanceMap2D:
    """
    计算结构重要性图 (S_map) 的独立模块【二维版本】。

    该模块接收一个 2D 图像（如 CT 投影或单个切片），并生成一个 2D 的重要性图。
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, device: str = 'cpu'):
        """
        初始化 2D S_map 计算器。

        Args:
            alpha (float): 梯度通道 G(x) 和其他通道之间的融合权重。
            beta (float): 窗宽窗位通道 W(x) 和密度通道 D(x) 之间的融合权重。
            device (str): 计算设备 ('cpu' or 'cuda').
        """
        self.alpha = alpha
        self.beta = beta
        self.device = device
        print(f"S_map 2D module initialized with alpha={alpha}, beta={beta} on device='{device}'")
        self._sobel_kernels = None

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量归一化到 [0, 1] 范围。"""
        min_val = tensor.min()
        max_val = tensor.max()
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
            raise ValueError("Input image_tensor must be a 4D tensor (B, C, H, W).")

        sobel_kernel_y, sobel_kernel_x = self._get_sobel_kernels()

        padding = 1  # for 3x3 kernel
        grad_y = F.conv2d(image_tensor, sobel_kernel_y, padding=padding)
        grad_x = F.conv2d(image_tensor, sobel_kernel_x, padding=padding)

        gradient_magnitude = torch.sqrt(grad_y ** 2 + grad_x ** 2)
        return self._normalize_tensor(gradient_magnitude)

    def _calculate_windowing_channel(self, image_tensor: torch.Tensor,
                                     window_settings: list[tuple[float, float]]) -> torch.Tensor:
        """
        计算窗宽窗位通道 W_max(x)。
        注意：对于非 HU 值的投影数据，这只是一个形式上的变换。
        """
        if not window_settings:
            return torch.zeros_like(image_tensor)

        windowed_maps = []
        for width, level in window_settings:
            lower_bound = level - width / 2
            upper_bound = level + width / 2

            windowed_map = (image_tensor - lower_bound) / (width + 1e-8)
            windowed_map = torch.clamp(windowed_map, 0.0, 1.0)
            windowed_maps.append(windowed_map)

        stacked_maps = torch.stack(windowed_maps, dim=0)
        max_windowed_map = torch.max(stacked_maps, dim=0)[0]

        return max_windowed_map

    def compute(self,
                image_2d: torch.Tensor,
                density_map_2d: torch.Tensor,
                window_settings: list[tuple[float, float]]) -> tuple[torch.Tensor, dict]:
        """
        执行完整的 2D S_map 计算。

        Args:
            image_2d (torch.Tensor): 原始 2D 图像，形状为 (H, W)。
            density_map_2d (torch.Tensor): 2D 密度图，形状为 (H, W)。
            window_settings (list[tuple[float, float]]): 窗宽窗位设置列表。

        Returns:
            torch.Tensor: 最终计算出的 2D S_map，形状为 (H, W)。
            dict: 包含所有中间通道的字典。
        """
        if image_2d.dim() != 2 or density_map_2d.dim() != 2:
            raise ValueError("Input images must be 2D tensors (H, W).")

        image_4d = image_2d.unsqueeze(0).unsqueeze(0).to(self.device)
        density_map_4d = density_map_2d.unsqueeze(0).unsqueeze(0).to(self.device)

        gradient_channel = self._calculate_gradient_channel(image_4d)

        # 对于投影数据，其值范围很小，需要调整窗设置以使其有意义
        # 这里我们根据输入数据的范围动态生成一些窗
        min_val, max_val = image_2d.min(), image_2d.max()
        data_range = max_val - min_val
        dynamic_window_settings = [
            (data_range, min_val + data_range / 2),  # 覆盖整个范围的窗
            (data_range / 2, min_val + data_range / 4),  # 覆盖低值区的窗
            (data_range / 2, max_val - data_range / 4),  # 覆盖高值区的窗
        ]
        print(f"Dynamically generated window settings for projection data: {dynamic_window_settings}")

        windowing_channel = self._calculate_windowing_channel(image_4d, dynamic_window_settings)
        density_channel_normalized = self._normalize_tensor(density_map_4d)

        s_map = self.alpha * gradient_channel + \
                (1 - self.alpha) * (self.beta * windowing_channel +
                                    (1 - self.beta) * density_channel_normalized)

        s_map_2d = s_map.squeeze(0).squeeze(0)

        intermediate_channels = {
            "gradient": gradient_channel.squeeze(0).squeeze(0).cpu().detach(),
            "windowing": windowing_channel.squeeze(0).squeeze(0).cpu().detach(),
            "density_normalized": density_channel_normalized.squeeze(0).squeeze(0).cpu().detach(),
        }

        return s_map_2d.cpu().detach(), intermediate_channels


def create_mock_density_image(h=512, w=512):
    """创建一个模拟的 2D 高斯密度图。"""
    print(f"Creating mock 2D Gaussian density map of size ({h}, {w})...")
    y, x = np.ogrid[-1:1:h * 1j, -1:1:w * 1j]
    density = np.exp(-((x ** 2 + y ** 2) / 0.5 ** 2))
    return torch.from_numpy(density.astype(np.float32))


def visualize_results_2d(original_image, original_density, final_s_map, channels):
    """可视化 2D 输入、中间通道和最终结果。"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(f"S_map 2D Generation - Verification", fontsize=16)

    titles = ["Original Image", "Mock Gaussian Density", "Gradient Channel G(x)", "Windowing Channel W(x)",
              "Final S_map"]
    images = [
        original_image,
        original_density,
        channels['gradient'],
        channels['windowing'],
        final_s_map
    ]
    cmaps = ['gray', 'hot', 'hot', 'hot', 'hot']

    for i, (ax, title, img, cmap) in enumerate(zip(axes, titles, images, cmaps)):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        if i == 4:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="S_map Core Module Prototype (2D Version)")
    parser.add_argument('--npy_path', type=str, required=True, help="Path to the input 2D .npy image file.")
    parser.add_argument('--alpha', type=float, default=0.6, help="Weight for the gradient channel.")
    parser.add_argument('--beta', type=float, default=0.7, help="Weight for the windowing channel.")
    args = parser.parse_args()

    # --- 1. 加载数据 ---
    if not os.path.exists(args.npy_path):
        raise FileNotFoundError(f"The specified file was not found: {args.npy_path}")

    print(f"Loading 2D image from {args.npy_path}...")
    image_np = np.load(args.npy_path)
    if image_np.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got shape {image_np.shape}")

    image_2d = torch.from_numpy(image_np.astype(np.float32))
    print(f"Image loaded with shape: {image_2d.shape}")
    H, W = image_2d.shape
    density_map = create_mock_density_image(h=H, w=W)

    # --- 2. 初始化 S_map 计算器 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    s_map_calculator = StructureImportanceMap2D(alpha=args.alpha, beta=args.beta, device=device)

    # --- 3. 计算 S_map ---
    # 窗设置将由 compute 方法根据数据范围动态生成
    final_s_map, intermediate_channels = s_map_calculator.compute(
        image_2d=image_2d,
        density_map_2d=density_map,
        window_settings=[]  # 传入空列表，触发动态生成
    )

    # --- 4. 可视化结果 ---
    visualize_results_2d(
        original_image=image_2d,
        original_density=density_map,
        final_s_map=final_s_map,
        channels=intermediate_channels
    )