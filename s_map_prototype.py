import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


class StructureImportanceMap:
    """
    计算结构重要性图 (S_map) 的独立模块。

    该模块接收一个 CT 体数据，并根据结构梯度、医学窗宽窗位先验以及
    当前的高斯密度分布，生成一个 3D 的重要性图。
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, device: str = 'cpu'):
        """
        初始化 S_map 计算器。

        Args:
            alpha (float): 梯度通道 G(x) 和其他通道之间的融合权重。
                           S = alpha * G + (1 - alpha) * (...).
            beta (float): 窗宽窗位通道 W(x) 和密度通道 D(x) 之间的融合权重。
                          (...) = beta * W + (1 - beta) * D.
            device (str): 计算设备 ('cpu' or 'cuda').
        """
        self.alpha = alpha
        self.beta = beta
        self.device = device
        print(f"S_map module initialized with alpha={alpha}, beta={beta} on device='{device}'")
        self._sobel_kernels = None

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量归一化到 [0, 1] 范围。"""
        min_val = tensor.min()
        max_val = tensor.max()
        # 加上一个很小的 epsilon 防止除以零
        return (tensor - min_val) / (max_val - min_val + 1e-8)

    def _get_sobel_kernels(self):
        """
        创建并缓存 3D Sobel 算子核。
        这是修正后的、更稳健的定义方式。
        """
        if self._sobel_kernels is None:
            # 定义 3x3x3 的 3D Sobel 核 (Numpy 数组)
            sobel_z_3d = np.array([
                [[-1, -2, -1], [-2, -3, -2], [-1, -2, -1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 1], [2, 3, 2], [1, 2, 1]],
            ], dtype=np.float32)

            sobel_y_3d = np.array([
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[-2, -3, -2], [0, 0, 0], [2, 3, 2]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            ], dtype=np.float32)

            sobel_x_3d = np.array([
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-2, 0, 2], [-3, 0, 3], [-2, 0, 2]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            ], dtype=np.float32)

            # 转换为 PyTorch 张量并增加 batch 和 channel 维度
            # 目标形状: (out_channels, in_channels, D, H, W) -> (1, 1, 3, 3, 3)
            sobel_kernel_z = torch.from_numpy(sobel_z_3d).unsqueeze(0).unsqueeze(0).to(self.device)
            sobel_kernel_y = torch.from_numpy(sobel_y_3d).unsqueeze(0).unsqueeze(0).to(self.device)
            sobel_kernel_x = torch.from_numpy(sobel_x_3d).unsqueeze(0).unsqueeze(0).to(self.device)

            self._sobel_kernels = (sobel_kernel_z, sobel_kernel_y, sobel_kernel_x)

        return self._sobel_kernels

    def _calculate_gradient_channel(self, ct_volume: torch.Tensor) -> torch.Tensor:
        """
        计算梯度通道 G(x)。
        使用 3D Sobel 算子计算体积数据的梯度幅值。
        """
        if ct_volume.dim() != 5:
            raise ValueError("Input ct_volume must be a 5D tensor (B, C, D, H, W).")

        sobel_kernel_z, sobel_kernel_y, sobel_kernel_x = self._get_sobel_kernels()

        # 使用 3D 卷积计算梯度
        padding = 1  # for 3x3x3 kernel
        grad_z = F.conv3d(ct_volume, sobel_kernel_z, padding=padding)
        grad_y = F.conv3d(ct_volume, sobel_kernel_y, padding=padding)
        grad_x = F.conv3d(ct_volume, sobel_kernel_x, padding=padding)

        # 计算梯度幅值
        gradient_magnitude = torch.sqrt(grad_z ** 2 + grad_y ** 2 + grad_x ** 2)

        # 归一化到 [0, 1]
        return self._normalize_tensor(gradient_magnitude)

    def _calculate_windowing_channel(self, ct_volume: torch.Tensor,
                                     window_settings: list[tuple[float, float]]) -> torch.Tensor:
        """
        计算窗宽窗位通道 W_max(x)。
        对每个窗应用变换，然后取最大值。
        """
        if not window_settings:
            print("Warning: No window settings provided. Windowing channel will be zero.")
            return torch.zeros_like(ct_volume)

        windowed_maps = []
        for width, level in window_settings:
            lower_bound = level - width / 2
            upper_bound = level + width / 2

            windowed_map = (ct_volume - lower_bound) / (width + 1e-8)
            windowed_map = torch.clamp(windowed_map, 0.0, 1.0)
            windowed_maps.append(windowed_map)

        stacked_maps = torch.stack(windowed_maps, dim=0)
        max_windowed_map = torch.max(stacked_maps, dim=0)[0]

        return max_windowed_map

    def compute(self,
                ct_volume: torch.Tensor,
                density_volume: torch.Tensor,
                window_settings: list[tuple[float, float]]) -> tuple[torch.Tensor, dict]:
        """
        执行完整的 S_map 计算。

        Args:
            ct_volume (torch.Tensor): 原始 CT 体数据，单位为亨氏单位 (HU)。
                                      形状为 (D, H, W)。
            density_volume (torch.Tensor): 模拟的或真实的高斯密度图。
                                           形状为 (D, H, W)。
            window_settings (list[tuple[float, float]]): 窗宽窗位设置列表。

        Returns:
            torch.Tensor: 最终计算出的 S_map，形状为 (D, H, W)。
            dict: 包含所有中间通道的字典，方便调试和可视化。
        """
        if ct_volume.dim() != 3 or density_volume.dim() != 3:
            raise ValueError("Input volumes must be 3D tensors (D, H, W).")

        ct_volume_5d = ct_volume.unsqueeze(0).unsqueeze(0).to(self.device)
        density_volume_5d = density_volume.unsqueeze(0).unsqueeze(0).to(self.device)

        gradient_channel = self._calculate_gradient_channel(ct_volume_5d)
        windowing_channel = self._calculate_windowing_channel(ct_volume_5d, window_settings)
        density_channel_normalized = self._normalize_tensor(density_volume_5d)

        s_map = self.alpha * gradient_channel + \
                (1 - self.alpha) * (self.beta * windowing_channel +
                                    (1 - self.beta) * density_channel_normalized)

        s_map_3d = s_map.squeeze(0).squeeze(0)

        intermediate_channels = {
            "gradient": gradient_channel.squeeze(0).squeeze(0).cpu().detach(),
            "windowing": windowing_channel.squeeze(0).squeeze(0).cpu().detach(),
            "density_normalized": density_channel_normalized.squeeze(0).squeeze(0).cpu().detach(),
        }

        return s_map_3d.cpu().detach(), intermediate_channels


def create_phantom_volume(d=128, h=128, w=128, background_hu=-1000, shell_hu=100, core_hu=400, lesion_hu=800):
    """创建一个简单的 3D 体模，模拟 CT 数据。"""
    print(f"Creating 3D phantom volume of size ({d}, {h}, {w})...")
    z, y, x = np.ogrid[-1:1:d * 1j, -1:1:h * 1j, -1:1:w * 1j]

    volume = np.full((d, h, w), background_hu, dtype=np.float32)

    mask_shell = x ** 2 + y ** 2 + z ** 2 < 0.7 ** 2
    volume[mask_shell] = shell_hu

    mask_core = x ** 2 + y ** 2 + z ** 2 < 0.4 ** 2
    volume[mask_core] = core_hu

    mask_lesion = (x - 0.5) ** 2 + (y - 0.5) ** 2 + z ** 2 < 0.1 ** 2
    volume[mask_lesion] = lesion_hu

    return torch.from_numpy(volume)


def create_mock_density_volume(d=128, h=128, w=128):
    """创建一个模拟的高斯密度图，尺寸与输入体积匹配。"""
    print(f"Creating mock Gaussian density volume of size ({d}, {h}, {w})...")
    z, y, x = np.ogrid[-1:1:d * 1j, -1:1:h * 1j, -1:1:w * 1j]

    density = np.exp(-((x ** 2 + y ** 2 + z ** 2) / 0.3 ** 2))
    density += 0.3 * np.exp(-(((x + 0.6) ** 2 + (y + 0.6) ** 2 + (z + 0.6) ** 2) / 0.05 ** 2))

    return torch.from_numpy(density.astype(np.float32))


def visualize_results(original_ct, original_density, final_s_map, channels, slice_idx):
    """可视化输入、中间通道和最终结果的指定切片。"""
    print(f"Visualizing slice {slice_idx}...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(f"S_map Generation - Verification (Slice {slice_idx})", fontsize=16)

    titles = ["Original CT", "Mock Gaussian Density", "Gradient Channel G(x)", "Windowing Channel W(x)", "Final S_map"]
    images = [
        original_ct[slice_idx, :, :],
        original_density[slice_idx, :, :],
        channels['gradient'][slice_idx, :, :],
        channels['windowing'][slice_idx, :, :],
        final_s_map[slice_idx, :, :]
    ]
    cmaps = ['gray', 'hot', 'hot', 'hot', 'hot']

    for i, (ax, title, img, cmap) in enumerate(zip(axes, titles, images, cmaps)):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        if i == 4:  # Add colorbar to the last plot
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="S_map Core Module Prototype")
    parser.add_argument('--npy_path', type=str, default=None, help="Path to the input .npy volume file.")
    parser.add_argument('--alpha', type=float, default=0.6, help="Weight for the gradient channel.")
    parser.add_argument('--beta', type=float, default=0.7, help="Weight for the windowing channel.")
    parser.add_argument('--slice', type=int, default=None,
                        help="Slice index to visualize. Defaults to the middle slice.")
    args = parser.parse_args()

    # --- 窗设置: (宽度, 水平) in HU ---
    WINDOW_SETTINGS = [
        (400, 50),  # Soft tissue window
        (1000, 300),  # Broader window covering all structures
        (200, 800)  # Window centered on a potential lesion
    ]

    # --- 1. 加载或创建数据 ---
    if args.npy_path:
        if not os.path.exists(args.npy_path):
            raise FileNotFoundError(f"The specified file was not found: {args.npy_path}")
        print(f"Loading volume from {args.npy_path}...")
        ct_volume_np = np.load(args.npy_path)
        ct_volume = torch.from_numpy(ct_volume_np.astype(np.float32))
        print(f"Volume loaded with shape: {ct_volume.shape}")
        D, H, W = ct_volume.shape
        density_map = create_mock_density_volume(d=D, h=H, w=W)
    else:
        print("No .npy file provided. Using generated phantom data.")
        D, H, W = 128, 128, 128
        ct_volume = create_phantom_volume(d=D, h=H, w=W)
        density_map = create_mock_density_volume(d=D, h=H, w=W)

    # --- 2. 初始化 S_map 计算器 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    s_map_calculator = StructureImportanceMap(alpha=args.alpha, beta=args.beta, device=device)

    # --- 3. 计算 S_map ---
    final_s_map, intermediate_channels = s_map_calculator.compute(
        ct_volume=ct_volume,
        density_volume=density_map,
        window_settings=WINDOW_SETTINGS
    )

    # --- 4. 可视化结果进行验证 ---
    if args.slice is None:
        slice_to_show = D // 2
    else:
        if 0 <= args.slice < D:
            slice_to_show = args.slice
        else:
            print(
                f"Warning: Slice index {args.slice} is out of bounds for volume depth {D}. Defaulting to middle slice {D // 2}.")
            slice_to_show = D // 2

    visualize_results(
        original_ct=ct_volume,
        original_density=density_map,
        final_s_map=final_s_map,
        channels=intermediate_channels,
        slice_idx=slice_to_show
    )