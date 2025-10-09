# r2_gaussian/utils/hu_utils.py
import torch
import numpy as np


def convert_mu_to_hu_torch(mu_image: torch.Tensor) -> torch.Tensor:
    """
    【PyTorch版】将原始的mu值（衰减系数）图像通过线性映射转换为HU（亨氏单位）图像。
    """
    HU_AIR = -1000.0
    HU_HIGH_DENSITY = 1000.0

    # torch.nan_to_num 替代 np.nan_to_num
    mu_image = torch.nan_to_num(mu_image)

    mu_min = torch.min(mu_image)
    mu_max = torch.max(mu_image)

    if mu_max - mu_min < 1e-6:
        return torch.full_like(mu_image, HU_AIR)
    else:
        a = (HU_HIGH_DENSITY - HU_AIR) / (mu_max - mu_min)
        b = HU_AIR - a * mu_min
        hu_image = a * mu_image + b
        return hu_image


def apply_windowing_torch(hu_image: torch.Tensor, window_level: int, window_width: int) -> torch.Tensor:
    """
    【PyTorch版】根据给定的窗位(WL)和窗宽(WW)将HU值图像转换为归一化的[0, 1]软掩码。
    这个函数现在完全可微。
    """
    min_hu = window_level - (window_width / 2.0)
    max_hu = window_level + (window_width / 2.0)

    # torch.clip 替代 np.clip
    soft_mask = torch.clip(hu_image, min_hu, max_hu)

    if window_width == 0:
        return torch.zeros_like(soft_mask)

    soft_mask = (soft_mask - min_hu) / window_width
    return soft_mask


# --- 验证部分 ---
def validate_hu_utils():
    print("\n*** 正在验证 PyTorch版本的 HU 工具函数 ***")
    # 创建一个模拟的 mu 图像 (B, C, H, W)
    mock_mu_image_np = np.random.rand(2, 1, 64, 64) * 0.5  # 模拟 mu 值
    mock_mu_image_torch = torch.from_numpy(mock_mu_image_np).float().cuda()

    # 1. 验证 convert_mu_to_hu_torch
    hu_image = convert_mu_to_hu_torch(mock_mu_image_torch)
    print(f"convert_mu_to_hu_torch 输出维度: {hu_image.shape}")
    print(f"HU 值范围: [{hu_image.min().item():.2f}, {hu_image.max().item():.2f}]")
    assert hu_image.min() >= -1000.0 and hu_image.max() <= 1000.0, "HU值转换验证失败"
    print("convert_mu_to_hu_torch 验证成功!")

    # 2. 验证 apply_windowing_torch
    # 使用典型的软组织窗
    wl, ww = 40, 400
    soft_mask = apply_windowing_torch(hu_image, wl, ww)
    print(f"\napply_windowing_torch 输出维度: {soft_mask.shape}")
    print(f"软掩码值范围: [{soft_mask.min().item():.2f}, {soft_mask.max().item():.2f}]")
    assert soft_mask.min() >= 0.0 and soft_mask.max() <= 1.0, "窗函数验证失败"
    print("apply_windowing_torch 验证成功!")

    # 3. 验证梯度传播
    hu_image.requires_grad = True
    soft_mask = apply_windowing_torch(hu_image, wl, ww)
    try:
        soft_mask.sum().backward()
        assert hu_image.grad is not None, "梯度传播失败！"
        print("梯度传播验证成功!")
    except Exception as e:
        print(f"梯度传播验证失败: {e}")


if __name__ == '__main__':
    # 如果独立运行此文件，执行验证
    if torch.cuda.is_available():
        validate_hu_utils()
    else:
        print("未检测到CUDA设备，跳过验证。")