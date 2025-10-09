import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn


def tv_3d_loss(vol, reduction="sum"):

    dx = torch.abs(torch.diff(vol, dim=0))
    dy = torch.abs(torch.diff(vol, dim=1))
    dz = torch.abs(torch.diff(vol, dim=2))

    tv = torch.sum(dx) + torch.sum(dy) + torch.sum(dz)

    if reduction == "mean":
        total_elements = (
            (vol.shape[0] - 1) * vol.shape[1] * vol.shape[2]
            + vol.shape[0] * (vol.shape[1] - 1) * vol.shape[2]
            + vol.shape[0] * vol.shape[1] * (vol.shape[2] - 1)
        )
        tv = tv / total_elements
    return tv


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gradient_difference_loss(pred, target, alpha=1.0):
    """
    【修正版】计算梯度差异损失 (GDL)。
    此版本经过简化，专门为单通道图像（如CT）设计，避免了复杂且易错的分组卷积。
    """
    # 检查输入是否为单通道，如果不是则报错，避免后续出现问题
    if pred.shape[1] != 1 or target.shape[1] != 1:
        raise ValueError(
            f"Gradient Difference Loss (GDL) 期望输入为单通道图像, "
            f"但收到的预测图像通道为 {pred.shape[1]}, "
            f"目标图像通道为 {target.shape[1]}."
        )

    # 定义Sobel算子核。对于单通道输入，核的形状应为 [out_channels, in_channels, kH, kW]
    # 这里 out_channels=1, in_channels=1
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)

    # 直接对单通道图像进行标准2D卷积，无需 'groups' 参数
    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
    target_grad_x = F.conv2d(target, sobel_x, padding=1)
    target_grad_y = F.conv2d(target, sobel_y, padding=1)

    # 计算梯度差异
    grad_diff_x = torch.abs(pred_grad_x - target_grad_x)
    grad_diff_y = torch.abs(pred_grad_y - target_grad_y)

    # 根据alpha计算最终损失
    if alpha == 1.0:
        loss = torch.mean(grad_diff_x + grad_diff_y)
    else:
        loss = torch.mean(torch.pow(grad_diff_x, alpha) + torch.pow(grad_diff_y, alpha))

    return loss
