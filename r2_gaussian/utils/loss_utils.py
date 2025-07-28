#r2_gaussian/utils/loss_utils.py
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


def bce_loss(pred_mask, gt_mask):
    """
    计算预测的置信度图和真实的二值掩码之间的二元交叉熵损失。

    Args:
        pred_mask (torch.Tensor): 模型的预测结果，值在 [0, 1] 之间。
        gt_mask (torch.Tensor): 真实的二值掩码，值为 0 或 1。

    Returns:
        torch.Tensor: 计算出的损失值。
    """
    return F.binary_cross_entropy(pred_mask, gt_mask)

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


class FrequencyLoss(nn.Module):
    """
    计算频域L1损失。
    直接惩罚预测图像和真实图像在傅里叶幅度谱上的差异。
    这对于恢复高频细节（如纹理、边缘）非常有效。
    """

    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def forward(self, pred_img, gt_img):
        """
        Args:
            pred_img (torch.Tensor): 预测图像，形状为 (B, C, H, W)
            gt_img (torch.Tensor): 真实图像，形状为 (B, C, H, W)
        """
        # 1. 对最后两个维度（H, W）进行二维傅里叶变换
        pred_fft = torch.fft.fft2(pred_img, dim=(-2, -1))
        gt_fft = torch.fft.fft2(gt_img, dim=(-2, -1))

        # 2. 计算幅度谱
        pred_magnitude = torch.abs(pred_fft)
        gt_magnitude = torch.abs(gt_fft)

        # 3. 计算幅度谱之间的L1损失
        loss = F.l1_loss(pred_magnitude, gt_magnitude)

        return loss
