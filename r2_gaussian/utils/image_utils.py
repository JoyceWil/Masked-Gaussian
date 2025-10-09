import sys
import numpy as np
import torch

# 保持原始的导入路径
sys.path.append("./")
from r2_gaussian.utils.loss_utils import ssim

# --- LPIPS 相关代码 (极简版) ---
try:
    import lpips

    LPIPS_AVAILABLE = True
    # 全局变量，用于缓存LPIPS模型，避免重复加载
    _lpips_model_cache = {}  # 使用字典来缓存不同设备上的模型
except ImportError:
    LPIPS_AVAILABLE = False


@torch.no_grad()
def compute_lpips(img1, img2):
    """
    计算LPIPS。输入张量应为 [B, C, H, W]，数据范围在 [0, 1]。
    这个函数会处理模型加载和设备问题。
    """
    if not LPIPS_AVAILABLE:
        if '_lpips_warned' not in globals():
            print("警告: lpips包未安装。LPIPS指标将不会被计算。")
            globals()['_lpips_warned'] = True
        return torch.tensor(float('nan'))

    device = img1.device

    # 从缓存中获取或创建模型
    if device not in _lpips_model_cache:
        print(f"正在设备 {device} 上首次初始化LPIPS(AlexNet)模型...")
        _lpips_model_cache[device] = lpips.LPIPS(net='alex').to(device)

    lpips_model = _lpips_model_cache[device]

    # LPIPS模型期望输入范围在 [-1, 1]
    img1_scaled = img1 * 2 - 1
    img2_scaled = img2 * 2 - 1

    return lpips_model(img1_scaled, img2_scaled)


# --- LPIPS 相关代码结束 ---


# ==============================================================================
# == 以下所有函数均保留原始Baseline的逻辑，只在需要的地方添加LPIPS分支 ==
# ==============================================================================

def mse(img1, img2, mask=None):
    """MSE error (原始Baseline逻辑)"""
    n_channel = img1.shape[1]
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)

        mask = mask.flatten(1).repeat(1, n_channel)
        mask = torch.where(mask != 0, True, False)

        mse_val = torch.stack(
            [
                (((img1[i, mask[i]] - img2[i, mask[i]])) ** 2).mean(0, keepdim=True)
                for i in range(img1.shape[0])
            ],
            dim=0,
        )

    else:
        mse_val = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return mse_val


def rmse(img1, img2, mask=None):
    """RMSE error (原始Baseline逻辑)"""
    mse_out = mse(img1, img2, mask)
    return mse_out ** 0.5


@torch.no_grad()
def psnr(img1, img2, mask=None, pixel_max=1.0):
    """PSNR (原始Baseline逻辑)"""
    mse_out = mse(img1, img2, mask)
    psnr_out = 10 * torch.log10(pixel_max ** 2 / mse_out.float())
    if mask is not None:
        if torch.isinf(psnr_out).any():
            print(mse_out.mean(), psnr_out.mean())
            psnr_out = 10 * torch.log10(pixel_max ** 2 / mse_out.float())
            psnr_out = psnr_out[~torch.isinf(psnr_out)]

    return psnr_out


@torch.no_grad()
def metric_vol(img1, img2, metric="psnr", pixel_max=1.0):
    """
    体积指标计算 (保留原始Baseline逻辑, 新增LPIPS分支)
    img1: GT, img2: Prediction
    """
    assert metric in ["psnr", "ssim", "lpips"]
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.copy())
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.copy())

    device = img1.device

    if metric == "psnr":
        if pixel_max is None:
            pixel_max = img1.max()
        mse_out = torch.mean((img1 - img2) ** 2)
        psnr_out = 10 * torch.log10(pixel_max ** 2 / mse_out.float())
        return psnr_out.item(), None

    elif metric == "ssim":
        ssims = []
        for axis in [0, 1, 2]:
            results = []
            count = 0
            n_slice = img1.shape[axis]
            for i in range(n_slice):
                if axis == 0:
                    slice1, slice2 = img1[i, :, :], img2[i, :, :]
                elif axis == 1:
                    slice1, slice2 = img1[:, i, :], img2[:, i, :]
                else:  # axis == 2
                    slice1, slice2 = img1[:, :, i], img2[:, :, i]

                if slice1.max() > 0:
                    result = ssim(slice1.unsqueeze(0).unsqueeze(0), slice2.unsqueeze(0).unsqueeze(0))
                    count += 1
                else:
                    result = 0
                results.append(result)

            if count > 0:
                mean_results = torch.sum(torch.tensor(results)) / count
                ssims.append(mean_results.item())
            else:
                ssims.append(0.0)
        return float(np.mean(ssims)), ssims

    elif metric == "lpips":
        all_axis_scores = []
        for axis in [0, 1, 2]:
            if axis == 0:
                slices1, slices2 = img1.unsqueeze(1), img2.unsqueeze(1)
            elif axis == 1:
                slices1, slices2 = img1.permute(1, 0, 2).unsqueeze(1), img2.permute(1, 0, 2).unsqueeze(1)
            else:  # axis == 2
                slices1, slices2 = img1.permute(2, 0, 1).unsqueeze(1), img2.permute(2, 0, 1).unsqueeze(1)

            slices1_rgb = slices1.repeat(1, 3, 1, 1)
            slices2_rgb = slices2.repeat(1, 3, 1, 1)

            batch_size = 32
            axis_scores = []
            for i in range(0, slices1_rgb.shape[0], batch_size):
                batch1 = slices1_rgb[i:i + batch_size].to(device)
                batch2 = slices2_rgb[i:i + batch_size].to(device)
                scores = compute_lpips(batch1, batch2)
                axis_scores.append(scores)

            if axis_scores:
                all_axis_scores.append(torch.cat(axis_scores).mean().item())

        return np.nanmean(all_axis_scores) if all_axis_scores else float('nan'), all_axis_scores


@torch.no_grad()
def metric_proj(img1, img2, metric="psnr", axis=2, pixel_max=1.0):
    """
    投影指标计算 (保留原始Baseline逻辑, 新增LPIPS分支)
    img1: GT [x, y, z], img2: Prediction [x, y, z]
    """
    assert axis == 2, "metric_proj 目前只支持沿着z轴(axis=2)的投影"
    assert metric in ["psnr", "ssim", "lpips"]
    if isinstance(img1, np.ndarray): img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray): img2 = torch.from_numpy(img2)

    device = img1.device

    if metric == "psnr" or metric == "ssim":
        n_slice = img1.shape[axis]
        results = []
        count = 0
        for i in range(n_slice):
            slice1, slice2 = img1[:, :, i], img2[:, :, i]

            if slice1.max() > 0:
                slice1 = slice1 / slice1.max()
                slice2 = slice2 / slice2.max()
                slice1_4d = slice1.unsqueeze(0).unsqueeze(0)
                slice2_4d = slice2.unsqueeze(0).unsqueeze(0)
                if metric == "psnr":
                    result = psnr(slice1_4d, slice2_4d, pixel_max=pixel_max)
                else:  # ssim
                    result = ssim(slice1_4d, slice2_4d)
                count += 1
            else:
                result = 0
            results.append(result)

        if count > 0:
            mean_results = torch.sum(torch.tensor(results)) / count
            return mean_results.item(), [r.item() for r in results]
        else:
            return 0.0, []

    elif metric == "lpips":
        slices1 = img1.permute(2, 0, 1).unsqueeze(1)
        slices2 = img2.permute(2, 0, 1).unsqueeze(1)

        slices1_rgb = slices1.repeat(1, 3, 1, 1)
        slices2_rgb = slices2.repeat(1, 3, 1, 1)

        batch_size = 16
        all_scores = []
        for i in range(0, slices1_rgb.shape[0], batch_size):
            batch1 = slices1_rgb[i:i + batch_size].to(device)
            batch2 = slices2_rgb[i:i + batch_size].to(device)
            scores = compute_lpips(batch1, batch2)
            all_scores.append(scores)

        results_tensor = torch.cat(all_scores)
        mean_result = results_tensor.mean().item()
        return mean_result, results_tensor.cpu().tolist()