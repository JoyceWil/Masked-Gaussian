# r2_gaussian/utils/image_utils.py (最终修正版 - 严格对齐基线)

import torch
import numpy as np
from r2_gaussian.utils.loss_utils import ssim
from skimage.metrics import peak_signal_noise_ratio

# --- LPIPS 相关代码 (复刻Baseline的缓存和调用逻辑, 保持不变) ---
try:
    import lpips

    LPIPS_AVAILABLE = True
    _lpips_model_cache = {}
except ImportError:
    LPIPS_AVAILABLE = False


@torch.no_grad()
def _compute_lpips_baseline(img1, img2):
    if not LPIPS_AVAILABLE:
        if '_lpips_warned' not in globals():
            print(
                "[WARN] LPIPS library not found. Please install it with 'pip install lpips'. LPIPS metrics will be skipped.")
            globals()['_lpips_warned'] = True
        return torch.tensor(float('nan'), device=img1.device)

    device = img1.device
    if device not in _lpips_model_cache:
        print(f"   - Initializing LPIPS(AlexNet) model on device: {device}...")
        _lpips_model_cache[device] = lpips.LPIPS(net='alex').to(device)

    lpips_model = _lpips_model_cache[device]
    img1_scaled = img1 * 2 - 1
    img2_scaled = img2 * 2 - 1
    return lpips_model(img1_scaled, img2_scaled)


@torch.no_grad()
def _psnr_baseline(img1, img2, pixel_max=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(pixel_max ** 2 / mse)


# --- 指标计算函数 (metric_vol 逻辑与基线一致, 保持不变) ---

@torch.no_grad()
def metric_vol(vol1, vol2, metric_type):
    if not isinstance(vol1, torch.Tensor): vol1 = torch.from_numpy(vol1.copy()).float()
    if not isinstance(vol2, torch.Tensor): vol2 = torch.from_numpy(vol2.copy()).float()
    vol1, vol2 = vol1.cuda(), vol2.cuda()

    if metric_type == "psnr":
        pixel_max = 1.0
        psnr_val = _psnr_baseline(vol1, vol2, pixel_max=pixel_max)
        return psnr_val.item(), None

    elif metric_type == "ssim":
        ssims = []
        for axis in range(3):
            axis_scores = []
            n_slice = vol1.shape[axis]
            for i in range(n_slice):
                if axis == 0:
                    slice1, slice2 = vol1[i, :, :], vol2[i, :, :]
                elif axis == 1:
                    slice1, slice2 = vol1[:, i, :], vol2[:, i, :]
                else:
                    slice1, slice2 = vol1[:, :, i], vol2[:, :, i]

                if slice1.max() > 0:
                    score = ssim(slice1.unsqueeze(0).unsqueeze(0), slice2.unsqueeze(0).unsqueeze(0))
                    axis_scores.append(score.item())

            if axis_scores:
                ssims.append(np.mean(axis_scores))
            else:
                ssims.append(0.0)
        return np.mean(ssims), ssims

    elif metric_type == "lpips":
        all_axis_scores = []
        for axis in range(3):
            if axis == 0:
                slices1, slices2 = vol1.unsqueeze(1), vol2.unsqueeze(1)
            elif axis == 1:
                slices1, slices2 = vol1.permute(1, 0, 2).unsqueeze(1), vol2.permute(1, 0, 2).unsqueeze(1)
            else:
                slices1, slices2 = vol1.permute(2, 0, 1).unsqueeze(1), vol2.permute(2, 0, 1).unsqueeze(1)

            slices1_rgb = slices1.repeat(1, 3, 1, 1)
            slices2_rgb = slices2.repeat(1, 3, 1, 1)

            batch_size = 32
            axis_scores = []
            for i in range(0, slices1_rgb.shape[0], batch_size):
                batch1 = slices1_rgb[i:i + batch_size]
                batch2 = slices2_rgb[i:i + batch_size]
                scores = _compute_lpips_baseline(batch1, batch2)
                if not torch.isnan(scores).any():
                    axis_scores.append(scores.cpu())

            if axis_scores: all_axis_scores.append(torch.cat(axis_scores).mean().item())

        return np.nanmean(all_axis_scores) if all_axis_scores else 0.0, all_axis_scores

    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


# --- [核心修正] 严格按照基线代码修正 metric_proj 函数 ---
@torch.no_grad()
def metric_proj(img1, img2, metric_type):
    if not isinstance(img1, torch.Tensor): img1 = torch.from_numpy(img1.copy()).float()
    if not isinstance(img2, torch.Tensor): img2 = torch.from_numpy(img2.copy()).float()
    img1, img2 = img1.cuda(), img2.cuda()

    n_slice = img1.shape[2]
    results = []

    if metric_type == "psnr" or metric_type == "ssim":
        for i in range(n_slice):
            slice1, slice2 = img1[:, :, i], img2[:, :, i]
            if slice1.max() > 0:
                # --- [逻辑修正] ---
                # 恢复与您提供的基线代码完全一致的独立归一化逻辑。
                # 这是本次修正的核心，确保评估标准与基线100%对齐。
                slice1_norm = slice1 / slice1.max()
                slice2_norm = slice2 / slice2.max()
                slice1_4d = slice1_norm.unsqueeze(0).unsqueeze(0)
                slice2_4d = slice2_norm.unsqueeze(0).unsqueeze(0)
                # --- [修正结束] ---

                if metric_type == "psnr":
                    score = _psnr_baseline(slice1_4d, slice2_4d, pixel_max=1.0)
                else:  # ssim
                    score = ssim(slice1_4d, slice2_4d)
                results.append(score.item())

        return np.mean(results) if results else 0.0, results

    elif metric_type == "lpips":
        # LPIPS 逻辑保持不变
        slices1 = img1.permute(2, 0, 1).unsqueeze(1)
        slices2 = img2.permute(2, 0, 1).unsqueeze(1)
        slices1_rgb = slices1.repeat(1, 3, 1, 1)
        slices2_rgb = slices2.repeat(1, 3, 1, 1)
        batch_size = 16
        for i in range(0, slices1_rgb.shape[0], batch_size):
            batch1 = slices1_rgb[i:i + batch_size]
            batch2 = slices2_rgb[i:i + batch_size]
            scores = _compute_lpips_baseline(batch1, batch2)
            if not torch.isnan(scores).any():
                results.extend(scores.cpu().tolist())

        return np.nanmean(results) if results else 0.0, results

    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


# --- [新增] 为单个4D图像张量计算指标的专用函数 (保持不变) ---
@torch.no_grad()
def metric_single_image(img1, img2, metric_type):
    """
    为单个4D图像张量 [B, C, H, W] 计算指标。
    这用于窗内指标计算，避免与 metric_proj 的堆叠逻辑冲突。
    其逻辑与基线中的 metric_proj 不同，但它服务于独立的训练过程监控目的。
    """
    if img1.dim() != 4 or img2.dim() != 4:
        print(f"[WARN] metric_single_image expects 4D tensors, but got {img1.shape} and {img2.shape}. Skipping.")
        return 0.0

    if metric_type == "psnr":
        return _psnr_baseline(img1, img2, pixel_max=1.0).item()
    elif metric_type == "ssim":
        return ssim(img1, img2).item()
    else:
        return 0.0