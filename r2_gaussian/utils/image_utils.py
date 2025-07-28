# r2_gaussian/utils/image_utils.py
import sys
import numpy as np
import torch
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

sys.path.append("./")
from r2_gaussian.utils.loss_utils import ssim as ssim_torch

try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("警告: lpips包未安装。LPIPS指标将不会被计算。")

_lpips_alex_model = None


def get_lpips_model(net_type='alex', device=None):
    global _lpips_alex_model
    if _lpips_alex_model is None and LPIPS_AVAILABLE:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"正在设备 {device} 上初始化LPIPS(AlexNet)模型...")
        _lpips_alex_model = lpips.LPIPS(net=net_type).to(device)
    return _lpips_alex_model


@torch.no_grad()
def compute_lpips(img1, img2):
    if not LPIPS_AVAILABLE:
        return torch.tensor([[float('nan')]], device=img1.device)
    lpips_model = get_lpips_model(device=img1.device)
    if lpips_model is None:
        return torch.tensor([[float('nan')]], device=img1.device)
    img1_scaled = img1 * 2 - 1
    img2_scaled = img2 * 2 - 1
    return lpips_model(img1_scaled, img2_scaled)


def mse(img1, img2, mask=None):
    n_channel = img1.shape[1]
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)
        mask = mask.flatten(1).repeat(1, n_channel)
        mask = torch.where(mask != 0, True, False)
        mse = torch.stack(
            [(((img1[i, mask[i]] - img2[i, mask[i]])) ** 2).mean(0, keepdim=True) for i in range(img1.shape[0])], dim=0)
    else:
        mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return mse


def rmse(img1, img2, mask=None):
    mse_out = mse(img1, img2, mask)
    return mse_out ** 0.5


@torch.no_grad()
def psnr(img1, img2, mask=None, pixel_max=1.0):
    mse_out = mse(img1, img2, mask)
    psnr_out = 10 * torch.log10(pixel_max ** 2 / mse_out.float())
    if mask is not None and torch.isinf(psnr_out).any():
        psnr_out = psnr_out[~torch.isinf(psnr_out)]
    return psnr_out


@torch.no_grad()
def metric_vol(gt_vol, pred_vol, metric="psnr", pixel_max=1.0):
    """
    【已修复】计算体积指标。
    修复了LPIPS计算中因permute维度不匹配导致的RuntimeError。
    """
    assert metric in ["psnr", "ssim", "lpips"]
    if isinstance(gt_vol, np.ndarray): gt_vol = torch.from_numpy(gt_vol.copy())
    if isinstance(pred_vol, np.ndarray): pred_vol = torch.from_numpy(pred_vol.copy())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gt_vol, pred_vol = gt_vol.to(device), pred_vol.to(device)

    if metric == "psnr":
        pixel_max_val = pixel_max if pixel_max is not None else gt_vol.max().item()
        mse_val = torch.mean((gt_vol - pred_vol) ** 2)
        if mse_val == 0: return float('inf'), None
        psnr_val = 10 * torch.log10(pixel_max_val ** 2 / mse_val.float())
        return psnr_val.item(), None

    elif metric == "ssim":
        ssims = []
        for axis in [0, 1, 2]:
            gt_slices = torch.unbind(gt_vol, dim=axis)
            pred_slices = torch.unbind(pred_vol, dim=axis)
            axis_ssims = []
            for s_gt, s_pred in zip(gt_slices, pred_slices):
                s_gt_np, s_pred_np = s_gt.cpu().numpy(), s_pred.cpu().numpy()
                data_range = s_gt_np.max() - s_gt_np.min()
                if data_range > 0:
                    axis_ssims.append(structural_similarity(s_gt_np, s_pred_np, data_range=data_range))
            if axis_ssims: ssims.append(np.mean(axis_ssims))
        return np.mean(ssims) if ssims else 0.0, ssims

    elif metric == "lpips":
        if not LPIPS_AVAILABLE: return float('nan'), [float('nan')] * 3

        lpips_scores = []
        # 假设体积数据维度为 (D, H, W) 或 (H, W, D)
        # 我们将沿三个轴向切片来计算LPIPS
        for axis in [0, 1, 2]:
            # --- 核心修复：根据axis正确地重排3D体积为2D切片批次 ---
            if axis == 0:  # 沿第一个轴切片
                gt_slices = gt_vol.unsqueeze(1)
                pred_slices = pred_vol.unsqueeze(1)
            elif axis == 1:  # 沿第二个轴切片
                gt_slices = gt_vol.permute(1, 0, 2).unsqueeze(1)
                pred_slices = pred_vol.permute(1, 0, 2).unsqueeze(1)
            else:  # axis == 2, 沿第三个轴切片
                gt_slices = gt_vol.permute(2, 0, 1).unsqueeze(1)
                pred_slices = pred_vol.permute(2, 0, 1).unsqueeze(1)

            # 归一化到 [0, 1] 以输入LPIPS模型
            gt_min, gt_max = gt_vol.min(), gt_vol.max()
            if gt_max > gt_min:
                gt_slices = (gt_slices - gt_min) / (gt_max - gt_min)
                pred_slices = (pred_slices - gt_min) / (gt_max - gt_min)

            # LPIPS需要3通道图像
            gt_slices = gt_slices.repeat(1, 3, 1, 1)
            pred_slices = pred_slices.repeat(1, 3, 1, 1)

            # 分批计算以防显存爆炸
            batch_size = 32
            axis_lpips = []
            for i in range(0, gt_slices.shape[0], batch_size):
                batch_gt = gt_slices[i:i + batch_size]
                batch_pred = pred_slices[i:i + batch_size]
                axis_lpips.append(compute_lpips(batch_gt, batch_pred))

            if axis_lpips:
                lpips_scores.append(torch.cat(axis_lpips).mean().item())
            else:
                lpips_scores.append(float('nan'))

        return np.nanmean(lpips_scores), lpips_scores


@torch.no_grad()
def metric_proj(gt_projs, pred_projs, metric="psnr", pixel_max=1.0):
    assert metric in ["psnr", "ssim", "lpips"]
    if isinstance(gt_projs, np.ndarray): gt_projs = torch.from_numpy(gt_projs)
    if isinstance(pred_projs, np.ndarray): pred_projs = torch.from_numpy(pred_projs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gt_projs, pred_projs = gt_projs.to(device), pred_projs.to(device)

    gt_projs = gt_projs.permute(2, 0, 1).unsqueeze(1)
    pred_projs = pred_projs.permute(2, 0, 1).unsqueeze(1)

    results = []
    if metric == "psnr":
        pixel_max_val = pixel_max if pixel_max is not None else 1.0
        mse_vals = torch.mean((gt_projs - pred_projs) ** 2, dim=[1, 2, 3])
        psnr_vals = 10 * torch.log10(pixel_max_val ** 2 / mse_vals)
        results = psnr_vals.cpu().numpy()

    elif metric == "ssim":
        for i in range(gt_projs.shape[0]):
            gt_slice, pred_slice = gt_projs[i], pred_projs[i]
            results.append(ssim_torch(gt_slice.unsqueeze(0), pred_slice.unsqueeze(0)).item())

    elif metric == "lpips":
        if not LPIPS_AVAILABLE: return float('nan'), [float('nan')] * gt_projs.shape[0]
        gt_projs_rgb = gt_projs.repeat(1, 3, 1, 1)
        pred_projs_rgb = pred_projs.repeat(1, 3, 1, 1)
        batch_size = 16
        lpips_scores = []
        for i in range(0, gt_projs_rgb.shape[0], batch_size):
            batch_gt = gt_projs_rgb[i:i + batch_size]
            batch_pred = pred_projs_rgb[i:i + batch_size]
            lpips_scores.append(compute_lpips(batch_gt, batch_pred))
        results = torch.cat(lpips_scores).squeeze().cpu().numpy()

    mean_result = np.mean(results) if len(results) > 0 else 0.0

    if isinstance(results, np.ndarray):
        return mean_result, results.tolist()
    else:
        return mean_result, results