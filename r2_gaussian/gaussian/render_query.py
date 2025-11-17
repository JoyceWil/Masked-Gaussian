import sys
import torch
import math
import numpy as np
import torch.nn.functional as F
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

sys.path.append("./")
from r2_gaussian.gaussian.gaussian_model import GaussianModel
from r2_gaussian.dataset.cameras import Camera
from r2_gaussian.arguments import PipelineParams


def query(
    pc: GaussianModel,
    center,
    nVoxel,
    sVoxel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
):
    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    means3D = pc.get_xyz
    density = pc.get_density

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    vol_pred, radii = voxelizer(
        means3D=means3D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "vol": vol_pred,
        "radii": radii,
    }


def _grid_sample_trilinear(vol_zyx: torch.Tensor, pts_hwk3: torch.Tensor):
    Z, Y, X = vol_zyx.shape
    vol = vol_zyx[None, None, ...]  # [1,1,Z,Y,X]

    x = pts_hwk3[..., 0]
    y = pts_hwk3[..., 1]
    z = pts_hwk3[..., 2]

    gx = (x / (X - 1)) * 2.0 - 1.0
    gy = (y / (Y - 1)) * 2.0 - 1.0
    gz = (z / (Z - 1)) * 2.0 - 1.0

    eps = 1e-6
    gx = torch.clamp(gx, -1.0 + eps, 1.0 - eps)
    gy = torch.clamp(gy, -1.0 + eps, 1.0 - eps)
    gz = torch.clamp(gz, -1.0 + eps, 1.0 - eps)

    grid = torch.stack([gx, gy, gz], dim=-1)  # [H,W,K,3]
    H, W, K = grid.shape[:3]
    grid = grid.permute(2, 0, 1, 3).unsqueeze(0)  # [1,K,H,W,3]

    samples = F.grid_sample(
        vol,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # [1,1,K,H,W]
    samples = samples.squeeze(0).squeeze(0)  # [K,H,W]
    samples = samples.permute(1, 2, 0)  # [H,W,K]
    return samples


def render_cached_projector(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    center,
    nVoxel,
    sVoxel,
    scaling_modifier=1.0,
):
    if not hasattr(viewpoint_camera, "cached_pts") or viewpoint_camera.cached_pts is None:
        return None

    vq = query(pc, center=center, nVoxel=nVoxel, sVoxel=sVoxel, pipe=pipe, scaling_modifier=scaling_modifier)
    vol = vq["vol"]  # [Z,Y,X], CUDA float32

    pts_np = viewpoint_camera.cached_pts
    stp_np = viewpoint_camera.cached_stp
    H, W, K = pts_np.shape[:3]

    # clone() to avoid "not writable" NumPy warning and ensure safety
    pts = torch.from_numpy(pts_np).to(device=vol.device, dtype=vol.dtype).clone()   # [H,W,K,3]
    steps = torch.from_numpy(stp_np).to(device=vol.device, dtype=vol.dtype).clone() # [H,W,K]

    # valid mask: coords >= 0
    valid = (pts[..., 0] >= 0) & (pts[..., 1] >= 0) & (pts[..., 2] >= 0)
    steps = steps * valid.to(steps.dtype)

    Z, Y, X = map(int, vol.shape)
    pts_idx = pts
    pts_idx[..., 0] = torch.clamp(pts_idx[..., 0] * (X - 1), 0.0, X - 1.0)
    pts_idx[..., 1] = torch.clamp(pts_idx[..., 1] * (Y - 1), 0.0, Y - 1.0)
    pts_idx[..., 2] = torch.clamp(pts_idx[..., 2] * (Z - 1), 0.0, Z - 1.0)

    mu_samples = _grid_sample_trilinear(vol, pts_idx)  # [H,W,K]

    lineint = (mu_samples * steps).sum(dim=-1)  # [H,W]
    s = getattr(viewpoint_camera, "s", 1.0)
    I0 = getattr(viewpoint_camera, "I0", 1.0)
    lineint = torch.clamp(lineint, min=0.0)
    trans = torch.exp(-lineint / float(s))
    I = float(I0) * trans
    render_img = 1.0 - I
    render_img = render_img.unsqueeze(0)  # [1,H,W]

    if not hasattr(pipe, "_cached_projector_logged"):
        print(f"[TPG] Using cached projector for {getattr(viewpoint_camera, 'image_name', 'unknown')} "
              f"(H={H}, W={W}, K={K})")
        pipe._cached_projector_logged = True

    dummy = torch.zeros((pc.get_xyz.shape[0],), device=vol.device, dtype=torch.int32)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device=pc.get_xyz.device, requires_grad=False)

    return {
        "render": render_img,
        "viewspace_points": screenspace_points,
        "visibility_filter": dummy > 0,
        "radii": dummy,
        "cached": True,  # 新增标志
    }


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
):
    if getattr(pipe, "use_cached_projector", False) and hasattr(viewpoint_camera, "cached_pts") and viewpoint_camera.cached_pts is not None:
        if hasattr(viewpoint_camera, "_voxel_center") and hasattr(viewpoint_camera, "_nVoxel") and hasattr(viewpoint_camera, "_sVoxel"):
            out = render_cached_projector(
                viewpoint_camera=viewpoint_camera,
                pc=pc,
                pipe=pipe,
                center=viewpoint_camera._voxel_center,
                nVoxel=viewpoint_camera._nVoxel,
                sVoxel=viewpoint_camera._sVoxel,
                scaling_modifier=scaling_modifier,
            )
            if out is not None:
                return out

    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    density = pc.get_density

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "cached": False,  # 常规路径
    }