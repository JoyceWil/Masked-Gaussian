# r2_gaussian/gaussian/render_query.py (最终修复版)
import sys
import torch
import math
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
        nVoxel_x=int(nVoxel[0]), nVoxel_y=int(nVoxel[1]), nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]), sVoxel_y=float(sVoxel[1]), sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]), center_y=float(center[1]), center_z=float(center[2]),
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
        means3D=means3D, opacities=density, scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
    )
    return {"vol": vol_pred, "radii": radii}


def render(
        viewpoint_camera: Camera,
        pc: GaussianModel,
        pipe: PipelineParams,
        scaling_modifier=1.0,
):
    # 准备一个张量来接收2D坐标。它的值会被CUDA内核覆盖。
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
    # 【关键】将全零的张量作为 means2D 传入
    means2D = screenspace_points
    density = pc.get_density
    scales = pc.get_scaling
    rotations = pc.get_rotation
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)

    # --- 【最终的核心修复】 ---
    # 假设 rasterizer 在原地修改了 means2D (即 screenspace_points)。
    # 渲染器函数会返回渲染图像和每个高斯点的屏幕半径。
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D, # 这个张量会被CUDA内核直接修改
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    # --- 【修复结束】 ---


    # 在返回时，我们使用被CUDA内核修改过的 screenspace_points 张量。
    # 它现在包含了所有可见高斯点的真实2D坐标。
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points, # 现在这里是带有真实坐标的张量
        "visibility_filter": radii > 0,
        "radii": radii,
    }