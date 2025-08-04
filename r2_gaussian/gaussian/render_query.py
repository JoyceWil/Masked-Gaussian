# r2_gaussian/gaussian/render_query.py (已集成DropGaussian)
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
    # query 函数不需要修改，保持原样
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
        active_sh_indices=None,  # 【修改1】: 增加 active_sh_indices 参数，默认值为 None
):
    # 准备一个与完整高斯点集大小相同的张量来接收2D坐标。
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

    # 【修改2】: 根据 active_sh_indices 是否存在，来决定使用全部高斯点还是子集
    if active_sh_indices is not None:
        # 如果提供了索引，就只选择这部分高斯点的属性进行渲染
        means3D = pc.get_xyz[active_sh_indices]
        means2D = screenspace_points[active_sh_indices] # 同样需要切片
        density = pc.get_density[active_sh_indices]
        scales = pc.get_scaling[active_sh_indices]
        rotations = pc.get_rotation[active_sh_indices]
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            # 注意: 如果使用预计算的协方差，也需要进行切片
            cov3D_precomp = pc.get_covariance(scaling_modifier)[active_sh_indices]
    else:
        # 如果没有提供索引（例如在评估阶段），则使用全部高斯点
        means3D = pc.get_xyz
        means2D = screenspace_points
        density = pc.get_density
        scales = pc.get_scaling
        rotations = pc.get_rotation
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)

    # 将准备好的数据传递给光栅化器
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # 【修改3】: 返回的 visibility_filter 和 radii 是针对被渲染的子集的，
    # 在 train.py 中我们已经处理了如何将这个子集的结果映射回完整点集。
    # 这里不需要做额外修改，但要理解其含义。
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points, # 仍然返回完整的张量，但只有可见部分被更新
        "visibility_filter": radii > 0,
        "radii": radii,
    }