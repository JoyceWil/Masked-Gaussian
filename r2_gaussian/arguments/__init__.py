import os
import sys
import os.path as osp
from argparse import ArgumentParser, Namespace

sys.path.append("./")
from r2_gaussian.utils.argument_utils import ParamGroup


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self.data_device = "cuda"
        self.ply_path = ""  # Path to initialization point cloud (if None, we will try to find `init_*.npy`.)
        self.scale_min = 0.0005  # percent of volume size
        self.scale_max = 0.5  # percent of volume size
        self.eval = True

        self.soft_mask_dir = ""
        self.core_mask_dir = ""

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = osp.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0002
        self.position_lr_final = 0.00002
        self.position_lr_max_steps = 30_000
        self.density_lr_init = 0.01
        self.density_lr_final = 0.001
        self.density_lr_max_steps = 30_000
        self.scaling_lr_init = 0.005
        self.scaling_lr_final = 0.0005
        self.scaling_lr_max_steps = 30_000
        self.rotation_lr_init = 0.001
        self.rotation_lr_final = 0.0001
        self.rotation_lr_max_steps = 30_000
        self.lambda_dssim = 0.25
        self.lambda_tv = 0.05
        self.tv_vol_size = 32
        self.density_min_threshold = 0.00001
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 5.0e-5
        self.densify_scale_threshold = 0.1  # percent of volume size
        self.max_screen_size = None
        self.max_scale = None  # percent of volume size
        self.max_num_gaussians = 500_000

        # 智能初始化参数
        self.intelligent_confidence_mode = "percentile"
        self.intelligent_confidence_percentile = 40.0   # 在'percentile'模式下，保留FDK密度最高的点的百分比
        self.intelligent_confidence_threshold = 0.2     # 在'fixed'模式下，使用的FDK密度阈值
        self.roi_init_bonus = 2.0    # 为被识别为核心的点赋予的初始置信度奖励值

        self.auto_mask_pre_threshold_ratio = 0.5

        self.enable_confidence_update = True
        self.roi_management_interval = 100
        self.roi_standard_reward = 0.15  # 落在软组织区域的标准奖励
        self.roi_core_bonus_reward = 0.15  # 落在核心骨架区域的额外奖励 (总奖励 = standard + core_bonus)
        self.roi_background_reward = -0.05  # 落在背景区域的惩罚
        self.use_reward_saturation = True  # 是否启用奖励饱和，防止置信度无限增长
        self.confidence_saturation_sensitivity = 0.3  # 饱和度敏感性，值越大，越容易饱和

        # 置信度
        self.confidence_max_val = 5.0
        self.confidence_min_val = -5.0
        self.confidence_densify_center = 0.0
        self.confidence_densify_sensitivity = 2.0
        self.clone_confidence_decay_factor = 0.5

        self.opacity_prune_threshold = 0.005
        self.confidence_densify_mode = 'add'
        self.confidence_densify_bonus_scale = 3.0e-5
        self.confidence_prune_center = -3.0
        self.confidence_prune_steepness = 0.7

        self.use_confidence_for_densify = True
        self.use_confidence_modulation = True

        # 置信度指导剪枝/致密化参数
        self.prune_confidence_threshold = 0.8  # 置信度高于此值的点将被保护，不会因低密度被剪枝
        self.densify_confidence_threshold = 0.0  # 置信度低于此值的点将不会被致密化（克隆或分裂）
        # self.lambda_gdl = 0.1   # psnr3d 35.197, ssim3d 0.928, lpips3d 0.073, psnr2d 55.406, ssim2d 0.995, lpips2d 0.006
        self.lambda_gdl = 0.3   # psnr3d 35.185, ssim3d 0.926, lpips3d 0.067, psnr2d 55.431, ssim2d 0.995, lpips2d 0.004
        # self.lambda_gdl = 0.5   # psnr3d 34.964, ssim3d 0.924, lpips3d 0.064, psnr2d 55.247, ssim2d 0.995, lpips2d 0.004

        self.compat_mode = True
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = osp.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
