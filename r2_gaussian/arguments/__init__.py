# r2_gaussian/arguments/__init__.py (最终修复版)
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
        self.ply_path = ""
        self.scale_min = 0.0005
        self.scale_max = 0.5
        self.eval = True

        # --- 【核心修改 1】 ---
        # 只保留我们需要的两个掩码参数，并移除 air_mask_dir
        self.soft_mask_dir = ""
        self.core_mask_dir = ""
        # --- 【修改结束】 ---

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
        self.position_lr_final = 0.000002
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

        self.intelligent_confidence_mode = 'percentile'
        self.intelligent_confidence_percentile = 40.0
        # self.intelligent_confidence_threshold = 0.07

        self.roi_management_interval = 100
        self.roi_prune_threshold = -3.0
        self.roi_protect_threshold = 0.8
        self.roi_candidate_threshold = 0.2
        self.roi_background_reward = -0.05
        self.roi_standard_reward = 0.05
        self.roi_core_bonus_reward = 0.05

        self.use_confidence_modulation = True
        self.confidence_prune_center = -2.0
        self.confidence_prune_steepness = 2.0

        # 添加新的调制致密化参数
        self.confidence_densify_sensitivity = 2.5

        self.confidence_densify_scale = 2.0

        self.opacity_prune_threshold = 0.005

        self.tv_vol_size = 32
        self.density_min_threshold = 0.005
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 5.0e-5
        self.densify_scale_threshold = 0.1
        self.max_screen_size = None
        self.max_scale = None
        self.max_num_gaussians = 500_000

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