# r2_gaussian/utils/log_utils.py (修正后)
import os
import sys
import os.path as osp
from argparse import Namespace
import yaml
import datetime

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

sys.path.append("./")
from r2_gaussian.utils.cfg_utils import args2string


def prepare_output_and_logger(args):
    # --- [修改开始] ---
    # 移除了所有 if not args.model_path: 的逻辑。
    # 我们现在假设 args.model_path 在调用此函数时已经是最终确定的路径。
    # --- [修改结束] ---

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(osp.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Save to yaml
    args_dict = vars(args)
    with open(osp.join(args.model_path, "cfg_args.yml"), "w") as f:
        yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        tb_writer.add_text("args", args2string(args_dict), global_step=0)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer