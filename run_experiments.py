# run_experiments.py
import os
import subprocess
import shutil
import yaml
import pandas as pd
from datetime import datetime

# ==============================================================================
# 1. 实验配置
# ==============================================================================

# --- 基础设置 ---
# 请根据您的环境修改
PYTHON_EXECUTABLE = "python"  # 或者您的conda环境中的python.exe的完整路径
TRAIN_SCRIPT = "train.py"
BASE_SOURCE_PATH = "data/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone"
BASE_OUTPUT_DIR = "output_experiments"  # 使用一个专门的目录存放所有实验结果

# --- 实验矩阵定义 ---
# 定义我们要运行的所有实验
experiments = []

# Group A: 绝对对照组 (True Control)
# 完全禁用ROI置信度系统，不使用--auto_mask
experiments.append({
    "name": "A_TrueControl",
    "description": "ROI置信度系统完全禁用 (无 --auto_mask)",
    "params": {
        "--intelligent_confidence_mode": "none"
    }
})

# Group B: 置信度系统基线组 (Baseline for Confidence System)
# 启用ROI系统，但不进行智能初始化
experiments.append({
    "name": "B_ConfBaseline",
    "description": "启用置信度系统, 但无智能初始化",
    "params": {
        "--auto_mask": True,
        "--intelligent_confidence_mode": "none"
    }
})

# Group C: 智能初始化实验组
# 启用ROI系统，并使用不同的初始奖励值
reward_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0]
for value in reward_values:
    experiments.append({
        "name": f"C_Reward_{value:.2f}".replace('.', '_'),  # 文件名友好
        "description": f"智能初始化, 初始奖励 = {value:.2f}",
        "params": {
            "--auto_mask": True,
            "--intelligent_confidence_mode": "percentile",
            "--roi_core_bonus_reward": value
        }
    })


# ==============================================================================
# 2. 实验执行器
# ==============================================================================

def run_all_experiments():
    """循环执行定义好的所有实验"""
    start_time = datetime.now()
    total_experiments = len(experiments)
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print(f"自动化实验开始，共计 {total_experiments} 个实验。")
    print(f"所有结果将保存在: {os.path.abspath(BASE_OUTPUT_DIR)}")
    print("=" * 80)

    for i, exp in enumerate(experiments):
        exp_name = exp["name"]
        exp_desc = exp["description"]
        exp_params = exp["params"]

        # 为每个实验创建唯一的输出目录
        model_path = os.path.join(BASE_OUTPUT_DIR, exp_name)

        print(f"\n--- [{i + 1}/{total_experiments}] 正在执行实验: {exp_name} ---")
        print(f"   描述: {exp_desc}")

        # 清理旧目录（可选，但推荐）
        if os.path.exists(model_path):
            print(f"   警告: 目录 {model_path} 已存在。")
            # answer = input("   是否删除并重新开始? (y/n): ").lower()
            # if answer == 'y':
            #     print("   正在删除旧目录...")
            #     shutil.rmtree(model_path)
            # else:
            #     print("   跳过此实验。")
            #     continue
            print("   正在删除旧目录...")
            shutil.rmtree(model_path)

        # 构建命令行
        cmd = [
            PYTHON_EXECUTABLE,
            TRAIN_SCRIPT,
            "-s", BASE_SOURCE_PATH,
            "-m", model_path,
            "--plot_confidence",  # 确保每个实验都生成置信度图
            # "--quiet"  # 减少不必要的tqdm输出，让日志更清晰
        ]

        # 添加特定于本实验的参数
        for key, value in exp_params.items():
            if isinstance(value, bool) and value:
                cmd.append(key)
            elif not isinstance(value, bool):
                cmd.append(key)
                cmd.append(str(value))

        print(f"   执行命令: {' '.join(cmd)}")

        try:
            # 执行训练脚本
            subprocess.run(cmd, check=True, text=True)
            print(f"--- 实验 {exp_name} 成功完成 ---")
        except subprocess.CalledProcessError as e:
            print(f"!!!!!! 实验 {exp_name} 失败 !!!!!!")
            print(f"返回码: {e.returncode}")
            print(f"错误输出:\n{e.stderr}")
            # 可选择在这里中断整个流程
            # break
        except KeyboardInterrupt:
            print("\n用户中断了实验。")
            return

    end_time = datetime.now()
    print("\n" + "=" * 80)
    print(f"所有实验执行完毕。总耗时: {end_time - start_time}")
    print("=" * 80)


# ==============================================================================
# 3. 结果汇总器
# ==============================================================================

def summarize_results():
    """扫描所有实验输出目录，收集评估数据并汇总到CSV文件。"""
    print("\n" + "=" * 80)
    print("开始汇总所有实验结果...")

    all_results = []
    eval_iterations = [1, 5000, 10000, 20000, 30000]  # 与train.py中的test_iterations保持一致

    for exp in experiments:
        exp_name = exp["name"]
        model_path = os.path.join(BASE_OUTPUT_DIR, exp_name)
        eval_path = os.path.join(model_path, "eval")

        if not os.path.exists(eval_path):
            print(f"  - 警告: 找不到实验 '{exp_name}' 的评估目录, 跳过。")
            continue

        for iter_num in eval_iterations:
            # 3D评估文件
            eval3d_file = os.path.join(eval_path, f"iter_{iter_num:06d}", "eval3d.yml")
            # 2D评估文件
            eval2d_file = os.path.join(eval_path, f"iter_{iter_num:06d}", "eval2d_render_test.yml")

            row_data = {
                "experiment_name": exp_name,
                "reward_value": exp['params'].get('--roi_core_bonus_reward', 'N/A'),
                "iteration": iter_num
            }

            # 读取3D结果
            if os.path.exists(eval3d_file):
                with open(eval3d_file, 'r') as f:
                    data_3d = yaml.safe_load(f)
                    row_data.update(data_3d)

            # 读取2D结果
            if os.path.exists(eval2d_file):
                with open(eval2d_file, 'r') as f:
                    data_2d = yaml.safe_load(f)
                    row_data.update(data_2d)

            # 只有在至少有一个评估文件存在时才添加数据
            if len(row_data) > 3:
                all_results.append(row_data)

    if not all_results:
        print("  - 错误: 未能收集到任何评估数据。请检查实验是否成功运行并生成了eval文件。")
        return

    # 创建DataFrame并保存
    df = pd.DataFrame(all_results)

    # 重新排列列的顺序，使其更具可读性
    cols_order = [
        "experiment_name", "reward_value", "iteration",
        "psnr_3d", "ssim_3d", "lpips_3d",
        "psnr_2d", "ssim_2d", "lpips_2d"
    ]
    # 获取所有其他列
    other_cols = [col for col in df.columns if col not in cols_order]
    final_cols = cols_order + sorted(other_cols)  # 排序以保持一致性
    df = df[final_cols]

    output_csv_path = os.path.join(BASE_OUTPUT_DIR, "experiment_summary.csv")
    df.to_csv(output_csv_path, index=False)

    print(f"\n结果汇总成功！数据已保存到: {os.path.abspath(output_csv_path)}")
    print("现在您可以使用Excel或Python Pandas打开此文件进行分析。")
    print("=" * 80)


# ==============================================================================
# 4. 主程序入口
# ==============================================================================

if __name__ == "__main__":
    run_all_experiments()
    summarize_results()