# r2_gaussian/inspect_pickle_v2.py

import pickle
import numpy as np

# --- 您的 pickle 文件路径 ---
pickle_path = "/media/data2/hezhipeng/synthetic_dataset_naf_format/cone_ntrain_50_angle_360/0_chest_cone.pickle"
# ----------------------------------------------------

print(f"--- 正在深入检查 Pickle 文件: {pickle_path} ---")

try:
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)


    # 这是一个辅助函数，用于打印字典内容
    def inspect_dict(d, name="字典"):
        print(f"\n--- 正在检查嵌套的 '{name}' ---")
        if not isinstance(d, dict):
            print(f"'{name}' 不是一个字典，它的类型是 {type(d)}。")
            return

        keys = list(d.keys())
        print(f"'{name}' 包含的键 (keys) 如下:")
        print(keys)

        print(f"\n--- '{name}' 中每个键的详细信息 ---")
        for key in keys:
            value = d[key]
            value_type = type(value)
            shape_info = ""
            if isinstance(value, np.ndarray):
                shape_info = f" | 形状 (Shape): {value.shape} | 数据类型 (dtype): {value.dtype}"
            elif isinstance(value, (list, tuple)):
                shape_info = f" | 长度 (Length): {len(value)}"

            print(f"  - 键 '{key}': 类型是 {value_type}{shape_info}")


    # 检查 'train' 字典
    if 'train' in data:
        inspect_dict(data['train'], name="train")
    else:
        print("\n未在顶层找到 'train' 键。")

    # 检查 'val' 字典
    if 'val' in data:
        inspect_dict(data['val'], name="val")
    else:
        print("\n未在顶层找到 'val' 键。")


except FileNotFoundError:
    print(f"[错误] 找不到文件: {pickle_path}")
except Exception as e:
    print(f"[错误] 加载或检查文件时出错: {e}")