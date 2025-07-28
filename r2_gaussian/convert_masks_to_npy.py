# r2_gaussian/convert_masks_to_npy.py

import numpy as np
import SimpleITK as sitk
import os
import glob

# --- 请确认以下路径 ---
# 1. 包含您手动标注的 .nii.gz 掩码的输入目录
#    (这是您在ITK-SNAP中保存标注文件的文件夹)
input_dir_nifti = '/media/data2/hezhipeng/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/masks/'

# 2. 用于存放最终 .npy 掩码文件的新目录
#    (这个目录将被训练脚本使用)
output_dir_npy = '/media/data2/hezhipeng/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/masks/'
# -----------------------

# 确保输出目录存在
os.makedirs(output_dir_npy, exist_ok=True)

# 查找输入目录中所有的 .nii.gz 文件
nifti_files = glob.glob(os.path.join(input_dir_nifti, '*.nii.gz'))

if not nifti_files:
    print(f"错误：在目录 '{input_dir_nifti}' 中没有找到任何 .nii.gz 文件。")
    print("请确认 'input_dir_nifti' 变量设置正确。")
    exit()

print(f"找到 {len(nifti_files)} 个 .nii.gz 掩码文件。开始转换...")

for nifti_path in nifti_files:
    try:
        # 读取 .nii.gz 文件
        mask_sitk = sitk.ReadImage(nifti_path)

        # 将SimpleITK图像对象转换为Numpy数组
        mask_array = sitk.GetArrayFromImage(mask_sitk)

        # --- 标准化处理 (非常重要) ---
        # ITK-SNAP保存的标签可能是1, 2, 3...
        # 我们需要一个二元掩码：0代表背景，1代表前景。
        # 这行代码会将所有非零的像素值都设置为1.0。
        mask_array[mask_array > 0] = 1.0

        # 将数据类型转换为float32，这在深度学习中很常见
        mask_array = mask_array.astype(np.float32)

        # 构建输出文件的路径
        # 例如：masks_manual_nifti/mask_0000.nii.gz -> masks_manual_npy/mask_0000.npy
        base_filename = os.path.basename(nifti_path)
        filename_without_ext = os.path.splitext(base_filename)[0]
        output_path = os.path.join(output_dir_npy, f"{filename_without_ext}.npy")

        # 保存为 .npy 文件
        np.save(output_path, mask_array)

        print(f"  -> 已转换: {nifti_path}  到  {output_path}")

    except Exception as e:
        print(f"处理文件 {nifti_path} 时发生错误: {e}")

print(f"\n✅ 批量转换完成！")
print(f"所有 .npy 格式的掩码文件已保存在 '{output_dir_npy}' 目录中。")
print("我们现在已经准备好，可以随时开始微调训练了！")