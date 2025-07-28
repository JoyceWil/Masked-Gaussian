# r2_gaussian/convert_npy_to_nifti.py

import numpy as np
import SimpleITK as sitk
import os

# --- 请修改以下路径 ---
# 输入的 .npy 文件路径
npy_input_path = '/media/data2/hezhipeng/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/vol_gt.npy'
# 希望保存的 .nii.gz 文件路径
nifti_output_path = '/home/hezhipeng/Workbench/r2_gaussian-main/r2_gaussian/vol_gt.nii.gz'
# -----------------------

print(f"正在从 {npy_input_path} 加载Numpy数组...")
try:
    volume_array = np.load(npy_input_path)
except Exception as e:
    print(f"错误：无法加载 .npy 文件。请检查路径是否正确。 {e}")
    exit()

print(f"加载成功，数据形状: {volume_array.shape}")

# 将Numpy数组转换为SimpleITK图像对象
# 注意：Numpy的坐标顺序通常是 (depth, height, width)
# SimpleITK的坐标顺序是 (x, y, z)，所以转换时会自动处理好
image = sitk.GetImageFromArray(volume_array)

# 【重要】设置像素间距 (Spacing)
# .npy文件不包含这个信息，但NIfTI格式需要它。
# 如果您知道原始CT的像素间距，请填入真实值。
# 如果不知道，使用 (1.0, 1.0, 1.0) 是一个安全的选择。
image.SetSpacing([1.0, 1.0, 1.0])

print(f"正在将图像保存到 {nifti_output_path}...")
try:
    sitk.WriteImage(image, nifti_output_path)
    print("🎉 转换成功！")
except Exception as e:
    print(f"错误：无法写入NIfTI文件。 {e}")