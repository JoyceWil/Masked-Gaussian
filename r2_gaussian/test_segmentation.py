# r2_gaussian/test_segmentation.py

import numpy as np
import matplotlib.pyplot as plt
import os

# 从我们新的模块文件导入新的2D分割器类
from segmentation_module import MedicalSegmenter2D


def main():
    print("--- 测试全新的2D U-Net分割模块 ---")

    # 加载数据 (这部分不变)
    image_path = "/media/data2/hezhipeng/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone/proj_train/proj_train_0000.npy"
    print(f"正在从图像文件: {image_path} 加载数据...")
    try:
        projection_image = np.load(image_path)
        print(f"测试图像形状: {projection_image.shape}")
    except Exception as e:
        print(f"[失败] 加载数据时出错: {e}")
        return

    # 初始化全新的2D分割器
    try:
        # 使用新的类
        segmenter = MedicalSegmenter2D(roi_size=(256, 256), device="cuda")
    except Exception as e:
        print(f"[失败] 2D分割模块初始化失败: {e}")
        return

    # 执行分割
    print("正在使用未经训练的U-Net执行分割...")
    binary_mask = segmenter.segment_single_image(projection_image)
    print("分割完成！")

    # 可视化结果
    # 我们期望看到的是随机的、无意义的噪声，这表明模型已准备好接受训练。
    print("正在生成并保存结果图像...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(projection_image, cmap='gray')
    axes[0].set_title('Original Projection')
    axes[0].axis('off')

    axes[1].imshow(binary_mask, cmap='gray')  # 直接显示黑白掩码，不叠加
    axes[1].set_title('Output from UNTRAINED U-Net (Expect Random Noise)')
    axes[1].axis('off')

    plt.tight_layout()
    output_filename = "segmentation_result_2d_untrained.png"
    plt.savefig(output_filename, dpi=300)

    print(f"🎉 成功！未经训练的U-Net输出结果已保存到: {output_filename}")
    print("请查看图片，如果看到的是随机噪声，那么我们已经为下一步的训练做好了准备！")


if __name__ == '__main__':
    main()