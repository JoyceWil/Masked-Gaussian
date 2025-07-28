# r2_gaussian/test_segmentation_pretrained.py (修正版)

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import segmentation_models_pytorch as smp
from monai.transforms import Resize


def main():
    print("--- 测试预训练的2D分割模型 (segmentation-models-pytorch) ---")

    # --- 1. 加载数据 (不变) ---
    image_path = "/media/data2/hezhipeng/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone/proj_train/proj_train_0000.npy"
    print(f"正在从图像文件: {image_path} 加载数据...")
    try:
        projection_image = np.load(image_path)
        print(f"测试图像形状: {projection_image.shape}")
    except Exception as e:
        print(f"[失败] 加载数据时出错: {e}")
        return

    # --- 2. 初始化预训练的U-Net模型 (不变) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在加载预训练模型到设备: {device}...")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ).to(device)
    model.eval()

    print("模型加载成功！使用的是在ImageNet上预训练的ResNet-34骨干。")

    # --- 3. 准备输入数据 (这里是关键的修正) ---
    # a. 归一化: 将0-255的图像缩放到0-1
    image_normalized = projection_image.astype(np.float32) / 255.0

    # b. 调整尺寸: 调整到模型期望的尺寸，例如256x256
    roi_size = (256, 256)
    resizer = Resize(spatial_size=roi_size, mode="area")

    # c. 增加通道维度，MONAI的变换期望输入是 (C, H, W)
    image_ch_first = np.expand_dims(image_normalized, axis=0)  # 形状变为 [1, 512, 512]

    # d. 应用Resize。输出的 image_resized_tensor 是一个MONAI的MetaTensor，其本质是torch.Tensor
    image_resized_tensor = resizer(image_ch_first)  # 形状变为 [1, 256, 256]

    # e. 增加批次维度并发送到设备
    #    因为 image_resized_tensor 已经是张量，我们不再需要 torch.from_numpy()
    #    我们直接在其基础上增加批次维度 (unsqueeze(0)) 即可
    input_tensor = image_resized_tensor.unsqueeze(0).to(device)  # 最终形状: [1, 1, 256, 256]

    # --- 4. 执行分割 (不变) ---
    print("正在使用预训练模型执行分割...")
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)

    # --- 5. 后处理和可视化 (不变) ---
    prob_map_resized = probs.squeeze().cpu().numpy()

    final_resizer = Resize(spatial_size=projection_image.shape, mode="bilinear")
    final_prob_map = final_resizer(np.expand_dims(prob_map_resized, axis=0))

    binary_mask = (final_prob_map > 0.5).astype(np.uint8).squeeze()

    print("分割完成！")

    # --- 6. 保存结果 (不变) ---
    print("正在生成并保存结果图像...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(projection_image, cmap='gray')
    axes[0].set_title('Original Projection')
    axes[0].axis('off')

    axes[1].imshow(projection_image, cmap='gray')
    axes[1].imshow(binary_mask, cmap='Reds', alpha=0.5)
    axes[1].set_title('Output from Pre-trained U-Net (ImageNet)')
    axes[1].axis('off')

    plt.tight_layout()
    output_filename = "segmentation_result_pretrained.png"
    plt.savefig(output_filename, dpi=300)

    print(f"🎉 成功！预训练模型输出结果已保存到: {output_filename}")
    print("请查看图片，结果应该是一个有结构的形状，不再是随机噪声。")


if __name__ == '__main__':
    main()