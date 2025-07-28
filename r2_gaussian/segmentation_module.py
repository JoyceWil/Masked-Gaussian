# r2_gaussian/segmentation_module.py

import torch
import numpy as np
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    EnsureTyped,
)
from monai.transforms import Resize  # 用于后处理


class MedicalSegmenter2D:
    """
    一个封装了为2D图像设计的U-Net模型的分割器。
    这个版本是为未来进行微调而准备的基础框架。
    """

    def __init__(self, roi_size=(256, 256), out_channels=1, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.roi_size = roi_size
        self.out_channels = out_channels

        print(f"2D分割模块正在加载U-Net模型到设备: {self.device}...")

        # 1. 加载一个为2D图像设计的标准U-Net模型
        # 注意：我们没有加载任何预训练权重。
        # 这是一个全新的、未经训练的模型，它的权重是随机初始化的。
        self.model = UNet(
            spatial_dims=2,  # 明确指定为2D模型
            in_channels=1,  # 输入是单通道灰度图
            out_channels=self.out_channels,  # 输出通道，对于二分类，1个通道就够了
            channels=(16, 32, 64, 128, 256),  # U-Net各层的通道数
            strides=(2, 2, 2, 2),  # U-Net各层的步长
            num_res_units=2,
        ).to(self.device)

        # 因为模型未经训练，所以它的输出在开始时将是随机的。
        # 我们将其设置为评估模式。
        self.model.eval()
        print("一个全新的、未经训练的2D U-Net模型已创建。")
        print("注意：它的初始输出将是随机噪声，这是训练前的正常现象。")

        # 2. 定义2D图像的预处理流程
        self.pre_transforms = Compose([
            # 输入已经是字典，我们只需要确保通道维度
            EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
            ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
            ),
            Resized(keys=["image"], spatial_size=self.roi_size, mode="area"),
            EnsureTyped(keys="image", device=self.device),
        ])

    @torch.no_grad()
    def segment_single_image(self, image_2d: np.ndarray):
        """
        对单张2D投影图进行分割。
        """
        original_shape = image_2d.shape
        input_data = {"image": image_2d.astype(np.float32)}

        # 1. 预处理
        processed_input = self.pre_transforms(input_data)
        input_tensor = processed_input['image'].unsqueeze(0)  # 增加批次维度 [1, 1, 256, 256]

        # 2. 模型推理
        logits = self.model(input_tensor)  # 输出形状 [1, 1, 256, 256]

        # 3. 后处理
        # 应用Sigmoid函数将logits转换为0-1之间的概率
        probs = torch.sigmoid(logits)

        # 将概率图从GPU移回CPU，并移除批次和通道维度
        prob_map_resized = probs.squeeze().cpu()  # [256, 256]

        # 使用双线性插值将掩码恢复到原始尺寸
        resizer = Resize(spatial_size=original_shape, mode="bilinear")
        final_prob_map = resizer(prob_map_resized.unsqueeze(0)).squeeze(0)

        # 将平滑后的概率图二值化得到最终掩码
        binary_mask = (final_prob_map > 0.5).numpy().astype(np.uint8)

        return binary_mask