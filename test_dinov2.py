import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from urllib.request import urlretrieve
import os

# --- 全局配置 ---
# 您可以更改这些路径
INPUT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
INPUT_IMAGE_FILENAME = "cat_example.jpg"


# 输出文件将根据输入文件名自动命名
# 例如，输出的可视化图将是 'cat_example_masks.png'
# 输出的掩码张量将是 'cat_example_soft_mask.pt' 和 'cat_example_core_mask.pt'

def generate_and_save_masks(
        image_path,
        output_dir='.',
        model_name='dinov2_vits14',
        patch_size=14,
        image_size=518
):
    """
    使用DINOv2为单张图片生成掩码，并保存可视化结果和原始张量。

    参数:
    - image_path (str): 输入图片的路径。
    - output_dir (str): 所有输出文件的保存目录。
    - model_name (str): DINOv2模型名称。
    - patch_size (int): ViT的patch大小。
    - image_size (int): 模型输入尺寸。
    """
    print("--- 开始处理 ---")

    # --- 1. 加载DINOv2模型 ---
    print(f"1. 正在加载DINOv2模型: {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.hub.load('facebookresearch/dinov2', model_name)
        model.to(device)
        model.eval()
        print("   模型加载成功。")
    except Exception as e:
        print(f"   模型加载失败，请检查网络连接或torch.hub缓存。错误: {e}")
        return

    # --- 2. 图像预处理 ---
    print(f"2. 正在预处理图像: {image_path}...")
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    transform = TF.Compose([
        TF.Resize(image_size, interpolation=TF.InterpolationMode.BICUBIC),
        TF.CenterCrop(image_size),
        TF.ToTensor(),
        TF.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    print("   图像预处理完成。")

    h_patches = image_size // patch_size
    w_patches = image_size // patch_size

    # --- 3. 获取自注意力图 ---
    print("3. 正在通过模型前向传播并提取注意力图...")
    with torch.no_grad():
        features_dict = model.forward_features(img_tensor)
        attentions = features_dict['x_norm_attn']
    print("   注意力图提取成功。")

    num_heads = attentions.shape[1]

    # --- 4. 生成“完整范围掩码 (soft_mask)” ---
    print("4. 正在生成'完整范围掩码' (soft_mask)...")
    cls_attentions = attentions[0, :, 0, 1:].reshape(num_heads, h_patches, w_patches)
    soft_mask = cls_attentions.mean(dim=0)

    soft_mask_resized = F.interpolate(
        soft_mask.unsqueeze(0).unsqueeze(0),
        size=(original_size[1], original_size[0]),
        mode='bilinear',
        align_corners=False
    ).squeeze()

    soft_mask_normalized = (soft_mask_resized - soft_mask_resized.min()) / (
                soft_mask_resized.max() - soft_mask_resized.min())
    print("   '完整范围掩码'生成成功。")

    # --- 5. 生成“核心骨架掩码 (core_mask)” ---
    print("5. 正在生成'核心骨架掩码' (core_mask)...")
    high_confidence_threshold = 0.7
    core_mask_binary = (soft_mask_normalized > high_confidence_threshold).float()
    core_mask_soft = soft_mask_normalized * core_mask_binary
    print("   '核心骨架掩码'生成成功。")

    # --- 6. 定义输出路径 ---
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    visualization_path = os.path.join(output_dir, f"{base_name}_masks_visualization.png")
    soft_mask_path = os.path.join(output_dir, f"{base_name}_soft_mask.pt")
    core_mask_path = os.path.join(output_dir, f"{base_name}_core_mask.pt")

    # --- 7. 保存可视化结果 ---
    print(f"6. 正在保存可视化对比图到: {visualization_path}...")
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(soft_mask_normalized.cpu().numpy(), cmap='viridis')
    axs[1].set_title('Soft Mask (Full Scope)')
    axs[1].axis('off')

    axs[2].imshow(core_mask_binary.cpu().numpy(), cmap='gray')
    axs[2].set_title('Core Mask (Binary Skeleton)')
    axs[2].axis('off')

    axs[3].imshow(core_mask_soft.cpu().numpy(), cmap='viridis')
    axs[3].set_title('Core Mask (Soft Skeleton)')
    axs[3].axis('off')

    plt.tight_layout()
    plt.savefig(visualization_path, bbox_inches='tight')
    plt.close(fig)  # 关闭图像以释放内存
    print(f"   可视化图保存成功。")

    # --- 8. 保存原始张量文件 ---
    print(f"7. 正在保存掩码张量文件...")
    torch.save(soft_mask_normalized.cpu(), soft_mask_path)
    print(f"   - '完整范围掩码'已保存到: {soft_mask_path}")

    torch.save(core_mask_soft.cpu(), core_mask_path)
    print(f"   - '核心骨架掩码'已保存到: {core_mask_path}")

    print("\n--- 处理完成！---")


if __name__ == '__main__':
    # --- 下载示例图片 ---
    if not os.path.exists(INPUT_IMAGE_FILENAME):
        print(f"未找到示例图片 '{INPUT_IMAGE_FILENAME}'。正在从网络下载...")
        try:
            urlretrieve(INPUT_IMAGE_URL, INPUT_IMAGE_FILENAME)
            print("下载成功。")
        except Exception as e:
            print(f"下载失败！请检查您的网络连接，或手动放置一张图片并命名为 '{INPUT_IMAGE_FILENAME}'。")
            print(f"错误信息: {e}")
            exit()  # 如果下载失败则退出程序

    # --- 执行主函数 ---
    # 所有输出文件将保存在当前脚本所在的目录
    generate_and_save_masks(image_path=INPUT_IMAGE_FILENAME, output_dir='.')