import torch

# 检查 GPU 是否可用
is_cuda_available = torch.cuda.is_available()
print(f"CUDA 可用: {is_cuda_available}")