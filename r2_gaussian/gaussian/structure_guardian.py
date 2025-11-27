import torch
import torch.nn.functional as F
import numpy as np
import os


class StructureGuardian:
    """
    一个用于加载和查询静态 P_vol 3D结构图的模块。
    这个模块从预先计算的 P_vol.npy (来自FDK) 中加载结构信息，
    并为剪枝保护和增密控制提供决策依据。
    """

    def __init__(self, device, scene_bbox, p_vol_path, alpha=10.0, beta=2.0,
                 default_protection_threshold=0.5, default_densify_threshold=0.5):
        """
        初始化 StructureGuardian (静态 P_vol 版本)。

        Args:
            device (torch.device): 计算设备 (e.g., 'cuda:0')。
            scene_bbox (torch.Tensor): 场景的边界框，形状为 (2, 3)。
            p_vol_path (str): 指向 P_vol.npy 文件的路径。
            alpha (float): 融合 G_mag 和 W 时的梯度权重。
            beta (float): 融合 G_mag 和 W 时的窗口权重。
            default_protection_threshold (float): 缺省的保护阈值。
            default_densify_threshold (float): 缺省的增密阈值。
        """
        self.device = device
        self.scene_bbox = scene_bbox.to(self.device)

        # 默认阈值（供 should_* 在未显式给阈值时使用）
        self.default_protection_threshold = float(default_protection_threshold)
        self.default_densify_threshold = float(default_densify_threshold)

        if not os.path.exists(p_vol_path):
            raise FileNotFoundError(f"结构先验 P_vol.npy 未在以下路径找到: {p_vol_path}")

        # 加载 P_vol 并将其移动到GPU
        p_vol = torch.from_numpy(np.load(p_vol_path)).float().to(self.device)

        # P_vol 形状应为 (D, H, W, 3) -> (Z, Y, X, C)
        if p_vol.dim() != 4 or p_vol.shape[3] != 3:
            raise ValueError(f"P_vol 形状不正确。应为 (D, H, W, 3)，但得到 {p_vol.shape}")

        self.resolution_z, self.resolution_y, self.resolution_x, _ = p_vol.shape

        # 计算体素大小，注意匹配坐标轴 (X, Y, Z)
        self.voxel_size = (self.scene_bbox[1] - self.scene_bbox[0]) / torch.tensor(
            [self.resolution_x, self.resolution_y, self.resolution_z],
            device=self.device, dtype=torch.float32
        )

        # 提取 G_mag (通道 1) 和 W (通道 2)
        g_mag_map = p_vol[..., 1]
        w_map = p_vol[..., 2]

        # 创建静态结构图，使用与初始化器一致的权重
        s_map_raw = alpha * g_mag_map + beta * w_map

        # 归一化到 [0, 1]，以便与阈值一起使用
        min_val = torch.min(s_map_raw)
        max_val = torch.max(s_map_raw)
        if max_val > min_val:
            self.static_structure_map = (s_map_raw - min_val) / (max_val - min_val)
        else:
            self.static_structure_map = torch.zeros_like(s_map_raw)

        print(f"StructureGuardian (Static) initialized with P_vol map. "
              f"Grid shape (ZxYxX): {self.static_structure_map.shape}")

    def _get_structure_values(self, means: torch.Tensor) -> torch.Tensor:
        """
        内部辅助函数：根据世界坐标 (means) 查询 P_vol 结构图的值。
        返回与输入同长度的一维张量，范围 [0, 1]。
        """
        if means.shape[0] == 0:
            return torch.tensor([], device=self.device, dtype=torch.float32)

        with torch.no_grad():
            # 1. 将世界坐标 (X, Y, Z) 转换为体素索引 (idx_x, idx_y, idx_z)
            indices = ((means - self.scene_bbox[0]) / self.voxel_size)

            # 2. P_vol 张量的维度是 (D, H, W) -> (Z, Y, X)
            # 我们需要正确地映射轴
            indices_x = torch.clamp(indices[:, 0].long(), 0, self.resolution_x - 1)
            indices_y = torch.clamp(indices[:, 1].long(), 0, self.resolution_y - 1)
            indices_z = torch.clamp(indices[:, 2].long(), 0, self.resolution_z - 1)

            # 3. 采样
            return self.static_structure_map[indices_z, indices_y, indices_x]

    def query_pvol(self, means: torch.Tensor) -> torch.Tensor:
        """
        连续查询函数：返回结构先验强度分数，范围 [0,1]。
        """
        return self._get_structure_values(means)

    def should_protect(self, means: torch.Tensor, protection_threshold: float = None, threshold: float = None) -> torch.Tensor:
        """
        判断给定位置的高斯核是否应该被保护（防止剪枝）。
        可选参数：
        - protection_threshold: 显式指定保护阈值
        - threshold: 兼容旧接口的别名
        """
        structure_values = self._get_structure_values(means)
        if structure_values.shape[0] == 0:
            return torch.tensor([], device=self.device, dtype=torch.bool)

        thr = self.default_protection_threshold
        if protection_threshold is not None:
            thr = float(protection_threshold)
        if threshold is not None:
            thr = float(threshold)

        return structure_values > thr

    def should_densify(self, means: torch.Tensor, densification_threshold: float = None, threshold: float = None) -> torch.Tensor:
        """
        判断给定位置的高斯核是否允许被增密。
        可选参数：
        - densification_threshold: 显式指定增密阈值
        - threshold: 兼容旧接口的别名
        """
        structure_values = self._get_structure_values(means)
        if structure_values.shape[0] == 0:
            return torch.tensor([], device=self.device, dtype=torch.bool)

        thr = self.default_densify_threshold
        if densification_threshold is not None:
            thr = float(densification_threshold)
        if threshold is not None:
            thr = float(threshold)

        return structure_values > thr

    def get_pvol_score(self, xyz: torch.Tensor):
        """
        返回 (scores, mask):
        - scores: 连续的结构先验强度 [0,1]，用于配额分配/排序
        - mask: 二值掩码，表示“是否属于结构区域”，这里按阈值 0.0 判定（只要有结构即为 True）
        """
        with torch.no_grad():
            scores = self.query_pvol(xyz)              # 连续分数
            mask = self.should_densify(xyz, threshold=0.0)  # 阈值0表示只要有结构就算1
            return scores, mask