import numpy as np
import torch
from torch.utils.data import TensorDataset
from .base import MultipleEnvironmentDataset


class EEGSleepDataset(MultipleEnvironmentDataset):
    """单通道脑电睡眠分期数据集（多域版本）"""

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.num_classes = 5  # 睡眠分期：Wake/N1/N2/N3/REM
        self.input_shape = (1, hparams["seq_len"])  # 单通道时序数据
        self.environments = self._load_environments(root, test_envs)

    def _load_environments(self, root, test_envs):
        """加载多个域数据（每个域对应不同被试/设备）"""
        envs = []
        for env_idx in range(hparams["num_domains"]):
            # 加载预处理后的脑电数据（参考DaSleep的.npy格式）
            data = np.load(f"{root}/domain{env_idx}_data.npy")  # 形状：(N, 1, seq_len)
            labels = np.load(f"{root}/domain{env_idx}_labels.npy")  # 0-4的整数标签

            # 转换为Tensor并划分训练/测试域
            is_test = env_idx in test_envs
            dataset = TensorDataset(
                torch.FloatTensor(data),
                torch.LongTensor(labels)
            )
            envs.append((dataset, "test" if is_test else "train"))
        return envs