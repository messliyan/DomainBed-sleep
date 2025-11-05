import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGFeaturizer(nn.Module):
    """单通道脑电特征提取器"""
    n_outputs = 128  # 输出特征维度

    def __init__(self, input_shape, hparams):
        super().__init__()
        seq_len = input_shape[1]  # 时序长度（如3000采样点）
        self.conv1 = nn.Conv1d(1, 32, kernel_size=31, stride=1, padding=15)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=7)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3)

        # 批归一化
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        # 池化后特征长度计算
        self.pool = nn.MaxPool1d(2)
        self.final_len = seq_len // (2 ** 3)  # 三次池化后长度

    def forward(self, x):
        # x shape: (batch, 1, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # 展平特征
        x = x.view(x.shape[0], -1)  # (batch, 128 * final_len)
        return x


class EEGClassifier(nn.Module):
    """睡眠分期分类器"""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)