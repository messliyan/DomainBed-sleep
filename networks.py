# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPoolingDownsample(nn.Module):
    """双池化降采样模块
    
    使用最大池化和平均池化的组合，通过1×1卷积融合，将长信号降采样到固定长度。
    适用于将125Hz的脑电信号(3750点)降采样到100Hz等效长度(3000点)。
    """
    def __init__(self, output_length=3000, in_channels=1, out_channels=1):
        super().__init__()
        # 双自适应池化：分别输出output_length点
        self.max_pool = nn.AdaptiveMaxPool1d(output_length)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_length)
        # 1×1卷积（一维）：融合2个通道为目标通道数
        self.conv1x1 = nn.Conv1d(
            in_channels=in_channels*2,  # 拼接后通道数：1×2=2
            out_channels=out_channels,
            kernel_size=1  # 1×1卷积，仅作用于通道维度
        )
    
    def forward(self, x):
        # x形状：(batch, channels, length) → 如(2, 1, 3750)
        max_out = self.max_pool(x)  # (2, 1, 3000)
        avg_out = self.avg_pool(x)  # (2, 1, 3000)
        # 通道维度拼接：(2, 2, 3000)
        fused = torch.cat([max_out, avg_out], dim=1)
        # 1×1卷积融合：(2, 1, 3000)
        output = self.conv1x1(fused)
        return output


class GELU(nn.Module):
    """高斯误差线性单元激活函数

    为较旧版本的PyTorch提供GELU实现。新版本可以直接使用nn.GELU()。
    """

    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        """前向传播

        Args:
            x: 输入张量

        Returns:
            应用GELU后的输出
        """
        x = torch.nn.functional.gelu(x)
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        x = self.activation(x) # for URM; does not affect other algorithms
        return x


class MRCNN(nn.Module):
    """多分辨率CNN特征提取器

    使用两个不同分辨率的卷积路径提取脑电信号的多尺度特征。
    首先对长信号进行降采样，确保所有输入标准化到相同长度。
    """
    n_outputs = 10240  # 更新为实际计算的输出维度
    
    def __init__(self):
        """初始化多分辨率CNN
        """
        super(MRCNN, self).__init__()
        drate = 0.5  # dropout率
        self.GELU = GELU()
        
        # 添加双池化降采样模块，将125Hz数据(3750点)降采样到3000点
        self.downsampler = DualPoolingDownsample(output_length=3000)

        # 路径1：小卷积核，提取局部细粒度特征
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        # 路径2：大卷积核，提取全局粗粒度特征
        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

    def forward(self, x):
        """前向传播

        Args:
            x: 输入脑电信号，形状为 [batch_size, 1, signal_length]
               支持不同长度的输入，如100Hz的3000点或125Hz的3750点

        Returns:
            提取的多尺度特征，形状统一为 [batch_size, 10240]
        """
        # 首先对输入信号进行降采样，确保所有输入都是3000点长度
        # 对于100Hz数据(3000点)，这一步不会改变信号
        # 对于125Hz数据(3750点)，会降采样到3000点
        x = self.downsampler(x)
        
        # 通过两个路径提取特征
        x1 = self.features1(x)
        x2 = self.features2(x)
        
        # 拼接两条路径的特征
        x_concat = torch.cat((x1, x2), dim=2)
        
        # 展平特征
        x_view = x_concat.view(len(x_concat), -1)
        
        # 确保输出维度正确
        assert x_view.size(1) == self.n_outputs, f"输出维度错误: {x_view.size(1)} != {self.n_outputs}"
        return x_view



def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 2:
        return MRCNN();
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层
    
    在反向传播时将梯度乘以负系数，用于域对抗训练。
    这是DANN算法的核心组件，允许特征提取器学习域不变的特征表示。
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        """前向传播
        
        简单地传递输入，同时保存alpha参数用于反向传播。
        
        Args:
            ctx: 上下文对象，用于保存反向传播所需的信息
            x: 输入特征
            alpha: 梯度反转系数，控制对抗训练的强度
            
        Returns:
            与输入相同的特征
        """
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播
        
        将梯度乘以负的alpha系数，实现梯度反转。
        
        Args:
            ctx: 上下文对象，包含保存的alpha参数
            grad_output: 来自后续层的梯度
            
        Returns:
            反转后的梯度和None（因为alpha不需要梯度）
        """
        output = grad_output.neg() * ctx.alpha
        return output, None

