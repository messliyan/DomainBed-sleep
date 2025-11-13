# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


# -------------------------- 从a.py导入的核心网络组件 --------------------------
class SELayer(nn.Module):  # 基础通道注意力
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1)


class SEBasicBlock(nn.Module):  # 残差SE块
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample: residual = self.downsample(x)
        return self.relu(out + residual)


class RCA_Net(nn.Module):  # 残差通道注意力（增强特征判别性）
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim//reduction, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim//reduction, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        residual = x
        attn = self.relu(self.conv1(x))
        attn = self.sigmoid(self.conv2(attn))
        return x * attn + residual


class TemporalAttention1D(nn.Module):  # 时域注意力（聚焦关键时序）
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(dim*2, dim//4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(dim//4, dim, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):  # x: [B, dim, T]
        avg_feat = self.avg_pool(x)  # [B, dim, 1]
        max_feat = self.max_pool(x)  # [B, dim, 1]
        attn = self.fc(torch.cat([avg_feat, max_feat], dim=1))  # [B, dim, 1]
        return x * attn  # 时序加权


class CrossDomainAttention(nn.Module):  # 交叉注意力（源-目标域对齐）
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Conv1d(dim, dim, kernel_size=1)  # 源域→Query
        self.kv_proj = nn.Conv1d(dim, dim*2, kernel_size=1)  # 目标域→Key/Value
        self.out_proj = nn.Conv1d(dim, dim, kernel_size=1)
    def forward(self, src_feat, tgt_feat):  # 源域/目标域特征: [B, dim, T]
        q = self.q_proj(src_feat)  # [B, dim, T_src]
        k, v = self.kv_proj(tgt_feat).chunk(2, dim=1)  # [B, dim, T_tgt]
        
        # 交叉注意力计算：源域关注目标域
        attn_scores = (q.transpose(1,2) @ k.transpose(1,2).transpose(2,3))  # [B, T_src, B, T_tgt]
        attn_scores = attn_scores / math.sqrt(q.shape[1])
        attn_probs = F.softmax(attn_scores.reshape(q.shape[0], q.shape[2], -1), dim=-1)
        attn_probs = attn_probs.reshape_as(attn_scores)
        
        out = (attn_probs @ v.transpose(1,2).transpose(0,1)).transpose(1,2)  # [B, dim, T_src]
        return self.out_proj(out)  # 对齐后的源域特征


class SparseSelfAttention(nn.Module):  # 稀疏自注意力（全局建模）
    def __init__(self, dim, num_heads=5, window_size=10, sparse_rate=0.3):
        super().__init__()
        self.dim = dim
        self.heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.sparse_rate = sparse_rate
        
        self.qkv = nn.Conv1d(dim, dim*3, kernel_size=1)
        self.out_proj = nn.Conv1d(dim, dim, kernel_size=1)
    def forward(self, x):  # x: [B, dim, T]
        B, C, T = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)  # 拆分QKV: [B, dim, T]
        q = q.view(B, self.heads, self.head_dim, T).transpose(2,3)  # [B, heads, T, head_dim]
        k = k.view(B, self.heads, self.head_dim, T).transpose(2,3)
        v = v.view(B, self.heads, self.head_dim, T).transpose(2,3)
        
        # 稀疏掩码：局部窗口+随机采样
        mask = torch.zeros(T, T, device=x.device)
        for i in range(T):  # 局部窗口
            mask[i, max(0, i-self.window_size//2):min(T, i+self.window_size//2+1)] = 1
        mask = mask | (torch.rand(T, T, device=x.device) < self.sparse_rate)  # 随机采样
        mask = mask.masked_fill(mask==0, -1e9)
        
        # 注意力计算
        attn_scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)  # [B, heads, T, T]
        attn_scores += mask[None, None, ...]
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        out = (attn_probs @ v).transpose(2,3).contiguous().view(B, C, T)  # [B, dim, T]
        return self.out_proj(out)


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


class SHHS_MRCN(nn.Module):  # SHHS专属（3750点输入）
    def __init__(self, dim=30):
        super().__init__()
        self.dim = dim
        drate = 0.5
        self.GELU = GELU()
        
        # 多路径卷积
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, padding=24),
            nn.BatchNorm1d(64), self.GELU,
            nn.MaxPool1d(8, 2, 4), nn.Dropout(drate),
            nn.Conv1d(64, 128, 8, 1, 4), nn.BatchNorm1d(128), self.GELU,
            nn.Conv1d(128, 128, 8, 1, 4), nn.BatchNorm1d(128), self.GELU,
            nn.MaxPool1d(4, 4, 2)
        )
        self.features2 = nn.Sequential(  # SHHS专属kernel=6
            nn.Conv1d(1, 64, kernel_size=400, stride=50, padding=200),
            nn.BatchNorm1d(64), self.GELU,
            nn.MaxPool1d(4, 2, 2), nn.Dropout(drate),
            nn.Conv1d(64, 128, 6, 1, 3), nn.BatchNorm1d(128), self.GELU,
            nn.Conv1d(128, 128, 6, 1, 3), nn.BatchNorm1d(128), self.GELU,
            nn.MaxPool1d(2, 2, 1)
        )
        
        # 残差SE+RCA
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, dim, 1)
        self.rca = RCA_Net(dim)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv1d(self.inplanes, planes, 1, stride, bias=False),
            nn.BatchNorm1d(planes)
        ) if (stride!=1 or self.inplanes!=planes) else None
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):  # x: [B, 1, 3750]
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat([x1, x2], dim=2)
        x_concat = self.AFR(x_concat)
        return self.rca(x_concat)  # [B, 30, T_shhs]


class SleepEDF_MRCN(nn.Module):  # Sleep-EDF专属（3000点输入）
    def __init__(self, dim=30):
        super().__init__()
        self.dim = dim
        drate = 0.5
        self.GELU = GELU()
        
        # 多路径卷积
        self.features1 = nn.Sequential(  # 与SHHS共享结构
            nn.Conv1d(1, 64, kernel_size=50, stride=6, padding=24),
            nn.BatchNorm1d(64), self.GELU,
            nn.MaxPool1d(8, 2, 4), nn.Dropout(drate),
            nn.Conv1d(64, 128, 8, 1, 4), nn.BatchNorm1d(128), self.GELU,
            nn.Conv1d(128, 128, 8, 1, 4), nn.BatchNorm1d(128), self.GELU,
            nn.MaxPool1d(4, 4, 2)
        )
        self.features2 = nn.Sequential(  # Sleep-EDF专属kernel=7
            nn.Conv1d(1, 64, kernel_size=400, stride=50, padding=200),
            nn.BatchNorm1d(64), self.GELU,
            nn.MaxPool1d(4, 2, 2), nn.Dropout(drate),
            nn.Conv1d(64, 128, 7, 1, 3), nn.BatchNorm1d(128), self.GELU,
            nn.Conv1d(128, 128, 7, 1, 3), nn.BatchNorm1d(128), self.GELU,
            nn.MaxPool1d(2, 2, 1)
        )
        
        # 残差SE+RCA（参数独立）
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, dim, 1)
        self.rca = RCA_Net(dim)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv1d(self.inplanes, planes, 1, stride, bias=False),
            nn.BatchNorm1d(planes)
        ) if (stride!=1 or self.inplanes!=planes) else None
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):  # x: [B, 1, 3000]
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat([x1, x2], dim=2)
        x_concat = self.AFR(x_concat)
        return self.rca(x_concat)  # [B, 30, T_edf]


class DomainDiscriminator(nn.Module):  # 精简域鉴别器（仅域分类）
    def __init__(self, dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # 二分类（sigmoid）
        )
    def forward(self, x):  # x: 展平的特征
        return self.classifier(x)  # 域分类预测（源域=1/目标域=0）


class DualBranchRobustUDA(nn.Module):
    """双分支鲁棒无监督域自适应模型"""
    n_outputs = 6000  # 固定输出维度：dim*2 * T_unified = 30*2*100 = 6000
    
    def __init__(self, dim=30):
        super().__init__()
        # 超参数
        self.dim = dim  # 分支输出通道数
        
        # 1. 双分支特征提取
        self.shhs_branch = SHHS_MRCN(dim)
        self.edf_branch = SleepEDF_MRCN(dim)
        
        # 2. 时域注意力
        self.temp_attn = TemporalAttention1D(dim)
        
        # 3. 交叉注意力（域对齐）
        self.cross_attn = CrossDomainAttention(dim)
        
        # 4. 稀疏自注意力（全局建模）
        self.sparse_attn = SparseSelfAttention(dim*2, num_heads=5)
        
    def forward(self, x, domain_label=None):
        """
        Args:
            x: 输入信号 [B, 1, T]（T=3750/SHHS 或 3000/Sleep-EDF）
            domain_label: 域标签 [B]（1=SHHS，0=Sleep-EDF）。如果为None，则根据输入长度自动判断
        Returns:
            提取的特征 [B, 6000]
        """
        B = x.shape[0]
        
        # 自动判断域标签（如果未提供）
        if domain_label is None:
            # 根据输入长度判断域：3750点为SHHS，3000点为Sleep-EDF
            domain_label = torch.zeros(B, dtype=torch.int64, device=x.device)
            domain_label[x.shape[2] == 3750] = 1
        
        # 步骤1：双分支特征提取（按域分离）
        shhs_mask = (domain_label == 1)
        edf_mask = (domain_label == 0)
        feat_shhs = self.shhs_branch(x[shhs_mask]) if shhs_mask.any() else None  # [B1, 30, T1]
        feat_edf = self.edf_branch(x[edf_mask]) if edf_mask.any() else None      # [B2, 30, T2]
        
        # 步骤2：时域注意力强化
        if feat_shhs is not None:
            feat_shhs = self.temp_attn(feat_shhs)
        if feat_edf is not None:
            feat_edf = self.temp_attn(feat_edf)
        
        # 步骤3：交叉注意力对齐
        feat_shhs_align = feat_shhs
        feat_edf_align = feat_edf
        if feat_shhs is not None and feat_edf is not None:
            feat_shhs_align = self.cross_attn(feat_shhs, feat_edf)  # 源域对齐目标域
            feat_edf_align = self.cross_attn(feat_edf, feat_shhs)  # 目标域对齐源域
        
        # 步骤4：特征拼接+稀疏自注意力
        T_unified = 100  # 固定时序长度
        feat_list = []
        if feat_shhs_align is not None:
            feat_shhs_align = F.adaptive_avg_pool1d(feat_shhs_align, T_unified)
            feat_list.append(feat_shhs_align)
        if feat_edf_align is not None:
            feat_edf_align = F.adaptive_avg_pool1d(feat_edf_align, T_unified)
            feat_list.append(feat_edf_align)
        fused_feat = torch.cat(feat_list, dim=0)  # [B, 30, 100]
        fused_feat = torch.cat([fused_feat, fused_feat], dim=1)  # [B, 60, 100]
        global_feat = self.sparse_attn(fused_feat)  # [B, 60, 100]
        
        # 展平特征
        global_feat_flat = global_feat.view(B, -1)  # [B, 6000]
        
        return global_feat_flat



def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 2:
        if hparams.get('use_dual_branch', False):
            return DualBranchRobustUDA(dim=hparams.get('dual_branch_dim', 30))
        else:
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

