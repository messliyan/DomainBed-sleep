"""CNNDA网络架构实现

此模块实现了用于单通道脑电睡眠分期的CNNDA模型。CNNDA使用多分辨率CNN进行特征提取，
能够有效捕获脑电信号中的局部多尺度特征。

模型主要组件：
1. 多分辨率CNN (MRCNN)：提取脑电信号的局部多尺度特征
2. 注意力增强模块 (SELayer)：自适应地重新校准特征通道重要性
3. 分类器：将提取的特征映射到睡眠阶段类别

该模型专为单通道脑电信号睡眠分期任务设计，可区分5个睡眠阶段。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# CNNDA模型超参数配置
NUM_CLASSES = 5  # 类别数量，对应5个睡眠阶段
afr_reduced_cnn_size = 30  # CNN输出特征的大小



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


class MRCNN(nn.Module):
    """多分辨率CNN特征提取器
    
    使用两个不同分辨率的卷积路径提取脑电信号的多尺度特征。
    """
    def __init__(self):
        """初始化多分辨率CNN
        """
        super(MRCNN, self).__init__()
        drate = 0.5  # dropout率
        self.GELU = GELU()
        
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
            
        Returns:
            提取的多尺度特征
        """
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        return x_concat


class Featurizer(nn.Module):
    """特征提取器
    
    使用多分辨率CNN从原始脑电信号中提取
    富含判别信息的特征表示。
    """
    def __init__(self):
        """初始化特征提取器
        """
        super(Featurizer, self).__init__()
        # 前端特征提取：多分辨率CNN
        self.mrcnn = MRCNN()
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入脑电信号，形状为 [batch_size, 1, signal_length]
            
        Returns:
            提取的特征表示
        """
        return self.mrcnn(x)


class MLP(nn.Module): 
    """Just an MLP""" 
    def __init__(self, n_inputs, n_outputs, hparams=None): 
        super(MLP, self).__init__()
        # 提供默认超参数，以防未传入
        if hparams is None:
            hparams = {
                'mlp_width': 256,
                'mlp_depth': 3,
                'mlp_dropout': 0.0
            }
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


class Classifier(nn.Module):
    """分类器
    
    使用MLP作为分类器，将特征提取器的输出映射到睡眠阶段类别。
    """
    def __init__(self):
        """初始化分类器
        """
        super(Classifier, self).__init__()
        # 使用MLP作为分类器
        self.mlp = MLP(80 * afr_reduced_cnn_size, NUM_CLASSES)
        self.n_outputs = NUM_CLASSES
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 特征提取器输出的特征
            
        Returns:
            分类器的输出logits
        """
        # 展平特征向量
        x_flat = x.contiguous().view(x.shape[0], -1)
        # 通过MLP输出类别概率分布
        output = self.mlp(x_flat)
        return output


class CNNDA(nn.Module):
    """CNNDA模型
    
    使用多分辨率CNN进行特征提取的脑电睡眠分期模型。
    支持返回特征向量和分类结果，以适应域对抗训练。
    """
    def __init__(self):
        """初始化CNNDA模型
        """
        super(CNNDA, self).__init__()
        # 1. 特征提取器：多分辨率CNN
        self.feature_extractor = Featurizer()
        # 2. 分类器：将特征映射到睡眠阶段类别，使用MLP
        self.classifier = Classifier()
        # 特征维度（用于域判别器）
        self.feature_dim = 80 * afr_reduced_cnn_size
        # 分类器输出维度
        self.n_outputs = self.classifier.n_outputs
    
    def forward(self, x, return_features=False):
        """前向传播
        
        Args:
            x: 输入脑电信号，形状为 [batch_size, 1, signal_length]
            return_features: 是否返回特征向量
            
        Returns:
            - 仅返回分类结果
            - 或返回 (分类结果, 特征向量) 元组
        """
        # 特征提取
        features = self.feature_extractor(x)
        # 将特征展平，适用于域判别器
        features_flat = features.contiguous().view(features.shape[0], -1)
        # 分类决策
        output = self.classifier(features)
        
        if return_features:
            return output, features_flat
        return output
