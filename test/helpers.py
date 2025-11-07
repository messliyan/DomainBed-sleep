# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import numpy as np
from torch.utils.data import Dataset

DEBUG_DATASETS = ['Debug28', 'Debug224', 'DebugSleep']

class DebugSleepDataset(Dataset):
    """用于测试的脑电睡眠数据集"""
    def __init__(self, env_name, num_samples=32):
        super().__init__()
        self.env_name = env_name
        self.num_samples = num_samples
        self.input_shape = (1, 3000)  # 单通道脑电信号，30秒@100Hz采样率
        self.num_classes = 5  # 5个睡眠阶段
        
        # 生成随机的脑电数据样本
        self.data = torch.randn(num_samples, 1, 3000)
        self.labels = torch.randint(0, 5, (num_samples,))
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def __len__(self):
        return self.num_samples

def make_minibatches(dataset, batch_size):
    """Test helper to make a minibatches array like train.py"""
    minibatches = []
    for env in dataset:
        X = torch.stack([env[i][0] for i in range(batch_size)]).cuda()
        y = torch.stack([torch.as_tensor(env[i][1])
            for i in range(batch_size)]).cuda()
        minibatches.append((X, y))
    return minibatches
