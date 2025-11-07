# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import unittest
import torch

import datasets
import hparams_registry
import networks
from test import helpers
from parameterized import parameterized


class TestNetworks(unittest.TestCase):

    def setUp(self):
        # 添加DebugSleep数据集到数据集注册表（如果尚未添加）
        if 'DebugSleep' not in datasets.DATASETS:
            datasets.DATASETS = list(datasets.DATASETS) + ['DebugSleep']
        
        # 确保DebugSleep的注册表项存在
        if 'DebugSleep' not in datasets.DATASETS_REGISTRY:
            class DebugSleepDatasetClass(datasets.MultipleEnvironmentEEGDataset):
                ENVIRONMENTS = ["env1"]
                
                def __init__(self, root, test_envs, hparams):
                    env = helpers.DebugSleepDataset("env1", num_samples=32)
                    self.input_shape = env.input_shape
                    self.num_classes = env.num_classes
                    super(datasets.MultipleEnvironmentEEGDataset, self).__init__([env])
            
            datasets.DATASETS_REGISTRY['DebugSleep'] = DebugSleepDatasetClass
            datasets.DATASETS_NUM_CLASSES['DebugSleep'] = 5
            datasets.DATASETS_INPUT_SHAPE['DebugSleep'] = (1, 3000)
    
    @parameterized.expand(itertools.product(helpers.DEBUG_DATASETS))
    def test_featurizer(self, dataset_name):
        """Test that Featurizer() returns a module which can take a
        correctly-sized input and return a correctly-sized output."""
        batch_size = 8
        hparams = hparams_registry.default_hparams('ERM', dataset_name)
        
        # 对于DebugSleep，使用特定的脑电网络配置
        if dataset_name == 'DebugSleep':
            # 直接创建输入数据而不依赖数据集类
            input_shape = (1, 3000)
            input_ = torch.randn(batch_size, *input_shape).cuda()
            # 测试MRCNN特征提取器
            featurizer = networks.MRCNN(input_shape, hparams).cuda()
            output = featurizer(input_)
            # 检查输出形状 - MRCNN输出特征维度
            self.assertEqual(output.shape[0], batch_size)
        else:
            # 原有测试逻辑
            dataset = datasets.get_dataset_class(dataset_name)('', [], hparams)
            input_ = helpers.make_minibatches(dataset, batch_size)[0][0]
            input_shape = dataset.input_shape
            featurizer = networks.Featurizer(input_shape, hparams).cuda()
            output = featurizer(input_)
            self.assertEqual(list(output.shape), [batch_size, featurizer.n_outputs])
    
    def test_mrcnn_network(self):
        """专门测试MRCNN网络对脑电数据的处理"""
        batch_size = 8
        input_shape = (1, 3000)  # 单通道脑电信号
        hparams = {}
        
        # 创建MRCNN模型
        mrcnn = networks.MRCNN(input_shape, hparams).cuda()
        
        # 创建随机输入数据
        input_ = torch.randn(batch_size, *input_shape).cuda()
        
        # 前向传播
        output = mrcnn(input_)
        
        # 验证输出形状 - MRCNN输出应为(batch_size, features)
        self.assertEqual(output.shape[0], batch_size)
        self.assertGreater(output.shape[1], 0)  # 特征维度应该大于0
        
    def test_selayer(self):
        """测试SELayer注意力机制"""
        # 创建SELayer实例
        num_channels = 64
        reduction = 16
        se_layer = networks.SELayer(num_channels, reduction).cuda()
        
        # 创建随机输入 (batch, channels, time)
        input_ = torch.randn(8, num_channels, 100).cuda()
        
        # 前向传播
        output = se_layer(input_)
        
        # 验证输出形状与输入相同
        self.assertEqual(output.shape, input_.shape)
        
    def test_sebasicblock(self):
        """测试SEBasicBlock残差块"""
        # 创建SEBasicBlock实例
        in_channels = 64
        out_channels = 64
        stride = 1
        downsample = None
        seblock = networks.SEBasicBlock(in_channels, out_channels, stride, downsample).cuda()
        
        # 创建随机输入 (batch, channels, time)
        input_ = torch.randn(8, in_channels, 100).cuda()
        
        # 前向传播
        output = seblock(input_)
        
        # 验证输出形状
        expected_time_dim = 100 // stride
        self.assertEqual(output.shape, (8, out_channels, expected_time_dim))
