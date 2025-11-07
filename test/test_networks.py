# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
import torch

import networks


class TestNetworks(unittest.TestCase):
    
    def test_mrcnn_network(self):
        """专门测试MRCNN网络对脑电数据的处理"""
        batch_size = 8
        input_shape = (1, 3000)  # 单通道脑电信号
        
        # 创建MRCNN模型
        mrcnn = networks.MRCNN().cuda()
        
        # 创建随机输入数据
        input_ = torch.randn(batch_size, *input_shape).cuda()
        
        # 前向传播
        output = mrcnn(input_)
        
        # 验证输出形状
        self.assertEqual(output.shape[0], batch_size)
        self.assertGreater(output.shape[1], 0)  # 特征维度应该大于0
    
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
        

        

