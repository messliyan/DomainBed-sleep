# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""Unit tests."""

import itertools
import os
import unittest

import algorithms
import datasets
import hparams_registry
from test import helpers
from parameterized import parameterized


class TestDatasets(unittest.TestCase):

    def setUp(self):
        # 添加DebugSleep数据集到数据集注册表
        if 'DebugSleep' not in datasets.DATASETS:
            datasets.DATASETS = list(datasets.DATASETS) + ['DebugSleep']
        
        # 为DebugSleep创建临时数据集类
        class DebugSleepDatasetClass(datasets.MultipleEnvironmentEEGDataset):
            ENVIRONMENTS = ["env1", "env2"]
            
            def __init__(self, root, test_envs, hparams):
                # 使用helpers中的DebugSleepDataset创建两个环境
                environments = []
                for env_name in self.ENVIRONMENTS:
                    env = helpers.DebugSleepDataset(env_name, num_samples=32)
                    environments.append(env)
                
                self.input_shape = environments[0].input_shape
                self.num_classes = environments[0].num_classes
                super(datasets.MultipleEnvironmentEEGDataset, self).__init__(environments)
        
        # 注册DebugSleep数据集类
        datasets.DATASETS_REGISTRY['DebugSleep'] = DebugSleepDatasetClass
        datasets.DATASETS_NUM_CLASSES['DebugSleep'] = 5
        datasets.DATASETS_INPUT_SHAPE['DebugSleep'] = (1, 3000)
    
    @parameterized.expand(itertools.product(['DebugSleep'] + datasets.DATASETS))
    def test_dataset_erm(self, dataset_name):
        """
        Test that ERM can complete one step on a given dataset without raising
        an error.
        Also test that num_environments() works correctly.
        """
        batch_size = 8
        hparams = hparams_registry.default_hparams('ERM', dataset_name)
        
        # 对于DebugSleep，不使用实际数据目录
        if dataset_name == 'DebugSleep':
            dataset = datasets.get_dataset_class(dataset_name)('', [], hparams)
        elif 'DATA_DIR' in os.environ:
            dataset = datasets.get_dataset_class(dataset_name)(
                os.environ['DATA_DIR'], [], hparams)
        else:
            self.skipTest('需要DATA_DIR环境变量（除DebugSleep外）')
        
        # 检查环境数量
        if dataset_name == 'DebugSleep':
            self.assertEqual(len(dataset), 2)  # DebugSleep有2个环境
        else:
            self.assertEqual(datasets.num_environments(dataset_name), len(dataset))
        
        # 测试算法更新
        algorithm = algorithms.get_algorithm_class('ERM')(
            dataset.input_shape,
            dataset.num_classes,
            len(dataset),
            hparams).cuda()
        
        # 限制batch_size不超过数据集大小
        batch_size = min(batch_size, min(len(env) for env in dataset))
        minibatches = helpers.make_minibatches(dataset, batch_size)
        algorithm.update(minibatches)
