# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import unittest

import algorithms
import datasets
import hparams_registry
from parameterized import parameterized


class TestHparamsRegistry(unittest.TestCase):

    def setUp(self):
        # 添加DebugSleep数据集到数据集注册表（如果尚未添加）
        if 'DebugSleep' not in datasets.DATASETS:
            datasets.DATASETS = list(datasets.DATASETS) + ['DebugSleep']
    
    def get_algorithms_and_datasets(self):
        """获取要测试的算法和数据集组合"""
        # 对于DebugSleep，我们只测试一部分核心算法
        core_algorithms = ['ERM', 'CDANN', 'DANN']
        combinations = []
        
        # 标准数据集使用所有算法
        for dataset in datasets.DATASETS:
            if dataset == 'DebugSleep':
                # DebugSleep只使用核心算法
                for algo in core_algorithms:
                    if algo in algorithms.ALGORITHMS:
                        combinations.append((algo, dataset))
            else:
                # 其他数据集使用所有算法
                for algo in algorithms.ALGORITHMS:
                    combinations.append((algo, dataset))
        
        return combinations
    
    @parameterized.expand(get_algorithms_and_datasets)
    def test_random_hparams_deterministic(self, algorithm_name, dataset_name):
        """Test that hparams_registry.random_hparams is deterministic"""
        # 为DebugSleep添加一个默认超参数集，如果不存在的话
        if dataset_name == 'DebugSleep' and not any(algo_name == algorithm_name and ds_name == dataset_name 
                                                  for algo_name, ds_name in hparams_registry.ALGORITHM_HPAMAMS.keys()):
            # 使用ERM的默认超参数作为基础
            hparams_registry.ALGORITHM_HPAMAMS[(algorithm_name, dataset_name)] = hparams_registry.ALGORITHM_HPAMAMS.get((algorithm_name, 'SleepDataset'), 
                                                                                                                  hparams_registry.ALGORITHM_HPAMAMS.get(('ERM', 'SleepDataset'), {}))
        
        a = hparams_registry.random_hparams(algorithm_name, dataset_name, 0)
        b = hparams_registry.random_hparams(algorithm_name, dataset_name, 0)
        self.assertEqual(a.keys(), b.keys())
        for key in a.keys():
            self.assertEqual(a[key], b[key], key)
    
    def test_default_hparams_for_sleep(self):
        """测试SleepDataset的默认超参数"""
        if 'SleepDataset' in datasets.DATASETS:
            # 测试主要算法的默认超参数
            for algo in ['ERM', 'CDANN', 'DANN']:
                if algo in algorithms.ALGORITHMS:
                    hparams = hparams_registry.default_hparams(algo, 'SleepDataset')
                    self.assertIsInstance(hparams, dict)
                    self.assertTrue(len(hparams) > 0)
