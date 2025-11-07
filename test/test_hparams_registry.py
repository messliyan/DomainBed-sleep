# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import unittest

import algorithms
import datasets
import hparams_registry
from parameterized import parameterized


class TestHparamsRegistry(unittest.TestCase):

    def get_algorithms_and_datasets(self):
        """获取要测试的算法和数据集组合"""
        combinations = []
        
        # 所有数据集使用所有算法
        for dataset in datasets.DATASETS:
            for algo in algorithms.ALGORITHMS:
                combinations.append((algo, dataset))
        
        return combinations
    
    @parameterized.expand(get_algorithms_and_datasets)
    def test_random_hparams_deterministic(self, algorithm_name, dataset_name):
        """Test that hparams_registry.random_hparams is deterministic"""
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
