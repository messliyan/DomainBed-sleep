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

    @parameterized.expand(itertools.product(datasets.DATASETS))
    def test_dataset_erm(self, dataset_name):
        """
        Test that ERM can complete one step on a given dataset without raising
        an error.
        Also test that num_environments() works correctly.
        """
        batch_size = 8
        hparams = hparams_registry.default_hparams('ERM', dataset_name)
        
        # 检查是否有DATA_DIR环境变量
        if 'DATA_DIR' in os.environ:
            dataset = datasets.get_dataset_class(dataset_name)(
                os.environ['DATA_DIR'], [], hparams)
        else:
            self.skipTest('需要DATA_DIR环境变量')
        
        # 检查环境数量
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
