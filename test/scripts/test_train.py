# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import os
import subprocess
import unittest
import uuid
import torch

class TestTrain(unittest.TestCase):

    @unittest.skipIf('DATA_DIR' not in os.environ, '需要DATA_DIR环境变量')
    def test_sleep_dataset_end_to_end(self):
        """Test that train.py can run with SleepDataset if DATA_DIR is available"""
        # 这个测试只在DATA_DIR可用时运行
        output_dir = os.path.join('d:\\', str(uuid.uuid4()))
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 使用SleepDataset运行一小步训练
            cmd = [
                'python', '-m', 'scripts.train',
                '--dataset', 'SleepDataset',
                '--data_dir', os.environ['DATA_DIR'],
                '--output_dir', output_dir,
                '--steps', '11',  # 非常少的步骤，仅验证能否运行
                '--algorithm', 'CDANN',
                '--test_envs', '1'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            
            # 只检查脚本是否能正常启动并运行，不检查准确率
            self.assertEqual(result.returncode, 0, "训练脚本执行失败")
            
        finally:
            # 清理：删除临时目录
            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir)
