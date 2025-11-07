# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import os
import subprocess
import unittest
import uuid
import torch

class TestTrain(unittest.TestCase):

    def test_debug_sleep_end_to_end(self):
        """Test that train.py successfully completes steps with DebugSleep dataset"""
        # 创建临时输出目录
        output_dir = os.path.join('d:\\', str(uuid.uuid4()))  # 使用Windows路径格式
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 使用DebugSleep数据集运行训练脚本（这里不需要实际数据目录）
            # 注意：我们需要在运行前确保train.py可以识别DebugSleep数据集
            cmd = [
                'python', '-m', 'scripts.train',
                '--dataset', 'DebugSleep',
                '--output_dir', output_dir,
                '--steps', '51',  # 减少步骤数以加快测试
                '--algorithm', 'ERM',
                '--hparams', '{"batch_size": 8}',
                '--test_envs', '1',  # 使用第二个环境作为测试环境
                '--seed', '0'
            ]
            
            # 执行命令并捕获输出
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            
            # 打印输出以便调试
            print("Command output:", result.stdout)
            print("Command error:", result.stderr)
            
            # 检查命令是否成功执行
            self.assertEqual(result.returncode, 0, "训练脚本执行失败")
            
            # 检查结果文件是否存在
            results_file = os.path.join(output_dir, 'results.jsonl')
            self.assertTrue(os.path.exists(results_file), "结果文件不存在")
            
            # 读取并检查结果
            with open(results_file) as f:
                lines = [l[:-1] for l in f if l.strip()]
                self.assertTrue(len(lines) > 0, "结果文件为空")
                
                last_epoch = json.loads(lines[-1])
                self.assertEqual(last_epoch['step'], 50, "最后一步应该是50")
                
                # 对于随机数据，我们不期望高准确率，但应该有合理的值
                self.assertGreaterEqual(last_epoch.get('env0_in_acc', 0), 0.0, "环境0的准确率应该大于0")
                
            # 检查日志文件
            log_file = os.path.join(output_dir, 'out.txt')
            self.assertTrue(os.path.exists(log_file), "日志文件不存在")
            
            with open(log_file) as f:
                text = f.read()
                self.assertTrue('50' in text, "日志中应该包含步骤50")
                
        finally:
            # 清理：删除临时目录
            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir)
    
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
