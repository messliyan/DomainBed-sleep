import sys
sys.path.insert(0, '.')

# 测试DebugSleep数据集是否能正确注册和使用
from test.helpers import DebugSleepDataset
import torch

print("测试DebugSleepDataset创建...")
dataset = DebugSleepDataset(env_name='env0')
print(f"数据集大小: {len(dataset)}")
sample, label = dataset[0]
print(f"样本形状: {sample.shape}, 标签: {label}")

print("\n测试成功! DebugSleep数据集能正常工作。")
