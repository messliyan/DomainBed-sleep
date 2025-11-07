"""命令启动器模块

此模块提供了不同的命令启动器实现，用于在各种计算环境中执行命令列表。
主要功能包括：
1. 本地串行执行命令
2. 模拟执行命令（仅打印）
3. 多GPU并行执行命令

通过实现自定义启动器，可以扩展支持各种集群环境。每个启动器都遵循相同的
接口：接受命令列表作为输入并执行它们。
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import subprocess
import time

import torch


def local_launcher(commands):
    """在本地机器上串行执行命令列表
    
    此启动器按顺序执行每个命令，等待一个命令完成后再执行下一个。
    适用于简单任务或调试场景。
    
    Args:
        commands: 命令字符串列表，每个字符串代表一个完整的shell命令
    """
    for cmd in commands:
        # 使用subprocess.call执行命令，shell=True表示在shell中执行
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """模拟执行命令（仅打印，不实际执行）
    
    此启动器不会执行任何命令，仅打印将要执行的命令。
    主要用于测试和验证命令生成逻辑是否正确。
    
    Args:
        commands: 命令字符串列表，每个字符串代表一个完整的shell命令
    """
    for cmd in commands:
        # 仅打印命令，不实际执行
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    """在本地机器上并行使用所有GPU执行命令
    
    此启动器会自动检测可用的GPU，并在多个GPU上并行执行命令。
    它维护每个GPU上的进程状态，并在GPU空闲时分配新命令。
    
    Args:
        commands: 命令字符串列表，每个字符串代表一个完整的shell命令
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    
    # 获取可用GPU列表
    try:
        # 从环境变量获取GPU列表，处理可能存在的额外逗号情况
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # 如果环境变量未设置，则使用所有可用GPU
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    
    n_gpus = len(available_gpus)
    # 用于跟踪每个GPU上运行的进程
    procs_by_gpu = [None]*n_gpus

    # 主循环：持续分配命令直到所有命令都被执行
    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            # 检查GPU是否空闲（进程为None或已完成）
            if (proc is None) or (proc.poll() is not None):
                # GPU空闲，分配下一个命令
                cmd = commands.pop(0)
                # 设置CUDA_VISIBLE_DEVICES环境变量限制只使用特定GPU
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break  # 分配一个命令后等待下一次循环
        # 短暂休眠避免CPU过度使用
        time.sleep(1)

    # 等待所有剩余任务完成后再返回
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

# 启动器注册表，用于通过名称查找和使用启动器
REGISTRY = {
    'local': local_launcher,      # 本地串行执行
    'dummy': dummy_launcher,      # 仅打印不执行
    'multi_gpu': multi_gpu_launcher  # 多GPU并行执行
}

# 尝试导入并注册Facebook特定的启动器
try:
    import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    # 如果无法导入Facebook模块，则忽略
    pass
