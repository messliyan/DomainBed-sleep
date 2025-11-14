"""超参数注册模块

此模块负责定义和管理所有算法和数据集的超参数。
它提供了一种统一的方式来管理：
1. 默认超参数设置
2. 随机超参数采样策略
3. 算法特定和数据集特定的超参数配置

现在的工作流程是：
- train.py 只使用默认超参数或用户传入的超参数
- sweep.py 负责生成随机超参数并传递给train.py

超参数通过全局注册表进行管理，每个算法和数据集组合都有其对应的
默认值和随机采样函数。这使得实验的配置和复现变得更加容易。
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from lib import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    """定义单个超参数并添加到注册表中
    
    Args:
        hparams: 超参数字典
        hparam_name: 超参数名称
        default_val: 超参数默认值
        random_val_fn: 随机值生成函数
    """
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """获取指定算法和数据集的超参数配置
    
    此函数是超参数管理的核心，为每个算法和数据集组合定义默认超参数和
    随机采样策略。超参数以（默认值，随机值）元组的形式存储。
    
    Args:
        algorithm: 算法名称
        dataset: 数据集名称
        random_seed: 随机种子，用于生成可复现的随机超参数
        
    Returns:
        包含所有超参数配置的字典
    """
    # 小图像数据集列表，用于特殊处理
    # SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']
    
    # 初始化超参数字典
    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """内部函数：定义单个超参数
        
        Args:
            name: 超参数名称
            default_val: 默认值
            random_val_fn: 接受RandomState对象并返回随机值的函数
        """
        # 确保超参数名称唯一
        assert(name not in hparams)
        # 创建基于种子和名称的随机状态，确保可复现性
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        # 存储（默认值，随机值）元组
        hparams[name] = (default_val, random_val_fn(random_state))

    # 无条件超参数定义 - 适用于所有算法和数据集
    
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    # 算法特定超参数定义
    # 每个代码块对应一种算法的超参数
    _hparam('lr_g', 1e-3, lambda r: 10 ** r.uniform(-5, -3.5))
    _hparam('lr_d', 1e-4, lambda r: 10 ** r.uniform(-5, -3.5))

    # 1. 域对抗训练算法
    if algorithm in ['DANN', 'CDANN']:  # 域对抗神经网络及其变体
        # 对抗训练基础参数
        #  “每更 1 次生成器，更几次判别器”（默认 0.5）
        _hparam('d_steps_per_g_step', 0.1, lambda r: round(r.uniform(0.2, 0.5), 1))  # 0-1之间的一位小数
        _hparam('warmup_steps', 4000, lambda r: int(r.choice([2000, 3000, 4000, 5000])))  # 1000-5000以1000为间隔
        _hparam('lambda_end', 0.4, lambda r: round(r.uniform(0.5, 1.0), 1))  # 0.5-2之间的一位小数
        
        # 双分支模型参数
        _hparam('use_dual_branch', True, lambda r: r.choice([True, False]))  # 是否使用双分支模型
        _hparam('dual_branch_dim', 30, lambda r: int(r.choice([15, 30, 60])))  # 双分支模型维度
        
        # 学习率调度器参数
        _hparam('scheduler_step_interval', 150, lambda r: int(10 ** r.uniform(1, 3)))  # 调度器调用间隔步数
        _hparam('source_scheduler_patience', 10, lambda r: int(r.choice([5, 10, 15])))  # 源域调度器耐心值
        _hparam('source_scheduler_factor', 0.5, lambda r: r.choice([0.3, 0.5, 0.7]))  # 源域调度器学习率衰减因子
        _hparam('target_scheduler_patience', 2, lambda r: int(r.choice([1, 2, 3])))  # 目标域调度器耐心值
        _hparam('target_scheduler_factor', 0.2, lambda r: r.choice([0.1, 0.2, 0.3]))  # 目标域调度器学习率衰减因子
        _hparam('min_lr', 1e-6, lambda r: 10 ** r.uniform(-7, -5))  # 最低学习率限制
        
        # DANN特定参0数
        _hparam('max_lambda', 0.3, lambda r: r.uniform(0.1, 0.5))  # 最大对抗强度权重
        _hparam('disc_loss_floor', 0.7, lambda r: r.uniform(0.3, 0.7))  # 判别器损失下限
        _hparam('disc_loss_critical_threshold', 0.1, lambda r: r.uniform(0.05, 0.2))  # 判别器损失临界阈值
        _hparam('disc_loss_history_length', 150, lambda r: int(10 ** r.uniform(1, 3)))  # 判别器损失历史长度
        _hparam('disc_grad_clip', 1.0, lambda r: r.uniform(0.5, 2.0))  # 判别器梯度裁剪值
        
        _hparam('beta1', 0.5, lambda r: r.choice([0.5]))
        _hparam('grad_penalty', 3.0, lambda r: 10**r.uniform(-2, 2))  # 梯度惩罚权重
        

        # 判别器网络结构参数
        _hparam('mlp_width', 128, lambda r: int(2 ** r.uniform(5, 9)))  # 判别器MLP宽度
        _hparam('mlp_depth', 2, lambda r: int(r.choice([2, 3, 4])))  # 判别器MLP深度
        _hparam('mlp_dropout', 0.3, lambda r: r.choice([0., 0.1, 0.2, 0.5]))  # 判别器dropout


    # 数据集和算法特定的超参数定义
    # 基础学习率
    _hparam('lr', 1e-3, lambda r: 10**r.uniform(-5, -3.5))

    _hparam('weight_decay', 0.01, lambda r: 10**r.uniform(-6, -2))


    _hparam('batch_size', 128, lambda r: int(2**r.uniform(3, 5.5)))



    return hparams


def default_hparams(algorithm, dataset):
    """获取指定算法和数据集的默认超参数
    
    这个函数现在只被train.py调用，用于获取默认超参数。
    
    Args:
        algorithm: 算法名称
        dataset: 数据集名称
        
    Returns:
        只包含默认超参数值的字典
    """
    # 从_hparams函数获取默认值，忽略随机值
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    """获取指定算法和数据集的随机超参数
    
    这个函数现在只被sweep.py调用，用于生成随机超参数，然后传递给train.py。
    基于给定的随机种子生成一组随机超参数，确保可复现性。
    
    Args:
        algorithm: 算法名称
        dataset: 数据集名称
        seed: 随机种子
        
    Returns:
        只包含随机超参数值的字典
    """
    # 从_hparams函数获取随机值，忽略默认值
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}

