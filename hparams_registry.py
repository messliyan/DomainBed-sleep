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
    _hparam('lr_g', 5e-4, lambda r: 10 ** r.uniform(-5, -3.5))
    _hparam('lr_d', 1e-4, lambda r: 10 ** r.uniform(-5, -3.5))

    # 1. 域对抗训练算法
    if algorithm in ['DANN', 'CDANN']:  # 域对抗神经网络及其变体
        # 对抗训练基础参数
        #  “每更 1 次生成器，更几次判别器”（默认 0.5）
        _hparam('d_steps_per_g_step', 0.1, lambda r: round(r.uniform(0.2, 0.5), 1))  # 0-1之间的一位小数
        _hparam('warmup_steps', 5000, lambda r: int(r.choice([2000, 3000, 4000, 5000])))  # 1000-5000以1000为间隔
        _hparam('lambda_end', 0.3, lambda r: round(r.uniform(0.5, 1.0), 1))  # 0.5-2之间的一位小数
        
        # 学习率调度器参数
        _hparam('scheduler_step_interval', 100, lambda r: int(10 ** r.uniform(1, 3)))  # 调度器调用间隔步数



        _hparam('beta1', 0.5, lambda r: r.choice([0.5]))
        _hparam('grad_penalty', 1.0, lambda r: 10**r.uniform(-2, 2))  # 梯度惩罚权重
        

        # 判别器网络结构参数
        _hparam('mlp_width', 128, lambda r: int(2 ** r.uniform(5, 9)))  # 判别器MLP宽度
        _hparam('mlp_depth', 2, lambda r: int(r.choice([2, 3, 4])))  # 判别器MLP深度
        _hparam('mlp_dropout', 0.1, lambda r: r.choice([0., 0.1, 0.2, 0.5]))  # 判别器dropout


    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    # 2. 分布匹配算法
    elif algorithm == "MMD" or algorithm == "CORAL":
        _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-1, 1))  # MMD核宽度参数

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))



    # 3. 因果启发算法
    elif algorithm == "CAD" or algorithm == "CondCAD":  # 因果自适应判别器
        _hparam('lmbda', 1e-1, lambda r: r.choice([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]))
        _hparam('temperature', 0.1, lambda r: r.choice([0.05, 0.1]))
        _hparam('is_normalized', False, lambda r: False)
        _hparam('is_project', False, lambda r: False)
        _hparam('is_flipped', True, lambda r: True)

    elif algorithm == "Transfer":
        # 增加对抗损失权重，增强生成器
        _hparam('t_lambda', 2.0, lambda r: 10**r.uniform(-1, 2))
        # 减小投影半径，限制判别器与生成器的差异
        _hparam('delta', 1.0, lambda r: r.uniform(0.5, 2.0))
        # 显著减少判别器更新频率
        _hparam('d_steps_per_g', 3, lambda r: int(r.choice([1, 2, 3])))
        # 增加判别器权重衰减
        _hparam('weight_decay_d', 1e-4, lambda r: 10**r.uniform(-6, -2))
        _hparam('gda', False, lambda r: True)
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        # 降低判别器学习率
        _hparam('lr_d', 1e-4, lambda r: 10**r.uniform(-5.5, -3.5))



    elif algorithm == 'ERMPlusPlus':
        _hparam('linear_lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    elif algorithm == 'URM':
        _hparam('urm', 'adversarial', lambda r: str(r.choice(['adversarial'])))
        
        # 增加对抗损失权重，增强生成器对抗能力
        _hparam('urm_adv_lambda', 0.3, lambda r: float(r.uniform(0.1,0.5)))
        # 添加标签平滑，避免判别器过于确信
        _hparam('urm_discriminator_label_smoothing', 0.1, lambda r: float(r.uniform(0, 0.2)))
        _hparam('urm_discriminator_optimizer', 'adam', lambda r: str(r.choice(['adam'])))
        # 减少隐藏层数量，降低判别器容量
        _hparam('urm_discriminator_hidden_layers', 1, lambda r: int(r.choice([1,2])))
        _hparam('urm_generator_output', 'tanh', lambda r: str(r.choice(['tanh', 'relu'])))
        # 降低判别器学习率，减缓其更新速度
        _hparam('urm_discriminator_lr', 1e-5, lambda r: 10**r.uniform(-7, -4.5))


    if algorithm == "ADRMX":
        _hparam('cnt_lambda', 1.0, lambda r: r.choice([1.0]))
        _hparam('dclf_lambda', 1.0, lambda r: r.choice([1.0]))
        # 增加对抗损失权重，增强生成器
        _hparam('disc_lambda', 1.0, lambda r: r.choice([0.75, 1.0, 1.5]))
        _hparam('rmxd_lambda', 1.0, lambda r: r.choice([1.0]))
        #  “每更 1 次生成器，更几次判别器”（默认 1:1）
        _hparam('d_steps_per_g_step', 1, lambda r: r.choice([1, 2]))
        _hparam('beta1', 0.5, lambda r: r.choice([0.5]))
        # 减小判别器网络宽度
        _hparam('mlp_width', 128, lambda r: r.choice([128, 256]))
        # 减少判别器网络深度
        _hparam('mlp_depth', 6, lambda r: int(r.choice([5, 6, 7])))
        # 增加dropout，减少判别器过拟合
        _hparam('mlp_dropout', 0.1, lambda r: r.choice([0., 0.1, 0.2]))


    # 数据集和算法特定的超参数定义
    # 基础学习率
    if algorithm == "ADRMX":
        _hparam('lr', 3e-5, lambda r: r.choice([2e-5, 3e-5, 4e-5, 5e-5]))
    else:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-5, -3.5))

    _hparam('weight_decay', 0.01, lambda r: 10**r.uniform(-6, -2))

    if algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5)))
    else:
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

