"""超参数注册模块

此模块负责定义和管理所有算法和数据集的超参数。
它提供了一种统一的方式来管理：
1. 默认超参数设置
2. 随机超参数采样策略
3. 算法特定和数据集特定的超参数配置

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
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']

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
    # 数据增强相关
    _hparam('data_augmentation', True, lambda r: True)  # 是否启用数据增强
    # 模型架构相关
    _hparam('resnet18', False, lambda r: False)  # 是否使用ResNet18
    _hparam('resnet50_augmix', True, lambda r: True)  # 是否使用ResNet50+AugMix
    _hparam('dinov2', False, lambda r: False)  # 是否使用DINOv2
    _hparam('vit', False, lambda r: False)  # 是否使用Vision Transformer
    _hparam('vit_attn_tune', False, lambda r: False)  # 是否微调ViT注意力
    _hparam('freeze_bn', False, lambda r: False)  # 是否冻结批归一化层
    # 训练策略相关
    _hparam('lars', False, lambda r: False)  # 是否使用LARS优化器
    _hparam('linear_steps', 500, lambda r: 500)  # 线性预热步数
    # 正则化相关
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))  # ResNet丢弃率
    _hparam('vit_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))  # ViT丢弃率
    # 分类器相关
    _hparam('class_balanced', False, lambda r: False)  # 是否使用类别平衡
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    # 算法特定超参数定义
    # 每个代码块对应一种算法的超参数

    # 1. 域对抗训练算法
    if algorithm in ['DANN', 'CDANN']:  # 域对抗神经网络及其变体
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))  # 域对抗损失权重
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))  # 判别器权重衰减
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))  # 每个生成器步对应的判别器步数
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))  # 梯度惩罚权重
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))  # Adam优化器的beta1参数
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))  # MLP隐藏层宽度
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))  # MLP深度
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))  # MLP丢弃率

    elif algorithm == 'Fish':
        _hparam('meta_lr', 0.5, lambda r:r.choice([0.05, 0.1, 0.5]))

    elif algorithm == "RDM": 
        if dataset in ['DomainNet']: 
            _hparam('rdm_lambda', 0.5, lambda r: r.uniform(0.1, 1.0))
        elif dataset in ['PACS', 'TerraIncognita']:
            _hparam('rdm_lambda', 5.0, lambda r: r.uniform(1.0, 10.0))
        else:
            _hparam('rdm_lambda', 5.0, lambda r: r.uniform(0.1, 10.0))
            
        if dataset == 'DomainNet':
            _hparam('rdm_penalty_anneal_iters', 2400, lambda r: int(r.uniform(1500, 3000)))
        else:
            _hparam('rdm_penalty_anneal_iters', 1500, lambda r: int(r.uniform(800, 2700)))
            
        if dataset in ['TerraIncognita', 'OfficeHome', 'DomainNet']:
            _hparam('variance_weight', 0.0, lambda r: r.choice([0.0]))
        else:
            _hparam('variance_weight', 0.004, lambda r: r.uniform(0.001, 0.007))
            
        _hparam('rdm_lr', 1.5e-5, lambda r: r.uniform(8e-6, 2e-5))

    elif algorithm == "RSC":
        _hparam('rsc_f_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "Mixup":
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "GroupDRO":
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    # 2. 分布匹配算法
    elif algorithm == "MMD" or algorithm == "CORAL" or algorithm == "CausIRL_CORAL" or algorithm == "CausIRL_MMD":
        _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-1, 1))  # MMD核宽度参数

    elif algorithm == "MLDG":
        _hparam('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1))
        _hparam('n_meta_test', 2, lambda r:  r.choice([1, 2]))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10**r.uniform(-1, 5))
        _hparam('vrex_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))



    # 3. 因果启发算法
    elif algorithm == "CAD" or algorithm == "CondCAD":  # 因果自适应判别器
        _hparam('lmbda', 1e-1, lambda r: r.choice([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]))
        _hparam('temperature', 0.1, lambda r: r.choice([0.05, 0.1]))
        _hparam('is_normalized', False, lambda r: False)
        _hparam('is_project', False, lambda r: False)
        _hparam('is_flipped', True, lambda r: True)

    elif algorithm == "Transfer":
        _hparam('t_lambda', 1.0, lambda r: 10**r.uniform(-2, 1))
        _hparam('delta', 2.0, lambda r: r.uniform(0.1, 3.0))
        _hparam('d_steps_per_g', 10, lambda r: int(r.choice([1, 2, 5])))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('gda', False, lambda r: True)
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))

    elif algorithm == 'EQRM':
        _hparam('eqrm_quantile', 0.75, lambda r: r.uniform(0.5, 0.99))
        _hparam('eqrm_burnin_iters', 2500, lambda r: 10 ** r.uniform(2.5, 3.5))
        _hparam('eqrm_lr', 1e-6, lambda r: 10 ** r.uniform(-7, -5))

    elif algorithm == 'ERMPlusPlus':
        _hparam('linear_lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    elif algorithm == 'URM':
        _hparam('urm', 'adversarial', lambda r: str(r.choice(['adversarial']))) # 'adversarial'
        
        _hparam('urm_adv_lambda', 0.1, lambda r: float(r.uniform(0,0.2)))
        _hparam('urm_discriminator_label_smoothing', 0, lambda r: float(r.uniform(0, 0)))
        _hparam('urm_discriminator_optimizer', 'adam', lambda r: str(r.choice(['adam'])))
        _hparam('urm_discriminator_hidden_layers', 1, lambda r: int(r.choice([1,2,3])))
        _hparam('urm_generator_output', 'tanh', lambda r: str(r.choice(['tanh', 'relu'])))
                
        if dataset in SMALL_IMAGES:
            _hparam('urm_discriminator_lr', 1e-3, lambda r: 10**r.uniform(-5.5, -3.5))
        else:
            _hparam('urm_discriminator_lr', 5e-5, lambda r: 10**r.uniform(-6, -4.5))


    if algorithm == "ADRMX":
        _hparam('cnt_lambda', 1.0, lambda r: r.choice([1.0]))
        _hparam('dclf_lambda', 1.0, lambda r: r.choice([1.0]))
        _hparam('disc_lambda', 0.75, lambda r: r.choice([0.75]))
        _hparam('rmxd_lambda', 1.0, lambda r: r.choice([1.0]))
        _hparam('d_steps_per_g_step', 2, lambda r: r.choice([2]))
        _hparam('beta1', 0.5, lambda r: r.choice([0.5]))
        _hparam('mlp_width', 256, lambda r: r.choice([256]))
        _hparam('mlp_depth', 9, lambda r: int(r.choice([8, 9, 10])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0]))


    # 数据集和算法特定的超参数定义
    # 以下代码块对应特定的超参数，根据数据集和算法进行调整
    # 基础学习率
    if dataset in SMALL_IMAGES:
        if algorithm == "ADRMX":
            _hparam('lr', 3e-3, lambda r: r.choice([5e-4, 1e-3, 2e-3, 3e-3]))
        else:
            _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    else:
        if algorithm == "ADRMX":
            _hparam('lr', 3e-5, lambda r: r.choice([2e-5, 3e-5, 4e-5, 5e-5]))
        else:
            _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
    else:
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))

    if dataset in SMALL_IMAGES:
        _hparam('batch_size', 64, lambda r: int(2**r.uniform(3, 9)))
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif algorithm == 'RDM':
        if dataset in ['DomainNet', 'TerraIncognita']:
            _hparam('batch_size', 40, lambda r: int(r.uniform(30, 60)))
        else:
            _hparam('batch_size', 88, lambda r: int(r.uniform(70, 100)))
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5)))
    else:
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5.5)))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2))

    return hparams


def default_hparams(algorithm, dataset):
    """获取指定算法和数据集的默认超参数
    
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
