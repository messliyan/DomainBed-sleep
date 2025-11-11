"""域泛化算法实现模块

此模块实现了多种域泛化(Domain Generalization)和域适应(Domain Adaptation)算法。
域泛化是机器学习中的一个重要研究方向，旨在训练一个能够在未见过的数据分布(域)上表现良好的模型。

主要包含以下类型的算法：
1. 经验风险最小化(ERM)及其变种
2. 不变性学习方法(如IRM, VREx)
3. 对抗学习方法(如DANN, CDANN)
4. 距离度量方法(如MMD, CORAL)
5. 多任务学习方法(如MTL)
6. 其他先进算法(如GroupDRO, Mixup, Fish等)
"""
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import networks

# 尝试导入backpack库，用于计算批量梯度
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

# 导入网络模块和工具函数
from lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, ErmPlusPlusMovingAvg, l2_between_dicts, proj, Nonparametric,
    LARS, SupConLossLambda
)


# 所有可用算法的列表
ALGORITHMS = [
    'ERM',             # 经验风险最小化（基准方法）
    'ERMPlusPlus',     # 改进版经验风险最小化
    'CORAL',           # 相关性对齐（UDA方法）
    'MMD',             # 最大均值差异（UDA方法）
    'DANN',            # 域对抗神经网络（UDA方法）
    'CDANN',           # 条件域对抗神经网络（UDA方法）
    'MTL',             # 多任务学习
    'SagNet',          # 风格对抗生成网络
    'ARM',             # 自适应风险最小化
    'CAD',             # 对比域适应（UDA方法）
    'CondCAD',         # 条件对比域适应（UDA方法）
    'Transfer',        # 迁移学习（UDA方法）
    'ADRMX',           # 自适应域正则化混合（UDA方法）
    'URM',             # 均匀风险最小化
]


def get_algorithm_class(algorithm_name):
    """获取指定名称的算法类
    
    根据算法名称返回对应的算法类实现。
    
    Args:
        algorithm_name: 算法名称字符串
        
    Returns:
        对应的算法类
        
    Raises:
        NotImplementedError: 如果指定的算法名称不存在
    """
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """算法抽象基类
    
    所有域泛化算法的基类，定义了算法的通用接口。子类需要实现具体的更新和预测方法。
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """执行一步更新
        
        实现完整的DANN更新逻辑，包括：
        1. 从源域数据学习分类任务
        2. 使用域对抗训练学习域不变特征
        3. 利用目标域无标签数据进行域适应
        
        Args:
            minibatches: (x, y)元组列表，来自源域
            unlabeled: 来自目标域的无标签数据列表
            
        Returns:
            包含各种损失值的字典
        """
        """执行一步更新
        
        给定所有环境的(x, y)元组列表，执行一次参数更新。
        当任务是域适应时，还可以接受来自测试域的无标签小批量数据。
        
        Args:
            minibatches: (x, y)元组列表，每个元组代表一个环境的小批量数据
            unlabeled: 可选，来自测试域的无标签小批量数据列表
            
        Returns:
            包含训练信息的字典，通常包含损失值等
        """
        raise NotImplementedError

    def predict(self, x):
        """预测输入数据的标签
        
        Args:
            x: 输入数据
            
        Returns:
            预测的标签或概率分布
        """
        raise NotImplementedError


class ERM(Algorithm):
    """经验风险最小化算法
    
    最基本的机器学习算法，直接最小化所有训练数据的经验风险。
    在域泛化场景中，它通常作为基准算法使用。
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化ERM算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # 创建特征提取器
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # 创建分类器
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # 构建完整网络
        self.network = nn.Sequential(self.featurizer, self.classifier)
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        """执行一步更新
        
        将所有环境的小批量数据合并，计算交叉熵损失并更新参数。
        
        Args:
            minibatches: (x, y)元组列表
            unlabeled: 未使用
            
        Returns:
            包含损失值的字典
        """
        # 合并所有环境的输入和标签
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        # 计算交叉熵损失
        loss = F.cross_entropy(self.predict(all_x), all_y)

        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        """预测输入数据的标签
        
        Args:
            x: 输入数据
            
        Returns:
            预测的对数概率
        """
        return self.network(x)


class ERMPlusPlus(Algorithm, ErmPlusPlusMovingAvg):
    """改进版经验风险最小化算法
    
    ERM++在标准ERM的基础上增加了一些改进，如：
    1. 支持LARS优化器
    2. 对线性层和特征提取器使用不同的学习率
    3. 实现了学习率调度
    4. 支持网络参数的移动平均
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化ERM++算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        # 创建特征提取器
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # 创建分类器
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # 构建完整网络
        self.network = nn.Sequential(self.featurizer, self.classifier)
        
        # 根据超参数选择优化器类型
        if self.hparams["lars"]:
            self.optimizer = LARS(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )

        # 收集分类器的参数
        linear_parameters = []
        for n, p in self.network[1].named_parameters():
            linear_parameters.append(p)

        # 为分类器创建单独的优化器
        if self.hparams["lars"]:
            self.linear_optimizer = LARS(
                linear_parameters,
                lr=self.hparams["linear_lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )
        else:
            self.linear_optimizer = torch.optim.Adam(
                linear_parameters,
                lr=self.hparams["linear_lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )
        
        # 初始化学习率调度相关变量
        self.lr_schedule = []
        self.lr_schedule_changes = 0
        # 创建学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=1)
        # 初始化移动平均
        ErmPlusPlusMovingAvg.__init__(self, self.network)

    def update(self, minibatches, unlabeled=None):
        """执行一步更新
        
        根据训练步数选择使用主优化器或线性优化器，并更新移动平均网络。
        
        Args:
            minibatches: (x, y)元组列表
            unlabeled: 未使用
            
        Returns:
            包含损失值的字典
        """
        # 根据训练步数选择优化器
        if self.global_iter > self.hparams["linear_steps"]:
            selected_optimizer = self.optimizer
        else:
            selected_optimizer = self.linear_optimizer

        # 合并所有环境的输入和标签
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        # 计算交叉熵损失
        loss = F.cross_entropy(self.network(all_x), all_y)

        # 梯度更新
        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        # 更新移动平均
        self.update_sma()
        # 如果不冻结BN层，更新移动平均网络的BN统计量
        if not self.hparams["freeze_bn"]:
            self.network_sma.train()
            self.network_sma(all_x)

        return {'loss': loss.item()}

    def predict(self, x):
        """预测输入数据的标签
        
        使用移动平均网络进行预测。
        
        Args:
            x: 输入数据
            
        Returns:
            预测的对数概率
        """
        self.network_sma.eval()
        return self.network_sma(x)

    def set_lr(self, eval_loaders_iid=None, schedule=None, device=None):
        """设置学习率
        
        根据验证损失调整学习率，或使用给定的学习率调度方案。
        
        Args:
            eval_loaders_iid: 验证数据加载器列表
            schedule: 可选的学习率调度方案
            device: 计算设备
            
        Returns:
            更新后的学习率调度方案
        """
        with torch.no_grad():
            # 只在线性训练步数后调整学习率
            if self.global_iter > self.hparams["linear_steps"]:
                if schedule is None:
                    # 计算验证损失
                    self.network_sma.eval()
                    val_losses = []
                    for loader in eval_loaders_iid:
                        loss = 0.0
                        for x, y in loader:
                            x = x.to(device)
                            y = y.to(device)
                            loss += F.cross_entropy(self.network_sma(x), y)
                        val_losses.append(loss / len(loader))
                    val_loss = torch.mean(torch.stack(val_losses))
                    # 更新学习率
                    self.scheduler.step(val_loss)
                    self.lr_schedule.append(self.scheduler._last_lr)
                    # 记录学习率变化次数
                    if len(self.lr_schedule) > 1:
                        if self.lr_schedule[-1] != self.lr_schedule[-2]:
                            self.lr_schedule_changes += 1
                    # 如果学习率变化超过3次，将学习率设为0
                    if self.lr_schedule_changes == 3:
                        self.lr_schedule[-1] = [0.0]
                    return self.lr_schedule
                else:
                    # 使用给定的学习率调度方案
                    self.optimizer.param_groups[0]['lr'] = (torch.Tensor(schedule[0]).requires_grad_(False))[0]
                    schedule = schedule[1:]
            return schedule

class URM(ERM):
    """
    均匀风险最小化算法实现
    来自论文: Uniformly Distributed Feature Representations for Fair and Robust Learning (TMLR 2024)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化URM算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        # 初始化基础ERM组件
        ERM.__init__(self, input_shape, num_classes, num_domains, hparams)

        # 设置对抗训练所需的判别器模型
        self._setup_adversarial_net()

        # 使用不降维的交叉熵损失
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def _modify_generator_output(self):
        """修改生成器输出激活函数
        
        根据超参数选择不同的激活函数。
        """
        print('--> Modifying encoder output:', self.hparams['urm_generator_output'])

        if self.hparams['urm_generator_output'] == 'tanh':
            self.featurizer.activation = nn.Tanh()
        elif self.hparams['urm_generator_output'] == 'sigmoid':
            self.featurizer.activation = nn.Sigmoid()
        elif self.hparams['urm_generator_output'] == 'identity':
            self.featurizer.activation = nn.Identity()
        elif self.hparams['urm_generator_output'] == 'relu':
            self.featurizer.activation = nn.ReLU()
        else:
            raise Exception('unrecognized output activation: %s' % self.hparams['urm_generator_output'])

    def _setup_adversarial_net(self):
        """设置对抗网络
        
        初始化判别器及其优化器，并修改生成器输出。
        """
        print('--> Initializing discriminator <--')        
        self.discriminator = self._init_discriminator()
        # 使用BCEWithLogitsLoss，它更数值稳定
        self.discriminator_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

        # 为判别器创建优化器
        if self.hparams["urm_discriminator_optimizer"] == 'sgd':
            self.discriminator_opt = torch.optim.SGD(
                self.discriminator.parameters(),
                lr=self.hparams['urm_discriminator_lr'],
                weight_decay=self.hparams['weight_decay'],
                momentum=0.9)
        elif self.hparams["urm_discriminator_optimizer"] == 'adam':
            self.discriminator_opt = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.hparams['urm_discriminator_lr'],
                weight_decay=self.hparams['weight_decay'])
        else:
            raise Exception('%s unimplemented' % self.hparams["urm_discriminator_optimizer"])

        # 修改生成器输出
        self._modify_generator_output()
        # 用于计算判别器准确率
        self.sigmoid = nn.Sigmoid()
            
    def _init_discriminator(self):
        """初始化判别器网络
        
        创建一个3层隐藏层的MLP作为判别器。
        
        Returns:
            判别器网络
        """
        model = nn.Sequential()
        model.add_module("dense1", nn.Linear(self.featurizer.n_outputs, 100))
        model.add_module("act1", nn.LeakyReLU())

        # 添加额外的隐藏层
        for _ in range(self.hparams['urm_discriminator_hidden_layers']):            
            model.add_module("dense%d" % (2+_), nn.Linear(100, 100))
            model.add_module("act2%d" % (2+_), nn.LeakyReLU())

        model.add_module("output", nn.Linear(100, 1)) 
        return model

    def _generate_noise(self, feats):
        """生成均匀噪声
        
        生成与特征相同形状的均匀分布噪声。
        
        Args:
            feats: 输入特征
            
        Returns:
            生成的噪声
        """
        if self.hparams['urm_generator_output'] == 'tanh':
            a, b = -1, 1
        elif self.hparams['urm_generator_output'] == 'relu':
            a, b = 0, 1
        elif self.hparams['urm_generator_output'] == 'sigmoid':
            a, b = 0, 1
        else:
            raise Exception('unrecognized output activation: %s' % self.hparams['urm_generator_output'])

        # 生成[0,1)均匀分布，然后映射到[a,b)
        uniform_noise = torch.rand(feats.size(), dtype=feats.dtype, layout=feats.layout, device=feats.device)
        n = ((b-a) * uniform_noise) + a
        return n

    def _generate_soft_labels(self, size, device, a, b):
        """生成软标签
        
        生成指定范围内的随机数作为软标签。
        
        Args:
            size: 标签大小
            device: 计算设备
            a: 范围下限
            b: 范围上限
            
        Returns:
            生成的软标签
        """
        uniform_noise = torch.rand(size, device=device)
        return ((b-a) * uniform_noise) + a

    def get_accuracy(self, y_true, y_prob):
        """计算二分类准确率
        
        Args:
            y_true: 真实标签
            y_prob: 预测概率
            
        Returns:
            准确率
        """
        assert y_true.ndim == 1 and y_true.size() == y_prob.size()
        y_prob = y_prob > 0.5
        return (y_true == y_prob).sum().item() / y_true.size(0)

    def return_feats(self, x):
        """提取特征
        
        Args:
            x: 输入数据
            
        Returns:
            提取的特征
        """
        return self.featurizer(x)

    def _update_discriminator(self, x, y, feats):
        """更新判别器
        
        使用真实噪声和生成的特征更新判别器。
        
        Args:
            x: 输入数据
            y: 标签
            feats: 提取的特征
        """
        # 分离特征，不通过编码器反向传播
        feats = feats.detach()
        # 生成噪声
        noise = self._generate_noise(feats)
        
        # 计算判别器输出
        noise_logits = self.discriminator(noise)
        feats_logits = self.discriminator(feats)

        # 创建硬标签
        hard_true_y = torch.tensor([1] * noise.shape[0], device=noise.device, dtype=noise.dtype)
        hard_fake_y = torch.tensor([0] * feats.shape[0], device=feats.device, dtype=feats.dtype)

        # 如果启用标签平滑，使用软标签
        if self.hparams['urm_discriminator_label_smoothing']:
            soft_true_y = self._generate_soft_labels(noise.shape[0], noise.device, 1-self.hparams['urm_discriminator_label_smoothing'], 1.0)
            soft_fake_y = self._generate_soft_labels(feats.shape[0], feats.device, 0, 0+self.hparams['urm_discriminator_label_smoothing'])
            true_y = soft_true_y
            fake_y = soft_fake_y
        else:
            true_y = hard_true_y
            fake_y = hard_fake_y

        # 计算损失
        noise_loss = self.discriminator_loss(noise_logits.squeeze(1), true_y)
        feats_loss = self.discriminator_loss(feats_logits.squeeze(1), fake_y)
        d_loss = 1*noise_loss + self.hparams['urm_adv_lambda']*feats_loss

        # 更新判别器
        self.discriminator_opt.zero_grad()
        d_loss.backward()
        self.discriminator_opt.step()

    def _compute_loss(self, x, y):
        """计算总损失
        
        包括分类损失和对抗损失。
        
        Args:
            x: 输入数据
            y: 标签
            
        Returns:
            总损失和提取的特征
        """
        feats = self.return_feats(x)
        # 计算分类损失
        ce_loss = self.loss(self.classifier(feats), y).mean()

        # 计算对抗损失，使判别器将特征分类为噪声（标签1）
        true_y = torch.tensor(feats.shape[0]*[1], device=feats.device, dtype=feats.dtype)
        g_logits = self.discriminator(feats)
        g_loss = self.discriminator_loss(g_logits.squeeze(1), true_y)
        loss = ce_loss + self.hparams['urm_adv_lambda']*g_loss

        return loss, feats

    def update(self, minibatches, unlabeled=None):
        """执行一步更新
        
        更新生成器和判别器参数。
        
        Args:
            minibatches: (x, y)元组列表
            unlabeled: 未使用
            
        Returns:
            包含损失值的字典
        """
        # 合并所有环境的输入和标签
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
            
        # 计算损失和特征
        loss, feats = self._compute_loss(all_x, all_y)

        # 更新生成器
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新判别器
        self._update_discriminator(all_x, all_y, feats)
    
        return {'loss': loss.item()}

class ARM(ERM):
    """自适应风险最小化算法"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化ARM算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        # 修改输入形状以包含上下文信息
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains, hparams)
        # 创建上下文网络
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        """预测输入数据的标签
        
        使用上下文信息增强输入特征进行预测。
        
        Args:
            x: 输入数据
            
        Returns:
            预测的对数概率
        """
        batch_size, c, h, w = x.shape
        # 计算元批次大小和支持大小
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        # 计算上下文信息
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        # 将上下文信息与输入连接
        x = torch.cat([x, context], dim=1)
        return self.network(x)

class AbstractDANN(Algorithm):
    """域对抗神经网络抽象基类
    
    实现了域对抗训练的核心逻辑，支持有监督的源域训练和无监督的目标域适应。
    """
    def __init__(self, input_shape, num_classes, num_domains, 
                 hparams, conditional, class_balance):
        """初始化DANN算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
            conditional: 是否使用条件对抗
            class_balance: 是否使用类别平衡
        """
        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance
        self.grl = networks.GradientReversalLayer.apply
        
        # 添加epoch计数器和steps_per_epoch信息
        self.register_buffer('current_epoch', torch.tensor([0]))
        self.steps_per_epoch = hparams.get('steps_per_epoch', 100)  # 默认值100

        # 创建网络组件
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # 域分类器
        self.discriminator = networks.MLP(
            self.featurizer.n_outputs,
            num_domains,  
            self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # 创建优化器
        self.disc_opt = torch.optim.AdamW(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.AdamW(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))
        
        # 初始化学习率调度相关变量
        self.lr_schedule = []
        self.lr_schedule_changes = 0
        # 创建源域学习率调度器 - 监控源域ACC
        # 优化参数以增加容错性：增大patience，减小factor变化率，增大threshold允许更大波动
        # 源域调度器（生成器）：精准贴合 0~5稳→5~15首降→15~25次降→25+稳
        self.source_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.gen_opt,
            mode='min',
            factor=0.65,  # 严格匹配理想路径的 0.001→0.00065→0.00042
            patience=6,  # 关键：5~15epoch内触发首次降LR（连续6个epoch降幅＜0.005）
            threshold=0.005,  # 适配gen_loss中期波动（日志中单次降幅＜0.005）
            threshold_mode='abs',  # 按绝对差值判定，避免源域小loss误判
            cooldown=5,  # 首次降后冷却5个epoch，刚好适配15~25次降节奏
            min_lr=8e-7,
            eps=1e-9,  # 处理gen_loss后期小数值精度问题
        )

        # 目标域调度器（判别器）：彻底解决LR不变，贴合 8~20首降→20~30次降→30+稳
        self.target_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.disc_opt,
            mode='min',
            factor=0.75,  # 严格匹配理想路径的 0.001→0.00075→0.00056
            patience=12,  # 8~20epoch内必触发（连续12个epoch降幅＜0.002）
            threshold=0.002,  # 关键：适配disc_loss后期极小波动（日志中单次降幅＜0.001）
            threshold_mode='abs',  # 核心修正！小loss场景下精准判定“无改善”
            cooldown=6,  # 冷却期适配对抗节奏，避免频繁调整
            min_lr=6e-7,
            eps=1e-10,  # 解决disc_loss接近0（如1e-6）时的浮点精度问题
        )

    def update(self, minibatches, unlabeled=None):
        """执行一步更新
        
        实现完整的DANN更新逻辑，包括：
        1. 从源域数据学习分类任务
        2. 使用域对抗训练学习域不变特征
        3. 利用目标域无标签数据进行域适应
        
        Args:
            minibatches: (x, y)元组列表，来自源域
            unlabeled: 来自目标域的无标签数据列表
            
        Returns:
            包含各种损失值的字典
        """
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        
        # 处理源域数据
        source_x = torch.cat([x for x, y in minibatches])
        source_y = torch.cat([y for x, y in minibatches])
        source_z = self.featurizer(source_x)
        
        # 处理目标域无标签数据
        has_unlabeled = unlabeled is not None and len(unlabeled) > 0
        target_z = None
        
        if has_unlabeled:
            target_x = torch.cat(unlabeled)
            target_z = self.featurizer(target_x)
        
        # 计算分类损失（仅使用源域数据）
        source_preds = self.classifier(source_z)
        classifier_loss = F.cross_entropy(source_preds, source_y)
        
        # 准备域分类器的输入和标签
        if has_unlabeled:
            # 合并源域和目标域特征
            all_z = torch.cat([source_z, target_z])
            
            # 源域标签为0，目标域标签为1（标准DANN二元域分类）
            source_domain_labels = torch.zeros(source_z.size(0), dtype=torch.int64, device=device)
            target_domain_labels = torch.ones(target_z.size(0), dtype=torch.int64, device=device)
            all_domain_labels = torch.cat([source_domain_labels, target_domain_labels])
        else:
            # 只有源域数据时，为每个源域环境分配不同的标签
            all_z = source_z
            all_domain_labels = torch.cat([
                torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
                for i, (x, y) in enumerate(minibatches)
            ])
        
        # 计算域对抗损失
        # 使用梯度反转层
        alpha = 2. / (1. + np.exp(-10 * self.update_count.item() / self.hparams.get('n_steps'))) - 1
        reversed_z = self.grl(all_z, alpha)
        
        # 根据是否条件对抗构造判别器输入
        if self.conditional and has_unlabeled:
            # 为目标域数据生成伪标签
            target_preds = self.classifier(target_z)
            target_pseudo_y = torch.argmax(target_preds, dim=1)
            all_y = torch.cat([source_y, target_pseudo_y])
            disc_input = reversed_z + self.class_embeddings(all_y)
        else:
            disc_input = reversed_z
        
        # 计算判别器输出
        disc_out = self.discriminator(disc_input)
        
        # 如果启用类别平衡，计算加权损失
        if self.class_balance:
            y_counts = F.one_hot(all_domain_labels).sum(dim=0)
            weights = 1. / (y_counts[all_domain_labels] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, all_domain_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, all_domain_labels)
        
        # 计算梯度惩罚
        if self.hparams['grad_penalty'] > 0:
            input_grad = autograd.grad(
                F.cross_entropy(disc_out, all_domain_labels, reduction='sum'),
                [disc_input], create_graph=True)[0]
            grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
            disc_loss += self.hparams['grad_penalty'] * grad_penalty
        
        # 交替更新判别器和生成器
        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g):
            # 更新判别器
                self.disc_opt.zero_grad()
                disc_loss.backward()
                self.disc_opt.step()
                
                # 目标域学习率调度 - 监控判别器损失
                with torch.no_grad():
                    # 只在epoch结束时调用学习率调度器
                    is_epoch_end = (self.update_count.item() % self.steps_per_epoch == 0)
                    if is_epoch_end:
                        self.target_scheduler.step(disc_loss)
                        if hasattr(self.target_scheduler, '_last_lr'):
                            current_disc_lr = self.target_scheduler._last_lr[0]
                            # 单独记录判别器LR变化，避免与生成器混淆
                            self.lr_schedule.append(('disc', current_disc_lr))
                            if len(self.lr_schedule) > 1 and self.lr_schedule[-2][0] == 'disc':
                                if self.lr_schedule[-1][1] != self.lr_schedule[-2][1]:
                                    self.lr_schedule_changes += 1
                
                return {
                    'disc_loss': disc_loss.item(),
                    'disc_lr': self.target_scheduler._last_lr[0] if hasattr(self.target_scheduler, '_last_lr') else self.hparams["lr_d"]
                }
        else:
            # 更新生成器（特征提取器和分类器）
            # 生成器的目标是最小化分类损失，同时最大化域对抗损失（通过负号实现）
            gen_loss = classifier_loss - self.hparams['lambda'] * disc_loss
            
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            
            # 使用分类器损失进行学习率调度（比准确率计算更高效）
            with torch.no_grad():
                # 只在epoch结束时调用学习率调度器
                is_epoch_end = (self.update_count.item() % self.steps_per_epoch == 0)
                # 更新生成器时的日志记录（修改后）
                if is_epoch_end:
                    self.source_scheduler.step(classifier_loss)
                    if hasattr(self.source_scheduler, '_last_lr'):
                        current_gen_lr = self.source_scheduler._last_lr[0]
                        self.lr_schedule.append(('gen', current_gen_lr))
                        if len(self.lr_schedule) > 1 and self.lr_schedule[-2][0] == 'gen':
                            if self.lr_schedule[-1][1] != self.lr_schedule[-2][1]:
                                self.lr_schedule_changes += 1
                    
                    # 更新epoch计数器
                    self.current_epoch += 1
            
            return {
                'gen_loss': gen_loss.item(),
                'class_lr': self.source_scheduler._last_lr[0] if hasattr(self.source_scheduler, '_last_lr') else self.hparams["lr_g"]
            }

    def predict(self, x):
        """预测输入数据的标签
        
        Args:
            x: 输入数据
            
        Returns:
            预测的对数概率
        """
        return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
    """无条件域对抗神经网络"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化DANN算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=False, class_balance=False)

class CDANN(AbstractDANN):
    """条件域对抗神经网络"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化CDANN算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)

class AbstractMMD(ERM):
    """基于MMD的域适应算法抽象基类"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        """初始化MMD算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
            gaussian: 是否使用高斯核
        """
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.kernel_type = "gaussian" if gaussian else "mean_cov"

    def my_cdist(self, x1, x2):
        """计算成对距离矩阵
        
        Args:
            x1: 第一个输入
            x2: 第二个输入
            
        Returns:
            距离矩阵
        """
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        """计算高斯核矩阵
        
        Args:
            x: 第一个输入
            y: 第二个输入
            gamma: 带宽参数列表
            
        Returns:
            核矩阵
        """
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)
        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))
        return K

    def mmd(self, x, y):
        """计算MMD距离
        
        Args:
            x: 第一个分布的样本
            y: 第二个分布的样本
            
        Returns:
            MMD距离
        """
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            # 使用均值和协方差差异
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        """执行一步更新
        
        计算分类损失和MMD损失，更新参数。
        
        Args:
            minibatches: (x, y)元组列表
            unlabeled: 未使用
            
        Returns:
            包含损失值的字典
        """
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        # 提取特征和预测
        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        # 计算分类损失
        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            # 计算域间MMD损失
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        # 平均化损失
        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        # 更新参数
        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}

class MMD(AbstractMMD):
    """使用高斯核的MMD算法"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化MMD算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)

class CORAL(AbstractMMD):
    """使用均值和协方差差异的MMD算法"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化CORAL算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)

class MTL(Algorithm):
    """多任务学习算法
    
    实现论文: Domain Generalization by Marginal Transfer Learning
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化MTL算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams)
        # 创建特征提取器和分类器
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        # 注册域嵌入
        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None):
        """执行一步更新
        
        对每个域计算损失并更新参数。
        
        Args:
            minibatches: (x, y)元组列表
            unlabeled: 未使用
            
        Returns:
            包含损失值的字典
        """
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        """更新域嵌入
        
        使用指数移动平均更新域嵌入。
        
        Args:
            features: 输入特征
            env: 域索引
            
        Returns:
            更新后的嵌入
        """
        return_embedding = features.mean(0)

        if env is not None:
            # 使用指数移动平均
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]
            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        """预测输入数据的标签
        
        使用域嵌入增强特征进行预测。
        
        Args:
            x: 输入数据
            env: 域索引
            
        Returns:
            预测的对数概率
        """
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))

class SagNet(Algorithm):
    """风格无关网络
    
    实现论文: https://arxiv.org/abs/1910.11645
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化SagNet算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams)
        # 创建特征提取器
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # 创建内容网络
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # 创建风格网络
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # 创建优化器
        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        """前向传播内容网络
        
        使用随机化的风格特征。
        
        Args:
            x: 输入数据
            
        Returns:
            内容网络的输出
        """
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        """前向传播风格网络
        
        使用随机化的内容特征。
        
        Args:
            x: 输入数据
            
        Returns:
            风格网络的输出
        """
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        """随机化特征
        
        对特征的风格或内容进行随机化。
        
        Args:
            x: 输入特征
            what: 随机化类型（"style"或"content"）
            eps: 数值稳定性参数
            
        Returns:
            随机化后的特征
        """
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            # 随机化风格
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            # 随机化内容
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        """执行一步更新
        
        交替更新内容网络、风格网络和对抗网络。
        
        Args:
            minibatches: (x, y)元组列表
            unlabeled: 未使用
            
        Returns:
            包含损失值的字典
        """
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # 更新内容网络
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # 更新风格网络
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # 更新对抗网络
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        """预测输入数据的标签
        
        使用内容网络进行预测。
        
        Args:
            x: 输入数据
            
        Returns:
            预测的对数概率
        """
        return self.network_c(self.network_f(x))

class AbstractCAD(Algorithm):
    """对比对抗域瓶颈抽象基类
    
    实现论文: Optimal Representations for Covariate Shift
    """
    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, is_conditional):
        """初始化CAD算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
            is_conditional: 是否使用条件瓶颈
        """
        super(AbstractCAD, self).__init__(input_shape, num_classes, num_domains, hparams)

        # 创建网络组件
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        params = list(self.featurizer.parameters()) + list(self.classifier.parameters())

        # 初始化瓶颈损失参数
        self.is_conditional = is_conditional
        self.base_temperature = 0.07
        self.temperature = hparams['temperature']
        self.is_project = hparams['is_project']
        self.is_normalized = hparams['is_normalized']
        self.is_flipped = hparams["is_flipped"]

        # 如果需要，添加投影头
        if self.is_project:
            self.project = nn.Sequential(
                nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs),
                nn.ReLU(inplace=True),
                nn.Linear(self.featurizer.n_outputs, 128),
            )
            params += list(self.project.parameters())

        # 创建优化器
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def bn_loss(self, z, y, dom_labels):
        """计算对比域瓶颈损失
        
        基于监督对比损失(SupCon)实现。
        
        Args:
            z: 特征表示
            y: 类别标签
            dom_labels: 域标签
            
        Returns:
            瓶颈损失
        """
        device = z.device
        batch_size = z.shape[0]

        # 创建掩码
        y = y.contiguous().view(-1, 1)
        dom_labels = dom_labels.contiguous().view(-1, 1)
        mask_y = torch.eq(y, y.T).to(device)
        mask_d = (torch.eq(dom_labels, dom_labels.T)).to(device)
        mask_drop = ~torch.eye(batch_size).bool().to(device)
        mask_y &= mask_drop
        mask_y_n_d = mask_y & (~mask_d)  # 相同类别不同域
        mask_y_d = mask_y & mask_d  # 相同类别相同域
        mask_y, mask_drop, mask_y_n_d, mask_y_d = mask_y.float(), mask_drop.float(), mask_y_n_d.float(), mask_y_d.float()

        # 计算logits
        if self.is_project:
            z = self.project(z)
        if self.is_normalized:
            z = F.normalize(z, dim=1)
        outer = z @ z.T
        logits = outer / self.temperature
        logits = logits * mask_drop
        # 数值稳定性
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        if not self.is_conditional:
            # 无条件CAD损失
            denominator = torch.logsumexp(logits + mask_drop.log(), dim=1, keepdim=True)
            log_prob = logits - denominator

            mask_valid = (mask_y.sum(1) > 0)
            log_prob = log_prob[mask_valid]
            mask_d = mask_d[mask_valid]

            if self.is_flipped:
                # 最大化不同域样本的对数概率
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (~mask_d).float().log(), dim=1)
            else:
                # 最小化相同域样本的对数概率
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (mask_d).float().log(), dim=1)
        else:
            # 条件CAD损失
            if self.is_flipped:
                mask_valid = (mask_y_n_d.sum(1) > 0)
            else:
                mask_valid = (mask_y_d.sum(1) > 0)

            mask_y = mask_y[mask_valid]
            mask_y_d = mask_y_d[mask_valid]
            mask_y_n_d = mask_y_n_d[mask_valid]
            logits = logits[mask_valid]

            # 计算相同标签的对数概率
            denominator = torch.logsumexp(logits + mask_y.log(), dim=1, keepdim=True)
            log_prob_y = logits - denominator

            if self.is_flipped:
                # 最大化不同域相同标签样本的对数概率
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_n_d.log(), dim=1)
            else:
                # 最小化相同域相同标签样本的对数概率
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_d.log(), dim=1)

        def finite_mean(x):
            """计算有限值的均值"""
            num_finite = (torch.isfinite(x).float()).sum()
            mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
            if num_finite != 0:
                mean = mean / num_finite
            else:
                return torch.tensor(0.0).to(x)
            return mean

        return finite_mean(bn_loss)

    def update(self, minibatches, unlabeled=None):
        """执行一步更新
        
        计算分类损失和瓶颈损失，更新参数。
        
        Args:
            minibatches: (x, y)元组列表
            unlabeled: 未使用
            
        Returns:
            包含损失值的字典
        """
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        # 合并所有环境的输入和标签
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        all_d = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        # 计算瓶颈损失和分类损失
        bn_loss = self.bn_loss(all_z, all_y, all_d)
        clf_out = self.classifier(all_z)
        clf_loss = F.cross_entropy(clf_out, all_y)
        total_loss = clf_loss + self.hparams['lmbda'] * bn_loss

        # 更新参数
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"clf_loss": clf_loss.item(), "bn_loss": bn_loss.item(), "total_loss": total_loss.item()}

    def predict(self, x):
        """预测输入数据的标签
        
        Args:
            x: 输入数据
            
        Returns:
            预测的对数概率
        """
        return self.classifier(self.featurizer(x))

class CAD(AbstractCAD):
    """对比对抗域瓶颈
    
    属性:
    - 最小化I(D;Z)
    - 需要域标签但不需要任务标签
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化CAD算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        super(CAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=False)

class CondCAD(AbstractCAD):
    """条件对比对抗域瓶颈
    
    属性:
    - 最小化I(D;Z|Y)
    - 需要域标签和任务标签
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化CondCAD算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        super(CondCAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=True)

class Transfer(Algorithm):
    """迁移学习算法
    
    实现论文: Quantifying and Improving Transferability in Domain Generalization
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化Transfer算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        super(Transfer, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.d_steps_per_g = hparams['d_steps_per_g']

        # 创建网络组件
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.adv_classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.adv_classifier.load_state_dict(self.classifier.state_dict())

        # 创建优化器
        if self.hparams['gda']:
            self.optimizer = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr'])
        else:
            self.optimizer = torch.optim.Adam(
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.adv_opt = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr_d'])

    def loss_gap(self, minibatches, device):
        """计算域间损失差距
        
        Args:
            minibatches: (x, y)元组列表
            device: 计算设备
            
        Returns:
            最大域损失与最小域损失的差值
        """
        max_env_loss, min_env_loss = torch.tensor([-float('inf')], device=device), torch.tensor([float('inf')], device=device)
        for x, y in minibatches:
            p = self.adv_classifier(self.featurizer(x))
            loss = F.cross_entropy(p, y)
            if loss > max_env_loss:
                max_env_loss = loss
            if loss < min_env_loss:
                min_env_loss = loss
        return max_env_loss - min_env_loss

    def update(self, minibatches, unlabeled=None):
        """执行一步更新
        
        交替更新主分类器和对抗分类器。
        
        Args:
            minibatches: (x, y)元组列表
            unlabeled: 未使用
            
        Returns:
            包含损失值的字典
        """
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        # 更新主分类器
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del all_x, all_y
        # 计算并更新损失差距
        gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
        self.optimizer.zero_grad()
        gap.backward()
        self.optimizer.step()
        self.adv_classifier.load_state_dict(self.classifier.state_dict())
        # 更新对抗分类器
        for _ in range(self.d_steps_per_g):
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
        return {'loss': loss.item(), 'gap': -gap.item()}

    def update_second(self, minibatches, unlabeled=None):
        """执行一步更新（备用实现）
        
        使用不同的更新策略。
        
        Args:
            minibatches: (x, y)元组列表
            unlabeled: 未使用
            
        Returns:
            包含损失值的字典
        """
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count = (self.update_count + 1) % (1 + self.d_steps_per_g)
        if self.update_count.item() == 1:
            # 更新主分类器
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            loss = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del all_x, all_y
            # 计算并更新损失差距
            gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            self.optimizer.zero_grad()
            gap.backward()
            self.optimizer.step()
            self.adv_classifier.load_state_dict(self.classifier.state_dict())
            return {'loss': loss.item(), 'gap': gap.item()}
        else:
            # 更新对抗分类器
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
            return {'gap': -gap.item()}

    def predict(self, x):
        """预测输入数据的标签
        
        Args:
            x: 输入数据
            
        Returns:
            预测的对数概率
        """
        return self.classifier(self.featurizer(x))

class ADRMX(Algorithm):
    """自适应域正则化混合算法
    
    实现论文: Additive Disentanglement of Domain Features with Remix Loss
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """初始化ADRMX算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
        """
        super(ADRMX, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

        # 初始化参数
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.mix_num = 1
        self.scl_int = SupConLossLambda(lamda=0.5)
        self.scl_final = SupConLossLambda(lamda=0.5)

        # 创建网络组件
        self.featurizer_label = networks.Featurizer(input_shape, self.hparams)
        self.featurizer_domain = networks.Featurizer(input_shape, self.hparams)

        self.discriminator = networks.MLP(self.featurizer_domain.n_outputs,
            num_domains, self.hparams)

        self.classifier_label_1 = networks.Classifier(
            self.featurizer_label.n_outputs,
            num_classes,
            is_nonlinear=True)

        self.classifier_label_2 = networks.Classifier(
            self.featurizer_label.n_outputs,
            num_classes,
            is_nonlinear=True)

        self.classifier_domain = networks.Classifier(
            self.featurizer_domain.n_outputs,
            num_domains,
            is_nonlinear=True)

        # 构建完整网络
        self.network = nn.Sequential(self.featurizer_label, self.classifier_label_1)

        # 创建优化器
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters())),
            lr=self.hparams["lr"],
            betas=(self.hparams['beta1'], 0.9))

        self.opt = torch.optim.Adam(
            (list(self.featurizer_label.parameters()) +
             list(self.featurizer_domain.parameters()) +
             list(self.classifier_label_1.parameters()) +
                list(self.classifier_label_2.parameters()) +
                list(self.classifier_domain.parameters())),
            lr=self.hparams["lr"],
            betas=(self.hparams['beta1'], 0.9))
                                                    
    def update(self, minibatches, unlabeled=None):
        """执行一步更新
        
        交替更新判别器和生成器网络。
        
        Args:
            minibatches: (x, y)元组列表
            unlabeled: 未使用
            
        Returns:
            包含损失值的字典
        """
        self.update_count += 1
        # 合并所有环境的输入和标签
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        # 提取特征
        feat_label = self.featurizer_label(all_x)
        feat_domain = self.featurizer_domain(all_x)
        feat_combined = feat_label - feat_domain

        # 获取域标签
        disc_labels = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=all_x.device)
            for i, (x, _) in enumerate(minibatches)
        ])
        # 计算判别器输出和损失
        disc_out = self.discriminator(feat_combined) 
        disc_loss = F.cross_entropy(disc_out, disc_labels)

        # 交替更新
        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):
            # 更新判别器
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'loss_disc': disc_loss.item()}
        else:
            # 更新生成器
            # 计算域分类损失
            domain_preds = self.classifier_domain(feat_domain)
            classifier_loss_domain = F.cross_entropy(domain_preds, disc_labels)
            classifier_remixed_loss = 0

            # 计算标签分类损失和对比损失
            int_preds = self.classifier_label_1(feat_label)
            classifier_loss_int = F.cross_entropy(int_preds, all_y)
            cnt_loss_int = self.scl_int(feat_label, all_y, disc_labels)

            # 计算最终分类损失和对比损失
            final_preds = self.classifier_label_2(feat_combined)
            classifier_loss_final = F.cross_entropy(final_preds, all_y)
            cnt_loss_final = self.scl_final(feat_combined, all_y, disc_labels)

            # 执行remix策略
            for i in range(self.num_classes):
                indices = torch.where(all_y == i)[0]
                for _ in range(self.mix_num):
                    # 获取同类不同域的两个样本
                    perm = torch.randperm(indices.numel())
                    if len(perm) < 2:
                        continue
                    idx1, idx2 = perm[:2]
                    # 执行remix
                    remixed_feat = feat_combined[idx1] + feat_domain[idx2]
                    # 计算预测和损失
                    pred = self.classifier_label_1(remixed_feat.view(1,-1))
                    classifier_remixed_loss += F.cross_entropy(pred.view(1, -1), all_y[idx1].view(-1))
            # 标准化remix损失
            classifier_remixed_loss /= (self.num_classes * self.mix_num)

            # 计算生成器总损失
            gen_loss = (classifier_loss_int +
                        classifier_loss_final +
                        self.hparams["dclf_lambda"] * classifier_loss_domain +
                        self.hparams["rmxd_lambda"] * classifier_remixed_loss +
                        self.hparams['cnt_lambda'] * (cnt_loss_int + cnt_loss_final) + 
                        (self.hparams['disc_lambda'] * -disc_loss))
            
            # 更新参数
            self.disc_opt.zero_grad()
            self.opt.zero_grad()
            gen_loss.backward()
            self.opt.step()

            return {'loss_total': gen_loss.item(), 
                'loss_cnt_int': cnt_loss_int.item(),
                'loss_cnt_final': cnt_loss_final.item(),
                'loss_clf_int': classifier_loss_int.item(), 
                'loss_clf_fin': classifier_loss_final.item(), 
                'loss_dmn': classifier_loss_domain.item(), 
                'loss_disc': disc_loss.item(),
                'loss_remixed': classifier_remixed_loss.item()}
    
    def predict(self, x):
        """预测输入数据的标签
        
        Args:
            x: 输入数据
            
        Returns:
            预测的对数概率
        """
        return self.network(x)
