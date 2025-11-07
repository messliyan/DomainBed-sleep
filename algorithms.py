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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入backpack库，用于计算批量梯度
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

# 导入网络模块和工具函数
import networks
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

        if self.global_iter > self.hparams["linear_steps"]:
            selected_optimizer = self.optimizer
        else:
            selected_optimizer = self.linear_optimizer



        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.network(all_x), all_y)

        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        self.update_sma()
        if not self.hparams["freeze_bn"]:
            self.network_sma.train()
            self.network_sma(all_x)

        return {'loss': loss.item()}

    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)

    def set_lr(self, eval_loaders_iid=None, schedule=None,device=None):
        with torch.no_grad():
             if self.global_iter > self.hparams["linear_steps"]:
                 if schedule is None:
                     self.network_sma.eval()
                     val_losses = []
                     for loader in eval_loaders_iid:
                         loss = 0.0
                         for x, y in loader:
                             x = x.to(device)
                             y = y.to(device)
                             loss += F.cross_entropy(self.network_sma(x),y)
                         val_losses.append(loss / len(loader ))
                     val_loss = torch.mean(torch.stack(val_losses))
                     self.scheduler.step(val_loss)
                     self.lr_schedule.append(self.scheduler._last_lr)
                     if len(self.lr_schedule) > 1:
                         if self.lr_schedule[-1] !=  self.lr_schedule[-2]:
                            self.lr_schedule_changes += 1
                     if self.lr_schedule_changes == 3:
                         self.lr_schedule[-1] = [0.0]
                     return self.lr_schedule
                 else:
                     self.optimizer.param_groups[0]['lr'] = (torch.Tensor(schedule[0]).requires_grad_(False))[0]
                     schedule = schedule[1:]
             return schedule

class URM(ERM):
    """
    Implementation of Uniform Risk Minimization, as seen in Uniformly Distributed Feature Representations for
    Fair and Robust Learning. TMLR 2024 (https://openreview.net/forum?id=PgLbS5yp8n)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        ERM.__init__(self, input_shape, num_classes, num_domains, hparams)

        # setup discriminator model for URM adversarial training
        self._setup_adversarial_net()

        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def _modify_generator_output(self):
        print('--> Modifying encoder output:', self.hparams['urm_generator_output'])
        
        # from lib import wide_resnet
        # assert type(self.featurizer) in [networks.MLP, networks.MNIST_CNN, wide_resnet.Wide_ResNet, networks.ResNet]

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
        print('--> Initializing discriminator <--')        
        self.discriminator = self._init_discriminator()
        self.discriminator_loss = torch.nn.BCEWithLogitsLoss(reduction="mean") # apply on logit

        # featurizer optimized by self.optimizer only
        if self.hparams["urm_discriminator_optimizer"] == 'sgd':
            self.discriminator_opt = torch.optim.SGD(self.discriminator.parameters(), lr=self.hparams['urm_discriminator_lr'], \
                weight_decay=self.hparams['weight_decay'], momentum=0.9)
        elif self.hparams["urm_discriminator_optimizer"] == 'adam':
            self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams['urm_discriminator_lr'], \
                weight_decay=self.hparams['weight_decay'])
        else:
            raise Exception('%s unimplemented' % self.hparams["urm_discriminator_optimizer"])

        self._modify_generator_output()
        self.sigmoid = nn.Sigmoid() # to compute discriminator acc.
            
    def _init_discriminator(self):
        """
        3 hidden layer MLP
        """
        model = nn.Sequential()
        model.add_module("dense1", nn.Linear(self.featurizer.n_outputs, 100))
        model.add_module("act1", nn.LeakyReLU())

        for _ in range(self.hparams['urm_discriminator_hidden_layers']):            
            model.add_module("dense%d" % (2+_), nn.Linear(100, 100))
            model.add_module("act2%d" % (2+_), nn.LeakyReLU())

        model.add_module("output", nn.Linear(100, 1)) 
        return model

    def _generate_noise(self, feats):
        """
        If U is a random variable uniformly distributed on [0, 1), then (b-a)*U + a is uniformly distributed on [a, b).
        """
        if self.hparams['urm_generator_output'] == 'tanh':
            a,b = -1,1
        elif self.hparams['urm_generator_output'] == 'relu':
            a,b = 0,1
        elif self.hparams['urm_generator_output'] == 'sigmoid':
            a,b = 0,1
        else:
            raise Exception('unrecognized output activation: %s' % self.hparams['urm_generator_output'])

        uniform_noise = torch.rand(feats.size(), dtype=feats.dtype, layout=feats.layout, device=feats.device) # U~[0,1]
        n = ((b-a) * uniform_noise) + a # n ~ [a,b)
        return n

    def _generate_soft_labels(self, size, device, a ,b):
        # returns size random numbers in [a,b]
         uniform_noise = torch.rand(size, device=device) # U~[0,1]
         return ((b-a) * uniform_noise) + a

    def get_accuracy(self, y_true, y_prob):
        # y_prob is binary probability
        assert y_true.ndim == 1 and y_true.size() == y_prob.size()
        y_prob = y_prob > 0.5
        return (y_true == y_prob).sum().item() / y_true.size(0)

    def return_feats(self, x):
        return self.featurizer(x)

    def _update_discriminator(self, x, y, feats):
        # feats = self.return_feats(x)
        feats = feats.detach() # don't backbrop through encoder in this step
        noise = self._generate_noise(feats)
        
        noise_logits = self.discriminator(noise) # (N,1)
        feats_logits = self.discriminator(feats) # (N,1)

        # hard targets
        hard_true_y = torch.tensor([1] * noise.shape[0], device=noise.device, dtype=noise.dtype) # [1,1...1] noise is true
        hard_fake_y = torch.tensor([0] * feats.shape[0], device=feats.device, dtype=feats.dtype) # [0,0...0] feats are fake (generated)

        if self.hparams['urm_discriminator_label_smoothing']:
            # label smoothing in discriminator
            soft_true_y = self._generate_soft_labels(noise.shape[0], noise.device, 1-self.hparams['urm_discriminator_label_smoothing'], 1.0) # random labels in range
            soft_fake_y = self._generate_soft_labels(feats.shape[0], feats.device, 0, 0+self.hparams['urm_discriminator_label_smoothing']) # random labels in range
            true_y = soft_true_y
            fake_y = soft_fake_y
        else:
            true_y = hard_true_y
            fake_y = hard_fake_y

        noise_loss = self.discriminator_loss(noise_logits.squeeze(1), true_y) # pass logits to BCEWithLogitsLoss
        feats_loss = self.discriminator_loss(feats_logits.squeeze(1), fake_y) # pass logits to BCEWithLogitsLoss

        d_loss = 1*noise_loss + self.hparams['urm_adv_lambda']*feats_loss

        # update discriminator
        self.discriminator_opt.zero_grad()
        d_loss.backward()
        self.discriminator_opt.step()

    def _compute_loss(self, x, y):
        feats = self.return_feats(x)
        ce_loss = self.loss(self.classifier(feats), y).mean()

        # train generator/encoder to make discriminator classify feats as noise (label 1)
        true_y = torch.tensor(feats.shape[0]*[1], device=feats.device, dtype=feats.dtype)
        g_logits = self.discriminator(feats)
        g_loss = self.discriminator_loss(g_logits.squeeze(1), true_y) # apply BCEWithLogitsLoss to discriminator's logit output
        loss = ce_loss + self.hparams['urm_adv_lambda']*g_loss

        return loss, feats

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
            
        loss, feats = self._compute_loss(all_x, all_y)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        self._update_discriminator(all_x, all_y, feats)
    
        return {'loss': loss.item()}




class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms - 使用CNNDA网络
        self.featurizer = networks.CNNDA()
        # 设置n_outputs属性，用于域判别器初始化
        self.featurizer.n_outputs = self.featurizer.feature_dim
        # 创建一个新的分类器接口，与原有架构兼容
        self.classifier = lambda features: self.featurizer(features, return_features=False)
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
            num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        # 使用CNNDA的return_features参数获取特征
        _, all_z = self.featurizer(all_x, return_features=True)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        input_grad = autograd.grad(
            F.cross_entropy(disc_out, disc_labels, reduction='sum'),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)






        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.predict(all_x)
        losses = torch.zeros(len(minibatches)).cuda()
        all_logits_idx = 0
        all_confs_envs = None

        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            losses[i] = F.cross_entropy(logits, y)
            
            nll = F.cross_entropy(logits, y, reduction = "none").unsqueeze(0)
        
            if all_confs_envs is None:
                all_confs_envs = nll
            else:
                all_confs_envs = torch.cat([all_confs_envs, nll], dim = 0)
                
        erm_loss = losses.mean()
        
        ## squeeze the risks
        all_confs_envs = torch.squeeze(all_confs_envs)
        
        ## find the worst domain
        worst_env_idx = torch.argmax(torch.clone(losses))
        all_confs_worst_env = all_confs_envs[worst_env_idx]

        ## flatten the risk
        all_confs_worst_env_flat = torch.flatten(all_confs_worst_env)
        all_confs_all_envs_flat = torch.flatten(all_confs_envs)
    
        matching_penalty = self.mmd(all_confs_worst_env_flat.unsqueeze(1), all_confs_all_envs_flat.unsqueeze(1)) 
        
        ## variance penalty
        variance_penalty = torch.var(all_confs_all_envs_flat)
        variance_penalty += torch.var(all_confs_worst_env_flat)
        
        total_loss = erm_loss + matching_penalty_weight * matching_penalty + variance_penalty_weight * variance_penalty
            
        if self.update_count == self.hparams['rdm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["rdm_lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.update_count += 1

        return {'update_count': self.update_count.item(), 'total_loss': total_loss.item(), 'erm_loss': erm_loss.item(), 'matching_penalty': matching_penalty.item(), 'variance_penalty': variance_penalty.item(), 'rdm_lambda' : self.hparams['rdm_lambda']}



        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
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
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))

class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
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
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))







    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0













    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)


class AbstractCAD(Algorithm):
    """Contrastive adversarial domain bottleneck (abstract class)
    from Optimal Representations for Covariate Shift <https://arxiv.org/abs/2201.00057>
    """

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, is_conditional):
        super(AbstractCAD, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        params = list(self.featurizer.parameters()) + list(self.classifier.parameters())

        # parameters for domain bottleneck loss
        self.is_conditional = is_conditional  # whether to use bottleneck conditioned on the label
        self.base_temperature = 0.07
        self.temperature = hparams['temperature']
        self.is_project = hparams['is_project']  # whether apply projection head
        self.is_normalized = hparams['is_normalized'] # whether apply normalization to representation when computing loss

        # whether flip maximize log(p) (False) to minimize -log(1-p) (True) for the bottleneck loss
        # the two versions have the same optima, but we find the latter is more stable
        self.is_flipped = hparams["is_flipped"]

        if self.is_project:
            self.project = nn.Sequential(
                nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs),
                nn.ReLU(inplace=True),
                nn.Linear(self.featurizer.n_outputs, 128),
            )
            params += list(self.project.parameters())

        # Optimizers
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def bn_loss(self, z, y, dom_labels):
        """Contrastive based domain bottleneck loss
         The implementation is based on the supervised contrastive loss (SupCon) introduced by
         P. Khosla, et al., in “Supervised Contrastive Learning“.
        Modified from  https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
        """
        device = z.device
        batch_size = z.shape[0]

        y = y.contiguous().view(-1, 1)
        dom_labels = dom_labels.contiguous().view(-1, 1)
        mask_y = torch.eq(y, y.T).to(device)
        mask_d = (torch.eq(dom_labels, dom_labels.T)).to(device)
        mask_drop = ~torch.eye(batch_size).bool().to(device)  # drop the "current"/"self" example
        mask_y &= mask_drop
        mask_y_n_d = mask_y & (~mask_d)  # contain the same label but from different domains
        mask_y_d = mask_y & mask_d  # contain the same label and the same domain
        mask_y, mask_drop, mask_y_n_d, mask_y_d = mask_y.float(), mask_drop.float(), mask_y_n_d.float(), mask_y_d.float()

        # compute logits
        if self.is_project:
            z = self.project(z)
        if self.is_normalized:
            z = F.normalize(z, dim=1)
        outer = z @ z.T
        logits = outer / self.temperature
        logits = logits * mask_drop
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        if not self.is_conditional:
            # unconditional CAD loss
            denominator = torch.logsumexp(logits + mask_drop.log(), dim=1, keepdim=True)
            log_prob = logits - denominator

            mask_valid = (mask_y.sum(1) > 0)
            log_prob = log_prob[mask_valid]
            mask_d = mask_d[mask_valid]

            if self.is_flipped:  # maximize log prob of samples from different domains
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (~mask_d).float().log(), dim=1)
            else:  # minimize log prob of samples from same domain
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (mask_d).float().log(), dim=1)
        else:
            # conditional CAD loss
            if self.is_flipped:
                mask_valid = (mask_y_n_d.sum(1) > 0)
            else:
                mask_valid = (mask_y_d.sum(1) > 0)

            mask_y = mask_y[mask_valid]
            mask_y_d = mask_y_d[mask_valid]
            mask_y_n_d = mask_y_n_d[mask_valid]
            logits = logits[mask_valid]

            # compute log_prob_y with the same label
            denominator = torch.logsumexp(logits + mask_y.log(), dim=1, keepdim=True)
            log_prob_y = logits - denominator

            if self.is_flipped:  # maximize log prob of samples from different domains and with same label
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_n_d.log(), dim=1)
            else:  # minimize log prob of samples from same domains and with same label
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_d.log(), dim=1)

        def finite_mean(x):
            # only 1D for now
            num_finite = (torch.isfinite(x).float()).sum()
            mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
            if num_finite != 0:
                mean = mean / num_finite
            else:
                return torch.tensor(0.0).to(x)
            return mean

        return finite_mean(bn_loss)

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        all_d = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        bn_loss = self.bn_loss(all_z, all_y, all_d)
        clf_out = self.classifier(all_z)
        clf_loss = F.cross_entropy(clf_out, all_y)
        total_loss = clf_loss + self.hparams['lmbda'] * bn_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"clf_loss": clf_loss.item(), "bn_loss": bn_loss.item(), "total_loss": total_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class CAD(AbstractCAD):
    """Contrastive Adversarial Domain (CAD) bottleneck

       Properties:
       - Minimize I(D;Z)
       - Require access to domain labels but not task labels
       """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=False)


class CondCAD(AbstractCAD):
    """Conditional Contrastive Adversarial Domain (CAD) bottleneck

    Properties:
    - Minimize I(D;Z|Y)
    - Require access to both domain labels and task labels
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CondCAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=True)


class Transfer(Algorithm):
    '''Algorithm 1 in Quantifying and Improving Transferability in Domain Generalization (https://arxiv.org/abs/2106.03632)'''
    ''' tries to ensure transferability among source domains, and thus transferabiilty between source and target'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Transfer, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.d_steps_per_g = hparams['d_steps_per_g']

        # Architecture
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

        # Optimizers
        if self.hparams['gda']:
            self.optimizer = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr'])
        else:
            self.optimizer = torch.optim.Adam(
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.adv_opt = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr_d'])

    def loss_gap(self, minibatches, device):
        ''' compute gap = max_i loss_i(h) - min_j loss_j(h), return i, j, and the gap for a single batch'''
        max_env_loss, min_env_loss =  torch.tensor([-float('inf')], device=device), torch.tensor([float('inf')], device=device)
        for x, y in minibatches:
            p = self.adv_classifier(self.featurizer(x))
            loss = F.cross_entropy(p, y)
            if loss > max_env_loss:
                max_env_loss = loss
            if loss < min_env_loss:
                min_env_loss = loss
        return max_env_loss - min_env_loss

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        # outer loop
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del all_x, all_y
        gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
        self.optimizer.zero_grad()
        gap.backward()
        self.optimizer.step()
        self.adv_classifier.load_state_dict(self.classifier.state_dict())
        for _ in range(self.d_steps_per_g):
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
        return {'loss': loss.item(), 'gap': -gap.item()}

    def update_second(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count = (self.update_count + 1) % (1 + self.d_steps_per_g)
        if self.update_count.item() == 1:
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            loss = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del all_x, all_y
            gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            self.optimizer.zero_grad()
            gap.backward()
            self.optimizer.step()
            self.adv_classifier.load_state_dict(self.classifier.state_dict())
            return {'loss': loss.item(), 'gap': gap.item()}
        else:
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
            return {'gap': -gap.item()}


    def predict(self, x):
        return self.classifier(self.featurizer(x))











class ADRMX(Algorithm):
    '''ADRMX: Additive Disentanglement of Domain Features with Remix Loss from (https://arxiv.org/abs/2308.06624)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ADRMX, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.register_buffer('update_count', torch.tensor([0]))

        self.num_classes = num_classes
        self.num_domains = num_domains
        self.mix_num = 1
        self.scl_int = SupConLossLambda(lamda=0.5)
        self.scl_final = SupConLossLambda(lamda=0.5)

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


        self.network = nn.Sequential(self.featurizer_label, self.classifier_label_1)

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

        self.update_count += 1
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        feat_label = self.featurizer_label(all_x)
        feat_domain = self.featurizer_domain(all_x)
        feat_combined = feat_label - feat_domain

        # get domain labels
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=all_x.device)
            for i, (x, _) in enumerate(minibatches)
        ])
        # predict domain feats from disentangled features
        disc_out = self.discriminator(feat_combined) 
        disc_loss = F.cross_entropy(disc_out, disc_labels) # discriminative loss for final labels (ascend/descend)

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        # alternating losses
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):
            # in discriminator turn
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'loss_disc': disc_loss.item()}
        else:
            # in generator turn

            # calculate CE from x_domain
            domain_preds = self.classifier_domain(feat_domain)
            classifier_loss_domain = F.cross_entropy(domain_preds, disc_labels) # domain clf loss
            classifier_remixed_loss = 0

            # calculate CE and contrastive loss from x_label
            int_preds = self.classifier_label_1(feat_label)
            classifier_loss_int = F.cross_entropy(int_preds, all_y) # intermediate CE Loss
            cnt_loss_int = self.scl_int(feat_label, all_y, disc_labels)

            # calculate CE and contrastive loss from x_dinv
            final_preds = self.classifier_label_2(feat_combined)
            classifier_loss_final = F.cross_entropy(final_preds, all_y) # final CE Loss
            cnt_loss_final = self.scl_final(feat_combined, all_y, disc_labels)

            # remix strategy
            for i in range(self.num_classes):
                indices = torch.where(all_y == i)[0]
                for _ in range(self.mix_num):
                    # get two instances from same class with different domains
                    perm = torch.randperm(indices.numel())
                    if len(perm) < 2:
                        continue
                    idx1, idx2 = perm[:2]
                    # remix
                    remixed_feat = feat_combined[idx1] + feat_domain[idx2]
                    # make prediction
                    pred = self.classifier_label_1(remixed_feat.view(1,-1))
                    # accumulate the loss
                    classifier_remixed_loss += F.cross_entropy(pred.view(1, -1), all_y[idx1].view(-1))
            # normalize
            classifier_remixed_loss /= (self.num_classes * self.mix_num)

            # generator loss negates the discrimination loss (negative update)
            gen_loss = (classifier_loss_int +
                        classifier_loss_final +
                        self.hparams["dclf_lambda"] * classifier_loss_domain +
                        self.hparams["rmxd_lambda"] * classifier_remixed_loss +
                        self.hparams['cnt_lambda'] * (cnt_loss_int + cnt_loss_final) + 
                        (self.hparams['disc_lambda'] * -disc_loss))
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
                'loss_remixed': classifier_remixed_loss.item(),
                }
    
    def predict(self, x):
        return self.network(x)
