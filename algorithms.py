"""域泛化算法实现模块

此模块实现了多种域泛化(Domain Generalization)和域适应(Domain Adaptation)算法。
域泛化是机器学习中的一个重要研究方向，旨在训练一个能够在未见过的数据分布(域)上表现良好的模型。

主要包含以下类型的算法：
1. 对抗学习方法(如DANN, CDANN)
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

# 所有可用算法的列表
ALGORITHMS = [
    'DANN',            # 域对抗神经网络（UDA方法）
    'CDANN',           # 条件域对抗神经网络（UDA方法）
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
        self.num_domains = num_domains

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


class AbstractDANN(Algorithm):
    """域对抗神经网络抽象基类
    
    实现了域对抗训练的核心逻辑，支持有监督的源域训练和无监督的目标域适应。
    """
    def __init__(self, input_shape, num_classes, num_domains, 
                 hparams, conditional):
        """初始化DANN算法
        
        Args:
            input_shape: 输入数据的形状
            num_classes: 类别数量
            num_domains: 域数量
            hparams: 超参数字典
            conditional: 是否使用条件对抗
        """
        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.grl = networks.GradientReversalLayer.apply
        
        # 调度器调用间隔（step）
        self.scheduler_step_interval = hparams.get('scheduler_step_interval', 100)  # 每100步调用一次调度器

        # 创建网络组件
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # 域分类器
        self.discriminator = networks.DomainDiscriminator(
            self.featurizer.n_outputs)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # 创建优化器
        self.disc_opt = torch.optim.AdamW(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            betas=(self.hparams['beta1'], 0.9)
        )  # 使用固定默认值

        self.gen_opt = torch.optim.AdamW(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            betas = (self.hparams['beta1'], 0.9)
        )  # 使用固定默认值
        
        # 创建源域学习率调度器 - 监控源域ACC
        self.source_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.gen_opt,
            patience=self.hparams.get('source_scheduler_patience', 10),
            factor=self.hparams.get('source_scheduler_factor', 0.5),
            min_lr=self.hparams.get('min_lr', 1e-6),  # 限制最低学习率
        )

        # 目标域调度器（判别器）
        self.target_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.disc_opt,
            patience=self.hparams.get('target_scheduler_patience', 1),  # 缩短耐心值
            factor=self.hparams.get('target_scheduler_factor', 0.2),  # 增大降价幅度
            min_lr=self.hparams.get('min_lr', 1e-6),  # 限制最低学习率
        )
        
        # 初始化调度器间隔参数
        self.scheduler_step_interval = self.hparams.get('scheduler_step_interval', 150)
        
        self.lambda_start =  0.0 # 初始对抗损失权重
      
        self.lambda_end = self.hparams['lambda_end']    # 最终对抗损失权重，默认1.0
        self.warmup_steps =self.hparams['warmup_steps']    # 预热步数
        self.current_lambda =  0.0
        self.total_steps =  self.hparams['n_steps'] # 总训练步数

        # 对抗强度的绝对上限，防止过强对抗
        self.max_lambda = self.hparams.get('max_lambda', 0.3) # 对抗强度的最大允许值
        # 判别器损失下限，用于动态调整对抗强度
        self.disc_loss_floor = self.hparams.get('disc_loss_floor', 0.7) # 判别器损失的理想下限值
        # 判别器损失严重过低阈值
        self.disc_loss_critical_threshold = self.hparams.get('disc_loss_critical_threshold', 0.1)
        # 记录历史判别器损失
        self.disc_loss_history = []
        self.disc_loss_history_length = self.hparams.get('disc_loss_history_length', 150)  # 保存最近的判别器损失
        # 保存原始判别器学习率，用于动态调整
        self.original_disc_lr = None

    def update(self, minibatches, unlabeled=None):
        """执行一步更新
        
        实现完整的DANN更新逻辑，包括：
        1. 从源域数据学习分类任务
        2. 使用域对抗训练学习域不变特征
        3. 利用目标域无标签数据进行域适应
        4. 方案三：自适应对抗强度（对抗损失权重线性增加）
        
        Args:
            minibatches: (x, y)元组列表，来自源域
            unlabeled: 来自目标域的无标签数据列表
            
        Returns:
            包含各种损失值的字典
        """
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        
        # 计算当前对抗损失权重（平衡版）
        current_step = self.update_count.item()
        if current_step <= self.warmup_steps:
            # 预热阶段：线性增加对抗损失权重
            self.current_lambda = self.lambda_start + (self.lambda_end - self.lambda_start) * (current_step / self.warmup_steps)
        else:
            # 预热后：更保守的增长策略，避免过强对抗
            # 线性增长，但增速更慢
            excess_ratio = (current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            growth_factor = 1.0 + 0.5 * excess_ratio  # 最多增长到初始lambda_end的1.5倍
            self.current_lambda = min(self.lambda_end * growth_factor, self.max_lambda)  # 应用绝对上限
        
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
        # 使用更平滑的梯度反转系数，避免对抗训练过强
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
        
        # 计算判别器损失（二元域适应场景不需要类别平衡）
        # DomainDiscriminator输出的是logits，使用二元交叉熵
        if self.num_domains == 2:
            # 二元域分类
            all_domain_labels = all_domain_labels.float()  # BCEWithLogitsLoss需要float类型
            disc_loss = F.binary_cross_entropy_with_logits(disc_out.squeeze(), all_domain_labels)
        else:
            # 多元域分类（保留原逻辑）
            disc_loss = F.cross_entropy(disc_out, all_domain_labels)
        
        # 计算梯度惩罚
        if self.hparams['grad_penalty'] > 0:
            if self.num_domains == 2:
                # 二元域分类
                input_grad = autograd.grad(
                    F.binary_cross_entropy_with_logits(disc_out.squeeze(), all_domain_labels, reduction='sum'),
                    [disc_input], create_graph=True)[0]
            else:
                # 多元域分类（保留原逻辑）
                input_grad = autograd.grad(
                    F.cross_entropy(disc_out, all_domain_labels, reduction='sum'),
                    [disc_input], create_graph=True)[0]
            grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
            disc_loss += self.hparams['grad_penalty'] * grad_penalty
        
        # 交替更新判别器和生成器
        d_steps_per_g = self.hparams['d_steps_per_g_step'] # 使用默认值1，确保配置不存在时仍能正常运行
        if (self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g):
            # 更新判别器 - 添加低损失保护机制
                should_update_discriminator = True
                
                # 初始化原始学习率（首次运行时）
                if self.original_disc_lr is None:
                    self.original_disc_lr = self.disc_opt.param_groups[0]['lr']
                
                # 检查判别器损失
                current_disc_loss = disc_loss.item()
                
                if current_disc_loss < self.disc_loss_critical_threshold:
                    # 当判别器损失过低时，暂停更新判别器并降低学习率
                    should_update_discriminator = False
                    # 降低判别器学习率
                    for param_group in self.disc_opt.param_groups:
                        param_group['lr'] = self.original_disc_lr * 0.1  # 降低到原始学习率的10%
                elif current_disc_loss < self.disc_loss_floor and current_disc_loss >= self.disc_loss_critical_threshold:
                    # 当判别器损失较低但未达到临界值时，降低学习率但不暂停更新
                    for param_group in self.disc_opt.param_groups:
                        # 动态调整学习率，损失越低，学习率越低
                        lr_factor = max(0.2, current_disc_loss / self.disc_loss_floor)
                        param_group['lr'] = self.original_disc_lr * lr_factor
                else:
                    # 当判别器损失恢复正常时，恢复原始学习率
                    for param_group in self.disc_opt.param_groups:
                        if param_group['lr'] < self.original_disc_lr:
                            # 渐进式恢复学习率，避免突变
                            param_group['lr'] = min(self.original_disc_lr, param_group['lr'] * 1.2)
                
                if should_update_discriminator:
                    self.disc_opt.zero_grad()
                    disc_loss.backward()
                    # 添加梯度裁剪，防止判别器更新过大
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.hparams.get('disc_grad_clip', 1.5))
                    self.disc_opt.step()
                
                # 记录判别器损失用于动态调整对抗强度
                self.disc_loss_history.append(disc_loss.item())
                if len(self.disc_loss_history) > self.disc_loss_history_length:
                    self.disc_loss_history.pop(0)
                
                # 目标域学习率调度 - 监控判别器损失
                with torch.no_grad():
                    # 按固定step间隔调用学习率调度器
                    current_step = self.update_count.item()
                    if current_step % self.scheduler_step_interval == 0:
                        self.target_scheduler.step(disc_loss)              
                
                return {
                    'disc_loss': disc_loss.item(),
                    'disc_lr': self.target_scheduler._last_lr[0] if hasattr(self.target_scheduler, '_last_lr') else self.hparams["lr_d"],
                    'current_lambda': self.current_lambda  # 记录当前对抗损失权重
                }
        else:
            # 更新生成器（特征提取器和分类器）
            # 动态调整对抗强度：如果判别器损失过低，降低对抗强度
            # 这有助于防止判别器过强导致的训练不平衡
            adaptive_lambda = self.current_lambda
            if len(self.disc_loss_history) > 0:
                avg_disc_loss = sum(self.disc_loss_history) / len(self.disc_loss_history)
                # 如果判别器损失过低，降低对抗强度
                if avg_disc_loss < self.disc_loss_floor:
                    # 损失越低，降低越多
                    reduction_factor = min(1.0, avg_disc_loss / self.disc_loss_floor)
                    adaptive_lambda = self.current_lambda * (0.9 + 0.1 * reduction_factor)
                    
            # 生成器的目标是最小化分类损失，同时最大化域对抗损失（通过负号实现）
            gen_loss = classifier_loss - adaptive_lambda * disc_loss
            
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            
            # 使用分类器损失进行学习率调度（比准确率计算更高效）
            with torch.no_grad():
                # 按固定step间隔调用学习率调度器
                current_step = self.update_count.item()
                if current_step % self.scheduler_step_interval == 0:
                    self.source_scheduler.step(classifier_loss)
                
            
            return {
                'gen_loss': gen_loss.item(),
                'class_lr': self.source_scheduler._last_lr[0] if hasattr(self.source_scheduler, '_last_lr') else self.hparams["lr_g"],
                'current_lambda': self.current_lambda  # 记录当前对抗损失权重
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
            hparams, conditional=False)

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
            hparams, conditional=True)
