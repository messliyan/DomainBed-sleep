"""
包含多种通用工具函数和辅助类，用于支持模型训练、数据处理、指标计算等基础任务，是项目的 “工具库”。
主要方法和类如下：
距离与投影函数
    distance(h1, h2)：计算两个网络（分类器）参数的欧氏距离（基于参数的 Frobenius 范数），用于衡量模型差异。
    proj(delta, adv_h, h)：将对抗网络adv_h投影到以h为中心、delta为半径的欧氏球内，用于对抗训练中的约束。
    l2_between_dicts(dict_1, dict_2)：计算两个参数字典（如模型状态字典）的 L2 距离，用于比较参数差异。
移动平均类
    ErmPlusPlusMovingAvg：维护网络参数的滑动平均（SMA），在训练迭代达到阈值后开始更新，用于稳定模型预测（可能用于 ERM++ 算法）。
    MovingAverage：基于指数移动平均（EMA）更新数据（如损失或特征），支持梯度校正，用于平滑训练过程中的波动。
数据处理工具
    make_weights_for_balanced_classes(dataset)：为不平衡数据集生成类别权重，使每个类别的样本在训练中贡献均衡的权重，缓解类别不平衡问题。
    split_dataset(dataset, n, seed)：将数据集按比例随机分割为两部分（如训练集和验证集），支持固定随机种子确保可复现性。
随机数据对生成函数
    random_pairs_of_minibatches(minibatches)：随机生成小批量数据对，用于对比学习或跨域数据增强。
评估与调试工具
    accuracy(network, loader, weights, device)：计算模型在给定数据加载器上的准确率，支持加权样本（如平衡权重）。
    pdb()：快速启动调试器（PDB），方便代码调试。
    seed_hash(*args)：将输入参数哈希为整数作为随机种子，确保实验的可复现性。
其他辅助功能
    print_separator()、print_row(row)：格式化输出实验结果（如表格），便于结果展示。
    Tee类：同时将输出写入控制台和文件，用于保存实验日志。
    ParamDict类：支持张量运算的有序字典，用于管理模型参数，方便参数的加减乘除等操作。
"""

import copy
import hashlib
import operator
import sys
from collections import Counter
from collections import OrderedDict
from itertools import cycle
from numbers import Number

import math
import numpy as np
import torch
from sklearn.metrics import f1_score


def distance(h1, h2):
    """计算两个网络模型参数之间的欧氏距离
    
    Args:
        h1: 第一个模型（分类器）对象
        h2: 第二个模型（分类器）对象
        
    Returns:
        torch.Tensor: 两个模型参数之间的欧氏距离
    """
    dist = 0.
    # 遍历模型的所有参数
    for param in h1.state_dict():
        # 获取两个模型的对应参数
        h1_param, h2_param = h1.state_dict()[param], h2.state_dict()[param]
        # 计算参数差异的Frobenius范数平方和
        dist += torch.norm(h1_param - h2_param) ** 2  # 对矩阵使用Frobenius范数
    # 返回欧氏距离（平方根）
    return torch.sqrt(dist)

def proj(delta, adv_h, h):
    """将对抗模型参数投影到以原始模型为中心的欧氏球内
    
    实现欧几里得投影：proj_{B(h, \delta)}(adv_h)
    确保对抗模型adv_h与原始模型h之间的距离不超过delta。
    
    Args:
        delta: 欧氏球的半径
        adv_h: 对抗模型（可能在球外）
        h: 原始模型，作为球的中心
        
    Returns:
        投影后的对抗模型
    """
    # 计算两个模型之间的距离
    dist = distance(adv_h, h)
    
    # 如果距离已经在允许范围内，直接返回
    if dist <= delta:
        return adv_h
    else:
        # 计算投影比例：将距离缩放到delta
        ratio = delta / dist
        # 对每个参数进行投影
        for param_h, param_adv_h in zip(h.parameters(), adv_h.parameters()):
            # 保留方向，缩放距离到delta
            param_adv_h.data = param_h + ratio * (param_adv_h - param_h)
        # 调试时可验证投影后距离是否正确
        # print("distance: ", distance(adv_h, h))
        return adv_h

def l2_between_dicts(dict_1, dict_2):
    """计算两个参数字典之间的L2距离
    
    常用于比较两个模型状态字典之间的差异。
    
    Args:
        dict_1: 第一个参数字典（如模型state_dict）
        dict_2: 第二个参数字典（如模型state_dict）
        
    Returns:
        torch.Tensor: 两个字典参数之间的平均L2距离
    """
    # 确保两个字典长度相同
    assert len(dict_1) == len(dict_2)
    
    # 按键排序获取参数值列表，确保顺序一致
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    
    # 计算所有参数的平均平方差
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()

class ErmPlusPlusMovingAvg:
    """维护网络参数的简单移动平均（SMA）
    
    用于ERM++算法中，在训练迭代达到阈值后开始更新移动平均模型，
    以稳定模型预测结果。
    """
    def __init__(self, network):
        """初始化移动平均对象
        
        Args:
            network: 要跟踪的原始网络模型
        """
        self.network = network  # 原始网络
        # 创建网络的深拷贝作为移动平均模型
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()  # 设置为评估模式
        self.sma_start_iter = 600  # 开始移动平均的迭代步数
        self.global_iter = 0  # 当前全局迭代步数
        self.sma_count = 0  # 移动平均更新计数器

    def update_sma(self):
        """更新移动平均模型参数
        
        在达到起始迭代步数后，使用简单移动平均更新模型参数。
        """
        self.global_iter += 1  # 增加迭代计数
        new_dict = {}  # 新的参数字典
        
        # 检查是否达到开始移动平均的迭代步数
        if self.global_iter >= self.sma_start_iter:
            self.sma_count += 1  # 增加移动平均计数器
            # 计算每个参数的移动平均值
            for (name, param_q), (_, param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                # 跳过批归一化的num_batches_tracked参数
                if 'num_batches_tracked' not in name:
                    # 计算简单移动平均：(prev * count + current) / (count + 1)
                    new_dict[name] = ((param_k.data.detach().clone() * self.sma_count + param_q.data.detach().clone()) / (1. + self.sma_count))
        else:
            # 未达到起始步数时，直接复制当前参数
            for (name, param_q), (_, param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    new_dict[name] = param_q.detach().data.clone()
        
        # 更新移动平均模型的参数
        self.network_sma.load_state_dict(new_dict)


class MovingAverage:
    """指数移动平均（EMA）计算类
    
    用于平滑数据序列，如损失值或特征表示。支持梯度校正，确保
    反向传播的梯度幅度不受EMA参数影响。
    """
    def __init__(self, ema, oneminusema_correction=True):
        """初始化移动平均对象
        
        Args:
            ema: 指数移动平均系数，范围[0,1)，接近1表示更平滑
            oneminusema_correction: 是否启用1/(1-ema)校正
        """
        self.ema = ema  # 指数移动平均系数
        self.ema_data = {}  # 存储每个数据项的当前EMA值
        self._updates = 0  # 更新计数器
        self._oneminusema_correction = oneminusema_correction  # 是否启用校正

    def update(self, dict_data):
        """使用新数据更新移动平均
        
        Args:
            dict_data: 包含需要更新的数据项的字典，键为名称，值为张量
            
        Returns:
            更新后的EMA数据字典
        """
        ema_dict_data = {}  # 存储更新后的EMA值
        
        for name, data in dict_data.items():
            # 将数据重塑为二维张量便于处理
            data = data.view(1, -1)
            
            # 获取前一个EMA值，如果是第一次更新则使用零张量
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            # 计算指数移动平均：ema * 前值 + (1-ema) * 新值
            ema_data = self.ema * previous_data + (1 - self.ema) * data
            
            # 应用梯度校正
            if self._oneminusema_correction:
                # 乘以1/(1-self.ema)，确保梯度幅度不受ema参数影响
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            
            # 保存当前EMA值（分离梯度）
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1  # 增加更新计数
        return ema_dict_data



def make_weights_for_balanced_classes(dataset):
    """为不平衡数据集生成类别权重
    
    计算每个类别的权重，使稀有类在训练中获得更高权重，
    缓解类别不平衡问题。
    
    Args:
        dataset: PyTorch数据集对象，每个样本返回(data, label)元组
        
    Returns:
        torch.Tensor: 长度等于数据集大小的权重张量，每个样本对应一个权重值
    """
    # 统计每个类别的样本数量
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    # 计算类别总数
    n_classes = len(counts)

    # 为每个类别计算权重
    weight_per_class = {}
    for y in counts:
        # 权重计算公式：1/(类别数量 * 总类别数)
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    # 生成样本权重张量
    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    """启动Python调试器
    
    恢复标准输出并重定向到PDB调试器，方便在代码中快速插入断点。
    提示用户输入'n'可跳转到父函数。
    """
    # 恢复标准输出（以防被重定向）
    sys.stdout = sys.__stdout__
    import pdb  # 导入Python调试器
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()  # 设置断点

def seed_hash(*args):
    """从输入参数生成确定性的随机种子
    
    将任意输入参数序列哈希为一个整数，用作随机种子，
    确保相同参数组合生成相同的随机种子，保证实验可复现性。
    
    Args:
        *args: 任意数量和类型的参数
        
    Returns:
        int: 用作随机种子的整数
    """
    # 将参数转换为字符串
    args_str = str(args)
    # 计算MD5哈希并转换为整数，限制在2^31范围内
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """数据集分割包装类
    
    用于创建原始数据集的子集，通过索引键列表来访问原始数据集中的特定元素。
    split_dataset函数的内部辅助类。
    """
    def __init__(self, underlying_dataset, keys):
        """初始化分割数据集
        
        Args:
            underlying_dataset: 原始数据集对象
            keys: 要包含在子集中的元素索引列表
        """
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset  # 原始数据集
        self.keys = keys  # 子集的索引键列表
    
    def __getitem__(self, key):
        """获取指定索引位置的元素
        
        Args:
            key: 在子集中的索引
            
        Returns:
            原始数据集中对应索引位置的元素
        """
        return self.underlying_dataset[self.keys[key]]
    
    def __len__(self):
        """返回子集的大小
        
        Returns:
            int: 子集中元素的数量
        """
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """随机分割数据集为两部分
    
    将给定的数据集随机分割为两个子集，第一个子集大小为n，第二个子集包含剩余的所有样本。
    使用固定种子确保分割结果可复现。
    
    Args:
        dataset: 要分割的原始数据集
        n: 第一个子集的大小
        seed: 随机种子，默认为0
        
    Returns:
        tuple: 包含两个_SplitDataset对象的元组，分别代表分割后的两个子集
    """
    assert(n <= len(dataset))  # 确保第一个子集大小不超过原始数据集
    
    # 创建索引列表并随机打乱
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)  # 使用固定种子打乱索引
    
    # 分割索引列表
    keys_1 = keys[:n]  # 第一个子集的索引
    keys_2 = keys[n:]  # 第二个子集的索引
    
    # 返回两个分割后的数据集
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    """生成随机的小批量数据对
    
    将输入的小批量列表随机打乱后，为每个小批量生成一个相邻的配对，
    用于对比学习或跨域数据增强等任务。确保配对的数据量相等。
    
    Args:
        minibatches: 小批量数据列表，每个元素是(数据,标签)元组
        
    Returns:
        list: 包含((数据1,标签1), (数据2,标签2))元组的列表
    """
    # 生成小批量索引的随机排列
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    # 为每个小批量创建一个配对
    for i in range(len(minibatches)):
        # 选择下一个索引，最后一个与第一个配对
        j = i + 1 if i < (len(minibatches) - 1) else 0

        # 获取配对的两个小批量数据和标签
        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        # 确定两个小批量中较小的大小
        min_n = min(len(xi), len(xj))

        # 截断到相同大小并添加到配对列表
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains-num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i,j in zip(meta_train, cycle(meta_test)):
         xi, yi = minibatches[i][0], minibatches[i][1]
         xj, yj = minibatches[j][0], minibatches[j][1]

         min_n = min(len(xi), len(xj))
         pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, weights, device):
    """计算模型在给定数据集上的准确率
    
    支持加权样本，可用于不平衡数据集的评估。
    
    Args:
        network: 要评估的神经网络模型
        loader: 数据加载器，提供批次数据和标签
        weights: 样本权重列表，None表示所有样本权重相等
        device: 计算设备（如'cuda'或'cpu'）
        
    Returns:
        float: 加权准确率
    """
    correct = 0  # 正确预测计数
    total = 0    # 总权重计数
    weights_offset = 0  # 权重偏移量
    
    network.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        for x, y in loader:
            # 将数据和标签移至指定设备
            x = x.to(device)
            y = y.to(device)
            
            # 获取模型预测
            p = network.predict(x)
            
            # 获取当前批次的样本权重
            if weights is None:
                batch_weights = torch.ones(len(x))  # 权重相等
            else:
                # 从权重列表中提取当前批次对应的权重
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            
            # 将权重移至相同设备
            batch_weights = batch_weights.to(device)
            
            # 根据输出维度判断是二分类还是多分类
            if p.size(1) == 1:  # 二分类
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:  # 多分类
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            
            # 累加总权重
            total += batch_weights.sum().item()
    
    network.train()  # 恢复训练模式
    return correct / total  # 返回加权准确率

def f1_score_metric(network, loader, weights, device, average='macro'):
    """计算模型在给定数据集上的F1分数
    
    支持加权样本，可用于不平衡数据集的评估。
    
    Args:
        network: 要评估的神经网络模型
        loader: 数据加载器，提供批次数据和标签
        weights: 样本权重列表，None表示所有样本权重相等
        device: 计算设备（如'cuda'或'cpu'）
        average: 平均方式，可选'macro', 'micro', 'weighted', 'binary'
        
    Returns:
        float: 加权F1分数
    """
    all_preds = []
    all_targets = []
    all_weights = []
    
    network.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        for x, y in loader:
            # 将数据和标签移至指定设备
            x = x.to(device)
            y = y.to(device)
            
            # 获取模型预测
            p = network.predict(x)
            
            # 获取预测类别
            if p.size(1) == 1:  # 二分类
                preds = p.gt(0).long().squeeze(1)
            else:  # 多分类
                preds = p.argmax(1)
            
            # 收集预测和目标标签
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
            # 收集权重（如果提供）
            if weights is not None:
                weights_offset = len(all_weights)
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                all_weights.extend(batch_weights.numpy())
    
    network.train()  # 恢复训练模式
    
    # 计算F1分数
    if weights is not None:
        # 使用加权F1分数
        return f1_score(all_targets, all_preds, average=average, sample_weight=all_weights)
    else:
        # 使用普通F1分数
        return f1_score(all_targets, all_preds, average=average)

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """支持张量运算的有序字典
    
    专门用于存储和操作模型参数（权重），支持参数之间的数学运算，
    便于实现动量优化等算法。
    """

    def __init__(self, *args, **kwargs):
        """初始化参数字典
        
        Args:
            *args: 传递给OrderedDict的位置参数
            **kwargs: 传递给OrderedDict的关键字参数
        """
        super().__init__(*args, **kwargs)

    def _prototype(self, other, op):
        """应用二元运算符到参数字典
        
        通用方法，将给定的二元运算符应用到字典中的每个元素。
        
        Args:
            other: 右操作数，可以是另一个字典或标量
            op: 二元运算符函数（如operator.add, operator.mul等）
            
        Returns:
            ParamDict: 包含运算结果的新参数字典
        """
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        """加法运算：self + other
        
        支持参数字典之间的加法或参数字典与标量的加法。
        
        Args:
            other: 右操作数
            
        Returns:
            ParamDict: 加法结果
        """
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        """乘法运算：other * self
        
        支持标量与参数字典的乘法。
        
        Args:
            other: 左操作数（标量）
            
        Returns:
            ParamDict: 乘法结果
        """
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__  # 乘法运算的简写形式

    def __neg__(self):
        """取负运算：-self
        
        Returns:
            ParamDict: 每个参数都取负值的新参数字典
        """
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        """减法运算：other - self
        
        支持标量与参数字典的减法或参数字典之间的减法。
        实现为 a - b := a + (-b)
        
        Args:
            other: 左操作数
            
        Returns:
            ParamDict: 减法结果
        """
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__  # 减法运算的简写形式

    def __truediv__(self, other):
        """除法运算：self / other
        
        支持参数字典与标量的除法或参数字典之间的除法。
        
        Args:
            other: 右操作数
            
        Returns:
            ParamDict: 除法结果
        """
        return self._prototype(other, operator.truediv)


############################################################
# A general PyTorch implementation of KDE. Builds on:
# https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/kde.py
############################################################

class Kernel(torch.nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, bw=None):
        super().__init__()
        self.bw = 0.05 if bw is None else bw

    def _diffs(self, test_Xs, train_Xs):
        """Computes difference between each x in test_Xs with all train_Xs."""
        test_Xs = test_Xs.view(test_Xs.shape[0], 1, *test_Xs.shape[1:])
        train_Xs = train_Xs.view(1, train_Xs.shape[0], *train_Xs.shape[1:])
        return test_Xs - train_Xs

    def forward(self, test_Xs, train_Xs):
        """Computes p(x) for each x in test_Xs given train_Xs."""

    def sample(self, train_Xs):
        """Generates samples from the kernel distribution."""


class GaussianKernel(Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_Xs, train_Xs):
        diffs = self._diffs(test_Xs, train_Xs)
        dims = tuple(range(len(diffs.shape))[2:])
        if dims == ():
            x_sq = diffs ** 2
        else:
            x_sq = torch.norm(diffs, p=2, dim=dims) ** 2

        var = self.bw ** 2
        exp = torch.exp(-x_sq / (2 * var))
        coef = 1. / torch.sqrt(2 * np.pi * var)

        return (coef * exp).mean(dim=1)

    def sample(self, train_Xs):
        # device = train_Xs.device
        noise = torch.randn(train_Xs.shape) * self.bw
        return train_Xs + noise

    def cdf(self, test_Xs, train_Xs):
        mus = train_Xs                                                      # kernel centred on each observation
        sigmas = torch.ones(len(mus), device=test_Xs.device) * self.bw      # bandwidth = stddev
        x_ = test_Xs.repeat(len(mus), 1).T                                  # repeat to allow broadcasting below
        return torch.mean(torch.distributions.Normal(mus, sigmas).cdf(x_))


def estimate_bandwidth(x, method="silverman"):
    x_, _ = torch.sort(x)
    n = len(x_)
    sample_std = torch.std(x_, unbiased=True)

    if method == 'silverman':
        # https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
        iqr = torch.quantile(x_, 0.75) - torch.quantile(x_, 0.25)
        bandwidth = 0.9 * torch.min(sample_std, iqr / 1.34) * n ** (-0.2)

    elif method.lower() == 'gauss-optimal':
        bandwidth = 1.06 * sample_std * (n ** -0.2)

    else:
        raise ValueError(f"Invalid method selected: {method}.")

    return bandwidth


class KernelDensityEstimator(torch.nn.Module):
    """The KernelDensityEstimator model."""

    def __init__(self, train_Xs, kernel='gaussian', bw_select='Gauss-optimal'):
        """Initializes a new KernelDensityEstimator.
        Args:
            train_Xs: The "training" data to use when estimating probabilities.
            kernel: The kernel to place on each of the train_Xs.
        """
        super().__init__()
        self.train_Xs = train_Xs
        self._n_kernels = len(self.train_Xs)

        if bw_select is not None:
            self.bw = estimate_bandwidth(self.train_Xs, bw_select)
        else:
            self.bw = None

        if kernel.lower() == 'gaussian':
            self.kernel = GaussianKernel(self.bw)
        else:
            raise NotImplementedError(f"'{kernel}' kernel not implemented.")

    @property
    def device(self):
        return self.train_Xs.device

    # TODO(eugenhotaj): This method consumes O(train_Xs * x) memory. Implement an iterative version instead.
    def forward(self, x):
        return self.kernel(x, self.train_Xs)

    def sample(self, n_samples):
        idxs = np.random.choice(range(self._n_kernels), size=n_samples)
        return self.kernel.sample(self.train_Xs[idxs])

    def cdf(self, x):
        return self.kernel.cdf(x, self.train_Xs)


############################################################
# PyTorch implementation of 1D distributions.
############################################################

EPS = 1e-16


class Distribution1D:
    def __init__(self, dist_function=None):
        """
        :param dist_function: function to instantiate the distribution (self.dist).
        :param parameters: list of parameters in the correct order for dist_function.
        """
        self.dist = None
        self.dist_function = dist_function

    @property
    def parameters(self):
        raise NotImplementedError

    def create_dist(self):
        if self.dist_function is not None:
            return self.dist_function(*self.parameters)
        else:
            raise NotImplementedError("No distribution function was specified during intialization.")

    def estimate_parameters(self, x):
        raise NotImplementedError

    def log_prob(self, x):
        return self.create_dist().log_prob(x)

    def cdf(self, x):
        return self.create_dist().cdf(x)

    def icdf(self, q):
        return self.create_dist().icdf(q)

    def sample(self, n=1):
        if self.dist is None:
            self.dist = self.create_dist()
        n_ = torch.Size([]) if n == 1 else (n,)
        return self.dist.sample(n_)

    def sample_n(self, n=10):
        return self.sample(n)


def continuous_bisect_fun_left(f, v, lo, hi, n_steps=32):
    val_range = [lo, hi]
    k = 0.5 * sum(val_range)
    for _ in range(n_steps):
        val_range[int(f(k) > v)] = k
        next_k = 0.5 * sum(val_range)
        if next_k == k:
            break
        k = next_k
    return k


class Normal(Distribution1D):
    def __init__(self, location=0, scale=1):
        self.location = location
        self.scale = scale
        super().__init__(torch.distributions.Normal)

    @property
    def parameters(self):
        return [self.location, self.scale]

    def estimate_parameters(self, x):
        mean = sum(x) / len(x)
        var = sum([(x_i - mean) ** 2 for x_i in x]) / (len(x) - 1)
        self.location = mean
        self.scale = torch.sqrt(var + EPS)

    def icdf(self, q):
        if q >= 0:
            return super().icdf(q)

        else:
            # To get q *very* close to 1 without numerical issues, we:
            # 1) Use q < 0 to represent log(y), where q = 1 - y.
            # 2) Use the inverse-normal-cdf approximation here:
            #    https://math.stackexchange.com/questions/2964944/asymptotics-of-inverse-of-normal-cdf
            log_y = q
            return self.location + self.scale * math.sqrt(-2 * log_y)


class Nonparametric(Distribution1D):
    def __init__(self, use_kde=True, bw_select='Gauss-optimal'):
        self.use_kde = use_kde
        self.bw_select = bw_select
        self.bw, self.data, self.kde = None, None, None
        super().__init__()

    @property
    def parameters(self):
        return []

    def estimate_parameters(self, x):
        self.data, _ = torch.sort(x)

        if self.use_kde:
            self.kde = KernelDensityEstimator(self.data, bw_select=self.bw_select)
            self.bw = torch.ones(1, device=self.data.device) * self.kde.bw

    def icdf(self, q):
        if not self.use_kde:
            # Empirical or step CDF. Differentiable as torch.quantile uses (linear) interpolation.
            return torch.quantile(self.data, float(q))

        if q >= 0:
            # Find quantile via binary search on the KDE CDF
            lo = torch.distributions.Normal(self.data[0], self.bw[0]).icdf(q)
            hi = torch.distributions.Normal(self.data[-1], self.bw[-1]).icdf(q)
            return continuous_bisect_fun_left(self.kde.cdf, q, lo, hi)

        else:
            # To get q *very* close to 1 without numerical issues, we:
            # 1) Use q < 0 to represent log(y), where q = 1 - y.
            # 2) Use the inverse-normal-cdf approximation here:
            #    https://math.stackexchange.com/questions/2964944/asymptotics-of-inverse-of-normal-cdf
            log_y = q
            v = torch.mean(self.data + self.bw * math.sqrt(-2 * log_y))
            return v

# --------------------------------------------------------
# LARS optimizer, implementation from MoCo v3:
# https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])


############################################################
# Supervised Contrastive Loss implementation from:
# https://arxiv.org/abs/2004.11362
############################################################
class SupConLossLambda(torch.nn.Module):
    def __init__(self, lamda: float=0.5, temperature: float=0.07):
        super(SupConLossLambda, self).__init__()
        self.temperature = temperature
        self.lamda = lamda

    def forward(self, features: torch.Tensor, labels: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        batch_size, _ = features.shape
        normalized_features = torch.nn.functional.normalize(features, p=2, dim=1)
        # create a lookup table for pairwise dot prods
        pairwise_dot_prods = torch.matmul(normalized_features, normalized_features.T)/self.temperature
        loss = 0
        nans = 0
        for i, (label, domain_label) in enumerate(zip(labels, domain_labels)):

            # take the positive and negative samples wrt in/out domain            
            cond_pos_in_domain = torch.logical_and(labels==label, domain_labels == domain_label) # take all positives
            cond_pos_in_domain[i] = False # exclude itself
            cond_pos_out_domain = torch.logical_and(labels==label, domain_labels != domain_label)
            cond_neg_in_domain = torch.logical_and(labels!=label, domain_labels == domain_label)
            cond_neg_out_domain = torch.logical_and(labels!=label, domain_labels != domain_label)

            pos_feats_in_domain = pairwise_dot_prods[cond_pos_in_domain]
            pos_feats_out_domain = pairwise_dot_prods[cond_pos_out_domain]
            neg_feats_in_domain = pairwise_dot_prods[cond_neg_in_domain]
            neg_feats_out_domain = pairwise_dot_prods[cond_neg_out_domain]


            # calculate nominator and denominator wrt lambda scaling
            scaled_exp_term = torch.cat((self.lamda * torch.exp(pos_feats_in_domain[:, i]), (1 - self.lamda) * torch.exp(pos_feats_out_domain[:, i])))
            scaled_denom_const = torch.sum(torch.cat((self.lamda * torch.exp(neg_feats_in_domain[:, i]), (1 - self.lamda) * torch.exp(neg_feats_out_domain[:, i]), scaled_exp_term))) + 1e-5

            # nof positive samples
            num_positives = pos_feats_in_domain.shape[0] + pos_feats_out_domain.shape[0] # total positive samples
            log_fraction = torch.log((scaled_exp_term / scaled_denom_const) + 1e-5) # take log fraction
            loss_i = torch.sum(log_fraction) / num_positives
            if torch.isnan(loss_i):
                nans += 1
                continue
            loss -= loss_i # sum and average over num positives
        return loss/(batch_size-nans+1) # avg over batch
