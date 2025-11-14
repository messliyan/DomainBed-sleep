"""
包含多种通用工具函数和辅助类，用于支持模型训练、数据处理、指标计算等基础任务，是项目的 "工具库"。
主要方法和类如下：
移动平均类
    MovingAverage：基于指数移动平均（EMA）更新数据（如损失或特征），支持梯度校正，用于平滑训练过程中的波动。
数据处理工具
    make_weights_for_balanced_classes(dataset)：为不平衡数据集生成类别权重，使每个类别的样本在训练中贡献均衡的权重，缓解类别不平衡问题。
    split_dataset(dataset, n, seed)：将数据集按比例随机分割为两部分（如训练集和验证集），支持固定随机种子确保可复现性。
评估与调试工具
    accuracy(network, loader, weights, device)：计算模型在给定数据加载器上的准确率，支持加权样本（如平衡权重）。
    pdb()：快速启动调试器（PDB），方便代码调试。
    seed_hash(*args)：将输入参数哈希为整数作为随机种子，确保实验的可复现性。
其他辅助功能
    print_separator()、print_row(row)：格式化输出实验结果（如表格），便于结果展示。
    Tee类：同时将输出写入控制台和文件，用于保存实验日志。
    ParamDict类：支持张量运算的有序字典，用于管理模型参数，方便参数的加减乘除等操作。
"""

import hashlib
import operator
import sys
from collections import Counter
from collections import OrderedDict
from numbers import Number

import math
import numpy as np
import torch
from sklearn.metrics import f1_score


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
        end_ = "\\"
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
