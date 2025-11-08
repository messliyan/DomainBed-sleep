"""数据集加载模块

此模块实现了用于单通道脑电睡眠分期任务的数据集加载和处理功能。
主要提供了以下功能：
1. 定义多种数据集类，支持多环境（域）数据加载
2. 提供通用的数据加载和处理方法
3. 支持从npz文件中加载脑电数据和对应的睡眠阶段标签
4. 实现数据预处理和格式转换，使其适用于深度学习模型

数据集结构设计支持域适应任务，每个子目录代表一个不同的域环境，
便于进行跨域泛化和迁移学习研究。
"""
import os  # 导入操作系统模块，用于处理文件和目录路径
import numpy as np  # 导入NumPy库，用于科学计算
import torch  # 导入PyTorch库，用于深度学习

# 可用数据集列表
DATASETS = [
    # EEG datasets - 定义数据集列表，包含脑电数据集
    "SleepDataset",
]


def get_dataset_class(dataset_name):
    """获取指定名称的数据集类
    
    Args:
        dataset_name: 数据集名称字符串
        
    Returns:
        对应的数据集类
        
    Raises:
        NotImplementedError: 当指定的数据集名称不存在时抛出
    """
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    """获取指定数据集的环境数量
    
    Args:
        dataset_name: 数据集名称字符串
        
    Returns:
        数据集包含的环境数量
    """
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    """多域数据集基类
    
    所有多域数据集的抽象基类，定义了共享的属性和方法。
    """
    # 默认训练步数
    N_STEPS = 5001
    # 检查点保存频率
    CHECKPOINT_FREQ = 100
    # 数据加载的工作线程数
    N_WORKERS = 8
    # 环境名称列表，需由子类定义
    ENVIRONMENTS = None
    # 输入数据形状，需由子类定义
    INPUT_SHAPE = None

    def __getitem__(self, index):
        """获取指定索引的环境数据集
        
        Args:
            index: 环境索引
            
        Returns:
            对应索引的环境数据集
        """
        return self.datasets[index]

    def __len__(self):
        """获取环境数据集的数量
        
        Returns:
            环境数据集的数量
        """
        return len(self.datasets)


class TensorDataset(torch.utils.data.Dataset):
    """简单的张量数据集
    
    用于包装脑电数据和对应标签的数据集类，继承自PyTorch的Dataset。
    """
    def __init__(self, data, labels):
        """初始化张量数据集
        
        Args:
            data: 脑电数据张量
            labels: 对应的标签张量
        """
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        """获取指定索引的数据样本
        
        Args:
            index: 样本索引
            
        Returns:
            (数据样本, 标签) 元组
        """
        return self.data[index], self.labels[index]

    def __len__(self):
        """获取数据集的样本数量
        
        Returns:
            数据集的样本数量
        """
        return len(self.data)


class MultipleEnvironmentEEGDataset(MultipleDomainDataset):
    """多环境脑电数据集
    
    基于目录结构的多环境脑电数据集类，每个子文件夹作为一个环境（域）。
    适用于域适应和域泛化任务。
    """
    # 脑电数据训练步数
    N_STEPS = 8000
    # 检查点保存频率
    CHECKPOINT_FREQ = 800
    # 数据加载的工作线程数
    N_WORKERS = 8
    # 输入形状：单通道，3000个数据点（对应30秒脑电信号，采样率100Hz）
    INPUT_SHAPE = (1, 3000)
    # 类别数量：5个睡眠阶段（清醒W, 浅睡N1, 中睡N2, 深睡N3, 快速眼动REM）
    num_classes = 5

    def __init__(self, root, test_envs, hparams):
        """初始化多环境脑电数据集
        
        Args:
            root: 数据集根目录路径
            test_envs: 测试环境的索引列表
            hparams: 超参数字典
        """
        super().__init__()
        # 动态获取环境列表（子目录名称）
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)  # 排序确保环境顺序一致
   
        # 初始化数据集列表，每个元素对应一个环境的数据集
        self.datasets = []
        
        # 为每个环境创建数据集
        for i, environment in enumerate(environments):
            # 构建环境路径
            env_path = os.path.join(root, environment)
            # 加载该环境下所有的npz文件数据
            env_data, env_labels = self._load_environment_data(env_path)
            
            # 创建环境数据集
            if len(env_data) > 0:
                # 将NumPy数组转换为PyTorch张量，并添加通道维度
                self.datasets.append(TensorDataset(
                    torch.FloatTensor(env_data).unsqueeze(1),  # 添加通道维度，形状变为 [N, 1, 3000]
                    torch.LongTensor(env_labels)  # 标签应为长整型
                ))
                print(f"成功加载环境 {environment}: {len(env_data)} 个样本")
            else:
                # 如果环境中没有数据，创建空数据集
                print(f"警告: 环境 {environment} 没有有效数据")
                self.datasets.append(TensorDataset(
                    torch.FloatTensor([]),
                    torch.LongTensor([])
                ))
        
        # 设置输入形状
        self.input_shape = self.INPUT_SHAPE

    def _load_environment_data(self, env_path):
        """加载指定环境路径下的所有npz文件数据
        
        Args:
            env_path: 环境目录路径
            
        Returns:
            (数据数组, 标签数组) 元组
        """
        all_x = []  # 存储所有数据样本
        all_y = []  # 存储所有标签
        
        try:
            # 遍历环境目录下的所有npz文件
            for filename in os.listdir(env_path):
                if filename.endswith('.npz'):  # 只处理npz文件
                    file_path = os.path.join(env_path, filename)
                    try:
                        # 加载npz文件
                        data = np.load(file_path)
                        
                        # 确保获取正确的x和y数据
                        if 'x' in data and 'y' in data:
                            x = data['x']
                            y = data['y']
                            
                            # 数据格式检查和处理
                            # 确保x是二维数组 (samples, seq_length)
                            if len(x.shape) == 1:
                                # 如果是一维数组，重塑为二维
                                x = x.reshape(-1, 3000)  # 假设每个样本长度为3000
                            
                            # 确保x和y的样本数匹配
                            if len(x) == len(y):
                                all_x.append(x)
                                all_y.append(y)
                            else:
                                print(f"警告: 文件 {filename} 的数据和标签长度不匹配")
                        else:
                            print(f"警告: 文件 {filename} 缺少x或y键")
                    except Exception as e:
                        print(f"加载文件 {filename} 时出错: {str(e)}")
        except Exception as e:
            print(f"读取目录 {env_path} 时出错: {str(e)}")
        
        # 数据合并处理
        if not all_x:  # 如果没有加载到数据
            return np.array([]), np.array([])
        
        # 合并所有数据样本和标签
        return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)


class SleepDataset(MultipleEnvironmentEEGDataset):
    """睡眠数据集
    
    特定的睡眠数据集类，继承自多环境脑电数据集，定义了固定的环境列表。
    用于统一处理所有睡眠数据集的域适应和泛化任务。
    """
    # 定义环境名称列表（数据集）
    # 包含：SHHS (Sleep Heart Health Study) 和 Sleep-EDF-78 数据集
    # ENVIRONMENTS = ["sleep-edf-20", "sleep-edf-78", "shhs"]
    ENVIRONMENTS = ["sleep-edf-20", "sleep-edf-78"]
    
    def __init__(self, root, test_envs, hparams):
        """初始化睡眠数据集
        
        Args:
            root: 数据集根目录路径
            test_envs: 测试环境的索引列表
            hparams: 超参数字典
        """
        # 直接使用传入的root路径作为数据集根目录
        self.dir = root
        # 调用父类初始化方法，加载数据集
        super().__init__(self.dir, test_envs, hparams)
