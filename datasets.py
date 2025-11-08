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
    N_STEPS = 15000
    # 检查点保存频率
    CHECKPOINT_FREQ = 1000
    # 数据加载的工作线程数
    N_WORKERS = 8
    # 默认输入形状：单通道，3000个数据点（对应30秒脑电信号，采样率100Hz）
    INPUT_SHAPE = (1, 3000)
    # 类别数量：5个睡眠阶段（清醒W, 浅睡N1, 中睡N2, 深睡N3, 快速眼动REM）
    num_classes = 5
    # 各环境特定参数配置
    ENVIRONMENT_PARAMS = {
        "shhs": {
            "samples_per_window": 3750  # 125Hz * 30s = 3750
        }
        # 其他环境默认为3000点(100Hz * 30s)
    }

    def __init__(self, root, test_envs, hparams):
        """初始化多环境脑电数据集
        
        Args:
            root: 数据集根目录路径
            test_envs: 测试环境的索引列表
            hparams: 超参数字典
        """
        super().__init__()
        # 优先使用子类定义的ENVIRONMENTS列表，如果存在的话
        if hasattr(self, 'ENVIRONMENTS') and self.ENVIRONMENTS is not None:
            environments = self.ENVIRONMENTS
            print(f"使用预定义的环境列表: {environments}")
        else:
            # 否则动态获取环境列表（子目录名称）
            environments = [f.name for f in os.scandir(root) if f.is_dir()]
            environments = sorted(environments)  # 排序确保环境顺序一致
            print(f"动态扫描到的环境列表: {environments}")
   
        # 初始化数据集列表，每个元素对应一个环境的数据集
        self.datasets = []
        # 存储每个环境的输入形状
        self.environment_input_shapes = []
        
        # 为每个环境创建数据集
        for i, environment in enumerate(environments):
            # 获取环境特定参数
            env_params = self._get_environment_params(environment)
            samples_per_window = env_params.get("samples_per_window", 3000)
            
            # 构建环境路径
            env_path = os.path.join(root, environment)
            # 加载该环境下所有的npz文件数据
            env_data, env_labels = self._load_environment_data(env_path)
            
            # 创建环境数据集
            if len(env_data) > 0:
                # 将NumPy数组转换为PyTorch张量，并确保正确的维度格式
                data_tensor = torch.FloatTensor(env_data)
                # 移除可能存在的额外维度
                data_tensor = torch.squeeze(data_tensor)
                # 确保数据是二维的 [N, samples_per_window]，然后添加通道维度
                if len(data_tensor.shape) == 1:
                    data_tensor = data_tensor.reshape(-1, samples_per_window)
              
                data_tensor = data_tensor.unsqueeze(1)
                self.datasets.append(TensorDataset(
                    data_tensor,
                    torch.LongTensor(env_labels)  # 标签应为长整型
                ))
                # 保存当前环境的输入形状
                self.environment_input_shapes.append((1, samples_per_window))
                print(f"成功加载环境 {environment}: {len(env_data)} 个样本, 输入形状: (1, {samples_per_window})")
            else:
                # 如果环境中没有数据，创建空数据集
                print(f"警告: 环境 {environment} 没有有效数据")
                self.datasets.append(TensorDataset(
                    torch.FloatTensor([]),
                    torch.LongTensor([])
                ))
                # 空数据集使用默认输入形状
                self.environment_input_shapes.append(self.INPUT_SHAPE)
        
        # 设置默认输入形状
        self.input_shape = self.INPUT_SHAPE

    def _get_environment_params(self, environment):
        """获取特定环境的参数
        
        Args:
            environment: 环境名称
            
        Returns:
            包含环境特定参数的字典
        """
        # 如果环境在预定义参数中，则返回对应的参数，否则使用默认参数
        if environment in self.ENVIRONMENT_PARAMS:
            return self.ENVIRONMENT_PARAMS[environment]
        else:
            return {}
            
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
    ENVIRONMENTS = ["sleep-edf-78", "shhs"]
    # ENVIRONMENTS = ["sleep-edf-20", "sleep-edf-78"]
    
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
