# DomainBed-sleep：脑电数据域适应与域泛化框架

## 项目概述

DomainBed-sleep是一个专注于脑电数据集域泛化和域适应的深度学习框架。该项目的核心目标是解决脑电数据在不同数据源（域）之间的分布差异问题，提高模型在新环境中的泛化能力。

## 核心功能

- 支持多种域适应和域泛化算法
- 针对睡眠脑电数据集的专用数据处理流程
- 灵活的数据集划分策略，支持监督学习、域泛化和无监督域适应(UDA)
- 高效的数据加载机制和训练框架
- 完整的实验结果记录和评估系统

## 支持的算法

项目实现了多种域适应和域泛化算法，所有算法均基于PyTorch神经网络框架：

### 1. 基础方法
- **ERM (Empirical Risk Minimization)**：经验风险最小化，作为基准方法
- **ERM++**：ERM的增强版本，添加了移动平均网络和学习率调度机制

### 2. 距离度量方法
- **CORAL (Correlation Alignment)**：通过对齐不同域之间的二阶统计信息实现域适应
- **MMD (Maximum Mean Discrepancy)**：使用最大均值差异度量不同域之间的分布差异

### 3. 对抗学习方法
- **DANN (Domain-Adversarial Neural Networks)**：无条件域对抗神经网络
- **CDANN (Conditional Domain-Adversarial Neural Networks)**：条件域对抗神经网络

### 4. 其他方法
- URM、MTL、SagNet等多种域适应和域泛化算法，总计16种算法实现

## 数据集处理

### 支持的数据集
- SHHS (Sleep Heart Health Study)
- Sleep-EDF-78
- 自定义睡眠数据集

### 数据格式
- 单通道3000点（30秒）的脑电数据
- 5个睡眠阶段的分类任务
- 从npz文件加载数据

### 数据集划分策略

#### 标准数据集划分
- **in_splits**：训练数据，用于源域训练
- **out_splits**：测试评估数据，用于性能评估
- **uda_splits**：未标记数据，仅用于域适应(UDA)任务

#### 无监督域适应(UDA)场景下的划分
1. 对所有环境（包括源域和目标域）按照比例分割为训练集和测试集
2. 对于目标域环境，进一步分割出未标记数据用于域适应
3. 源域的训练数据用于有监督训练
4. 目标域的未标记数据用于域适应训练

## 核心算法实现

### DANN (无条件域对抗神经网络)

#### 模型结构
- **特征提取器(Featurizer)**：将输入脑电信号转换为特征表示
- **分类器(Classifier)**：基于特征表示预测睡眠阶段类别
- **判别器(Discriminator)**：尝试区分特征来自哪个域

#### 训练过程
1. **交替更新策略**：按比例更新判别器和生成器（特征提取器+分类器）
2. **判别器更新**：最小化域分类损失，添加梯度惩罚稳定训练
3. **生成器更新**：联合优化分类损失和对抗损失（最大化判别器错误率）

### CDANN (条件域对抗神经网络)

CDANN在DANN基础上的增强：
- 添加类别嵌入信息，使判别器输入包含特征表示和类别信息
- 实现类别平衡机制，更好地处理类别不平衡问题
- 特别适合类别分布在不同域之间差异较大的情况

## 数据加载机制

### 两种数据加载器

1. **InfiniteDataLoader**（用于训练）
   - 支持无限循环采样，适合迭代次数固定的训练模式
   - 可设置类别平衡权重，解决类别不平衡问题

2. **FastDataLoader**（用于评估）
   - 高效执行完整数据遍历，适合评估场景
   - 较大的批处理大小提高评估速度

## 环境要求

- Python 3.6+
- PyTorch 1.5+
- NumPy
- CUDA支持（推荐）

## 使用方法

### 训练模型

```bash
cd scripts
python train.py --dataset SleepDataset --algorithm DANN --task domain_adaptation --target_env 0 --data_dir ../eeg_data
```

### 参数说明
- `--dataset`：数据集名称
- `--algorithm`：使用的算法
- `--task`：任务类型（domain_generalization或domain_adaptation）
- `--target_env`：目标域环境索引
- `--data_dir`：数据目录路径
- `--hparams`：自定义超参数（JSON格式）
- `--steps`：训练步数

## 目录结构

```
DomainBed-sleep/
├── algorithms.py       # 算法实现
├── datasets.py         # 数据集加载
├── networks.py         # 网络结构定义
├── model_selection.py  # 模型选择方法
├── lib/                # 工具库
│   ├── fast_data_loader.py  # 高效数据加载器
│   └── misc.py        # 辅助函数
├── scripts/           # 训练和评估脚本
│   └── train.py       # 主训练脚本
└── eeg_data/          # 数据目录
```

## 研究意义

该项目专注于解决睡眠脑电数据在不同设备、不同被试或不同采集条件下的域适应问题。通过实现多种先进的域适应算法，提高了模型在新环境中的泛化能力，为睡眠研究和临床应用提供了可靠的技术支持。

## 许可证

[MIT License](LICENSE)