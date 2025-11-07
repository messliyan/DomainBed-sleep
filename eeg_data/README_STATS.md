# 睡眠数据集统计工具

本目录包含用于统计睡眠数据集信息的工具脚本，以及如何使用预计算的统计信息来优化模型训练的说明。

## 1. 统计脚本功能

`stats_generator.py` 是一个用于生成睡眠数据集统计信息的脚本，主要功能包括：

- 统计每个数据文件中的睡眠阶段数量（总阶段数和各类别阶段数）
- 计算整个数据集的类别分布
- 生成CSV格式的统计报表

## 2. 使用方法

### 基本使用

```bash
python stats_generator.py
```

这将使用默认参数处理 `shhs` 数据集，并在项目根目录生成统计文件。

### 自定义参数

```bash
python stats_generator.py --dataset_dir /path/to/dataset --output_file /path/to/output.csv
```

参数说明：
- `--dataset_dir`: 数据集目录路径，包含所有 npz 文件
- `--output_file`: 统计结果输出CSV文件路径，默认输出到项目根目录

## 3. 输出文件说明

### 3.1 CSV统计文件

生成的CSV文件（默认位于项目根目录的sleep_stats.csv）包含以下字段：
- `filename`: 文件名
- `total_stages`: 该文件中的总睡眠阶段数量
- `wake_count`: 清醒阶段(Wake)的数量
- `n1_count`: N1阶段的数量
- `n2_count`: N2阶段的数量
- `n3_count`: N3阶段的数量
- `rem_count`: REM阶段的数量
- `sampling_rate`: 采样率




### 4.2 懒加载优化建议

为了优化大型数据集的加载性能，可以结合预计算的统计信息进行懒加载：

1. **使用文件路径列表**: 只存储文件路径，需要时再加载数据
2. **预计算阶段数量**: 使用脚本生成的 `total_stages` 信息，可以避免在初始化时加载所有文件
3. **批量处理**: 根据文件大小和阶段数量，合理分组处理文件

## 5. 代码修改建议

如果需要修改训练代码以使用预计算的统计信息，可以参考以下方法：

### 修改 `data_loader.py`

可以在 `SleepDataset` 类的初始化中添加从统计文件加载信息的功能：

```python
import csv

def _load_precomputed_stats(self, stats_file='sleep_stats.csv'):
    """从预计算的统计文件加载信息（默认从根目录的sleep_stats.csv加载）"""
    file_stages_map = {}
    with open(stats_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_stages_map[row['filename']] = int(row['total_stages'])
    
    # 更新 self.file_stages_count
    self.file_stages_count = []
    for file in self.valid_files:
        filename = os.path.basename(file)
        self.file_stages_count.append(file_stages_map.get(filename, 0))
```



## 6. 注意事项

- 统计脚本只处理 `.npz` 格式的文件
- 如果数据集发生变化，需要重新运行统计脚本
- 预计算的统计信息应该与实际使用的数据集完全匹配
- 对于大型数据集，统计过程可能需要一些时间

