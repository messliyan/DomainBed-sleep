import numpy as np
import os
import csv
from glob import glob
import argparse


def process_dataset(dataset_dir, output_file):
    """
    处理数据集，统计每个文件的睡眠阶段数量和数据标准化所需的均值、标准差
    
    Args:
        dataset_dir: 数据集目录路径
        output_file: 输出CSV文件路径
    """
    # 将相对路径转换为绝对路径
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dataset_dir)
    
    if not os.path.isabs(output_file):
        output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), output_file)
    # 获取所有npz文件
    npz_files = glob(os.path.join(dataset_dir, "*.npz"))
    print(f"找到 {len(npz_files)} 个npz文件")
    
    # 存储每个文件的统计信息
    file_stats = []
    
    # 存储所有文件的类别计数，用于显示整体统计
    total_class_counts = None
    
    # 存储所有数据用于计算整体标准化统计信息
    all_data = []
    
    # 处理每个文件
    for i, file_path in enumerate(npz_files, 1):
        file_name = os.path.basename(file_path)
        print(f"处理文件 {i}/{len(npz_files)}: {file_name}")
        
        try:
            # 加载文件
            data = np.load(file_path)
            x = data['x']
            y = data['y']
            fs = data.get('fs', 100)
            
            # 获取阶段数量
            stages_count = len(x)
            
            # 统计每个类别的数量
            # 假设标签范围是0-4（5个睡眠阶段）
            class_counts = np.bincount(y, minlength=5)
            
            # 更新总体类别计数
            if total_class_counts is None:
                total_class_counts = class_counts.copy()
            else:
                total_class_counts += class_counts
            
            # 收集数据用于计算整体标准化统计信息
            all_data.append(x)
            
            # 计算文件级别的均值和标准差
            file_mean = np.mean(x)
            file_std = np.std(x)
            
            # 保存文件统计信息
            file_stat = {
                'filename': file_name,
                'total_stages': stages_count,
                'wake_count': int(class_counts[0]),
                'n1_count': int(class_counts[1]),
                'n2_count': int(class_counts[2]),
                'n3_count': int(class_counts[3]),
                'rem_count': int(class_counts[4]),
                'sampling_rate': int(fs),
                'data_mean': float(file_mean),
                'data_std': float(file_std)
            }
            file_stats.append(file_stat)
            
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
            continue
    
    # 显示整体类别统计
    if total_class_counts is not None:
        print("\n整体类别统计:")
        print(f"清醒(Wake): {total_class_counts[0]}")
        print(f"N1: {total_class_counts[1]}")
        print(f"N2: {total_class_counts[2]}")
        print(f"N3: {total_class_counts[3]}")
        print(f"REM: {total_class_counts[4]}")
        
        # 计算并显示整体标准化统计信息
        if all_data:
            all_data_array = np.concatenate(all_data)
            overall_mean = np.mean(all_data_array)
            overall_std = np.std(all_data_array)
            
            print("\n整体数据标准化统计:")
            print(f"均值: {overall_mean}")
            print(f"标准差: {overall_std}")
            
            # 将整体统计信息添加到文件
            overall_stat = {
                'filename': '__OVERALL_STATS__',
                'total_stages': sum(total_class_counts),
                'wake_count': int(total_class_counts[0]),
                'n1_count': int(total_class_counts[1]),
                'n2_count': int(total_class_counts[2]),
                'n3_count': int(total_class_counts[3]),
                'rem_count': int(total_class_counts[4]),
                'sampling_rate': 100,  # 默认采样率
                'data_mean': float(overall_mean),
                'data_std': float(overall_std)
            }
            file_stats.insert(0, overall_stat)
    else:
        print("\n警告: 未能处理任何文件")
    
    # 保存统计信息到CSV文件（追加模式）
    # 文件已在main函数中创建并写入表头，这里直接追加数据
    try:
        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'total_stages', 'wake_count', 'n1_count', 
                          'n2_count', 'n3_count', 'rem_count', 'sampling_rate',
                          'data_mean', 'data_std']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 写入所有文件的统计信息
            for stat in file_stats:
                writer.writerow(stat)
        print(f"\n统计信息已追加到: {output_file}")
    except Exception as e:
        print(f"\n写入文件时出错: {str(e)}")

    return None


def main():
    parser = argparse.ArgumentParser(description='睡眠数据集统计信息生成器')
    parser.add_argument('--dataset_dir', type=str, 
                        default=os.path.join('eeg_data', 'shhs'),
                        help='数据集目录路径')
    parser.add_argument('--output_file', type=str, 
                        default=os.path.join('eeg_data/shhs', 'sleep_stats.csv'),
                        help='统计结果输出CSV文件路径')
    parser.add_argument('--all-datasets', action='store_true',
                        help='是否统计dataset目录下的所有数据集')
    parser.add_argument('--overwrite', action='store_true',
                        help='是否覆盖已存在的输出文件')

    args = parser.parse_args()


    if args.all_datasets:
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_dataset_dir = os.path.join(project_root, 'eeg_data')

        # 将相对路径转换为绝对路径
        if not os.path.isabs(args.output_file):
            args.output_file = os.path.join(project_root, args.output_file)

        print(f"开始统计 {main_dataset_dir} 目录下的所有数据集...")
        print(f"所有数据将写入到文件: {args.output_file}")

        # 检查是否需要创建新文件并写入表头
        
        # 每次都创建新文件并写入表头
        try:
            with open(args.output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'total_stages', 'wake_count', 'n1_count',
                                'n2_count', 'n3_count', 'rem_count', 'sampling_rate',
                                'data_mean', 'data_std']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            print(f"已创建新的输出文件: {args.output_file}")
        except Exception as e:
            print(f"创建输出文件时出错: {str(e)}")
            return

        # 获取dataset目录下的所有子目录
        if os.path.exists(main_dataset_dir):
            all_dataset_dirs = []
            for item in os.listdir(main_dataset_dir):
                item_path = os.path.join(main_dataset_dir, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    all_dataset_dirs.append(item_path)

            if not all_dataset_dirs:
                print(f"警告: 在 {main_dataset_dir} 目录下未找到子数据集目录")
            else:
                print(f"找到 {len(all_dataset_dirs)} 个数据集目录")

                # 为每个数据集生成统计信息，全部写入同一个文件
                for i, dataset_dir in enumerate(all_dataset_dirs, 1):
                    dataset_name = os.path.basename(dataset_dir)
                    print(f"\n[{i}/{len(all_dataset_dirs)}] 处理数据集: {dataset_name}")

                    # 处理单个数据集，直接使用同一个输出文件
                    process_dataset(dataset_dir, args.output_file)

                print("\n所有数据集统计完成！")
        else:
            print(f"错误: 数据集主目录 {main_dataset_dir} 不存在")
    else:
        # 处理单个数据集
        print(f"开始处理单个数据集: {args.dataset_dir}")
        print(f"统计信息将写入到文件: {args.output_file}")
        
        # 每次都创建新文件并写入表头
        try:
            with open(args.output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'total_stages', 'wake_count', 'n1_count',
                              'n2_count', 'n3_count', 'rem_count', 'sampling_rate',
                              'data_mean', 'data_std']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            print(f"已创建新的输出文件: {args.output_file}")
        except Exception as e:
            print(f"创建输出文件时出错: {str(e)}")
            return
        
        process_dataset(args.dataset_dir, args.output_file)


if __name__ == "__main__":
    main()