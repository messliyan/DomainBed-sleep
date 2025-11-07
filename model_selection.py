"""模型选择模块

此模块实现了不同的模型选择策略，用于在超参数和训练时间步之间选择最佳模型。
主要包括Oracle选择、训练域验证集选择、自动学习率训练域验证集选择和留一域交叉验证等方法。
"""
import itertools
import numpy as np


def get_test_records(records):
    """获取测试记录
    
    给定具有共同测试环境的记录，提取仅包含单个测试环境且没有其他测试环境的记录。
    
    Args:
        records: 记录集合，通常是一个数据集或记录列表
        
    Returns:
        过滤后的记录集合，只包含单一测试环境的记录
    """
    return records.filter(lambda r: len(r['args']['test_envs']) == 1)


class SelectionMethod:
    """模型选择方法的抽象基类
    
    定义了模型选择策略的通用接口和方法。子类需要实现具体的模型选择逻辑，
    用于在超参数和训练时间步之间选择最佳模型。
    """

    def __init__(self):
        """初始化方法
        
        抽象类不能直接实例化，抛出TypeError异常
        """
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """计算单次运行的最佳验证准确率和对应的测试准确率
        
        给定一次运行的所有记录，返回一个包含最佳验证准确率和对应测试准确率的字典。
        这是一个抽象方法，子类必须实现。
        
        Args:
            run_records: 单次运行的记录集合
            
        Returns:
            字典，格式为{'val_acc': 验证准确率, 'test_acc': 测试准确率}
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):
        """获取所有超参数组合的性能
        
        给定来自单个(dataset, algorithm, test env)组合的所有记录，
        返回按验证准确率排序的(运行准确率, 记录)元组列表。
        
        Args:
            records: 记录集合
            
        Returns:
            按验证准确率降序排序的元组列表，每个元组包含运行准确率字典和对应的记录
        """
        # 按超参数种子分组
        return (records.group('args.hparams_seed')
            # 对每个种子的记录计算运行准确率
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            )
            # 过滤掉无效结果
            .filter(lambda x: x[0] is not None)
            # 按验证准确率降序排序
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )

    @classmethod
    def sweep_acc(self, records):
        """获取超参数搜索的最佳测试准确率
        
        给定来自单个(dataset, algorithm, test env)组合的所有记录，
        返回验证准确率最高的运行的测试准确率。
        
        Args:
            records: 记录集合
            
        Returns:
            验证准确率最高的运行的测试准确率，如果没有有效记录则返回None
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            # 返回验证准确率最高的运行的测试准确率
            return _hparams_accs[0][0]['test_acc']
        else:
            return None


class OracleSelectionMethod(SelectionMethod):
    """Oracle选择方法
    
    使用测试域作为验证集的选择方法。与在所有检查点中选择最佳验证准确率不同，
    此方法选择最后一个检查点，即不进行早停。这是一种理论上的上限方法，
    因为在实际应用中通常无法访问测试数据。
    """
    name = "test-domain validation set (oracle)"

    @classmethod
    def run_acc(self, run_records):
        """计算Oracle选择方法的运行准确率
        
        选择最后一个检查点，并使用测试域的输出准确率作为验证准确率，
        测试域的输入准确率作为测试准确率。
        
        Args:
            run_records: 单次运行的记录集合
            
        Returns:
            字典，格式为{'val_acc': 验证准确率, 'test_acc': 测试准确率}，如果没有有效记录则返回None
        """
        # 过滤出只包含单个测试环境的记录
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        
        # 获取测试环境索引
        test_env = run_records[0]['args']['test_envs'][0]
        
        # 构建准确率键名
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        
        # 选择最后一个检查点
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        
        return {
            'val_acc': chosen_record[test_out_acc_key],  # 使用测试域输出准确率作为验证准确率
            'test_acc': chosen_record[test_in_acc_key]  # 使用测试域输入准确率作为测试准确率
        }


class IIDAccuracySelectionMethod(SelectionMethod):
    """训练域验证集选择方法
    
    使用所有训练域的平均输出准确率作为验证标准，选择验证准确率最高的模型。
    这是一种常用的模型选择策略，避免使用测试数据进行验证。
    """
    name = "training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """计算单个记录的准确率
        
        给定单个记录，计算训练域的平均输出准确率作为验证准确率，
        以及测试域的输入准确率作为测试准确率。
        
        Args:
            record: 单个训练记录
            
        Returns:
            字典，格式为{'val_acc': 验证准确率, 'test_acc': 测试准确率}
        """
        # 获取测试环境索引
        test_env = record['args']['test_envs'][0]
        
        # 收集所有训练环境的输出准确率键名
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break  # 当没有更多环境时停止
            if i != test_env:
                val_env_keys.append(f'env{i}_out_acc')  # 只添加训练环境
        
        # 构建测试准确率键名
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        
        # 计算平均验证准确率和测试准确率
        return {
            'val_acc': np.mean([record[key] for key in val_env_keys]),  # 训练域平均输出准确率
            'test_acc': record[test_in_acc_key]  # 测试域输入准确率
        }

    @classmethod
    def run_acc(self, run_records):
        """计算训练域验证集方法的运行准确率
        
        过滤出测试记录，对每个记录计算准确率，然后选择验证准确率最高的记录。
        
        Args:
            run_records: 单次运行的记录集合
            
        Returns:
            验证准确率最高的记录对应的{'val_acc': 验证准确率, 'test_acc': 测试准确率}字典，
            如果没有有效记录则返回None
        """
        # 获取测试记录
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        
        # 对每个记录计算准确率并选择验证准确率最高的
        return test_records.map(self._step_acc).argmax('val_acc')


class IIDAutoLRAccuracySelectionMethod(SelectionMethod):
    """自动学习率训练域验证集选择方法
    
    与IIDAccuracySelectionMethod类似，但使用自动学习率优化后的测试域输入准确率。
    这是为了支持自动学习率调整功能。
    """
    name = "auto lr training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """计算单个记录的准确率
        
        给定单个记录，计算训练域的平均输出准确率作为验证准确率，
        以及自动学习率优化后的测试域输入准确率作为测试准确率。
        
        Args:
            record: 单个训练记录
            
        Returns:
            字典，格式为{'val_acc': 验证准确率, 'test_acc': 测试准确率}
        """
        # 获取测试环境索引
        test_env = record['args']['test_envs'][0]
        
        # 收集所有训练环境的输出准确率键名
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break  # 当没有更多环境时停止
            if i != test_env:
                val_env_keys.append(f'env{i}_out_acc')  # 只添加训练环境
        
        # 构建自动学习率优化后的测试准确率键名
        test_in_acc_key = 'fd_env{}_in_acc'.format(test_env)
        
        # 计算平均验证准确率和自动学习率优化后的测试准确率
        return {
            'val_acc': np.mean([record[key] for key in val_env_keys]),  # 训练域平均输出准确率
            'test_acc': record[test_in_acc_key]  # 自动学习率优化后的测试域输入准确率
        }

    @classmethod
    def run_acc(self, run_records):
        """计算自动学习率训练域验证集方法的运行准确率
        
        过滤出测试记录，对每个记录计算准确率，然后选择验证准确率最高的记录。
        
        Args:
            run_records: 单次运行的记录集合
            
        Returns:
            验证准确率最高的记录对应的{'val_acc': 验证准确率, 'test_acc': 测试准确率}字典，
            如果没有有效记录则返回None
        """
        # 获取测试记录
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        
        # 对每个记录计算准确率并选择验证准确率最高的
        return test_records.map(self._step_acc).argmax('val_acc')


class LeaveOneOutSelectionMethod(SelectionMethod):
    """留一域交叉验证选择方法
    
    使用留一域交叉验证的策略选择最佳模型。对于每个测试环境，
    使用其他环境作为验证集，避免直接使用测试数据。
    """
    name = "leave-one-domain-out cross-validation"

    @classmethod
    def _step_acc(self, records):
        """计算单个步骤的准确率
        
        给定对应于单个步骤的记录组，计算留一域交叉验证的准确率。
        
        Args:
            records: 对应于单个步骤的记录组
            
        Returns:
            字典，格式为{'val_acc': 验证准确率, 'test_acc': 测试准确率}，如果数据不完整则返回None
        """
        # 获取测试记录
        test_records = get_test_records(records)
        if len(test_records) != 1:
            return None  # 需要恰好一个测试记录

        # 获取测试环境索引
        test_env = test_records[0]['args']['test_envs'][0]
        
        # 计算环境总数
        n_envs = 0
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break  # 当没有更多环境时停止
            n_envs += 1
        
        # 初始化验证准确率数组，-1表示未计算
        val_accs = np.zeros(n_envs) - 1
        
        # 计算每个验证环境的准确率
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2):
            # 找出除了测试环境外的另一个环境作为验证环境
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop()
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)]
        
        # 移除测试环境的准确率，只保留训练环境
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:])
        
        # 检查是否有缺失的验证环境准确率
        if any([v == -1 for v in val_accs]):
            return None  # 数据不完整
        
        # 计算平均验证准确率
        val_acc = np.sum(val_accs) / (n_envs - 1)
        
        return {
            'val_acc': val_acc,  # 交叉验证平均准确率
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)]  # 测试域输入准确率
        }

    @classmethod
    def run_acc(self, records):
        """计算留一域交叉验证方法的运行准确率
        
        按步骤分组，对每个步骤计算准确率，然后选择验证准确率最高的步骤。
        
        Args:
            records: 单次运行的记录集合
            
        Returns:
            验证准确率最高的步骤对应的{'val_acc': 验证准确率, 'test_acc': 测试准确率}字典，
            如果没有有效记录则返回None
        """
        # 按步骤分组并计算每个步骤的准确率
        step_accs = records.group('step').map(lambda step, step_records:
            self._step_acc(step_records)
        ).filter_not_none()  # 过滤掉无效结果
        
        if len(step_accs):
            # 选择验证准确率最高的步骤
            return step_accs.argmax('val_acc')
        else:
            return None
