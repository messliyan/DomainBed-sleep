# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run sweeps
"""

import argparse
import copy
import hashlib
import json
import os
import shlex
import shutil

import numpy as np
import tqdm
import algorithms
import command_launchers
import datasets
from lib import misc


class Job:
    # 定义任务状态常量
    NOT_LAUNCHED = 'Not launched'  # 未启动
    INCOMPLETE = 'Incomplete'      # 未完成
    DONE = 'Done'                  # 已完成

    def __init__(self, train_args, sweep_output_dir):
        # 根据训练参数生成唯一哈希值
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        # 设置输出目录路径
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        # 深拷贝训练参数并设置输出目录
        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        
        # 构建训练命令
        command = ['python', '-m', 'scripts.train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        # 根据输出目录状态设置任务状态
        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        # 返回任务信息的字符串表示
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        # 启动任务
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        # 创建任务输出目录
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        # 获取所有命令并启动
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        # 删除任务
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')

def all_test_env_combinations(n):
    """
    生成所有可能的测试环境组合
    对于包含n >= 3个环境的数据集，返回所有1个和2个测试环境的组合
    """
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]

def make_args_list(n_trials, dataset_names, algorithms, n_hparams_from, n_hparams, steps,
    data_dir, task, holdout_fraction, single_test_envs, hparams):
    """
    生成参数列表
    包含所有可能的试验、数据集、算法、测试环境等组合
    
    现在此函数负责生成随机超参数并将其转换为JSON字符串传递给train.py
    """
    import json
    import hparams_registry
    
    args_list = []
    for trial_seed in range(n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                if single_test_envs:
                    # 单个测试环境
                    all_test_envs = [
                        [i] for i in range(datasets.num_environments(dataset))]
                else:
                    # 多个测试环境组合
                    all_test_envs = all_test_env_combinations(
                        datasets.num_environments(dataset))
                for test_envs in all_test_envs:
                    for hparams_seed in range(n_hparams_from, n_hparams):
                        # 构建训练参数字典
                        train_args = {}
                        train_args['dataset'] = dataset
                        train_args['algorithm'] = algorithm
                        train_args['test_envs'] = test_envs
                        train_args['holdout_fraction'] = holdout_fraction
                        # 设置hparams_seed为0，因为我们现在在sweep.py中生成随机超参数
                        train_args['hparams_seed'] = 0
                        train_args['data_dir'] = data_dir
                        train_args['task'] = task
                        train_args['trial_seed'] = trial_seed
                        train_args['seed'] = misc.seed_hash(dataset,
                            algorithm, test_envs, hparams_seed, trial_seed)
                        if steps is not None:
                            train_args['steps'] = steps
                        
                        # 生成随机超参数
                        seed = misc.seed_hash(dataset, algorithm, test_envs, hparams_seed, trial_seed)
                        random_hparams_dict = hparams_registry.random_hparams(algorithm, dataset, seed)
                        
                        # 如果用户提供了额外的超参数，将其与随机超参数合并
                        if hparams is not None:
                            # 解析用户提供的hparams
                            user_hparams = json.loads(hparams)
                            # 合并超参数，用户提供的超参数优先级更高
                            random_hparams_dict.update(user_hparams)
                        
                        # 将超参数转换为JSON字符串并添加到train_args
                        train_args['hparams'] = json.dumps(random_hparams_dict)
                        
                        args_list.append(train_args)
    return args_list

def ask_for_confirmation():
    # 请求用户确认
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

# 获取所有非调试数据集
DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS)
    parser.add_argument('--task', type=str, default="domain_generalization")
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--single_test_envs', action='store_true')
    parser.add_argument('--skip_confirmation', action='store_true')
    args = parser.parse_args()

    # 生成参数列表
    args_list = make_args_list(
        n_trials=args.n_trials,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        data_dir=args.data_dir,
        task=args.task,
        holdout_fraction=args.holdout_fraction,
        single_test_envs=args.single_test_envs,
        hparams=args.hparams
    )

    # 创建任务列表
    jobs = [Job(train_args, args.output_dir) for train_args in args_list]

    # 打印任务信息
    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    # 根据命令执行相应操作
    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)
