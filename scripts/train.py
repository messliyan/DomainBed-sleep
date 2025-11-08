# 导入必要的Python库
import argparse  # 命令行参数解析
import collections  # 高性能容器数据类型
import json  # JSON数据处理
import os  # 操作系统接口
import random  # 随机数生成
import sys  # 系统相关参数和函数
import time  # 时间相关函数

import numpy as np  # 科学计算库
import torch  # 深度学习框架
import torch.utils.data  # PyTorch数据工具
import algorithms  # 自定义算法模块
import datasets  # 自定义数据集模块
import hparams_registry  # 超参数注册表
from lib import misc  # 自定义工具函数
from lib.fast_data_loader import InfiniteDataLoader, FastDataLoader  # 高效数据加载器

# 主程序入口
if __name__ == "__main__":
   
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Domain adaptation')
    # 添加命令行参数
    parser.add_argument('--data_dir', type=str, default='eeg_data', 
                        help='数据目录路径，相对路径将被解析为项目根目录下的路径')

    parser.add_argument('--dataset', type=str, default="SleepDataset")
    parser.add_argument('--algorithm', type=str, default="DANN")
    parser.add_argument('--task', type=str, default="domain_adaptation",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=34,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=34,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0],
                        help='指定索引为1的环境作为测试环境（目标域）')
    parser.add_argument('--output_dir', type=str, default="results/dann_test_env0")
    parser.add_argument('--holdout_fraction', type=float, default=0.2,
                        help="")
    parser.add_argument('--uda_holdout_fraction', type=float, default=0.8,
        help="")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()  # 解析命令行参数
    
    # 智能处理数据路径：如果是相对路径，则视为相对于项目根目录
    if args.data_dir and not os.path.isabs(args.data_dir):
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    
    # 如果我们想实现检查点功能，只需每隔一段时间保存这些值，然后从磁盘加载它们。
    start_step = 0
    algorithm_dict = None # 加载预训练模型的算法字典
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    # 将标准输出和标准错误重定向到文件
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    # 打印环境信息
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))

    # 打印命令行参数
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    # 设置超参数
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    # 打印超参数
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    # 设置随机种子以确保结果可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置计算设备
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 加载数据集
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = [] # ：标记训练数据，用于源域训练
    out_splits = [] # 测试评估数据，用于性能评估
    uda_splits = [] # 未标记数据，用于域适应训练

    """
    数据集处理和模型训练的主循环代码
    包含数据集分割、加载器创建、模型训练、评估和保存等功能
    """
    # 遍历数据集中的每个环境
    for env_i, env in enumerate(dataset):
        uda = []  # 用于存储无标签数据的列表

    # 将数据集分割为训练集和测试集
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),  # 按比例分割
            misc.seed_hash(args.trial_seed, env_i))  # 使用随机种子确保可重复性

    # 如果当前环境是目标域环境，进一步分割出无标签数据
        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

    # 如果需要类别平衡，为每个数据集创建权重
        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    # 检查域适应任务是否有足够的无标签样本
    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    # 创建训练数据加载器
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    # 创建无标签数据加载器
    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

    # 创建评估数据加载器
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=128,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    # 获取算法类并初始化模型
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - len(args.test_envs), hparams)

    # 如果有预训练模型，加载其状态
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    # 将模型移动到指定设备
    algorithm.to(device)

    # 创建数据迭代器
    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    # 计算每个周期的步数
    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    # 设置总训练步数和检查点频率
    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    # 定义保存检查点的函数
    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    # 训练主循环
    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        # 获取并准备训练批次数据
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        # 如果是域适应任务，获取无标签数据
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        # 更新模型
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        # 收集检查点值
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        # 定期进行评估和保存
        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            # 计算并存储各种指标的平均值
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            # 在所有数据集上评估模型
            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc

            # 记录内存使用情况
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            # 打印结果
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            # 保存结果到文件
            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            # 保存模型状态
            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            # 如果需要，保存每个检查点的模型
            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    # 保存最终模型
    save_checkpoint('model.pkl')

    # 创建完成标记文件
    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
