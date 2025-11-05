# 在原有训练脚本中添加脑电相关配置
def add_args(parser):
    parser.add_argument("--dataset", default="EEGSleepDataset")
    parser.add_argument("--seq_len", type=int, default=3000)  # 脑电序列长度
    parser.add_argument("--num_domains", type=int, default=3)  # 域数量
    parser.add_argument("--domain_loss_weight", type=float, default=0.1)

# 加载脑电数据集
if args.dataset == "EEGSleepDataset":
    dataset = datasets.EEGSleepDataset(
        root=args.data_dir,
        test_envs=args.test_env,
        hparams=hparams
    )