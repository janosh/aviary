import torch


def add_common_args(parser):

    add_test_val_args(parser)
    add_dataloader_args(parser)
    add_optimizer_args(parser)
    add_ensemble_args(parser)
    add_restart_args(parser)
    add_task_args(parser)

    args, _ = parser.parse_known_args()

    task_err = f"task must be regression or classification, got {args.task}"
    assert args.task in ["regression", "classification"], task_err

    if torch.cuda.is_available() and (not args.disable_cuda):
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    return args


def add_test_val_args(parser):
    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument(
        "--val-path",
        type=str,
        metavar="PATH",
        help="Path to independent validation set",
    )
    valid_group.add_argument(
        "--val-size",
        default=0.0,
        type=float,
        metavar="FLOAT",
        help="Proportion of data used for validation",
    )
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--test-path", type=str, metavar="PATH", help="Path to independent test set"
    )
    test_group.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        metavar="FLOAT",
        help="Proportion of data set for testing",
    )


def add_dataloader_args(parser):
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        metavar="INT",
        help="Number of data loading workers (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        "--bsize",
        default=128,
        type=int,
        metavar="INT",
        help="Mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--data-seed",
        default=0,
        type=int,
        metavar="INT",
        help="Seed used when splitting data sets (default: 0)",
    )
    parser.add_argument(
        "--sample",
        default=1,
        type=int,
        metavar="INT",
        help="Sub-sample the training set for learning curves",
    )


def add_optimizer_args(parser):
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="INT",
        help="Number of training epochs to run (default: 100)",
    )
    parser.add_argument(
        "--loss",
        default="L1",
        type=str,
        metavar="STR",
        help="Loss function if regression (default: 'L1')",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Specifies whether to use heteroscedastic loss variants",
    )
    parser.add_argument(
        "--optim",
        default="AdamW",
        type=str,
        metavar="STR",
        help="Optimizer used for training (default: 'AdamW')",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        default=3e-4,
        type=float,
        metavar="FLOAT",
        help="Initial learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="FLOAT [0,1]",
        help="Optimizer momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-6,
        type=float,
        metavar="FLOAT [0,1]",
        help="Optimizer weight decay (default: 1e-6)",
    )


def add_ensemble_args(parser):
    parser.add_argument(
        "--ensemble",
        default=1,
        type=int,
        metavar="INT",
        help="Number models to ensemble",
    )
    parser.add_argument(
        "--run-id",
        default="run_1",
        type=str,
        metavar="STR",
        help="Index for model in an ensemble of models",
    )


def add_restart_args(parser):
    use_group = parser.add_mutually_exclusive_group()
    use_group.add_argument(
        "--fine-tune", type=str, metavar="PATH", help="Checkpoint path for fine tuning"
    )
    use_group.add_argument(
        "--transfer",
        type=str,
        metavar="PATH",
        help="Checkpoint path for transfer learning",
    )


def add_task_args(parser):
    parser.add_argument(
        "--task",
        default="regression",
        type=str,
        metavar="STR",
        help="Regression or classification",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model/ensemble",
    )
    parser.add_argument("--train", action="store_true", help="Train the model/ensemble")

    # misc
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--log", action="store_true", help="Log training metrics to tensorboard"
    )
