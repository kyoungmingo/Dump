from argparse import ArgumentParser


__all__ = ["argument_parsing"]


def argument_parsing():
    parser = ArgumentParser(description="Action recognition model training/testing")

    # basic settings
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--cuda", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")

    parser.add_argument("--output_dir", default="./outputs", type=str)

    # dataset settings
    parser.add_argument(
        "--dataset_name",
        default="dv_yatav",
        type=str,
        choices=["dv_yatav", "dv_test", "dv_jeju"],
    )
    parser.add_argument("--dataset_root", default="./data", type=str)
    parser.add_argument("--num_workers", default=0, type=int)

    parser.add_argument("--vid_size", default=[128], nargs="+", type=int)
    parser.add_argument("--num_sample_frames", default=8, type=int)

    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=16, type=int)

    # BGR normalization parameters
    parser.add_argument(
        "--norm_mean", default=[0.37645, 0.394666, 0.43216], nargs=3, type=float
    )
    parser.add_argument(
        "--norm_std", default=[0.216989, 0.22145, 0.22803], nargs=3, type=float
    )

    # model settings
    parser.add_argument(
        "--model_name", default="r2plus1d_18", type=str, choices=["r2plus1d_18"]
    )
    parser.add_argument("--backbone_pretrained", default="", type=str)
    parser.add_argument("--pretrained", default="", type=str)

    # loss settings
    parser.add_argument("--label_smooth", default=False, action="store_true")

    parser.add_argument("--base_lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.95, type=float)

    parser.add_argument("--step", default=[20, 40], nargs="+", type=int)
    parser.add_argument("--max_epoch", default=60, type=int)
    parser.add_argument("--warmup_step", default=10, type=int)
    parser.add_argument("--warmup_factor", default=0.1, type=float)
    parser.add_argument("--gamma", default=0.1, type=float)

    parser.add_argument(
        "--optimizer", default="adam", type=str, choices=["sgd", "adam"]
    )

    # data augmentation settings
    parser.add_argument("--color_jitter", default=False, action="store_true")

    parser.add_argument("--random_horizontal_flip", default=0.5, type=float)
    parser.add_argument("--random_shift", default=False, action="store_true")
    parser.add_argument("--shift_ratio", default=3, type=float)

    parser.add_argument("--random_crop", default=False, action="store_true")
    parser.add_argument("--random_crop_scales", default=None, nargs=2, type=float)
    parser.add_argument("--crop_scale", default=1.1, type=float)

    parser.add_argument("--log_period", default=10, type=int)
    parser.add_argument("--eval_period", default=1, type=int)
    parser.add_argument("--save_period", default=4, type=int)

    # yolo box crop Test!!
    parser.add_argument("--box_crop", default=False, action="store_true")

    return parser.parse_args()
