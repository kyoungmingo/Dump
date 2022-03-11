import os
import sys
from pathlib import Path
from datetime import datetime

import nvsmi

from scripts import train, test
from utils.logging import Logger
from utils.config import argument_parsing
from utils.print_colors import print_warning, print_green



if __name__ == "__main__":
    import torch
    torch.multiprocessing.set_start_method('spawn')

    args = argument_parsing()

    MODE = "train" if args.train else "test"

    args.output_dir = Path(args.output_dir) / MODE
    args.output_dir.mkdir(exist_ok=True, parents=True)

    sys.stdout = Logger(
        args.output_dir / f"log_{MODE}({str(datetime.now()).replace(' ', '_')}).txt"
    )

    if args.cuda:
        GPU_IS_AVAILABLE = False
        for gpu_info in nvsmi.get_gpus():
            if gpu_info.gpu_util < 5 and gpu_info.mem_util < 5:
                GPU_IS_AVAILABLE = True
                print_green(f"This process will be handled in GPU:{gpu_info.id}!!")

                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_info.id
                break

            print_warning(f"GPU:{gpu_info.id} is handling other process...")

        if GPU_IS_AVAILABLE is False:
            print_warning("There is no available GPUs!")
            args.cuda = False
    print(f"Running with {'GPU' if args.cuda else 'CPU'}s...")

    print_green("User Config".center(30, "="))
    for k, v in args.__dict__.items():
        print_green(f"{k}: {v}")
    print_green("End".center(30, "="))

    if args.train:
        train(args)
    else:
        test(args)
