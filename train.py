# import debugpy
# import torch.distributed as dist
# import os

# # Determine the rank of the current process
# rank = int(os.environ.get("RANK", 0))

# # Attach debugger to a specific rank, e.g., rank 0
# if rank == 0:
#     debugpy.listen(("localhost", 10002))  # Choose an available port
#     print("Waiting for debugger attach...")
#     debugpy.wait_for_client()
#     print("Debugger attached, continuing execution...")


import argparse
import os
import random
import sys

# Get the directory of the current file
# current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
# sys.path.append(current_dir)

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import hawk.tasks as tasks
from hawk.common.config import Config
from hawk.common.dist_utils import get_rank, init_distributed_mode
from hawk.common.logger import setup_logger
from hawk.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from hawk.common.registry import registry
from hawk.common.utils import now

# imports modules for registration
from hawk.datasets.builders import *
from hawk.models import *
from hawk.processors import *
from hawk.runners import *
from hawk.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=False, default="/remote-home/share/jiaqitang/Hawk/train_configs/visionbranch_stage2_finetune.yaml", help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)

    # datasets['webvid']['train'][0]
    # datasets
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
