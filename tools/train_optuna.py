import sys
sys.path.insert(0, '.')

import argparse
import os

import torch
import torch.distributed as dist
import optuna
from optuna import Study, Trial

from configs import set_cfg_from_file
from tools.train_tune import train, setup_logger

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
            default='configs/bisenetv2_coco_accessibility_stage_1.py',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()

def objective(trial: Trial):
    args = parse_args()
    cfg = set_cfg_from_file(args.config)

    cfg.__dict__['finetune_from'] = args.finetune_from

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    setup_logger('train', 'train.log')

    # Set hyperparameters to be optimized
    cfg.lr_start = 5*trial.suggest_float('lr_start', 1e-3, 1e-2, log=True)
    cfg.weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-3, log=True)
    cfg.warmup_iters = trial.suggest_int('warmup_iters', 10, 100, step=45)
    # cfg.max_iter = trial.suggest_int('max_iter', 10000, 20000, step=5000)
    cfg.max_iter = 100

    print(f"Hyperparameters: lr_start={cfg.lr_start}, weight_decay={cfg.weight_decay}, warmup_iters={cfg.warmup_iters}, max_iter={cfg.max_iter}")

    return train(cfg)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///res/optuna/bisenetv2_custom.db",
                                study_name="bisenetv2_coco_accessibility_stage_1",
                                load_if_exists=True)
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:")
    print(study.best_params)