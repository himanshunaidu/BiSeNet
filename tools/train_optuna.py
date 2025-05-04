import sys
sys.path.insert(0, '.')

import argparse
import os
import subprocess

import torch
import torch.distributed as dist
import optuna
from optuna import Study, Trial

from configs import set_cfg_from_file
from tools.train_tune import train, setup_logger

def objective(trial: Trial):
    # Set hyperparameters to be optimized
    lr_start = 5*trial.suggest_float('lr_start', 1e-3, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-3, log=True)
    warmup_iters = trial.suggest_int('warmup_iters', 10, 100, step=45)
    # max_iter = trial.suggest_int('max_iter', 10000, 20000, step=5000)
    max_iter = 100

    print(f"Hyperparameters: lr_start={lr_start}, weight_decay={weight_decay}, warmup_iters={warmup_iters}, max_iter={max_iter}")

    # mious, fw_mious, cat_ious, f1_scores, macro_f1, micro_f1 = train(cfg)

    # return fw_mious[-1]  # Return the last value of fw_mious as the objective value

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///res/optuna/bisenetv2_custom.db",
                                study_name="bisenetv2_coco_accessibility_stage_1",
                                load_if_exists=True)
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:")
    print(study.best_params)