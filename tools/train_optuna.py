import sys
sys.path.insert(0, '.')

import argparse
import os
import subprocess
import json

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
    warmup_iters = trial.suggest_int('warmup_iters', 100, 1000, step=450)
    max_iter = trial.suggest_int('max_iter', 1000, 2000, step=5000)
    # warmup_iters = 1
    # max_iter = 5

    respth = './res/optuna/bisenetv2_coco_accessibility_stage_1'

    print(f"Hyperparameters: lr_start={lr_start}, weight_decay={weight_decay}, warmup_iters={warmup_iters}, max_iter={max_iter}")

    # Run the train_tune script

    # Set the environment variable for distributed training
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['NGPUS'] = '1'
    ## Set the arguments for the script
    cmd = [
        'torchrun', '--nproc_per_node=1',
        'tools/train_tune.py',
        '--config', 'configs/bisenetv2_coco_accessibility_stage_1.py',
        '--lr-start', str(lr_start),
        '--weight-decay', str(weight_decay),
        '--warmup-iters', str(warmup_iters),
        '--max-iter', str(max_iter),
        '--respth', respth
    ]

    # Execute the command
    subprocess.run(cmd, env=env)
    score = float('-inf')
    # Load the results from the training
    # Assuming the results are saved in a file named 'results.json'
    print("Current path: ", os.getcwd())
    try:
        with open(os.path.join(respth, 'result.json'), 'r') as f:
            results = json.load(f)
            # Extract the desired metric (e.g., mIoU) from the results
            # Assuming the results contain a key 'mIoU' for the metric
            score = results['fw_mious'][-1]  # Get the last value of fw_mious
    except Exception as e:
        print(f"Error loading results: {e}")
        # Handle the error (e.g., return a default value or raise an exception)
        score = float('-inf')
    
    return score  # Return the score for optimization

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///res/optuna/bisenetv2_coco_accessibility_stage_1.db",
                                study_name="bisenetv2_coco_accessibility_stage_1",
                                load_if_exists=True)
    study.optimize(objective, n_trials=5)

    print("Best hyperparameters:")
    print(study.best_params)