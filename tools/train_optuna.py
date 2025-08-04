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
    lr_start = trial.suggest_float('lr_start', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    freeze_type = trial.suggest_categorical('freeze_type', ['DETAIL', 'SEGMENT', 'HEAD', 'DETAIL_AND_SEGMENT'])
    # warmup_iters = trial.suggest_int('warmup_iters', 100, 1000, step=450)
    # max_iter = trial.suggest_int('max_iter', 1000, 2000, step=5000)
    warmup_iters = 10
    max_iter = 500

    respth = './res/optuna/bisenetv2_mapillary_accessibility_ios_point_mapper'

    print(f"Hyperparameters: lr_start={lr_start}, weight_decay={weight_decay}, warmup_iters={warmup_iters}, max_iter={max_iter}")
    print(f"Freeze type: {freeze_type}")

    # Run the train_tune script

    # Set the environment variable for distributed training
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['NGPUS'] = '1'
    ## Set the arguments for the script
    cmd = [
        'torchrun', '--nproc_per_node=1',
        'tools/train_tune.py',
        '--config', 'configs/mapillary_accessibility/bisenetv2_mapillary_accessibility_ios_point_mapper.py',
        '--finetune-from', 'res/bisenetv2_mapillary_accessibility_ios_point_mapper/model_final_mapillary_accessibility_stage_2.pth',
        '--freeze-type', freeze_type,
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
                                storage="sqlite:///res/optuna/bisenetv2_mapillary_accessibility_ios_point_mapper.db",
                                study_name="bisenetv2_mapillary_accessibility_ios_point_mapper",
                                load_if_exists=True)
    study.optimize(objective, n_trials=5)

    print("Best hyperparameters:")
    print(study.best_params)
    with open('res/optuna/bisenetv2_mapillary_accessibility_ios_point_mapper/best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)