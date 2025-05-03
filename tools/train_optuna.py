import argparse

from configs import set_cfg_from_file


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
            default='configs/bisenetv2.py',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()

if __name__ == "__main__":
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=20)

    print("Best hyperparameters:")
    # print(study.best_params)