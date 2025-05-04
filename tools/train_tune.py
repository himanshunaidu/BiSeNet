#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
This file will be used for specific hyper-parameter tuning, where fw_miou is the main metric.
"""

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import json
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.data import get_data_loader
from evaluate import eval_model, get_eval_model_results_single_scale
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
# from lib.logger import setup_logger, log_msg
from lib.logger import log_msg

from tqdm import tqdm

def setup_logger(name, logpth):
    """
    Overriding the default setup_logger function to prevent logging to a file.
    Instead, it will only log to the console.
    """
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and dist.get_rank() != 0:
        log_level = logging.WARNING
    try:
        logging.basicConfig(level=log_level, format=FORMAT, force=True)
    except Exception:
        for hl in logging.root.handlers: logging.root.removeHandler(hl)
        logging.basicConfig(level=log_level, format=FORMAT)
    logging.root.addHandler(logging.StreamHandler())

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
            default='configs/bisenetv2_coco_accessibility_stage_1.py',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    parse.add_argument('--lr-start', type=float, default=None,
            help='learning rate for the first iteration')
    parse.add_argument('--weight-decay', type=float, default=None,
            help='weight decay for the optimizer')
    parse.add_argument('--warmup-iters', type=int, default=None,
            help='number of warmup iterations')
    parse.add_argument('--max-iter', type=int, default=None,
            help='number of iterations for training')
    parse.add_argument('--respth', type=str, default='res/optuna',
            help='the path to save the trained model')
    return parse.parse_args()

def override_cfg(cfg, args):
    """
    Override the default config with command line arguments.
    """
    if 'finetune_from' in args and args.finetune_from is not None:
        cfg.finetune_from = args.finetune_from
    else:
        cfg.finetune_from = None
    if args.lr_start is not None:
        cfg.lr_start = args.lr_start
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.warmup_iters is not None:
        cfg.warmup_iters = args.warmup_iters
    if args.max_iter is not None:
        cfg.max_iter = args.max_iter

    if args.respth is not None:
        cfg.respth = args.respth
    print('cfg:', cfg.__dict__)
    return cfg

def set_model(cfg, lb_ignore=255):
    logger = logging.getLogger()
    net = model_factory[cfg.model_type](cfg.n_cats)
    if not cfg.finetune_from is None:
        logger.info(f'load pretrained weights from {cfg.finetune_from}')
        msg = net.load_state_dict(torch.load(cfg.finetune_from,
            map_location='cpu'), strict=False)
        logger.info('\tmissing keys: ' + json.dumps(msg.missing_keys))
        logger.info('\tunexpected keys: ' + json.dumps(msg.unexpected_keys))
    if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7, lb_ignore)
    criteria_aux = [OhemCELoss(0.7, lb_ignore)
            for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux


def set_optimizer(cfg, model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim


def set_model_dist(cfg, net):
    """
    Set the model to distributed mode.
    Is not required in the case of single GPU training.
    """
    local_rank = int(os.environ['LOCAL_RANK'])
    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank, ],
        #  find_unused_parameters=True,
        output_device=local_rank
        )
    return net


def set_meters(cfg):
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters



def train(cfg):
    logger = logging.getLogger()

    ## dataset
    dl = get_data_loader(cfg, mode='train')

    ## model
    net, criteria_pre, criteria_aux = set_model(cfg, dl.dataset.lb_ignore)

    ## optimizer
    optim = set_optimizer(cfg, net)

    ## mixed precision training
    scaler = amp.GradScaler()

    ## ddp training
    net = set_model_dist(cfg, net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters(cfg)

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    ## train loop
    for it, (im, lb) in tqdm(enumerate(dl), total=len(dl)):
        im = im.cuda()
        lb = lb.cuda()

        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        with amp.autocast(enabled=cfg.use_fp16):
            logits, *logits_aux = net(im)
            loss_pre = criteria_pre(logits, lb)
            loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
            loss = loss_pre + sum(loss_aux)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        torch.cuda.synchronize()

        time_meter.update()
        loss_meter.update(loss.item())
        loss_pre_meter.update(loss_pre.item())
        _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

        ## print training log message
        if (it + 1) % 100 == 0:
            lr = lr_schdr.get_lr()
            lr = sum(lr) / len(lr)
            msg = log_msg(
                it, cfg.max_iter, lr, time_meter, loss_meter,
                loss_pre_meter, loss_aux_meters)
            logger.info(msg)
        lr_schdr.step()

    ## dump the final model and evaluate the result
    save_pth = osp.join(cfg.respth, 'model_final.pth')
    logger.info('\nsave models to {}'.format(save_pth))
    state = net.module.state_dict()
    if dist.get_rank() == 0: torch.save(state, save_pth)

    logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()

    mious, fw_mious, cat_ious, f1_scores, macro_f1, micro_f1 = get_eval_model_results_single_scale(cfg, net.module)

    return {
        'mious': mious,
        'fw_mious': fw_mious,
        'cat_ious': cat_ious,
        'f1_scores': f1_scores,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1
    }


def main(cfg):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-train', cfg.respth)
    result = train(cfg)
    with open(osp.join(cfg.respth, 'result.json'), 'w') as f:
        f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    cfg = set_cfg_from_file(args.config)
    cfg = override_cfg(cfg, args)
    main(cfg)
