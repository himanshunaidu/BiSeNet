#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
This script also saves the intermediate model with the best validation loss.
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
from torchinfo import summary

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.data import get_data_loader
from evaluate import eval_model, get_eval_model_results_single_scale
from lib.ohem_ce_loss import OhemCELoss, OhemCEWeightedLoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, log_msg

from tqdm import tqdm

# For fine-tuning
from lib.models.bisenetv2 import FreezeType


## fix all random seeds
#  torch.manual_seed(123)
#  torch.cuda.manual_seed(123)
#  np.random.seed(123)
#  random.seed(123)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
            default='configs/bisenetv2.py',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    parse.add_argument('--freeze-type', type=str, default="NONE", 
            choices=[e.value for e in FreezeType],
            help='freeze type for fine-tuning: all, detail, segment, head, none')
    return parse.parse_args()

args = parse_args()
cfg = set_cfg_from_file(args.config)

def set_model(lb_ignore=255):
    logger = logging.getLogger()
    net = model_factory[cfg.model_type](cfg.n_cats)
    if not args.finetune_from is None:
        logger.info(f'load pretrained weights from {args.finetune_from}')
        msg = net.load_state_dict(torch.load(args.finetune_from,
            map_location='cpu'), strict=False)
        logger.info('\tmissing keys: ' + json.dumps(msg.missing_keys))
        logger.info('\tunexpected keys: ' + json.dumps(msg.unexpected_keys))
    if args.freeze_type != 'NONE':
        net.fine_tune_freeze(FreezeType[args.freeze_type])
    if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    if hasattr(cfg, 'custom_mapping_weights') and cfg.custom_mapping_weights is not None and len(cfg.custom_mapping_weights) > 0:
        criteria_pre = OhemCEWeightedLoss(0.7, cfg.custom_mapping_weights, lb_ignore)
        criteria_aux = [OhemCEWeightedLoss(0.7, cfg.custom_mapping_weights, lb_ignore)
                for _ in range(cfg.num_aux_heads)]
    else:
        criteria_pre = OhemCELoss(0.7, lb_ignore)
        criteria_aux = [OhemCELoss(0.7, lb_ignore)
                for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
        
        # Filter out frozen parameters
        wd_params = [p for p in wd_params if p.requires_grad]
        nowd_params = [p for p in nowd_params if p.requires_grad]
        lr_mul_wd_params = [p for p in lr_mul_wd_params if p.requires_grad]
        lr_mul_nowd_params = [p for p in lr_mul_nowd_params if p.requires_grad]
        
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
        
        # Filter out frozen parameters
        wd_params = [p for p in wd_params if p.requires_grad]
        non_wd_params = [p for p in non_wd_params if p.requires_grad]        
        
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


def set_model_dist(net):
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


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters



def train():
    logger = logging.getLogger()

    ## dataset
    dl = get_data_loader(cfg, mode='train')

    ## model
    net, criteria_pre, criteria_aux = set_model(dl.dataset.lb_ignore)

    ## optimizer
    optim = set_optimizer(net)

    ## mixed precision training
    scaler = amp.GradScaler()

    ## ddp training
    # if torch.cuda.device_count() > 1:
    net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    best_val_iou = 0.0

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

        ## save intermediate model if the validation loss is lower
        if (it + 1) % 10000 == 0:
            mious, fw_mious, cat_ious, f1_scores, macro_f1, micro_f1 = get_eval_model_results_single_scale(
                cfg, net.module
            )
            val_miou = sum(mious) / len(mious)
            logger.info('\nSaving intermediate model if the validation loss is lower')
            if val_miou > best_val_iou:
                save_pth = osp.join(cfg.respth, 'model_intermediate_{}.pth'.format(it))
                logger.info('\nsave models to {}'.format(save_pth))
                state = net.module.state_dict()
                if dist.get_rank() == 0: torch.save(state, save_pth)
                best_val_iou = val_miou
                logger.info('\nIntermediate model saved successfully')
            else:
                logger.info('\nIntermediate model not saved, validation loss did not improve')

        lr_schdr.step()

    ## dump the final model and evaluate the result
    save_pth = osp.join(cfg.respth, 'model_final.pth')
    logger.info('\nsave models to {}'.format(save_pth))
    state = net.module.state_dict()
    if dist.get_rank() == 0: torch.save(state, save_pth)

    logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()
    iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net.module)
    logger.info('\neval results of f1 score metric:')
    logger.info('\n' + tabulate(f1_content, headers=f1_heads, tablefmt='orgtbl'))
    logger.info('\neval results of miou metric:')
    logger.info('\n' + tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))

    return


def main():
    # if torch.cuda.device_count() > 1:
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    # else:
    #     torch.cuda.set_device(0)

    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-train', cfg.respth)
    train()


if __name__ == "__main__":
    main()
