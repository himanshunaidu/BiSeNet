
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

import lib.data.transform_cv2 as T
from lib.data.sampler import RepeatedDistSampler

from lib.data.base_dataset import BaseDataset, BaseDatasetKwargs
from lib.data.cityscapes_cv2 import CityScapes
from lib.data.coco import CocoStuff
from lib.data.coco_accessibility import CocoStuffAccessibility
# from lib.data.coco_accessibility_2 import CocoStuffAccessibility2
from lib.data.coco_custom_edge_mapping import CocoStuffAccessibilityCustomEdgeMapping
# from lib.data.coco_custom_edge_mapping_2 import CocoStuffAccessibilityCustomEdgeMapping2
from lib.data.custom_ios_point_mapper import CustomIOSPointMapper
from lib.data.ade20k import ADE20k
from lib.data.customer_dataset import CustomerDataset





def get_data_loader(cfg, mode='train'):
    if mode == 'train':
        trans_func = T.TransformationTrain(cfg.scales, cfg.cropsize)
        batchsize = cfg.ims_per_gpu
        annpath = cfg.train_im_anns
        shuffle = True
        drop_last = True
    elif mode == 'val':
        trans_func = T.TransformationVal()
        batchsize = cfg.eval_ims_per_gpu
        annpath = cfg.val_im_anns
        shuffle = False
        drop_last = False

    kwargs: BaseDatasetKwargs = {}
    if hasattr(cfg, 'n_cats'): kwargs['n_cats'] = cfg.n_cats
    if hasattr(cfg, 'lb_ignore'): kwargs['lb_ignore'] = cfg.lb_ignore
    if hasattr(cfg, 'custom_mapping_key'): kwargs['custom_mapping_key'] = cfg.custom_mapping_key
    if hasattr(cfg, 'custom_mapping_weights'): kwargs['custom_mapping_weights'] = cfg.custom_mapping_weights
    if hasattr(cfg, 'custom_mapping_dict'): kwargs['custom_mapping_dict'] = cfg.custom_mapping_dict
    ds = eval(cfg.dataset)(cfg.im_root, annpath, trans_func=trans_func, mode=mode, **kwargs)

    if dist.is_initialized():
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            assert not cfg.max_iter is None
            n_train_imgs = cfg.ims_per_gpu * dist.get_world_size() * cfg.max_iter
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.DistributedSampler(
                ds, shuffle=shuffle)
        batchsampler = torch.utils.data.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dl = DataLoader(
            ds,
            batch_sampler=batchsampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
        )
    return dl
