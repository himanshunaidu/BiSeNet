#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset, BaseDatasetKwargs
from lib.data.custom_maps.coco import cocoStuff_dict, cocoStuff_continuous_53_dict, cocoStuff_continuous_35_dict, \
    cocoStuff_continuous_11_dict, cocoStuff_continuous_9_dict, cocoStuff_cityscapes_dict

'''
The following dataset is used for COCO-Stuff dataset with an accessibility mapping.
'''
# Custom mapping dictionary for COCOStuff
custom_mapping_dicts = {
    'city': cocoStuff_cityscapes_dict,
    '53': cocoStuff_continuous_53_dict,
    '35': cocoStuff_continuous_35_dict,
    '11': cocoStuff_continuous_11_dict,
    '9': cocoStuff_continuous_9_dict
}

class CocoStuffAccessibility(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train', custom_mapping='53', **kwargs: BaseDatasetKwargs):
        super(CocoStuffAccessibility, self).__init__(
                dataroot, annpath, trans_func, mode, **kwargs)
        self.n_cats = kwargs.get('n_cats', 53)
        self.lb_ignore = kwargs.get('lb_ignore', 255)

        self.custom_mapping_dict = self._get_custom_mappings(**kwargs)
        # print(f"Using custom mapping: {custom_mapping} with {len(self.custom_mapping_dict)} classes.")

        ## label mapping, map cocoStuff to cocoStuff with accessibility (use cocoStuff_continuous_dict)
        self.lb_map = np.arange(256)
        for ind in range(256):
            if ind in self.custom_mapping_dict.keys():
                self.lb_map[ind] = self.custom_mapping_dict[ind]
            else:
                self.lb_map[ind] = self.lb_ignore

        self.to_tensor = T.ToTensor(
            mean=(0.46962251, 0.4464104,  0.40718787), # coco, rgb
            std=(0.27469736, 0.27012361, 0.28515933),
        )
    
    def _get_custom_mappings(self, **kwargs):
        """
        Returns the custom mapping dictionary based on the provided key.
        """
        if kwargs.get('custom_mapping_dict') is not None:
            return kwargs['custom_mapping_dict']
        
        custom_mapping_key = kwargs.get('custom_mapping_key', '53')
        if custom_mapping_key not in custom_mapping_dicts:
            raise ValueError(f"Invalid custom mapping key: {custom_mapping_key}. "
                             f"Available keys are: {list(custom_mapping_dicts.keys())}")
        return custom_mapping_dicts[custom_mapping_key]

if __name__ == "__main__":
    dataroot = '../../datasets/coco'
    annpath = '../../datasets/coco/train.txt'
    dataset = CocoStuffAccessibility(dataroot, annpath, custom_mapping_key='7')

    for i in range(10):
        img, label = dataset[i]
        print(img.shape, label.shape)
        print(torch.unique(label))
    
    print(dataset.lb_map)
    # Get all non-255 label mappings
    label_mappings = {i: int(dataset.lb_map[i]) for i in range(256) if dataset.lb_map[i] != 255}
    print(label_mappings)