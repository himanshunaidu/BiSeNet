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
from lib.data.custom_maps.edge_mapping import edge_mapping_dict, edge_mapping_to_cocoStuff_custom_53_dict, \
    edge_mapping_to_cocoStuff_custom_35_dict, edge_mapping_to_cocoStuff_custom_11_dict

'''
The following dataset is used for COCO-Stuff dataset with an accessibility mapping.
'''
# Custom mapping dictionary for Edge Mapping
custom_mapping_dicts = {
    '53': edge_mapping_to_cocoStuff_custom_53_dict,
    '35': edge_mapping_to_cocoStuff_custom_35_dict,
    '11': edge_mapping_to_cocoStuff_custom_11_dict
}

# custom_to_cocoStuff_dict = {0:41, 1:35, 2:19, 3:50, 4:24, 5:0, 6:8, 7:11, 8:31, 9:27,
#                             10:0, 11:1, 12:1, 13:3, 14:12, 15:5, 16:6, 17:2, 18:2, 19:0}

class CocoStuffAccessibilityCustomEdgeMapping(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train', **kwargs: BaseDatasetKwargs):
        super(CocoStuffAccessibilityCustomEdgeMapping, self).__init__(
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
    dataroot = '../../datasets/custom_images'
    annpath = '../../datasets/custom_images/train.txt'
    dataset = CocoStuffAccessibilityCustomEdgeMapping(dataroot, annpath, custom_mapping_key='11')

    for i in range(10):
        img, label = dataset[i]
        print(img.shape, label.shape)
        print(torch.unique(label))
    
    print(dataset.lb_map)