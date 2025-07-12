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

import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset, BaseDatasetKwargs
from lib.data.custom_maps.mapillary_vistas import mapillary_vistas_dict, mapillary_vistas_continuous_49_dict, \
    mapillary_vistas_continuous_11_dict
    
'''
The following dataset is used for Mapillary Vistas dataset with an accessibility mapping.
'''
# Custom mapping dictionary for Mapillary Vistas
custom_mapping_dicts = {
    '49': mapillary_vistas_continuous_49_dict,
    '11': mapillary_vistas_continuous_11_dict
}

class MapillaryVistas(BaseDataset):
    def __init__(self, dataroot, annpath, trans_func=None, mode='train', **kwargs: BaseDatasetKwargs):
        super(MapillaryVistas, self).__init__(
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
    
    
