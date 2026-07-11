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
from lib.data.mappings.builder import MappingSpec, MappingBuilder
from lib.data.mappings.coco import COCO_STUFF_ACCESSIBILITY_11, COCO_STUFF_ACCESSIBILITY_12, COCO_STUFF_ACCESSIBILITY_13

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

custom_mapping_specs: dict[str, MappingSpec] = {
    'accessibility_11': COCO_STUFF_ACCESSIBILITY_11,
    'accessibility_12': COCO_STUFF_ACCESSIBILITY_12,
    'accessibility_13': COCO_STUFF_ACCESSIBILITY_13
}

class CocoStuffAccessibility(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train', **kwargs: BaseDatasetKwargs):
        super(CocoStuffAccessibility, self).__init__(
                dataroot, annpath, trans_func, mode, **kwargs)
        self.n_cats = kwargs.get('n_cats', 53)
        self.lb_ignore = kwargs.get('lb_ignore', 255)

        # self.custom_mapping_dict = self._get_custom_mappings(**kwargs)
        # print(f"Using custom mapping: {custom_mapping} with {len(self.custom_mapping_dict)} classes.")

        ## label mapping, map cocoStuff to cocoStuff with accessibility (use cocoStuff_continuous_dict)
        # self.lb_map = np.arange(256)
        # for ind in range(256):
        #     if ind in self.custom_mapping_dict.keys():
        #         self.lb_map[ind] = self.custom_mapping_dict[ind]
        #     else:
        #         self.lb_map[ind] = self.lb_ignore
        self.lb_map = self._get_custom_mappings(**kwargs)

        self.to_tensor = T.ToTensor(
            mean=(0.46962251, 0.4464104,  0.40718787), # coco, rgb
            std=(0.27469736, 0.27012361, 0.28515933),
        )
    
    def _get_custom_mappings(self, **kwargs):
        """
        Returns the custom mapping dictionary based on the provided key.
        """
        # if kwargs.get('custom_mapping_spec_key') is not None:
        spec_key = kwargs.get('custom_mapping_spec_key', 'accessibility_11')
        return self._get_custom_mapping_from_spec(spec_key)
        
        # if kwargs.get('custom_mapping_dict') is not None:
        #     return self._get_custom_mapping_from_dict(kwargs['custom_mapping_dict'])
        
        # custom_mapping_key = kwargs.get('custom_mapping_key', '53')
        # return self._get_custom_mapping_from_dict_key(custom_mapping_key)
    
    def _get_custom_mapping_from_spec(self, spec_key):
        """
        Returns the custom mapping dictionary based on the provided spec key.
        """
        if spec_key not in custom_mapping_specs:
            raise ValueError(f"Invalid custom mapping spec key: {spec_key}. "
                             f"Available keys are: {list(custom_mapping_specs.keys())}")
        mapping_spec = custom_mapping_specs[spec_key]
        # Create a mapping dictionary from the mapping specification
        lb_map = mapping_spec.compile_label_map()
        return lb_map
    
    def _get_custom_mapping_from_dict(self, mapping_dict):
        """
        Returns the custom mapping dictionary based on the provided mapping dictionary.
        """
        if not isinstance(mapping_dict, dict):
            raise ValueError(f"Invalid custom mapping dictionary: {mapping_dict}. "
                             f"Expected a dictionary.")
        lb_map = np.arange(256)
        for ind in range(256):
            if ind in mapping_dict.keys():
                lb_map[ind] = mapping_dict[ind]
            else:
                lb_map[ind] = self.lb_ignore
        return lb_map
    
    def _get_custom_mapping_from_dict_key(self, dict_key):
        """
        Returns the custom mapping dictionary based on the provided dictionary key.
        """
        if dict_key not in custom_mapping_dicts:
            raise ValueError(f"Invalid custom mapping dictionary key: {dict_key}. "
                             f"Available keys are: {list(custom_mapping_dicts.keys())}")
        custom_mapping_dict = custom_mapping_dicts[dict_key]
        return self._get_custom_mapping_from_dict(custom_mapping_dict)

if __name__ == "__main__":
    dataroot = '../../datasets/coco'
    annpath = '../../datasets/coco/train.txt'
    dataset = CocoStuffAccessibility(dataroot, annpath, custom_mapping_spec_key='accessibility_11', n_cats=11, lb_ignore=255)

    for i in range(10):
        img, label = dataset[i]
        print(img.shape, label.shape)
        print(torch.unique(label))
    
    print(dataset.lb_map)
    # Get all non-255 label mappings
    label_mappings = {i: int(dataset.lb_map[i]) for i in range(256) if dataset.lb_map[i] != 255}
    print(label_mappings)