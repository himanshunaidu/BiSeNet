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
from PIL import Image

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset, BaseDatasetKwargs
from lib.data.custom_maps.mapillary_vistas import mapillary_vistas_dict, mapillary_vistas_continuous_49_dict, \
    mapillary_vistas_continuous_11_dict
from lib.data.mappings.builder import MappingSpec, MappingBuilder
from lib.data.mappings.mapillary_vistas import MAPILLARY_VISTAS_ACCESSIBILITY_11, MAPILLARY_VISTAS_ACCESSIBILITY_12, \
    MAPILLARY_VISTAS_ACCESSIBILITY_13, MAPILLARY_VISTAS_ACCESSIBILITY_3_CURB_RAMP
    
'''
The following dataset is used for Mapillary Vistas dataset with an accessibility mapping.
'''
# Custom mapping dictionary for Mapillary Vistas
custom_mapping_dicts = {
    '49': mapillary_vistas_continuous_49_dict,
    '11': mapillary_vistas_continuous_11_dict
}

custom_mapping_specs: dict[str, MappingSpec] = {
    'accessibility_11': MAPILLARY_VISTAS_ACCESSIBILITY_11,
    'accessibility_12': MAPILLARY_VISTAS_ACCESSIBILITY_12,
    'accessibility_13': MAPILLARY_VISTAS_ACCESSIBILITY_13,
    'accessibility_3_curb_ramp': MAPILLARY_VISTAS_ACCESSIBILITY_3_CURB_RAMP
}

class MapillaryVistas(BaseDataset):
    def __init__(self, dataroot, annpath, trans_func=None, mode='train', **kwargs: BaseDatasetKwargs):
        super(MapillaryVistas, self).__init__(
                dataroot, annpath, trans_func, mode, **kwargs)
        self.n_cats = kwargs.get('n_cats', 11)
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

        # TODO: Recalculate mean and std for Mapillary Vistas dataset
        self.to_tensor = T.ToTensor(
            mean=(0.41663339, 0.45617131, 0.46773973),
            std=(0.25652909, 0.26454683, 0.29007679),
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
        
        # custom_mapping_key = kwargs.get('custom_mapping_key', '11')
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
    
    # def get_image(self, impth, lbpth):
    #     img = cv2.imread(impth)[:, :, ::-1].copy()
    #     # Label image is paletted, so we need to read it properly
    #     # label = cv2.imread(lbpth, 0)
    #     label = np.array(Image.open(lbpth)).astype(np.uint8)
    #     return img, label
    
if __name__ == "__main__":
    dataroot = '../../datasets/mapillary_vistas'
    annpath = '../../datasets/mapillary_vistas/train.txt'
    dataset = MapillaryVistas(dataroot, annpath, custom_mapping_spec_key='accessibility_11', n_cats=11, lb_ignore=255)

    for i in range(10):
        img, label = dataset[i]
        print(img.shape, label.shape)
        print(torch.unique(label))
    
    print(dataset.lb_map, dataset.lb_map.shape)
    # Get all non-255 label mappings
    label_mappings = {i: int(dataset.lb_map[i]) for i in range(256) if dataset.lb_map[i] != 255}
    print(label_mappings)
    