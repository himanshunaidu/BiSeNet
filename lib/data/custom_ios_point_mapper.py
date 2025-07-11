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
from lib.data.base_dataset import BaseDataset

'''
The following dataset is used for the custom annotated iOSPointMapper dataset with an accessibility mapping.
'''

cocoStuff_dict = {0:'background', 1:'person', 2:'bicycle', 3:'car', 4:'motorcycle', 6:'bus', 7:'train', 8:'truck',
                  10:'traffic light', 11:'fire hydrant', 12:'street sign', 13:'stop sign', 14:'parking meter',
                  15:'bench', # 33: 'suitcase', 41:'skateboard', 
                  64:'potted plant', 92:'banner', 94:'branch',
                  96:'building-other', 97:'bush', 99:'cage', 100:'cardboard', 111:'dirt', 113:'fence', 
                #   115:'floor-other', 116:'floor-stone', 
                  124:'grass', 125:'gravel', 126:'ground-other', 
                  128:'house', 129:'leaves', # 130:'light', 
                  132: 'metal', 134:'moss', 136:'mud', 140:'pavement', 142:'plant-other', 144:'platform',
                  145:'playfield', 146:'railing', 147:'railroad', 149:'road', 150:'rock', 151:'roof', 154:'sand', 159:'snow',
                  161:'stairs', 162:'stone', 164:'structural-other', 169:'tree', 171: 'wall-brick', 172:'wall-concrete', 
                  173:'wall-other', 174:'wall-panel', 175:'wall-stone', 176:'wall-tile', 177:'wall-wood', # 178:'water-other', 
                  182:'wood' }

# This is not valid anymore because multiple classes are mapped to the same class in cocoStuff_continuous_dict
cos2cocoStuff_dict = {0:149, 1:140, 2:96, 3:173, 4:113, 5:132, 6:10, 7:13, 8:129, 9:124,
                      10:0, 11:1, 12:1, 13:3, 14:8, 15:6, 16:7, 17:2, 18:2, 19:0}

# cocoStuff_continuous_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 6:5, 7:6, 8:7,
#                   10:8, 11:9, 12:11, 13:11, 14:12,
#                   15:13, 33:14, 41:15, 64:16, 92:17, 94:31,
#                   96:19, 97:31, 99:21, 100:22, 111:27, 113:24, 
#                   115:25, 116:26, 124:27, 125:28, 126:29, 128:30,
#                   129:31, 130:32, 132: 33, 134:27, 136:34, 140:35, 142:31, 144:37,
#                   145:38, 146:39, 147:40, 149:41, 150:42, 151:43, 154:27, 159:27,
#                   161:46, 162:47, 164:48, 169:49, 171: 50, 172:50, 
#                   173:50, 174:50, 175:50, 176:50, 177:50, 178:51, 182:52 }
# The following dict is to map the relevant cocostuff classes to a continuous set of labels.
# Multiple classes to one include: traffic sign, vegetation, terrain
# traffic sign (10): 12, 13
# vegetation (15): 94, 97, 129, 142, 169
# terrain (19): 111, 124, 125, 126 (cancelled), 134, 136, 154, 159
# building (16): 96, 128
# wall (33): 171, 172, 173, 174, 175, 176, 177
# cocoStuff_continuous_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 6:5, 7:6, 8:7,
#                   10:8, 11:9, 
#                   12:10, 13:10, # traffic sign
#                   14:11, 15:12, 64:13, 92:14, 
#                   94:15, 97:15, 129:15, 142:15, 169:15, # vegetation
#                   96:16, 128:16, # building
#                   99:17, 100:18, 
#                   111:19, 124:19, 125:19, 126:0, 134:19, 136:19, 154:19, 159:19, # terrain
#                   113:20, 132: 21, 140:22, 144:23,
#                   145:24, 146:25, 147:26, 149:27, 150:28, 151:29,
#                   161:30, 162:31, 164:32, 
#                   171:33, 172:33, 173:33, 174:33, 175:33, 176:33, 177:33, # wall
#                   182:34 }

ios_point_mapper_dict = {
    0: 'background', 1: 'bicycle', 2: 'bike rack', 3: 'bridge', 4: 'building',
    5: 'bus', 6: 'car', 7: 'dynamic', 8: 'fence', 9: 'ground',
    10: 'guard rail', 11: 'motorcycle', 12: 'parking', 13: 'person',
    14: 'pole', 15: 'rail track', 16: 'rider', 17: 'road',
    18: 'sidewalk', 19: 'sky', 20: 'static',
    21: 'terrain', 22: 'traffic light', 23: 'traffic sign',
    24: 'train', 25: 'truck', 26: 'tunnel',
    27: 'vegetation', 28: 'wall'
}

# The following dict is to map the custom classes to the continuous set of cocoStuff classes (cocoStuff_continuous_dict)
# not the original cocostuff classes.
# ios_point_mapper_to_cocoStuff_dict = {0:0, 1:2, 2:0, 3:0, 4:16, 5:5, 6:3, 7:0, 8:20, 9:0, 10:25,
#                                       11:4, 12:0, 13:1, 14:21, 15:26, 16:1, 17:27, 18:22, 19:0, 20:0,
#                                       21:19, 22:8, 23:10, 24:6, 25:7, 26:0, 27:15, 28:33}
# Final classes: road, pavement, building, traffic light, traffic sign, pole, vegetation, terrain
ios_point_mapper_to_cocoStuff_dict = {17:0, 18:1, 4:2, 22:3, 23:4, 14:5, 27:6, 21:7}


class CustomIOSPointMapper(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(CustomIOSPointMapper, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 8 # actually 35: equal to length of cocoStuff_continuous_dict
        self.lb_ignore = 255

        ## label mapping, map cocoStuff to cocoStuff with accessibility (use cocoStuff_continuous_dict)
        self.lb_map = np.arange(256)
        for ind in range(256):
            if ind in ios_point_mapper_to_cocoStuff_dict.keys():
                self.lb_map[ind] = ios_point_mapper_to_cocoStuff_dict[ind]
            else:
                self.lb_map[ind] = self.lb_ignore

        self.to_tensor = T.ToTensor(
            mean=(0.46962251, 0.4464104,  0.40718787), # coco, rgb
            std=(0.27469736, 0.27012361, 0.28515933),
        )

if __name__ == "__main__":
    dataroot = '../../datasets/ios_point_mapper/iOSPointMapper_1_Cityscapes'
    annpath = '../../datasets/ios_point_mapper/iOSPointMapper_1_Cityscapes/dataset.txt'
    dataset = CustomIOSPointMapper(dataroot, annpath)
    print(f"Number of categories: {dataset.n_cats}")
    print(f"Label mapping: {dataset.lb_map}")
    print(f"To tensor transform: {dataset.to_tensor}")
    
    # Example of how to use the dataset
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for images, labels in dataloader:
        print(f"Batch images shape: {images.shape}, labels shape: {labels.shape}")
        break  # Just to show one batch
