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
The following dataset is used for COCO-Stuff dataset with an accessibility mapping.
'''

cocoStuff_dict = {0:'background', 1:'person', 2:'bicycle', 3:'car', 4:'motorcycle', 6:'bus', 7:'train', 8:'truck',
                  10:'traffic light', 11:'fire hydrant', 12:'street sign', 13:'stop sign', 14:'parking meter',
                  15:'bench', 33: 'suitcase', 41:'skateboard', 64:'potted plant', 92:'banner', 94:'branch',
                  96:'building-other', 97:'bush', 99:'cage', 100:'cardboard', 111:'dirt', 113:'fence', 
                  115:'floor-other', 116:'floor-stone', 124:'grass', 125:'gravel', 126:'ground-other', 128:'house',
                  129:'leaves', 130:'light', 134:'moss', 136:'mud', 140:'pavement', 142:'plant-other', 144:'platform',
                  145:'playfield', 146:'railing', 147:'railroad', 149:'road', 150:'rock', 151:'roof', 154:'sand', 159:'snow',
                  161:'stairs', 162:'stone', 164:'structural-other', 169:'tree', 171: 'wall-brick', 172:'wall-concrete', 
                  173:'wall-other', 174:'wall-panel', 175:'wall-stone', 176:'wall-tile', 177:'wall-wood', 178:'water-other', 182:'wood' }

cos2cocoStuff_dict = {0:149, 1:140, 2:96, 3:173, 4:113, 5:0, 6:10, 7:13, 8:129, 9:124,
                      10:0, 11:1, 12:1, 13:3, 14:8, 15:6, 16:7, 17:2, 18:2, 19:0}

cocoStuff_continuous_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 6:5, 7:6, 8:7,
                  10:8, 11:9, 12:10, 13:11, 14:12,
                  15:13, 33:14, 41:15, 64:16, 92:17, 94:18,
                  96:19, 97:20, 99:21, 100:22, 111:23, 113:24, 
                  115:25, 116:26, 124:27, 125:28, 126:29, 128:30,
                  129:31, 130:32, 134:33, 136:34, 140:35, 142:36, 144:37,
                  145:38, 146:39, 147:40, 149:41, 150:42, 151:43, 154:44, 159:45,
                  161:46, 162:47, 164:48, 169:49, 171: 50, 172:50, 
                  173:50, 174:50, 175:50, 176:50, 177:50, 178:51, 182:52 }

custom_to_cocoStuff_dict = {0:41, 1:35, 2:19, 3:50, 4:24, 5:0, 6:8, 7:11, 8:31, 9:27,
                            10:0, 11:1, 12:1, 13:3, 14:12, 15:5, 16:6, 17:2, 18:2, 19:0}

custom_id_to_class = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light',
                7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
                14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle', 19: 'undefined', 20:'road marking', 
                21:'footpath', 22:'pedestrian traffic light',23:'curb',24:'lowered curb',25:'covered bus station',
                26:'bench',27:'wheeled pedestrian'}

class CocoStuffAccessibilityCustomEdgeMapping(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(CocoStuffAccessibilityCustomEdgeMapping, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 53 # actually 53: equal to length of cocoStuff_continuous_dict
        self.lb_ignore = 255

        ## label mapping, map cocoStuff to cocoStuff with accessibility (use cocoStuff_continuous_dict)
        self.lb_map = np.arange(256)
        for ind in range(256):
            if ind in custom_to_cocoStuff_dict.keys():
                self.lb_map[ind] = custom_to_cocoStuff_dict[ind]
            else:
                self.lb_map[ind] = self.lb_ignore

        self.to_tensor = T.ToTensor(
            mean=(0.46962251, 0.4464104,  0.40718787), # coco, rgb
            std=(0.27469736, 0.27012361, 0.28515933),
        )
    

if __name__ == "__main__":
    dataroot = '../../datasets/coco'
    annpath = '../../datasets/coco/train.txt'
    dataset = CocoStuffAccessibility(dataroot, annpath)

    for i in range(10):
        img, label = dataset[i]
        print(img.shape, label.shape)
        print(torch.unique(label))
    
    print(dataset.lb_map)