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
from lib.data.coco_accessibility import CocoStuffAccessibility, cocoStuff_continuous_dict, cocoStuff_dict

import logging
import lib.logger as logger
from tqdm import tqdm

if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    logger.setup_logger(name="analyze_coco_accessibility", logpth='log')
    local_logger = logging.getLogger("analyze_coco_accessibility")
    local_logger.info("Start analyzing COCO-Stuff dataset with accessibility mapping")

    # Load the training dataset
    dataroot = '../../datasets/coco'
    annpath = '../../datasets/coco/train.txt'
    dataset = CocoStuffAccessibility(dataroot, annpath, mode='train')

    COUNT = len(dataset)
    local_logger.info("Analyzing {0} images".format(COUNT))

    ignore_count = 0 # images with only 0 and 255 labels

    # Analyze the pixel-wise distribution of labels
    label_counts = np.zeros((256,), dtype=np.int64)
    for i in tqdm(range(COUNT)):
        img, label = dataset[i]
        local_bin_count = np.bincount(label.numpy().flatten(), minlength=256)
        label_counts += local_bin_count

        if np.sum(local_bin_count) - local_bin_count[0] - local_bin_count[255] == 0:
            ignore_count += 1

        # local_logger.info("{}: {} {} {}".format(i, 
        #     np.sum(local_bin_count) - local_bin_count[0] - local_bin_count[255],
        #     local_bin_count[0], local_bin_count[255])
        # )
        # print(np.bincount(label.numpy().flatten(), minlength=dataset.n_cats).shape)
    # local_logger.info("Label share: ", label_counts)

    # Using cocostuff_dict and cocoStuff_continuous_dict to map the labels
    label_name_dict = {}
    for k, v in cocoStuff_continuous_dict.items():
        label_name_dict[v] = cocoStuff_dict[k]
    # local_logger.info(label_name_dict)
    # Add placeholder class name for 255
    label_name_dict[255] = 'ignore'

    label_count_dict = {}
    for k, v in label_name_dict.items():
        label_count_dict[v] = label_counts[k].item()
    # local_logger.info(label_count_dict)

    label_count_share_dict = {}
    tot = sum(label_count_dict.values())
    for k, v in label_count_dict.items():
        label_count_share_dict[k] = v / tot
    local_logger.info(label_count_share_dict)

    local_logger.info("Number of images with only 0 and 255 labels: " + str(ignore_count))