"""
This script analyzes the images in the specified directory to find class proportions in the annotations.
"""
import glob
import cv2
import numpy as np
from tqdm import tqdm
import os
from collections import namedtuple

IMAGE_DIR_PATH = "labels/train2017"
IMAGES = glob.glob(IMAGE_DIR_PATH + "/*.png")

classes = {'road': {149:'road'}, 'sidewalk': {140:'pavement'}, 'traffic light': {10:'traffic light'}, 'pole': {132: 'metal'},
           'traffic sign': {12:'street sign', 13:'stop sign'}, 'building': {96:'building-other', 128:'house'},
           'vegetation': {94: 'branch', 97:'bush', 129:'leaves', 142:'plant-other', 169:'tree'},
           'terrain': {111:'dirt', 124:'grass', 125:'gravel', 126:'ground-other', 134:'moss', 136:'mud', 154:'sand', 159:'snow'},
           'rail track': {147: 'railroad'}}
class_proportions = {key: 0 for key in classes.keys()}

class_label = 12

print(f"Number of images: {len(IMAGES)}")
for index, image_path in enumerate(IMAGES):
    image_full_path = os.path.join(os.getcwd(), image_path)
    # Read the image as a grayscale image
    image = cv2.imread(image_full_path, cv2.IMREAD_GRAYSCALE)
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Image at path {image_full_path} could not be loaded. Please check the path and try again.")

    # Get the unique values in the image
    unique_values = np.unique(image)   
    
    # Check if the class label is present in the unique values
    if class_label in unique_values:# and len(unique_values) < 5:
        print("Unique values in image:", unique_values)
        print(f"Class label {class_label} found in image: {image_full_path.replace('labels', 'images').replace('.png', '.jpg')}")
        print(f"Label Image: {image_full_path}")
        print("\n")