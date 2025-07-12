"""
This script stores all the custom dictionaries for the OASIS Edge Mapping dataset used for mapping classes in semantic segmentation tasks.
"""
edge_mapping_dict = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light',
                7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
                14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle', 19: 'undefined', 20:'road marking', 
                21:'footpath', 22:'pedestrian traffic light',23:'curb',24:'lowered curb',25:'covered bus station',
                26:'bench',27:'wheeled pedestrian'}

# Mapping from edge mapping classes to **custom** cocostuff classes (53 classes)
## This customization of cocostuff classes comes from edge mapping repository
## done to map the fewer relevant classes to a continuous range of classes
edge_mapping_to_cocoStuff_custom_53_dict = {0:41, 1:35, 2:18, 3:50, 4:23, 5:32, 6:7, 7:10, 8:30, 9:26,
                            10:255, 11:0, 12:0, 13:2, 14:6, 15:4, 16:5, 17:3, 18:1, 19:255}

# Mapping from edge mapping classes to **custom** cocostuff classes (35 classes)
## This customization of cocostuff classes comes from edge mapping repository
## done to map the fewer relevant classes to a continuous range of classes
edge_mapping_to_cocoStuff_custom_35_dict = {0:26, 1:21, 2:15, 3:32, 4:19, 5:20, 6:7, 7:9, 8:14, 9:18,
                                            10:255, 11:0, 12:0, 13:2, 14:6, 15:4, 16:5, 17:3, 18:1, 19:255}

# Mapping from edge mapping classes to **custom** cocostuff classes (11 classes)
edge_mapping_to_cocoStuff_custom_11_dict = {0:0, 1:1, 2:2, 3:8, 4:8, 5:3, 6:4, 7:5, 8:6, 9:7, 
                                            10:8, 11:9, 12:9, 13:9, 14:9, 15:9, 16:9, 17:9, 18:9, 19:255}
