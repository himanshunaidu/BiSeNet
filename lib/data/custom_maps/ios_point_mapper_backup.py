"""
This script stores all the custom dictionaries for the iOSPointMapper dataset used for mapping classes in semantic segmentation tasks.
"""
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

ios_point_mapper_to_cocoStuff_custom_35_dict = {0:255, 1:1, 2:255, 3:255, 4:15, 5:4, 6:2, 7:255, 8:19, 9:255,
                                                10:255, 11:3, 12:255, 13:0, 14:20, 15:255, 16:0, 17:26, 18:21,
                                                19:255, 20:255, 21:18, 22:7, 23:9, 24:5, 25:6, 26:255, 27:14, 28:32}

ios_point_mapper_to_cocoStuff_custom_11_dict = {0:10, 1:9, 2:255, 3:255, 4:2, 5:9, 6:9, 7:9, 8:8, 9:8,
                                                10:8, 11:9, 12:8, 13:9, 14:3, 15:8, 16:9, 17:0, 18:1,
                                                19:8, 20:8, 21:7, 22:4, 23:5, 24:9, 25:9, 26:255, 27:6, 28:8}

ios_point_mapper_to_cocoStuff_custom_9_dict = {0:8, 1:7, 2:255, 3:255, 4:2, 5:7, 6:7, 7:7, 8:6, 9:6,
                                                10:6, 11:7, 12:6, 13:7, 14:3, 15:6, 16:7, 17:0, 18:1,
                                                19:6, 20:6, 21:6, 22:4, 23:5, 24:7, 25:7, 26:255, 27:6, 28:6}

## Archived Code
# ios_point_mapper_to_cocoStuff_custom_35_dict_id_name = {}
# for key, value in ios_point_mapper_to_cocoStuff_custom_35_dict.items():
#     if key == 255: continue
#     ios_point_mapper_to_cocoStuff_custom_35_dict_id_name[value] = ios_point_mapper_dict[key]
# print(f"ios_point_mapper_to_cocoStuff_custom_35_dict_id_name: {ios_point_mapper_to_cocoStuff_custom_35_dict_id_name}")

# ios_point_mapper_to_cocoStuff_custom_11_dict_id_name = {}
# for key, value in ios_point_mapper_to_cocoStuff_custom_11_dict.items():
#     if key == 255: continue
#     ios_point_mapper_to_cocoStuff_custom_11_dict_id_name[value] = ios_point_mapper_dict[key]
# print(f"ios_point_mapper_to_cocoStuff_custom_11_dict_id_name: {ios_point_mapper_to_cocoStuff_custom_11_dict_id_name}")

## Output of Archive Code
# ios_point_mapper_to_cocoStuff_custom_35_dict_id_name: {255: 'tunnel', 1: 'bicycle', 15: 'building', 
# 4: 'bus', 2: 'car', 19: 'fence', 3: 'motorcycle', 0: 'rider', 20: 'pole', 26: 'road', 21: 'sidewalk', 
# 18: 'terrain', 7: 'traffic light', 9: 'traffic sign', 5: 'train', 6: 'truck', 14: 'vegetation', 32: 'wall'}

# ios_point_mapper_to_cocoStuff_custom_11_dict_id_name: {10: 'background', 9: 'truck', 255: 'tunnel', 2: 'building', 
# 8: 'wall', 3: 'pole', 0: 'road', 1: 'sidewalk', 7: 'terrain', 4: 'traffic light', 5: 'traffic sign', 6: 'vegetation'}



## Older maps
# The following dict is to map the custom classes to the continuous set of cocoStuff classes (cocoStuff_continuous_dict)
# not the original cocostuff classes.
# ios_point_mapper_to_cocoStuff_dict = {0:0, 1:2, 2:0, 3:0, 4:16, 5:5, 6:3, 7:0, 8:20, 9:0, 10:25,
#                                       11:4, 12:0, 13:1, 14:21, 15:26, 16:1, 17:27, 18:22, 19:0, 20:0,
#                                       21:19, 22:8, 23:10, 24:6, 25:7, 26:0, 27:15, 28:33}
# Final classes: road, pavement, building, traffic light, traffic sign, pole, vegetation, terrain
# ios_point_mapper_to_cocoStuff_dict = {17:0, 18:1, 4:2, 22:3, 23:4, 14:5, 27:6, 21:7}