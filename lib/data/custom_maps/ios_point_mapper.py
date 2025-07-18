"""
This script stores all the custom dictionaries for the iOSPointMapper dataset used for mapping classes in semantic segmentation tasks.
"""

# Current
ios_point_mapper_dict = {
    0: "background",
    1: "bicycle",
    2: "bike rack",
    3: "bridge",
    4: "building",
    5: "bus",
    6: "car",
    7: "crosswalk",
    8: "curb",
    9: "curb ramp",
    10: "dynamic",
    11: "fence",
    12: "ground",
    13: "guard rail",
    14: "license plate",
    15: "motorcycle",
    16: "parking",
    17: "person",
    18: "pole",
    19: "rail track",
    20: "rider",
    21: "road",
    22: "sidewalk",
    23: "sky",
    24: "static",
    25: "tactile paving",
    26: "terrain",
    27: "traffic light",
    28: "traffic sign",
    29: "train",
    30: "truck",
    31: "tunnel",
    32: "vegetation",
    33: "wall"
}


# cross-walk is treated as road
# curb, curb ramp and tactile paving is treated as sidewalk
# license plate is treated as car
ios_point_mapper_to_cocoStuff_custom_35_dict = {0:255, 1:1, 2:255, 3:255, 4:15, 5:4, 6:2, 7:26, 8:21, 9:21, 10:255,
                                                11:19, 12:255, 13:255, 14:2, 15:3, 16:255, 17:0, 18:20, 19:255,
                                                20:0, 21:26, 22:21, 23:255, 24:255, 25:21, 26:18, 27:7, 28:9, 29:5,
                                                30:6, 31:255, 32:14, 33:32}

ios_point_mapper_to_cocoStuff_custom_11_dict = {0:10, 1:9, 2:255, 3:255, 4:2, 5:9, 6:9, 7:0, 8:1, 9:1, 10:9, 
                                                11:8, 12:8, 13:8, 14:9, 15:9, 16:8, 17:9, 18:3, 19:8, 20:9,
                                                21:0, 22:1, 23:8, 24:8, 25:1, 26:7, 27:4, 28:5, 29:9, 30:9,
                                                31:255, 32:6, 33:8}

ios_point_mapper_to_cocoStuff_custom_9_dict = {0:8, 1:7, 2:255, 3:255, 4:2, 5:7, 6:7, 7:0, 8:1, 9:1, 10:7, 
                                               11:6, 12:6, 13:6, 14:7, 15:7, 16:6, 17:7, 18:3, 19:6, 20:7,
                                               21:0, 22:1, 23:6, 24:6, 25:1, 26:6, 27:4, 28:5, 29:7, 30:7,
                                               31:255, 32:6, 33:6}

## Archived Code
# ios_point_mapper_to_cocoStuff_custom_35_dict_id_name = {}
# for key, value in ios_point_mapper_to_cocoStuff_custom_35_dict.items():
#     if key == 255: continue
#     ios_point_mapper_to_cocoStuff_custom_35_dict_id_name[ios_point_mapper_dict[key]] = value
# print(f"ios_point_mapper_to_cocoStuff_custom_35_dict_id_name: {ios_point_mapper_to_cocoStuff_custom_35_dict_id_name}")

# ios_point_mapper_to_cocoStuff_custom_11_dict_id_name = {}
# for key, value in ios_point_mapper_to_cocoStuff_custom_11_dict.items():
#     if key == 255: continue
#     ios_point_mapper_to_cocoStuff_custom_11_dict_id_name[ios_point_mapper_dict[key]] = value
# print(f"ios_point_mapper_to_cocoStuff_custom_11_dict_id_name: {ios_point_mapper_to_cocoStuff_custom_11_dict_id_name}")

# ios_point_mapper_to_cocoStuff_custom_9_dict_id_name = {}
# for key, value in ios_point_mapper_to_cocoStuff_custom_9_dict.items():
#     if key == 255: continue
#     ios_point_mapper_to_cocoStuff_custom_9_dict_id_name[ios_point_mapper_dict[key]] = value
# print(f"ios_point_mapper_to_cocoStuff_custom_9_dict_id_name: {ios_point_mapper_to_cocoStuff_custom_9_dict_id_name}")



## Older maps
# The following dict is to map the custom classes to the continuous set of cocoStuff classes (cocoStuff_continuous_dict)
# not the original cocostuff classes.
# ios_point_mapper_to_cocoStuff_dict = {0:0, 1:2, 2:0, 3:0, 4:16, 5:5, 6:3, 7:0, 8:20, 9:0, 10:25,
#                                       11:4, 12:0, 13:1, 14:21, 15:26, 16:1, 17:27, 18:22, 19:0, 20:0,
#                                       21:19, 22:8, 23:10, 24:6, 25:7, 26:0, 27:15, 28:33}
# Final classes: road, pavement, building, traffic light, traffic sign, pole, vegetation, terrain
# ios_point_mapper_to_cocoStuff_dict = {17:0, 18:1, 4:2, 22:3, 23:4, 14:5, 27:6, 21:7}