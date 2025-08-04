"""
This script stores all the custom dictionaries for the iOSPointMapper dataset used for mapping classes in semantic segmentation tasks.
"""

# Current
ios_point_mapper_combined_dict = {
    0: "background",
    1: "bench",
    2: "bicycle",
    3: "bike rack",
    4: "bridge",
    5: "building",
    6: "bus",
    7: "bus station",
    8: "car",
    9: "crosswalk",
    10: "curb",
    11: "curb ramp",
    12: "dynamic",
    13: "fence",
    14: "ground",
    15: "guard rail",
    16: "license plate",
    17: "motorcycle",
    18: "parking",
    19: "person",
    20: "pole",
    21: "rail track",
    22: "rider",
    23: "road",
    24: "sidewalk",
    25: "sky",
    26: "static",
    27: "tactile paving",
    28: "terrain",
    29: "traffic light",
    30: "traffic sign",
    31: "train",
    32: "truck",
    33: "tunnel",
    34: "vegetation",
    35: "wall"
}


# cross-walk is treated as road
# curb, curb ramp and tactile paving is treated as sidewalk
## New modification: curb is now part of road
# license plate is treated as car
ios_point_mapper_combined_to_cocoStuff_custom_35_dict = {0:255, 1:11, 2:1, 3:255, 4:255, 5:15, 6:4, 7:15, 8:2, 9:26,
                                                10:26, 11:21, 12:255, 13:19, 14:255, 15:255, 16:2, 17:3, 18:255,
                                                19:0, 20:20, 21:255, 22:0, 23:26, 24:21, 25:255, 26:266, 27:21,
                                                28:18, 29:7, 30:9, 31:5, 32:6, 33:255, 34:14, 35:32}

ios_point_mapper_combined_to_cocoStuff_custom_11_dict = {0:10, 1:8, 2:9, 3:255, 4:255, 5:2, 6:9, 7:2, 8:9, 9:0,
                                                10:0, 11:1, 12:9, 13:8, 14:8, 15:8, 16:9, 17:9, 18:8,
                                                19:9, 20:3, 21:8, 22:9, 23:0, 24:1, 25:8, 26:8, 27:1,
                                                28:7, 29:4, 30:5, 31:9, 32:9, 33:255, 34:6, 35:8}

ios_point_mapper_combined_to_cocoStuff_custom_9_dict = {0:8, 1:6, 2:7, 3:255, 4:255, 5:2, 6:7, 7:2, 8:7, 9:0,
                                                10:0, 11:1, 12:7, 13:6, 14:6, 15:6, 16:7, 17:7, 18:6, 
                                                19:7, 20:3, 21:6, 22:7, 23:0, 24:1, 25:6, 26:6, 27:1,
                                                28:6, 29:4, 30:5, 31:7, 32:7, 33:255, 34:6, 35:6}

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


