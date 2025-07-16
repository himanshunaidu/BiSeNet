"""
This script stores all the custom dictionaries for the iOSPointMapper dataset used for mapping classes in semantic segmentation tasks.
"""

# Previous
# edge_mapping_ios_dict = {
#     0: "background",
#     1: "bicycle",
#     2: "bike rack",
#     3: "bridge",
#     4: "building",
#     5: "bus",
#     6: "car",
#     7: "crosswalk",
#     8: "curb",
#     9: "curb ramp",
#     10: "dynamic",
#     11: "fence",
#     12: "ground",
#     13: "guard rail",
#     14: "license plate",
#     15: "motorcycle",
#     16: "parking",
#     17: "person",
#     18: "pole",
#     19: "rail track",
#     20: "rider",
#     21: "road",
#     22: "sidewalk",
#     23: "sky",
#     24: "static",
#     25: "tactile paving",
#     26: "terrain",
#     27: "traffic light",
#     28: "traffic sign",
#     29: "train",
#     30: "truck",
#     31: "tunnel",
#     32: "vegetation",
#     33: "wall"
# }

# cross-walk is treated as road
# curb, curb ramp and tactile paving is treated as sidewalk
# license plate is treated as car
# edge_mapping_ios_to_cocoStuff_custom_35_dict = {0:255, 1:1, 2:255, 3:255, 4:15, 5:4, 6:2, 7:26, 8:21, 9:21, 10:255,
#                                                 11:19, 12:255, 13:255, 14:2, 15:3, 16:255, 17:0, 18:20, 19:255,
#                                                 20:0, 21:26, 22:21, 23:255, 24:255, 25:21, 26:18, 27:7, 28:9, 29:5,
#                                                 30:6, 31:255, 32:14, 33:32}

# edge_mapping_ios_to_cocoStuff_custom_11_dict = {0:10, 1:9, 2:255, 3:255, 4:2, 5:9, 6:9, 7:0, 8:1, 9:1, 10:9, 
#                                                 11:8, 12:8, 13:8, 14:9, 15:9, 16:8, 17:9, 18:3, 19:8, 20:9,
#                                                 21:0, 22:1, 23:8, 24:8, 25:1, 26:7, 27:4, 28:5, 29:9, 30:9,
#                                                 31:255, 32:6, 33:8}

# edge_mapping_ios_to_cocoStuff_custom_9_dict = {0:8, 1:7, 2:255, 3:255, 4:2, 5:7, 6:7, 7:0, 8:1, 9:1, 10:7, 
#                                                11:6, 12:6, 13:6, 14:7, 15:7, 16:6, 17:7, 18:3, 19:6, 20:7,
#                                                21:0, 22:1, 23:6, 24:6, 25:1, 26:6, 27:4, 28:5, 29:7, 30:7,
#                                                31:255, 32:6, 33:6}

# Current
edge_mapping_ios_dict = {
    0: "background",
    1: "bench",
    2: "bicycle",
    3: "building",
    4: "bus",
    5: "car",
    6: "covered bus station",
    7: "curb",
    8: "fence",
    9: "footpath",
    10: "lowered curb",
    11: "motorcycle",
    12: "pedestrian traffic light",
    13: "person",
    14: "pole",
    15: "rider",
    16: "road",
    17: "road marking",
    18: "sidewalk",
    19: "sky",
    20: "terrain",
    21: "traffic light",
    22: "traffic sign",
    23: "train",
    24: "truck",
    25: "undefined",
    26: "vegetation",
    27: "wall",
    28: "wheeled pedestrian"
}

# covered bus stations is treated as building
# road marking, cross-walk is treated as road
# footpath, curb, curb ramp and tactile paving are treated as sidewalk
# license plate is treated as car
# pedestrian traffic light is treated as traffic light
# wheeled pedestrian is treated as person
edge_mapping_ios_to_cocoStuff_custom_35_dict = {0:255, 1:11, 2:1, 3:15, 4:4, 5:2, 6:15, 7:21, 8:19, 9:21, 10:21,
                                                11:3, 12:7, 13:0, 14:20, 15:0, 16:26, 17:26, 18:21, 19:255,
                                                20:18, 21:7, 22:9, 23:5, 24:6, 25:255, 26:14, 27:32, 28:0}

edge_mapping_ios_to_cocoStuff_custom_11_dict = {0:10, 1:8, 2:9, 3:2, 4:9, 5:9, 6:2, 7:1, 8:8, 9:1, 10:1, 
                                                11:9, 12:4, 13:9, 14:3, 15:9, 16:0, 17:0, 18:1, 19:8, 20:7,
                                                21:4, 22:5, 23:9, 24:9, 25:10, 26:6, 27:8, 28:9}

edge_mapping_ios_to_cocoStuff_custom_9_dict = {0:8, 1:6, 2:7, 3:2, 4:7, 5:7, 6:2, 7:1, 8:6, 9:1, 10:1, 
                                               11:7, 12:4, 13:7, 14:3, 15:7, 16:0, 17:0, 18:1, 19:6, 20:6,
                                               21:4, 22:5, 23:7, 24:7, 25:8, 26:6, 27:6, 28:7}

## Archived Code
edge_mapping_ios_to_cocoStuff_custom_35_dict_id_name = {}
for key, value in edge_mapping_ios_to_cocoStuff_custom_35_dict.items():
    if key == 255: continue
    edge_mapping_ios_to_cocoStuff_custom_35_dict_id_name[value] = edge_mapping_ios_dict[key]
print(f"edge_mapping_ios_to_cocoStuff_custom_35_dict_id_name: {edge_mapping_ios_to_cocoStuff_custom_35_dict_id_name}")

edge_mapping_ios_to_cocoStuff_custom_11_dict_id_name = {}
for key, value in edge_mapping_ios_to_cocoStuff_custom_11_dict.items():
    if key == 255: continue
    edge_mapping_ios_to_cocoStuff_custom_11_dict_id_name[value] = edge_mapping_ios_dict[key]
print(f"edge_mapping_ios_to_cocoStuff_custom_11_dict_id_name: {edge_mapping_ios_to_cocoStuff_custom_11_dict_id_name}")

edge_mapping_ios_to_cocoStuff_custom_9_dict_id_name = {}
for key, value in edge_mapping_ios_to_cocoStuff_custom_9_dict.items():
    if key == 255: continue
    edge_mapping_ios_to_cocoStuff_custom_9_dict_id_name[value] = edge_mapping_ios_dict[key]
print(f"edge_mapping_ios_to_cocoStuff_custom_9_dict_id_name: {edge_mapping_ios_to_cocoStuff_custom_9_dict_id_name}")



## Older maps
# The following dict is to map the custom classes to the continuous set of cocoStuff classes (cocoStuff_continuous_dict)
# not the original cocostuff classes.
# ios_point_mapper_to_cocoStuff_dict = {0:0, 1:2, 2:0, 3:0, 4:16, 5:5, 6:3, 7:0, 8:20, 9:0, 10:25,
#                                       11:4, 12:0, 13:1, 14:21, 15:26, 16:1, 17:27, 18:22, 19:0, 20:0,
#                                       21:19, 22:8, 23:10, 24:6, 25:7, 26:0, 27:15, 28:33}
# Final classes: road, pavement, building, traffic light, traffic sign, pole, vegetation, terrain
# ios_point_mapper_to_cocoStuff_dict = {17:0, 18:1, 4:2, 22:3, 23:4, 14:5, 27:6, 21:7}