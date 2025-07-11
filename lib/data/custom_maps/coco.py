"""
This script stores all the custom dictionaries for COCO-Stuff used for mapping classes in semantic segmentation tasks.
"""

# This dictionary stores the mapping of relevant COCO-Stuff dataset classes to their respective names.
cocoStuff_dict_no_bg = {0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 5:'bus', 6:'train', 7:'truck',
                        9:'traffic light', 10:'fire hydrant', 11:'street sign', 12:'stop sign', 13:'parking meter',
                        14:'bench', # 32: 'suitcase', 40:'skateboard', 
                        63:'potted plant', 91:'banner', 93:'branch',
                        95:'building-other', 96:'bush', 98:'cage', 99:'cardboard', 110:'dirt', 112:'fence',
                      #   114:'floor-other', 115:'floor-stone',
                        123:'grass', 124:'gravel', 125:'ground-other',
                        127:'house', 128:'leaves', # 129:'light',
                        131: 'metal', 133:'moss', 135:'mud', 139:'pavement', 141:'plant-other', 143:'platform',
                        144:'playfield', 145:'railing', 146:'railroad', 148:'road', 149:'rock', 150:'roof', 153:'sand', 158:'snow',
                        160:'stairs', 161:'stone', 163:'structural-other', 168:'tree', 170: 'wall-brick', 171:'wall-concrete', 
                        172:'wall-other', 173:'wall-panel', 174:'wall-stone', 175:'wall-tile', 176:'wall-wood', # 177:'water-other', 
                        181:'wood' }

# This dictionary maps edge_mapping and cityscapes classes to COCO-Stuff classes.
# Not in use, but kept for reference.
cos2cocoStuff_dict_no_bg = {0:148, 1:139, 2:95, 3:172, 4:112, 5:131, 6:9, 7:12, 8:128, 9:123, # Having metal as 5 (pole) is not ideal, but it is the only way to keep the mapping consistent.
                            10:255, 11:0, 12:0, 13:2, 14:7, 15:5, 16:6, 17:1, 18:1, 19:255}

# The following dictionary is to map the relevant cocostuff classes to a continuous set of labels.
## This specific dictionary takes all the 53 relevant classes from COCO-Stuff and maps them one-by-one to a continuous set of labels.
cocoStuff_continuous_53_dict_no_bg = {
    0:0, 1:1, 2:2, 3:3, 5:4, 6:5, 7:6,
    9:7, 10:8, 11:9, 12:10, 13:11,
    14:12, 32:13, 40:14,
    63:15, 91:16, 93:17, 
    95:18, 96:19, 98:20, 99:21, 110:22, 112:23,
    114:24, 115:25,
    123:26, 124:27, 125:28,
    127:29, 128:30, 129:31,
    131:32, 133:33, 135:34, 139:35, 141:36, 143:37,
    144:38, 145:39, 146:40, 148:41, 149:42, 150:43, 153:44, 158:45,
    160:46, 161:47, 163:48, 168:49, 170:50, 171:50,
    172:50, 173:50, 174:50, 175:50, 176:50, 177:50, 181:51,
    255:52  # Background is mapped to 255, which is not used in the continuous labels.
}

# The following dictionary is to map the relevant cocostuff classes to a continuous set of labels.
## This specific dictionary takes all the relevant classes and maps them to 35 continuous labels.
## Thus, in some cases, multiple classes are mapped to the same label.
## Multiple classes to one include: traffic sign, vegetation, terrain
### traffic sign (9): 11, 12
### vegetation (14): 93, 96, 128, 141, 168
### terrain (18): 110, 123, 124, 125 (cancelled), 133, 135, 153, 158
### building (15): 95, 127
### wall (32): 170, 171, 172, 173, 174, 175, 176
cocoStuff_continuous_35_dict_no_bg = {
    0:0, 1:1, 2:2, 3:3, 5:4, 6:5, 7:6,
    9:7, 10:8,
    11:9, 12:9, # traffic sign
    13:10, 14:11, 63:12, 91:13,
    93:14, 96:14, 128:14, 141:14, 168:14, # vegetation
    95:15, 127:15, # building
    98:16, 99:17,
    110:18, 123:18, 124:18, 133:18, 135:18, 153:18, 158:18, # terrain
    112:19, 131:20, 139:21, 143:22,
    144:23, 145:24, 146:25, 148:26, 149:27, 150:28,
    160:29, 161:30, 163:31,
    170:32, 171:32, 172:32, 173:32, 174:32, 175:32, 176:32, # wall
    181:33, 255:34  # Background is mapped to 255, which is not used in the continuous labels.
}

# The following dictionary is to map a very small relevant subset of cocostuff classes to a continuous set of labels.
## Classes: road, sidewalk, building, pole, traffic light, traffic sign, vegetation, terrain, background
cocoStuff_continuous_7_dict_no_bg = {
    148:0, 139:1,
    95:2, 127:2, # building
    131:3, 9:4,
    11:5, 12:5, 13:5,  # traffic sign
    93:6, 96:6, 128:6, 141:6, 168:6, # vegetation
    110:7, 123:7, 124:7, 125:7, 133:7, 135:7, 153:7, 158:7, # terrain
    255:8  # Background is mapped to 255, which is not used in the continuous labels.
}
# Weight mapping for the continuous set of labels
cocoStuff_continuous_7_weights_no_bg = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1]  # Background has less weight

# Map for cityscapes
cocoStuff_cityscapes_dict_no_bg = {
    148: 0, 139: 1,
    95: 2, 127: 2,
    131: 5,
    9: 6,
    11: 7, 12: 7
}