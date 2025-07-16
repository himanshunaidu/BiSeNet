"""
This script stores all the custom dictionaries for the Mapillary Vistas dataset used for mapping classes in semantic segmentation tasks.
"""
mapillary_vistas_dict = {
    "labels": [{'name': 'animal--bird', 'readable': 'Bird', 'instances': True, 'evaluate': True, 'color': [165, 42, 42], 'id': 0}, {'name': 'animal--ground-animal', 'readable': 'Ground Animal', 'instances': True, 'evaluate': True, 'color': [0, 192, 0], 'id': 1}, {'name': 'construction--barrier--ambiguous', 'readable': 'Ambiguous Barrier', 'instances': False, 'evaluate': False, 'color': [250, 170, 31], 'id': 2}, {'name': 'construction--barrier--concrete-block', 'readable': 'Concrete Block', 'instances': False, 'evaluate': True, 'color': [250, 170, 32], 'id': 3}, {'name': 'construction--barrier--curb', 'readable': 'Curb', 'instances': False, 'evaluate': True, 'color': [196, 196, 196], 'id': 4}, {'name': 'construction--barrier--fence', 'readable': 'Fence', 'instances': False, 'evaluate': True, 'color': [190, 153, 153], 'id': 5}, {'name': 'construction--barrier--guard-rail', 'readable': 'Guard Rail', 'instances': False, 'evaluate': True, 'color': [180, 165, 180], 'id': 6}, {'name': 'construction--barrier--other-barrier', 'readable': 'Barrier', 'instances': False, 'evaluate': True, 'color': [90, 120, 150], 'id': 7}, {'name': 'construction--barrier--road-median', 'readable': 'Road Median', 'instances': False, 'evaluate': True, 'color': [250, 170, 33], 'id': 8}, {'name': 'construction--barrier--road-side', 'readable': 'Road Side', 'instances': False, 'evaluate': True, 'color': [250, 170, 34], 'id': 9}, {'name': 'construction--barrier--separator', 'readable': 'Lane Separator', 'instances': False, 'evaluate': True, 'color': [128, 128, 128], 'id': 10}, {'name': 'construction--barrier--temporary', 'readable': 'Temporary Barrier', 'instances': True, 'evaluate': True, 'color': [250, 170, 35], 'id': 11}, {'name': 'construction--barrier--wall', 'readable': 'Wall', 'instances': False, 'evaluate': True, 'color': [102, 102, 156], 'id': 12}, {'name': 'construction--flat--bike-lane', 'readable': 'Bike Lane', 'instances': False, 'evaluate': True, 'color': [128, 64, 255], 'id': 13}, {'name': 'construction--flat--crosswalk-plain', 'readable': 'Crosswalk - Plain', 'instances': True, 'evaluate': True, 'color': [140, 140, 200], 'id': 14}, {'name': 'construction--flat--curb-cut', 'readable': 'Curb Cut', 'instances': False, 'evaluate': True, 'color': [170, 170, 170], 'id': 15}, {'name': 'construction--flat--driveway', 'readable': 'Driveway', 'instances': True, 'evaluate': True, 'color': [250, 170, 36], 'id': 16}, {'name': 'construction--flat--parking', 'readable': 'Parking', 'instances': False, 'evaluate': True, 'color': [250, 170, 160], 'id': 17}, {'name': 'construction--flat--parking-aisle', 'readable': 'Parking Aisle', 'instances': False, 'evaluate': True, 'color': [250, 170, 37], 'id': 18}, {'name': 'construction--flat--pedestrian-area', 'readable': 'Pedestrian Area', 'instances': False, 'evaluate': True, 'color': [96, 96, 96], 'id': 19}, {'name': 'construction--flat--rail-track', 'readable': 'Rail Track', 'instances': False, 'evaluate': True, 'color': [230, 150, 140], 'id': 20}, {'name': 'construction--flat--road', 'readable': 'Road', 'instances': False, 'evaluate': True, 'color': [128, 64, 128], 'id': 21}, {'name': 'construction--flat--road-shoulder', 'readable': 'Road Shoulder', 'instances': False, 'evaluate': True, 'color': [110, 110, 110], 'id': 22}, {'name': 'construction--flat--service-lane', 'readable': 'Service Lane', 'instances': False, 'evaluate': True, 'color': [110, 110, 110], 'id': 23}, {'name': 'construction--flat--sidewalk', 'readable': 'Sidewalk', 'instances': False, 'evaluate': True, 'color': [244, 35, 232], 'id': 24}, {'name': 'construction--flat--traffic-island', 'readable': 'Traffic Island', 'instances': False, 'evaluate': True, 'color': [128, 196, 128], 'id': 25}, {'name': 'construction--structure--bridge', 'readable': 'Bridge', 'instances': False, 'evaluate': True, 'color': [150, 100, 100], 'id': 26}, {'name': 'construction--structure--building', 'readable': 'Building', 'instances': False, 'evaluate': True, 'color': [70, 70, 70], 'id': 27}, {'name': 'construction--structure--garage', 'readable': 'Garage', 'instances': False, 'evaluate': True, 'color': [150, 150, 150], 'id': 28}, {'name': 'construction--structure--tunnel', 'readable': 'Tunnel', 'instances': False, 'evaluate': True, 'color': [150, 120, 90], 'id': 29}, {'name': 'human--person--individual', 'readable': 'Person', 'instances': True, 'evaluate': True, 'color': [220, 20, 60], 'id': 30}, {'name': 'human--person--person-group', 'readable': 'Person Group', 'instances': False, 'evaluate': False, 'color': [220, 20, 60], 'id': 31}, {'name': 'human--rider--bicyclist', 'readable': 'Bicyclist', 'instances': True, 'evaluate': True, 'color': [255, 0, 0], 'id': 32}, {'name': 'human--rider--motorcyclist', 'readable': 'Motorcyclist', 'instances': True, 'evaluate': True, 'color': [255, 0, 100], 'id': 33}, {'name': 'human--rider--other-rider', 'readable': 'Other Rider', 'instances': True, 'evaluate': True, 'color': [255, 0, 200], 'id': 34}, {'name': 'marking--continuous--dashed', 'readable': 'Lane Marking - Dashed Line', 'instances': False, 'evaluate': True, 'color': [255, 255, 255], 'id': 35}, {'name': 'marking--continuous--solid', 'readable': 'Lane Marking - Straight Line', 'instances': False, 'evaluate': True, 'color': [255, 255, 255], 'id': 36}, {'name': 'marking--continuous--zigzag', 'readable': 'Lane Marking - Zigzag Line', 'instances': False, 'evaluate': True, 'color': [250, 170, 29], 'id': 37}, {'name': 'marking--discrete--ambiguous', 'readable': 'Lane Marking - Ambiguous', 'instances': False, 'evaluate': False, 'color': [250, 170, 28], 'id': 38}, {'name': 'marking--discrete--arrow--left', 'readable': 'Lane Marking - Arrow (Left)', 'instances': True, 'evaluate': True, 'color': [250, 170, 26], 'id': 39}, {'name': 'marking--discrete--arrow--other', 'readable': 'Lane Marking - Arrow (Other)', 'instances': True, 'evaluate': True, 'color': [250, 170, 25], 'id': 40}, {'name': 'marking--discrete--arrow--right', 'readable': 'Lane Marking - Arrow (Right)', 'instances': True, 'evaluate': True, 'color': [250, 170, 24], 'id': 41}, {'name': 'marking--discrete--arrow--split-left-or-straight', 'readable': 'Lane Marking - Arrow (Split Left or Straight)', 'instances': True, 'evaluate': True, 'color': [250, 170, 22], 'id': 42}, {'name': 'marking--discrete--arrow--split-right-or-straight', 'readable': 'Lane Marking - Arrow (Split Right or Straight)', 'instances': True, 'evaluate': True, 'color': [250, 170, 21], 'id': 43}, {'name': 'marking--discrete--arrow--straight', 'readable': 'Lane Marking - Arrow (Straight)', 'instances': True, 'evaluate': True, 'color': [250, 170, 20], 'id': 44}, {'name': 'marking--discrete--crosswalk-zebra', 'readable': 'Lane Marking - Crosswalk', 'instances': True, 'evaluate': True, 'color': [255, 255, 255], 'id': 45}, {'name': 'marking--discrete--give-way-row', 'readable': 'Lane Marking - Give Way (Row)', 'instances': True, 'evaluate': True, 'color': [250, 170, 19], 'id': 46}, {'name': 'marking--discrete--give-way-single', 'readable': 'Lane Marking - Give Way (Single)', 'instances': True, 'evaluate': True, 'color': [250, 170, 18], 'id': 47}, {'name': 'marking--discrete--hatched--chevron', 'readable': 'Lane Marking - Hatched (Chevron)', 'instances': False, 'evaluate': True, 'color': [250, 170, 12], 'id': 48}, {'name': 'marking--discrete--hatched--diagonal', 'readable': 'Lane Marking - Hatched (Diagonal)', 'instances': False, 'evaluate': True, 'color': [250, 170, 11], 'id': 49}, {'name': 'marking--discrete--other-marking', 'readable': 'Lane Marking - Other', 'instances': True, 'evaluate': True, 'color': [255, 255, 255], 'id': 50}, {'name': 'marking--discrete--stop-line', 'readable': 'Lane Marking - Stop Line', 'instances': True, 'evaluate': True, 'color': [255, 255, 255], 'id': 51}, {'name': 'marking--discrete--symbol--bicycle', 'readable': 'Lane Marking - Symbol (Bicycle)', 'instances': True, 'evaluate': True, 'color': [250, 170, 16], 'id': 52}, {'name': 'marking--discrete--symbol--other', 'readable': 'Lane Marking - Symbol (Other)', 'instances': True, 'evaluate': True, 'color': [250, 170, 15], 'id': 53}, {'name': 'marking--discrete--text', 'readable': 'Lane Marking - Text', 'instances': True, 'evaluate': True, 'color': [250, 170, 15], 'id': 54}, {'name': 'marking-only--continuous--dashed', 'readable': 'Lane Marking (only) - Dashed Line', 'instances': False, 'evaluate': True, 'color': [255, 255, 255], 'id': 55}, {'name': 'marking-only--discrete--crosswalk-zebra', 'readable': 'Lane Marking (only) - Crosswalk', 'instances': False, 'evaluate': True, 'color': [255, 255, 255], 'id': 56}, {'name': 'marking-only--discrete--other-marking', 'readable': 'Lane Marking (only) - Other', 'instances': False, 'evaluate': True, 'color': [255, 255, 255], 'id': 57}, {'name': 'marking-only--discrete--text', 'readable': 'Lane Marking (only) - Test', 'instances': False, 'evaluate': True, 'color': [255, 255, 255], 'id': 58}, {'name': 'nature--mountain', 'readable': 'Mountain', 'instances': False, 'evaluate': True, 'color': [64, 170, 64], 'id': 59}, {'name': 'nature--sand', 'readable': 'Sand', 'instances': False, 'evaluate': True, 'color': [230, 160, 50], 'id': 60}, {'name': 'nature--sky', 'readable': 'Sky', 'instances': False, 'evaluate': True, 'color': [70, 130, 180], 'id': 61}, {'name': 'nature--snow', 'readable': 'Snow', 'instances': False, 'evaluate': True, 'color': [190, 255, 255], 'id': 62}, {'name': 'nature--terrain', 'readable': 'Terrain', 'instances': False, 'evaluate': True, 'color': [152, 251, 152], 'id': 63}, {'name': 'nature--vegetation', 'readable': 'Vegetation', 'instances': False, 'evaluate': True, 'color': [107, 142, 35], 'id': 64}, {'name': 'nature--water', 'readable': 'Water', 'instances': False, 'evaluate': True, 'color': [0, 170, 30], 'id': 65}, {'name': 'object--banner', 'readable': 'Banner', 'instances': True, 'evaluate': True, 'color': [255, 255, 128], 'id': 66}, {'name': 'object--bench', 'readable': 'Bench', 'instances': True, 'evaluate': True, 'color': [250, 0, 30], 'id': 67}, {'name': 'object--bike-rack', 'readable': 'Bike Rack', 'instances': True, 'evaluate': True, 'color': [100, 140, 180], 'id': 68}, {'name': 'object--catch-basin', 'readable': 'Catch Basin', 'instances': True, 'evaluate': True, 'color': [220, 128, 128], 'id': 69}, {'name': 'object--cctv-camera', 'readable': 'CCTV Camera', 'instances': True, 'evaluate': True, 'color': [222, 40, 40], 'id': 70}, {'name': 'object--fire-hydrant', 'readable': 'Fire Hydrant', 'instances': True, 'evaluate': True, 'color': [100, 170, 30], 'id': 71}, {'name': 'object--junction-box', 'readable': 'Junction Box', 'instances': True, 'evaluate': True, 'color': [40, 40, 40], 'id': 72}, {'name': 'object--mailbox', 'readable': 'Mailbox', 'instances': True, 'evaluate': True, 'color': [33, 33, 33], 'id': 73}, {'name': 'object--manhole', 'readable': 'Manhole', 'instances': True, 'evaluate': True, 'color': [100, 128, 160], 'id': 74}, {'name': 'object--parking-meter', 'readable': 'Parking Meter', 'instances': True, 'evaluate': True, 'color': [20, 20, 255], 'id': 75}, {'name': 'object--phone-booth', 'readable': 'Phone Booth', 'instances': True, 'evaluate': True, 'color': [142, 0, 0], 'id': 76}, {'name': 'object--pothole', 'readable': 'Pothole', 'instances': False, 'evaluate': True, 'color': [70, 100, 150], 'id': 77}, {'name': 'object--sign--advertisement', 'readable': 'Signage - Advertisement', 'instances': True, 'evaluate': True, 'color': [250, 171, 30], 'id': 78}, {'name': 'object--sign--ambiguous', 'readable': 'Signage - Ambiguous', 'instances': True, 'evaluate': False, 'color': [250, 172, 30], 'id': 79}, {'name': 'object--sign--back', 'readable': 'Signage - Back', 'instances': True, 'evaluate': True, 'color': [250, 173, 30], 'id': 80}, {'name': 'object--sign--information', 'readable': 'Signage - Information', 'instances': True, 'evaluate': True, 'color': [250, 174, 30], 'id': 81}, {'name': 'object--sign--other', 'readable': 'Signage - Other', 'instances': True, 'evaluate': True, 'color': [250, 175, 30], 'id': 82}, {'name': 'object--sign--store', 'readable': 'Signage - Store', 'instances': True, 'evaluate': True, 'color': [250, 176, 30], 'id': 83}, {'name': 'object--street-light', 'readable': 'Street Light', 'instances': True, 'evaluate': True, 'color': [210, 170, 100], 'id': 84}, {'name': 'object--support--pole', 'readable': 'Pole', 'instances': True, 'evaluate': True, 'color': [153, 153, 153], 'id': 85}, {'name': 'object--support--pole-group', 'readable': 'Pole Group', 'instances': False, 'evaluate': False, 'color': [153, 153, 153], 'id': 86}, {'name': 'object--support--traffic-sign-frame', 'readable': 'Traffic Sign Frame', 'instances': True, 'evaluate': True, 'color': [128, 128, 128], 'id': 87}, {'name': 'object--support--utility-pole', 'readable': 'Utility Pole', 'instances': True, 'evaluate': True, 'color': [0, 0, 80], 'id': 88}, {'name': 'object--traffic-cone', 'readable': 'Traffic Cone', 'instances': True, 'evaluate': True, 'color': [210, 60, 60], 'id': 89}, {'name': 'object--traffic-light--general-single', 'readable': 'Traffic Light - General (Single)', 'instances': True, 'evaluate': True, 'color': [250, 170, 30], 'id': 90}, {'name': 'object--traffic-light--pedestrians', 'readable': 'Traffic Light - Pedestrians', 'instances': True, 'evaluate': True, 'color': [250, 170, 30], 'id': 91}, {'name': 'object--traffic-light--general-upright', 'readable': 'Traffic Light - General (Upright)', 'instances': True, 'evaluate': True, 'color': [250, 170, 30], 'id': 92}, {'name': 'object--traffic-light--general-horizontal', 'readable': 'Traffic Light - General (Horizontal)', 'instances': True, 'evaluate': True, 'color': [250, 170, 30], 'id': 93}, {'name': 'object--traffic-light--cyclists', 'readable': 'Traffic Light - Cyclists', 'instances': True, 'evaluate': True, 'color': [250, 170, 30], 'id': 94}, {'name': 'object--traffic-light--other', 'readable': 'Traffic Light - Other', 'instances': True, 'evaluate': True, 'color': [250, 170, 30], 'id': 95}, {'name': 'object--traffic-sign--ambiguous', 'readable': 'Traffic Sign - Ambiguous', 'instances': True, 'evaluate': False, 'color': [192, 192, 192], 'id': 96}, {'name': 'object--traffic-sign--back', 'readable': 'Traffic Sign (Back)', 'instances': True, 'evaluate': True, 'color': [192, 192, 192], 'id': 97}, {'name': 'object--traffic-sign--direction-back', 'readable': 'Traffic Sign - Direction (Back)', 'instances': True, 'evaluate': True, 'color': [192, 192, 192], 'id': 98}, {'name': 'object--traffic-sign--direction-front', 'readable': 'Traffic Sign - Direction (Front)', 'instances': True, 'evaluate': True, 'color': [220, 220, 0], 'id': 99}, {'name': 'object--traffic-sign--front', 'readable': 'Traffic Sign (Front)', 'instances': True, 'evaluate': True, 'color': [220, 220, 0], 'id': 100}, {'name': 'object--traffic-sign--information-parking', 'readable': 'Traffic Sign - Parking', 'instances': True, 'evaluate': True, 'color': [0, 0, 196], 'id': 101}, {'name': 'object--traffic-sign--temporary-back', 'readable': 'Traffic Sign - Temporary (Back)', 'instances': True, 'evaluate': True, 'color': [192, 192, 192], 'id': 102}, {'name': 'object--traffic-sign--temporary-front', 'readable': 'Traffic Sign - Temporary (Front)', 'instances': True, 'evaluate': True, 'color': [220, 220, 0], 'id': 103}, {'name': 'object--trash-can', 'readable': 'Trash Can', 'instances': True, 'evaluate': True, 'color': [140, 140, 20], 'id': 104}, {'name': 'object--vehicle--bicycle', 'readable': 'Bicycle', 'instances': True, 'evaluate': True, 'color': [119, 11, 32], 'id': 105}, {'name': 'object--vehicle--boat', 'readable': 'Boat', 'instances': True, 'evaluate': True, 'color': [150, 0, 255], 'id': 106}, {'name': 'object--vehicle--bus', 'readable': 'Bus', 'instances': True, 'evaluate': True, 'color': [0, 60, 100], 'id': 107}, {'name': 'object--vehicle--car', 'readable': 'Car', 'instances': True, 'evaluate': True, 'color': [0, 0, 142], 'id': 108}, {'name': 'object--vehicle--caravan', 'readable': 'Caravan', 'instances': True, 'evaluate': True, 'color': [0, 0, 90], 'id': 109}, {'name': 'object--vehicle--motorcycle', 'readable': 'Motorcycle', 'instances': True, 'evaluate': True, 'color': [0, 0, 230], 'id': 110}, {'name': 'object--vehicle--on-rails', 'readable': 'On Rails', 'instances': True, 'evaluate': True, 'color': [0, 80, 100], 'id': 111}, {'name': 'object--vehicle--other-vehicle', 'readable': 'Other Vehicle', 'instances': True, 'evaluate': True, 'color': [128, 64, 64], 'id': 112}, {'name': 'object--vehicle--trailer', 'readable': 'Trailer', 'instances': True, 'evaluate': True, 'color': [0, 0, 110], 'id': 113}, {'name': 'object--vehicle--truck', 'readable': 'Truck', 'instances': True, 'evaluate': True, 'color': [0, 0, 70], 'id': 114}, {'name': 'object--vehicle--vehicle-group', 'readable': 'Vehicle Group', 'instances': False, 'evaluate': False, 'color': [0, 0, 142], 'id': 115}, {'name': 'object--vehicle--wheeled-slow', 'readable': 'Wheeled Slow', 'instances': True, 'evaluate': True, 'color': [0, 0, 192], 'id': 116}, {'name': 'object--water-valve', 'readable': 'Water Valve', 'instances': True, 'evaluate': True, 'color': [170, 170, 170], 'id': 117}, {'name': 'void--car-mount', 'readable': 'Car Mount', 'instances': False, 'evaluate': True, 'color': [32, 32, 32], 'id': 118}, {'name': 'void--dynamic', 'readable': 'Dynamic', 'instances': False, 'evaluate': True, 'color': [111, 74, 0], 'id': 119}, {'name': 'void--ego-vehicle', 'readable': 'Ego Vehicle', 'instances': False, 'evaluate': True, 'color': [120, 10, 10], 'id': 120}, {'name': 'void--ground', 'readable': 'Ground', 'instances': False, 'evaluate': True, 'color': [81, 0, 81], 'id': 121}, {'name': 'void--static', 'readable': 'Static', 'instances': False, 'evaluate': True, 'color': [111, 111, 0], 'id': 122}, {'name': 'void--unlabeled', 'readable': 'Unlabeled', 'instances': False, 'evaluate': False, 'color': [0, 0, 0], 'id': 123}],
    "version": "2.0",
    "mapping": "mvd20",
    "folder_structure": "{split}/{content}/{key:.{22}}.{ext}"
}

marking_labels = [
    {'name': 'marking--continuous--dashed', 'readable': 'Lane Marking - Dashed Line', 'instances': False, 'evaluate': True, 'color': [255, 255, 255], 'id': 35}, 
    {'name': 'marking--continuous--solid', 'readable': 'Lane Marking - Straight Line', 'instances': False, 'evaluate': True, 'color': [255, 255, 255], 'id': 36}, 
    {'name': 'marking--continuous--zigzag', 'readable': 'Lane Marking - Zigzag Line', 'instances': False, 'evaluate': True, 'color': [250, 170, 29], 'id': 37}, 
    {'name': 'marking--discrete--ambiguous', 'readable': 'Lane Marking - Ambiguous', 'instances': False, 'evaluate': False, 'color': [250, 170, 28], 'id': 38}, 
    {'name': 'marking--discrete--arrow--left', 'readable': 'Lane Marking - Arrow (Left)', 'instances': True, 'evaluate': True, 'color': [250, 170, 26], 'id': 39}, 
    {'name': 'marking--discrete--arrow--other', 'readable': 'Lane Marking - Arrow (Other)', 'instances': True, 'evaluate': True, 'color': [250, 170, 25], 'id': 40}, 
    {'name': 'marking--discrete--arrow--right', 'readable': 'Lane Marking - Arrow (Right)', 'instances': True, 'evaluate': True, 'color': [250, 170, 24], 'id': 41}, 
    {'name': 'marking--discrete--arrow--split-left-or-straight', 'readable': 'Lane Marking - Arrow (Split Left or Straight)', 'instances': True, 'evaluate': True, 'color': [250, 170, 22], 'id': 42}, 
    {'name': 'marking--discrete--arrow--split-right-or-straight', 'readable': 'Lane Marking - Arrow (Split Right or Straight)', 'instances': True, 'evaluate': True, 'color': [250, 170, 21], 'id': 43}, 
    {'name': 'marking--discrete--arrow--straight', 'readable': 'Lane Marking - Arrow (Straight)', 'instances': True, 'evaluate': True, 'color': [250, 170, 20], 'id': 44}, 
    {'name': 'marking--discrete--crosswalk-zebra', 'readable': 'Lane Marking - Crosswalk', 'instances': True, 'evaluate': True, 'color': [255, 255, 255], 'id': 45}, 
    {'name': 'marking--discrete--give-way-row', 'readable': 'Lane Marking - Give Way (Row)', 'instances': True, 'evaluate': True, 'color': [250, 170, 19], 'id': 46}, 
    {'name': 'marking--discrete--give-way-single', 'readable': 'Lane Marking - Give Way (Single)', 'instances': True, 'evaluate': True, 'color': [250, 170, 18], 'id': 47}, 
    {'name': 'marking--discrete--hatched--chevron', 'readable': 'Lane Marking - Hatched (Chevron)', 'instances': False, 'evaluate': True, 'color': [250, 170, 12], 'id': 48}, 
    {'name': 'marking--discrete--hatched--diagonal', 'readable': 'Lane Marking - Hatched (Diagonal)', 'instances': False, 'evaluate': True, 'color': [250, 170, 11], 'id': 49}, 
    {'name': 'marking--discrete--other-marking', 'readable': 'Lane Marking - Other', 'instances': True, 'evaluate': True, 'color': [255, 255, 255], 'id': 50}, 
    {'name': 'marking--discrete--stop-line', 'readable': 'Lane Marking - Stop Line', 'instances': True, 'evaluate': True, 'color': [255, 255, 255], 'id': 51}, 
    {'name': 'marking--discrete--symbol--bicycle', 'readable': 'Lane Marking - Symbol (Bicycle)', 'instances': True, 'evaluate': True, 'color': [250, 170, 16], 'id': 52}, 
    {'name': 'marking--discrete--symbol--other', 'readable': 'Lane Marking - Symbol (Other)', 'instances': True, 'evaluate': True, 'color': [250, 170, 15], 'id': 53}, 
    {'name': 'marking--discrete--text', 'readable': 'Lane Marking - Text', 'instances': True, 'evaluate': True, 'color': [250, 170, 15], 'id': 54}, 
    {'name': 'marking-only--continuous--dashed', 'readable': 'Lane Marking (only) - Dashed Line', 'instances': False, 'evaluate': True, 'color': [255, 255, 255], 'id': 55}, 
    {'name': 'marking-only--discrete--crosswalk-zebra', 'readable': 'Lane Marking (only) - Crosswalk', 'instances': False, 'evaluate': True, 'color': [255, 255, 255], 'id': 56}, 
    {'name': 'marking-only--discrete--other-marking', 'readable': 'Lane Marking (only) - Other', 'instances': False, 'evaluate': True, 'color': [255, 255, 255], 'id': 57}, 
    {'name': 'marking-only--discrete--text', 'readable': 'Lane Marking (only) - Test', 'instances': False, 'evaluate': True, 'color': [255, 255, 255], 'id': 58}]

# Map ids to continuous labels
mapillary_vistas_continuous_detail_dict = {
    2: {'name': 'construction--barrier--ambiguous', 'id': 0},
    3: {'name': 'construction--barrier--concrete-block', 'id': 1},
    4: {'name': 'construction--barrier--curb', 'id': 2},
    5: {'name': 'construction--barrier--fence', 'id': 3},
    6: {'name': 'construction--barrier--guard-rail', 'id': 4},
    7: {'name': 'construction--barrier--other-barrier', 'id': 5},
    8: {'name': 'construction--barrier--road-median', 'id': 6},
    9: {'name': 'construction--barrier--road-side', 'id': 7},
    10: {'name': 'construction--barrier--separator', 'id': 8},
    # 11: {'name': 'construction--barrier--temporary', 'id': 9},
    12: {'name': 'construction--barrier--wall', 'id': 9},
    13: {'name': 'construction--flat--bike-lane', 'id': 10},
    14: {'name': 'construction--flat--crosswalk-plain', 'id': 11},
    15: {'name': 'construction--flat--curb-cut', 'id': 12},
    16: {'name': 'construction--flat--driveway', 'id': 13},
    # parking types
    17: {'name': 'construction--flat--parking', 'id': 14},
    18: {'name': 'construction--flat--parking-aisle', 'id': 14}, # same as parking
    # sidewalk types
    19: {'name': 'construction--flat--pedestrian-area', 'id': 15}, # same as sidewalk (for now)
    24: {'name': 'construction--flat--sidewalk', 'id':15},
    
    20: {'name': 'construction--flat--rail-track', 'id': 16},
    # road types
    21: {'name': 'construction--flat--road', 'id': 17},
    23: {'name': 'construction--flat--service-lane', 'id': 17}, # same as road
    
    22: {'name': 'construction--flat--road-shoulder', 'id': 18},
    
    # other vehicle types
    25: {'name': 'construction--flat--traffic-island', 'id': 19}, # same as other vehicle
    109: {'name': 'object--vehicle--caravan', 'id': 19}, # same as other vehicle
    111: {'name': 'object--vehicle--on-rails', 'id': 19}, # same as other vehicle
    112: {'name': 'object--vehicle--other-vehicle', 'id': 19},
    113: {'name': 'object--vehicle--trailer', 'id': 19}, # same as other vehicle
    115: {'name': 'object--vehicle--vehicle-group', 'id': 19}, # same as other vehicle
    116: {'name': 'object--vehicle--wheeled-slow', 'id': 19}, # same as other vehicle
    
    26: {'name': 'construction--structure--bridge', 'id': 20},
    # building types
    27: {'name': 'construction--structure--building', 'id': 21},
    28: {'name': 'construction--structure--garage', 'id': 21}, # same as building
    
    29: {'name': 'construction--structure--tunnel', 'id': 22},
    # person types
    30: {'name': 'human--person--individual', 'id': 23},
    31: {'name': 'human--person--person-group', 'id': 23}, # same as individual
    32: {'name': 'human--rider--bicyclist', 'id': 23}, # same as individual
    33: {'name': 'human--rider--motorcyclist', 'id': 23}, # same as individual
    34: {'name': 'human--rider--other-rider', 'id': 23}, # same as individual
    
    # (skipping most of the lane markings)
    # crosswalk types
    45: {'name': 'marking--discrete--crosswalk-zebra', 'id': 24},
    56: {'name': 'marking-only--discrete--crosswalk-zebra', 'id': 24}, # same as crosswalk zebra
    
    # terrain types
    60: {'name': 'nature--sand', 'id': 25}, # same as terrain
    62: {'name': 'nature--snow', 'id': 25}, # same as terrain
    63: {'name': 'nature--terrain', 'id': 25},
    
    # vegetation types
    64: {'name': 'nature--vegetation', 'id': 28},
    
    67: {'name': 'object--bench', 'id': 29},
    68: {'name': 'object--bike-rack', 'id': 30},
    69: {'name': 'object--catch-basin', 'id': 31},
    71: {'name': 'object--fire-hydrant', 'id': 32},
    72: {'name': 'object--junction-box', 'id': 33},
    73: {'name': 'object--mailbox', 'id': 34},
    75: {'name': 'object--parking-meter', 'id': 35},
    76: {'name': 'object--phone-booth', 'id': 36},
    # 77: {'name': 'object--pothole', 'id': 37},
    
    # other signage types
    78: {'name': 'object--sign--advertisement', 'id': 37},
    79: {'name': 'object--sign--ambiguous', 'id': 37},
    80: {'name': 'object--sign--back', 'id': 37},
    81: {'name': 'object--sign--information', 'id': 37},
    82: {'name': 'object--sign--other', 'id': 37},
    83: {'name': 'object--sign--store', 'id': 37},
    
    84: {'name': 'object--street-light', 'id': 38},
    # pole types
    85: {'name': 'object--support--pole', 'id': 39},
    86: {'name': 'object--support--pole-group', 'id': 39}, # same as pole
    87: {'name': 'object--support--traffic-sign-frame', 'id': 40},
    88: {'name': 'object--support--utility-pole', 'id': 39}, # same as pole
    
    # traffic sign types
    96: {'name': 'object--traffic-sign--ambiguous', 'id': 40},
    97: {'name': 'object--traffic-sign--back', 'id': 40},
    98: {'name': 'object--traffic-sign--direction-back', 'id': 40},
    99: {'name': 'object--traffic-sign--direction-front', 'id': 40},
    100: {'name': 'object--traffic-sign--front', 'id': 40},
    101: {'name': 'object--traffic-sign--information-parking', 'id': 40},
    102: {'name': 'object--traffic-sign--temporary-back', 'id': 40},
    103: {'name': 'object--traffic-sign--temporary-front', id: 40},
    
    # traffic light types
    90: {'name': 'object--traffic-light--general-single', 'id': 41},
    91: {'name': 'object--traffic-light--pedestrians', 'id': 41},
    92: {'name': 'object--traffic-light--general-upright', 'id': 41},
    93: {'name': 'object--traffic-light--general-horizontal', 'id': 41},
    94: {'name': 'object--traffic-light--cyclists', 'id': 41},
    95: {'name': 'object--traffic-light--other', 'id': 41},
    
    104: {'name': 'object--trash-can', 'id': 42},
    105: {'name': 'object--vehicle--bicycle', 'id': 43},
    106: {'name': 'object--vehicle--boat', 'id': 44},
    107: {'name': 'object--vehicle--bus', 'id': 45},
    108: {'name': 'object--vehicle--car', 'id': 46},
    110: {'name': 'object--vehicle--motorcycle', 'id': 47},    
    114: {'name': 'object--vehicle--truck', 'id': 48},
    
    # lane marking types
    35: {'name': 'marking--continuous--dashed', 'id': 49},
    36: {'name': 'marking--continuous--solid', 'id': 49},
    37: {'name': 'marking--continuous--zigzag', 'id': 49},
    38: {'name': 'marking--discrete--ambiguous', 'id': 49},
    39: {'name': 'marking--discrete--arrow--left', 'id': 49},
    40: {'name': 'marking--discrete--arrow--other', 'id': 49},
    41: {'name': 'marking--discrete--arrow--right', 'id': 49},
    42: {'name': 'marking--discrete--arrow--split-left-or-straight', 'id': 49},
    43: {'name': 'marking--discrete--arrow--split-right-or-straight', 'id': 49},
    44: {'name': 'marking--discrete--arrow--straight', 'id': 49},
    46: {'name': 'marking--discrete--give-way-row', 'id': 49},
    47: {'name': 'marking--discrete--give-way-single', 'id': 49},
    48: {'name': 'marking--discrete--hatched--chevron', 'id': 49},
    49: {'name': 'marking--discrete--hatched--diagonal', 'id': 49},
    50: {'name': 'marking--discrete--other-marking', 'id': 49},
    51: {'name': 'marking--discrete--stop-line', 'id': 49},
    52: {'name': 'marking--discrete--symbol--bicycle', 'id': 49},
    53: {'name': 'marking--discrete--symbol--other', 'id': 49},
    54: {'name': 'marking--discrete--text', 'id': 49},
    55: {'name': 'marking-only--continuous--dashed', 'id': 49},
    57: {'name': 'marking-only--discrete--other-marking', 'id': 49},
    58: {'name': 'marking-only--discrete--text', 'id': 49}
}

mapillary_vistas_continuous_49_dict = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 12: 9, 13: 10, 14: 11, 15: 12, 
                                    16: 13, 17: 14, 18: 14, 19: 15, 24: 15, 20: 16, 21: 17, 23: 17, 22: 18, 25: 19, 109: 19, 
                                    111: 19, 112: 19, 113: 19, 115: 19, 116: 19, 26: 20, 27: 21, 28: 21, 29: 22, 30: 23, 
                                    31: 23, 32: 23, 33: 23, 34: 23, 45: 24, 56: 24, 60: 25, 62: 25, 63: 25, 64: 28, 67: 29, 
                                    68: 30, 69: 31, 71: 32, 72: 33, 73: 34, 75: 35, 76: 36, 78: 37, 79: 37, 80: 37, 81: 37, 
                                    82: 37, 83: 37, 84: 38, 85: 39, 86: 39, 87: 39, 88: 39, 96: 40, 97: 40, 98: 40, 99: 40, 
                                    100: 40, 101: 40, 102: 40, 103: 40, 90: 41, 91: 41, 92: 41, 93: 41, 94: 41, 95: 41, 
                                    104: 42, 105: 43, 106: 44, 107: 45, 108: 46, 110: 47, 114: 48,
                                    35: 49, 36: 49, 37: 49, 38: 49, 39: 49, 40: 49, 41: 49, 42: 49, 43: 49, 44: 49, 46: 49, 47: 49, 48: 49, 49: 49, 50: 49, 51: 49, 52: 49, 53: 49, 54: 49, 55: 49, 57: 49, 58: 49}

mapillary_vistas_continuous_11_dict = {
    21:0, 23:0, # road
    
    19:1, 24:1, # sidewalk
    27:2, 28:2, # building
    
    85:3, 86:3, 87:3, 88:3, # pole
    90:4, 91:4, 92:4, 93:4, 94:4, 95:4, # traffic light
    96:5, 97:5, 98:5, 99:5, 100:5, 101:5, 102:5, 103:5, # traffic sign
    
    64:6, # vegetation
    60:7, 62:7, 63:7, # terrain
    
    61:8, # static
    # dynamic
    16:10, 17:10, 18:10, # ground
    # background
    
    ## lane markings will also be part of road
    35:0, 36:0, 37:0, 38:0, 39:0, 40:0, 41:0, 42:0, 43:0, 44:0, 45:0,
    46:0, 47:0, 48:0, 49:0, 50:0, 51:0, 52:0, 53:0, 54:0, 55:0, 56:0, 57:0, 58:0,
    ## other road types (e.g. bike lane, barrier separator, etc.) will also be part of road
    13:0, 10:0, 14:0, 22:0,
    
    # curb cut will be part of sidewalk
    15:1
}