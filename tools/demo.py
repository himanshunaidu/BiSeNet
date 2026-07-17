
import sys
sys.path.insert(0, '.')
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file


# uncomment the following line if you want to reduce cpu usage, see issue #231
#  torch.set_num_threads(4)

torch.set_grad_enabled(False)
np.random.seed(255)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2.py',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)


# palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
# cityscapes palette
palette_dict = {
    "background": [0, 0, 0],
    "road": [128, 64, 128],
    "sidewalk": [244, 35, 232],
    "building": [70, 70, 70],
    "pole": [153, 153, 153],
    "traffic light": [250, 170, 30],
    "traffic sign": [220, 220, 0],
    "curb_ramp": [220, 220, 0],
    "vegetation": [107, 142, 35],
    "terrain": [152, 251, 152],
    "static": [20, 20, 20],
    "dynamic": [240, 240, 240],
}
palette = np.zeros((256, 3), dtype=np.uint8)
for i, color in enumerate(palette_dict.values()):
    palette[i] = color
print("Palette for visualization:", palette[:len(palette_dict)])

# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)
print('Loading image:', args.img_path)
im = cv2.imread(args.img_path)[:, :, ::-1]
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

# shape divisor
org_size = im.size()[2:]
new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

# inference
im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
out = net(im)[0]
out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
out = out.argmax(dim=1)

# visualize
out = out.squeeze().detach().cpu().numpy()
print(np.unique(out))
pred = palette[out]
pred_rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
cv2.imwrite('./res.jpg', pred_rgb)
