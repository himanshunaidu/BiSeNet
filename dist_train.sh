
'''
NOTE: replace torchrun with torch.distributed.launch if you use older version of pytorch. I suggest you use the same version as I do since I have not tested compatibility with older version after updating.
'''


## bisenetv1 cityscapes
export CUDA_VISIBLE_DEVICES=0,1
cfg_file=configs/bisenetv1_city.py
NGPUS=2
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file


## bisenetv2 cityscapes
export CUDA_VISIBLE_DEVICES=0,1
cfg_file=configs/bisenetv2_city.py
NGPUS=2
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file


## bisenetv1 cocostuff
export CUDA_VISIBLE_DEVICES=0,1,2,3
cfg_file=configs/bisenetv1_coco.py
NGPUS=4
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file


## bisenetv2 cocostuff
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cfg_file=configs/bisenetv2_coco.py
NGPUS=8
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file


## bisenetv1 ade20k
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cfg_file=configs/bisenetv1_ade20k.py
NGPUS=8
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file


## bisenetv2 ade20k
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cfg_file=configs/bisenetv2_ade20k.py
NGPUS=8
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file


export CUDA_VISIBLE_DEVICES=0
cfg_file=configs/bisenetv2_coco_accessibility_stage_1.py
NGPUS=1
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file
mv ./res/coco_accessibility_2/model_final.pth ./res/coco_accessibility_2/model_final_stage_1.pth
torchrun --nproc_per_node=1 tools/train_amp.py --finetune-from ./res/coco_accessibility_2/model_final_stage_1.pth --config ./configs/bisenetv2_coco_accessibility_stage_2.py

## Hyperparameter tuning
export CUDA_VISIBLE_DEVICES=0
cfg_file=configs/bisenetv2_coco_accessibility_stage_1.py
NGPUS=1
torchrun --nproc_per_node=$NGPUS tools/train_tune.py --config $cfg_file

export CUDA_VISIBLE_DEVICES=0
cfg_file=configs/bisenetv2_coco_accessibility_stage_1.py
NGPUS=1
torchrun --nproc_per_node=$NGPUS tools/train_optuna.py --config $cfg_file

export CUDA_VISIBLE_DEVICES=0
cfg_file=configs/bisenetv2_coco_accessibility_stage_1.py
NGPUS=1
torchrun --nproc_per_node=$NGPUS tools/train_amp2.py --config $cfg_file &&
cp ./res/bisenetv2_coco_accessibility_stage_1/model_final.pth ./res/bisenetv2_coco_accessibility_stage_2 &&
mv ./res/bisenetv2_coco_accessibility_stage_2/model_final.pth ./res/bisenetv2_coco_accessibility_stage_2/model_final_coco_accessibility_stage_1.pth &&
export CUDA_VISIBLE_DEVICES=0
cfg_file=configs/bisenetv2_coco_accessibility_stage_2.py
NGPUS=1
torchrun --nproc_per_node=$NGPUS tools/train_amp2.py --config $cfg_file --finetune-from ./res/bisenetv2_coco_accessibility_stage_2/model_final_coco_accessibility_stage_1.pth
# 
export CUDA_VISIBLE_DEVICES=0
cfg_file=configs/bisenetv2_coco_custom_edge_mapping.py
NGPUS=1
torchrun --nproc_per_node=$NGPUS tools/train_amp2.py --config $cfg_file --finetune-from ./res/bisenetv2_coco_custom_edge_mapping/model_final_coco_accessibility_stage_2.pth
#
export CUDA_VISIBLE_DEVICES=0
cfg_file=configs/bisenetv2_coco_ios_point_mapper.py
NGPUS=1
torchrun --nproc_per_node=$NGPUS tools/train_amp2.py --config $cfg_file --finetune-from ./res/bisenetv2_coco_ios_point_mapper/model_final_coco_custom_edge_mapping.pth
