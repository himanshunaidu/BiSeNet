
## Run inference on a single image

```bash
python tools/demo.py --config configs/bisenetv2_city.py --weight-path ./lib/models/model_zoo/model_final_v2_city.pth --img-path ./datasets/custom_images/test.jpg
```

## Finetune on a custom dataset

```bash
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 tools/train_amp.py --finetune-from ./res/coco_accessibility/model_final_coco_171.pth --config ./configs/bisenetv2_coco_custom_edge_mapping.py
```

## New Commands

// train_amp2.py instead of train_amp.py, which is a new version of the training script that supports additional features such as fine-grained freeze control, custom dataset mappings and weights, etc. 

### Custom Commands

1. Pre-train on Coco-Stuff

```bash
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 tools/train_amp2.py --config ./configs/mapillary_accessibility_12_exp/bisenetv2_mapillary_accessibility_stage_1.py
```

```bash
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 tools/train_amp2.py --config ./configs/mapillary_accessibility_12_exp/bisenetv2_mapillary_accessibility_stage_2.py --finetune-from ./res/bisenetv2_mapillary_accessibility_stage_1/model_final.pth
```

```bash
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 tools/train_amp2.py --config ./configs/mapillary_accessibility_12_exp/bisenetv2_mapillary_accessibility_ios_point_mapper.py --finetune-from ./res/bisenetv2_mapillary_accessibility_stage_2/model_final.pth
```

// Combine above three commands
```bash
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 tools/train_amp2.py --config ./configs/mapillary_accessibility_12_exp/bisenetv2_mapillary_accessibility_stage_1.py  &&
torchrun --nproc_per_node=1 tools/train_amp2.py --config ./configs/mapillary_accessibility_12_exp/bisenetv2_mapillary_accessibility_stage_2.py --finetune-from ./res/bisenetv2_mapillary_accessibility_stage_1/model_final.pth &&
torchrun --nproc_per_node=1 tools/train_amp2.py --config ./configs/mapillary_accessibility_12_exp/bisenetv2_mapillary_accessibility_ios_point_mapper.py --finetune-from ./res/bisenetv2_mapillary_accessibility_stage_2/model_final.pth
```

```bash
export CUDA_VISIBLE_DEVICES=0
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_12_exp/bisenetv2_mapillary_accessibility_stage_1.py &&
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_12_exp/bisenetv2_mapillary_accessibility_stage_2.py --finetune-from ./res/bisenetv2_mapillary_accessibility_stage_1/model_final.pth &&
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_12_exp/bisenetv2_mapillary_accessibility_ios_point_mapper.py --finetune-from ./res/bisenetv2_mapillary_accessibility_stage_2/model_final.pth
```


#### Running 12 class full training loop
```bash
export CUDA_VISIBLE_DEVICES=0
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_12/bisenetv2_mapillary_accessibility_stage_1.py &&
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_12/bisenetv2_mapillary_accessibility_stage_2.py --finetune-from ./res/bisenetv2_mapillary_accessibility_stage_1/model_final.pth &&
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_12/bisenetv2_mapillary_accessibility_ios_point_mapper.py --finetune-from ./res/bisenetv2_mapillary_accessibility_stage_2/model_final.pth
```

#### Running short full training loop
```bash
export CUDA_VISIBLE_DEVICES=2
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_12_short/bisenetv2_mapillary_accessibility_stage_1.py &&
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_12_short/bisenetv2_mapillary_accessibility_stage_2.py --finetune-from ./res/bisenetv2_mapillary_accessibility_stage_1_short/model_final.pth &&
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_12_short/bisenetv2_mapillary_accessibility_ios_point_mapper.py --finetune-from ./res/bisenetv2_mapillary_accessibility_stage_2_short/model_final.pth
```

#### Running short 0 full training loop
```bash
export CUDA_VISIBLE_DEVICES=1
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_12_short_0/bisenetv2_mapillary_accessibility_stage_1.py &&
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_12_short_0/bisenetv2_mapillary_accessibility_stage_2.py --finetune-from ./res/bisenetv2_mapillary_accessibility_stage_1_short_0/model_final.pth &&
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_12_short_0/bisenetv2_mapillary_accessibility_ios_point_mapper_curb.py --finetune-from ./res/bisenetv2_mapillary_accessibility_stage_2_short_0/model_final.pth
```

#### Running short 3 class (curb ramp) full training loop
```bash
export CUDA_VISIBLE_DEVICES=1
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_3_curb_ramp/bisenetv2_mapillary_accessibility_stage_1.py &&
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_3_curb_ramp/bisenetv2_mapillary_accessibility_stage_2.py --finetune-from ./res/curb_ramp/bisenetv2_mapillary_accessibility_stage_1/model_final.pth &&
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--rdzv-id=train_gpu0 \
tools/train_amp2.py --config ./configs/mapillary_accessibility_3_curb_ramp/bisenetv2_mapillary_accessibility_ios_point_mapper.py --finetune-from ./res/curb_ramp/bisenetv2_mapillary_accessibility_stage_2/model_final.pth
```


2. Run inference on a single image

```bash
python tools/demo.py --config ./configs/mapillary_accessibility_12_short_0/bisenetv2_mapillary_accessibility_ios_point_mapper.py --weight-path ./res/bisenetv2_mapillary_accessibility_ios_point_mapper_short_0/model_final.pth --img-path ./datasets/ios_point_mapper/leftImg8bit/default/3d543e4362_frame_000000_leftImg8bit.png
```