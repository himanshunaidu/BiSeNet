
## Run inference on a single image

```bash
python tools/demo.py --config configs/bisenetv2_city.py --weight-path ./lib/models/model_zoo/model_final_v2_city.pth --img-path ./datasets/custom_images/test.jpg
```

## Finetune on a custom dataset

```bash
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 tools/train_amp.py --finetune-from ./res/coco_accessibility/model_final_coco_171.pth --config ./configs/bisenetv2_coco_custom_edge_mapping.py
```