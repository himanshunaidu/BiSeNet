
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    n_cats=12,
    num_aux_heads=4,
    lr_start=5e-3,
    weight_decay=1e-4,
    warmup_iters=1000,
    max_iter=12500,
    dataset='CocoStuffAccessibility',
    im_root='./datasets/coco/coco_subset',
    train_im_anns='./datasets/coco/coco_subset/train.txt',
    val_im_anns='./datasets/coco/coco_subset/val.txt',
    custom_mapping_spec_key='accessibility_12',
    custom_mapping_weights=[0.2, 1.5, 3.0, 1.5, 2.5, 2.0, 2.0, 2.0, 0.5, 0.5, 0.3, 0.3], # Arbitrary weights for the classes in cocoStuff_continuous_main_dict
    scales=[0.75, 2.],
    cropsize=[640, 640],
    eval_crop=[640, 640],
    eval_scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
    ims_per_gpu=64,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res/coco/bisenetv2_coco_accessibility_stage_2',
)
