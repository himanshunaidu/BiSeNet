
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    n_cats=8,
    num_aux_heads=4,
    lr_start=5e-3,
    weight_decay=1e-4,
    warmup_iters=100,
    max_iter=30000,
    dataset='CocoStuffAccessibilityCustomEdgeMapping2',
    im_root='./datasets/custom_edge_mapping',
    train_im_anns='./datasets/custom_edge_mapping/train.txt',
    val_im_anns='./datasets/custom_edge_mapping/val.txt',
    scales=[0.75, 2.],
    cropsize=[640, 640],
    eval_crop=[720, 1280],
    eval_scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
    ims_per_gpu=8,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res/bisenetv2_coco_custom_subset_edge_mapping',
)
