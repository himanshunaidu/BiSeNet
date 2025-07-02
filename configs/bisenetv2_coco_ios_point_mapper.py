
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    n_cats=35,
    num_aux_heads=4,
    lr_start=5e-3,
    weight_decay=1e-4,
    warmup_iters=100,
    max_iter=25000,
    dataset='CustomIOSPointMapper',
    im_root='./datasets/ios_point_mapper/iOSPointMapper_1_Cityscapes',
    train_im_anns='./datasets/ios_point_mapper/iOSPointMapper_1_Cityscapes/dataset.txt',
    val_im_anns='./datasets/ios_point_mapper/iOSPointMapper_1_Cityscapes/dataset.txt',
    scales=[0.75, 2.],
    cropsize=[640, 640],
    eval_crop=[720, 1280],
    eval_scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
    ims_per_gpu=8,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res/bisenetv2_coco_ios_point_mapper',
)
