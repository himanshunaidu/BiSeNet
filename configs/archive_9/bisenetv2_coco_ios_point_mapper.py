
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    n_cats=9,
    num_aux_heads=4,
    lr_start=5e-3,
    weight_decay=1e-4,
    warmup_iters=100,
    max_iter=10000,
    dataset='CustomIOSPointMapper',
    im_root='./datasets/ios_point_mapper/iOSPointMapper_1_Cityscapes',
    train_im_anns='./datasets/ios_point_mapper/iOSPointMapper_1_Cityscapes/dataset.txt',
    val_im_anns='./datasets/ios_point_mapper/iOSPointMapper_1_Cityscapes/dataset.txt',
    custom_mapping_key='9',
    custom_mapping_weights=[1.5, 3.0, 1.5, 2.5, 2.0, 2.0, 0.1, 0.1, 0.2], # Arbitrary weights for the classes in cocoStuff_continuous_main_dict
    scales=[0.75, 2.],
    cropsize=[640, 640],
    eval_crop=[640, 640],
    eval_scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
    ims_per_gpu=8,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res/bisenetv2_coco_ios_point_mapper',
)
