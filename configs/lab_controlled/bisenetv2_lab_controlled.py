
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    n_cats=11,
    num_aux_heads=4,
    lr_start=1e-3,
    weight_decay=1e-4,
    warmup_iters=100,
    max_iter=10000,
    dataset='LabControlled',
    im_root='./datasets/lab_controlled',
    train_im_anns='./datasets/lab_controlled/dataset.txt',
    val_im_anns='./datasets/lab_controlled/dataset.txt',
    custom_mapping_key='11',
    custom_mapping_weights=[1.5, 3.0, 1.5, 2.5, 2.0, 2.0, 0.5, 0.5, 0.3, 0.3, 0.2], # Arbitrary weights for the classes in cocoStuff_continuous_main_dict
    scales=[0.5, 0.75, 1.0],
    cropsize=[640, 640],
    eval_crop=[640, 640],
    eval_scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
    ims_per_gpu=8,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res/lab_controlled',
)
