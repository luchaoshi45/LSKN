_base_ = [
    'lsk_t_fpn_1x_dota_le90.py'
]

angle_version = 'le90'
# gpu_number = 1
# fp16 = dict(loss_scale='dynamic')

model = dict(
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=True)),
)

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(400, 400), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=[9, 11]),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    dataset=dict(
        pipeline=train_pipeline))

