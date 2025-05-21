# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,  # 总 epochs
        by_epoch=True,
        milestones=[12, 18, 22],
        gamma=0.1)
]


# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.0001),  # Sparse R-CNN 常使用 AdamW
    clip_grad=dict(max_norm=1, norm_type=2)
)

# Option 2: Switch to SGD (more traditional for Mask R-CNN)
# optim_wrapper = dict(
#     type='OptimWrapper', # AmpOptimWrapper can also be used with SGD
#     optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001), # batch_size=2 (0.02 / 16 * 2 = 0.0025)
#     clip_grad=dict(max_norm=35, norm_type=2) # Common for SGD
# )

auto_scale_lr = dict(enable=False, base_batch_size=16)
