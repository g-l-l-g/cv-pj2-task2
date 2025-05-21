# dataset settings
dataset_type = 'VOCDataset'
data_root = 'D:/python object/computer vision/project2/task2/mmdetection/data/VOCdevkit/'
backend_args = None

# === 训练数据处理流水线 ===
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),  # 加载标注框
    dict(type='Resize', scale=(800, 600), keep_ratio=True),  # 调整图像尺寸
    dict(type='RandomFlip', prob=0.5),  # 随机水平翻转
    dict(type='PhotoMetricDistortion'),  # <<--- 新增数据增强
    dict(type='PackDetInputs')  # 整理打包模型输入所需的数据
]

# === 测试/验证数据处理流水线 ===
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(800, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))  # 定义需要传递给模型的元信息
]

# === 训练数据加载器 ===
train_dataloader = dict(
    batch_size=2,
    num_workers=2,  # 根据你的 CPU 核心数和内存调整
    persistent_workers=True,  # 避免每个 epoch 重建 worker 进程
    sampler=dict(type='DefaultSampler', shuffle=True),  # 默认采样器，打乱数据
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 按图像长宽比分组打包，提高效率
    dataset=dict(
        type='RepeatDataset',  # 重复数据集，增加每个 epoch 的迭代次数
        times=1,  # 每个 epoch 中数据集重复 1 次
        dataset=dict(
            # *** 这里是关键修改：不再使用 ConcatDataset ***
            # 直接配置单个 VOCDataset
            type=dataset_type,  # 类型为 'VOCDataset'
            data_root=data_root,  # 数据集根目录
            ann_file='VOC2007/ImageSets/Main/trainval.txt',  # *** 只使用 VOC2007 的 trainval 文件 ***
            data_prefix=dict(sub_data_root='VOC2007/'),  # 数据前缀指向 VOC2007 文件夹
            filter_cfg=dict(  # 过滤配置
                filter_empty_gt=True,  # 过滤没有标注框的图像
                min_size=32,          # 过滤尺寸过小的图像 (可选)
                bbox_min_size=32      # 过滤尺寸过小的标注框 (可选)
            ),
            pipeline=train_pipeline,  # 使用上面定义的训练流水线
            backend_args=backend_args
        )
    )
)

# === 验证数据加载器 ===
val_dataloader = dict(
    batch_size=1,  # 验证时通常 batch_size 为 1
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/test.txt',  # *** 使用 VOC2007 的 test 文件进行验证 ***
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

# === 测试数据加载器 ===
test_dataloader = val_dataloader

# === 评估器配置 ===
# Pascal VOC2007 默认使用 11点插值计算 mAP
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator  # 测试评估器与验证评估器相同
