import mmcv
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import os
import torch

# 1. 指定配置文件和 checkpoint 文件路径
config_file = '../work_dirs/sparse-rcnn_r50_fpn_1x_voc_3/sparse-rcnn_r50_fpn_1x_voc.py'
checkpoint_file = '../work_dirs/sparse-rcnn_r50_fpn_1x_voc_3/best_pascal_voc_mAP_epoch_22.pth'

# 2. 指定要检测的图片路径
img_path = '../mmdetection/data/my images/000017.jpg'
# img_path = '../mmdetection/data/VOCdevkit/VOC2007/JPEGImages/000445.jpg'  # 你可以更改为任何你想测试的图片

# 3. 指定可视化结果保存的路径和文件名
output_dir = '../work_dirs/images/my images'  # 定义保存图片的目录

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = os.path.join(output_dir, f"vis_{os.path.basename(img_path)}")  # 例如 vis_demo.jpg

# 4. 指定设备
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 5. 指定显示检测框的置信度阈值 (可选，但推荐)
score_thr = 0.25

# 6. 加载配置文件
cfg = Config.fromfile(config_file)

# 7. 初始化模型
print(f"Initializing detector with config: {config_file} and checkpoint: {checkpoint_file}")
model = init_detector(config_file, checkpoint_file, device=device)
print("Detector initialized successfully.")

# 8. 读取图片
print(f"Reading image: {img_path}")
img = mmcv.imread(img_path)
if img is None:
    print(f"Error: Could not read image at {img_path}")
    exit()

# 9. 执行推理
print(f"Performing inference on {img_path}...")

result = inference_detector(model, img)
print("Inference completed.")

# 10. 初始化 Visualizer 并进行可视化
print(f"Visualizing results and saving to {output_filename}...")
try:
    if 'visualizer' in cfg:
        visualizer_cfg = cfg.visualizer
    elif ('default_hooks' in cfg
          and 'visualization' in cfg.default_hooks
          and 'visualizer' in cfg.default_hooks.visualization):
        visualizer_cfg = cfg.default_hooks.visualization.visualizer
    else:
        print("Warning: Visualizer config not found in cfg.visualizer "
              "or cfg.default_hooks.visualization.visualizer. Using a default DetLocalVisualizer.")
        visualizer_cfg = dict(type='DetLocalVisualizer', name='visualizer')

    visualizer = VISUALIZERS.build(visualizer_cfg)

    if hasattr(model, 'dataset_meta'):
        visualizer.dataset_meta = model.dataset_meta
    elif 'dataset_meta' in cfg:  # 有些配置文件可能直接定义了dataset_meta
        visualizer.dataset_meta = cfg.dataset_meta
    else:

        print("Warning: model.dataset_meta not found. Manually setting for VOC (20 classes).")
        voc_classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                       'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                       'train', 'tvmonitor')
        visualizer.dataset_meta = {'classes': voc_classes, 'palette': [([0]*3)]*len(voc_classes)}

    if hasattr(result, 'pred_instances'):
        visualizer.add_datasample(
            name=os.path.basename(img_path),
            image=img,  # BGR image
            data_sample=result,
            draw_gt=False,
            pred_score_thr=score_thr,
            out_file=output_filename,
            show=False
        )
        print(f"Visualization saved to {output_filename}")
    else:
        print("Result is not a DetDataSample. Attempting MMDetection 2.x style visualization (model.show_result).")
        model.show_result(
            img,
            result,
            score_thr=score_thr,
            show=False,
            out_file=output_filename
        )
        print(f"Visualization saved to {output_filename} using model.show_result")

except Exception as e:
    print(f"Error during visualization: {e}")
    print("Make sure your MMDetection and MMEngine versions are compatible and visualizer is configured correctly.")
    print("Alternatively, consider using tools/test.py with --show-dir for robust visualization.")
