# cv-pj2-task2
# 基于VOC数据集的Mask R-CNN与Sparse R-CNN模型训练与测试

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 1.9+](https://img.shields.io/badge/PyTorch-1.9%2B-orange.svg)](https://pytorch.org/)
[![MMDetection 3.x](https://img.shields.io/badge/MMDetection-3.x-brightgreen.svg)](https://github.com/open-mmlab/mmdetection)

本项目基于 [MMDetection](https://github.com/open-mmlab/mmdetection) 框架，在 PASCAL VOC 数据集上训练与测试两类模型：

- **Mask R-CNN**：用于实例分割；
- **Sparse R-CNN**：用于目标检测。

项目涵盖数据预处理、模型配置、训练测试流程、结果可视化等模块。通过解决如 `KeyError: 'mask'` 等实际问题，加深对目标检测与实例分割流程的理解。

完整实验流程、配置、关键代码修改、分析结果等内容详见 `[你的实验报告文件名].pdf`（请替换为实际文件名）。

---

## 🚀 功能特性

- ✅ 支持在 VOC 数据集上训练 Mask R-CNN 与 Sparse R-CNN
- ✅ 修改 `XMLDataset` 实现 Mask 加载支持
- ✅ 使用 COCO API 对 VOC 测试集进行评估
- ✅ 可视化检测框和分割掩码，支持平滑后处理
- ✅ 训练过程可通过 TensorBoard 监控

---
## 📂 项目结构
```
.
├── work_dirs/ # 训练日志、模型权重、可视化结果
│ ├── mask-rcnn_r50_fpn_1x_voc/ # Mask R-CNN 实验输出
│ ├── sparse-rcnn_r50_fpn_1x_voc/ # Sparse R-CNN 实验输出
│ └── images/
│
├── visualize/ # 自定义可视化脚本
│ ├── mask_plt.py
│ └── sparse_plt.py
│
├── mask_voc07.py
├── mask_schedule_1x
├── mask-rcnn_r50_fpn_1x_voc.py
├── sparse_voc07.py
├── sparse_schedule_1x
├── sparse-rcnn_r50_fpn_1x_voc.py
│
├── default_runtime.py
│
├── tools/
│    ├── train.py
│    └── test.py
│
├── data/
│ └── VOCdevkit/
│     ├── VOC2007/
│     └── VOC2012/
│
├── requirements.txt
├── LICENSE
└── README.md
```
---

## 🧠 Mask 加载修改说明

MMDetection 中默认的 `XMLDataset` 并不完全支持 VOC 中的实例分割掩码加载，会导致训练时报错：`KeyError: 'mask'`。

### 解决方案（详见实验报告）：

在python环境中的mmdet库目录下找到`./datasets/xml_style.py` 文件，将其替换为该repo中mmdet_modification/目录下的`xml_style.py`文件


---

## 📦 安装指南

### ✅ 环境要求

- Python 3.8+
- PyTorch 建议使用 PyTorch 1.13.1 + CUDA 11.7
- mmdet 3.0.0
- mmcv 2.0.0
- 依赖库：`mmcv-full`、`mmengine`、`pycocotools`、`opencv-python`、`matplotlib`、`tensorboard` 等

### ✅ 安装步骤

```bash
# 创建环境
conda create -n "虚拟环境名称" python=3.8 -y
conda activate "虚拟环境名称"

# 安装 PyTorch（根据实际CUDA版本）
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

## 快速安装
####安装依赖
pip install -r requirements.txt

## 🛠 使用说明
### 📁 数据准备
下载 VOC2007 和 VOC2012 数据集；
解压至 ./data/VOCdevkit/；
确保 SegmentationObject/ 目录存在于 VOC2007 中；
配置文件中设置：
- data_root = 'data/'
- ann_file = 'VOC2007/ImageSets/Segmentation/trainval.txt'
- data_prefix = dict(img_path='VOC2007/')
- seg_subdir = 'VOC2007/SegmentationObject'

## 🏋️‍♀️ 训练模型
#### Mask R-CNN 训练
- python ./tools/train.py ./mask-rcnn_r50_fpn_1x_voc.py --work-dir ./work_dirs/mask-rcnn_experiment
#### Sparse R-CNN 训练
- python ./tools/train.py ./sparse-rcnn_r50_fpn_1x_voc.py --work-dir ./work_dirs/sparse-rcnn_experiment
## 🧪 测试与评估
#### Mask R-CNN 测试
- python ./tools/test.py ./work_dirs/mask-rcnn_experiment/mask-rcnn_r50_fpn_1x_voc.py ./work_dirs/mask-rcnn_experiment/best_pascal_voc_mAP_epoch_22.pth \
#### Sparse R-CNN 测试
- python ./tools/test.py ./work_dirs/sparse-rcnn_experiment/sparse-rcnn_r50_fpn_1x_voc.py ./work_dirs/sparse-rcnn_experiment/best_pascal_voc_mAP_epoch_22.pth \

## 🎨 结果可视化
#### mask
在python环境中的mmdet库目录下找到`./models/detectors/two_stage.py` 文件，将其替换为该repo中mmdet_modification/目录下的`two_stage.py`文件，在指定要检测的图片路径，配置文件，权重文件路径，可视化结果保存的路径和文件名之后直接运行`./visualize/mask_plot.py`文件即可
#### sparse
在指定要检测的图片路径，配置文件，权重文件路径，可视化结果保存的路径和文件名之后直接运行`./visualize/sparse_plot.py`文件即可
