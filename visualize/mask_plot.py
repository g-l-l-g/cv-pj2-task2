import mmcv
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmdet.structures.mask import BitmapMasks
import os
import torch
import numpy as np
import cv2  # 导入 OpenCV


# 1. 指定配置文件和 checkpoint 文件路径
config_file = '../work_dirs/mask-rcnn_r50_fpn_1x_voc_4/mask-rcnn_r50_fpn_1x_voc.py'
checkpoint_file = '../work_dirs/mask-rcnn_r50_fpn_1x_voc_4/best_pascal_voc_mAP_epoch_21.pth'

# 2. 指定要检测的图片路径
image_index = '000012'
img_path = f'../mmdetection/data/my images/{image_index}.jpg'

# 3. 指定可视化结果保存的路径和文件名
output_dir = os.path.join('../work_dirs/images/my images', image_index)
os.makedirs(output_dir, exist_ok=True)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = os.path.join(output_dir, f"vis_smoothed_{os.path.basename(img_path)}")  # 改个名以区分

# 4. 指定设备
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 5. 指定显示检测框的置信度阈值
score_thr = 0.5

# 6. 加载配置文件
cfg = Config.fromfile(config_file)

# 7. 初始化模型
print(f"Initializing detector with config: {config_file} and checkpoint: {checkpoint_file}")
model = init_detector(config_file, checkpoint_file, device=device)
print("Detector initialized successfully.")

# 8. 读取图片
print(f"Reading image: {img_path}")
img_bgr = mmcv.imread(img_path)  # 通常 mmcv.imread 返回 BGR 格式
if img_bgr is None:
    print(f"Error: Could not read image at {img_path}")
    exit()

# 9. 执行推理
print(f"Performing inference on {img_path}...")
result = inference_detector(model, img_bgr.copy())  # 传入图像的副本，以防被修改
print("Inference completed.")


# --- 提取最终预测 ---
final_pred_instances = result.pred_instances
final_bboxes_tensor = final_pred_instances.bboxes
final_labels_tensor = final_pred_instances.labels
final_scores_tensor = final_pred_instances.scores
# final_masks_tensor = final_pred_instances.masks # 如果有平滑处理，这会被覆盖

# --- 提取RPN Proposals ---
rpn_bboxes_tensor = None
rpn_scores_tensor = None

if hasattr(result, 'rpn_proposals_bboxes') and result.rpn_proposals_bboxes is not None:
    rpn_bboxes_tensor = result.rpn_proposals_bboxes
    print(f"Captured {rpn_bboxes_tensor.shape[0]} RPN proposal bboxes.")
else:
    print("Warning: 'rpn_proposals_bboxes' not found in the result.")

if hasattr(result, 'rpn_proposals_scores') and result.rpn_proposals_scores is not None:
    rpn_scores_tensor = result.rpn_proposals_scores
    print(f"Captured {rpn_scores_tensor.shape[0]} RPN proposal scores.")
else:
    print("Warning: 'rpn_proposals_scores' not found in the result.")


# --- 在可视化之前进行掩码平滑处理 ---
if hasattr(result, 'pred_instances') and \
        hasattr(result.pred_instances, 'masks') and \
        result.pred_instances.masks is not None and \
        len(result.pred_instances.masks) > 0:  # 确保有掩码且不为空

    print("Smoothing masks...")
    masks_tensor = result.pred_instances.masks  # 这是一个 (N, H, W) 的布尔型 Tensor

    # 将 Tensor 移到 CPU 并转换为 NumPy 数组
    masks_np_bool = masks_tensor.cpu().numpy()  # (N, H, W) 布尔数组

    processed_masks_list_np = []
    for i in range(masks_np_bool.shape[0]):  # 遍历每个实例的掩码
        single_mask_bool = masks_np_bool[i]  # (H, W) 布尔数组
        single_mask_uint8 = single_mask_bool.astype(np.uint8) * 255  # 转换为 0-255 的 uint8

        # --- 选择并应用平滑操作 ---
        # 示例：中值滤波
        # ksize 必须是奇数。值越大，平滑效果越强，但细节损失也越多。
        # 你可以尝试 3, 5, 7 等值。
        ksize = 5
        smoothed_mask_uint8 = cv2.medianBlur(single_mask_uint8, ksize)

        # 示例：高斯模糊 + 阈值化
        # blurred_mask = cv2.GaussianBlur(single_mask_uint8, (ksize, ksize), 0)
        # _, smoothed_mask_uint8 = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)
        # --- 结束平滑操作 ---

        processed_masks_list_np.append(smoothed_mask_uint8.astype(bool))  # 转回布尔型，用于 BitmapMasks

    if processed_masks_list_np:
        # 用处理后的掩码更新 result 对象中的 masks
        # 注意：直接修改 result.pred_instances.masks 需要确保类型兼容。
        # MMDetection 3.x 的 DetDataSample.pred_instances.masks 通常期望是 BitmapMasks 或 PolygonMasks 类型。
        # 如果原始是 Tensor，直接替换可能导致 visualizer 出错。
        # 更稳妥的做法是创建一个新的 BitmapMasks 对象。
        # 获取原始掩码的高和宽
        original_height, original_width = masks_tensor.shape[1], masks_tensor.shape[2]
        result.pred_instances.masks = BitmapMasks(processed_masks_list_np, original_height, original_width)
        print("Masks smoothed and updated in result.")
else:
    print("No masks found in pred_instances or pred_instances is missing.")
# --- 结束掩码平滑处理 ---


# --- 可视化 ---
img_to_draw_rpn = img_bgr.copy()
img_to_draw_final_with_rpn_combined = img_bgr.copy()  # 用于在一张图上画两者

# 绘制 RPN Proposals
if rpn_bboxes_tensor is not None:
    rpn_bboxes_np = rpn_bboxes_tensor.cpu().numpy()
    num_proposals_to_draw = min(100, rpn_bboxes_np.shape[0])  # 最多画100个

    # 如果有分数，可以按分数选择top-k
    if rpn_scores_tensor is not None:
        rpn_scores_np = rpn_scores_tensor.cpu().numpy()
        # 确保分数是一维的
        if rpn_scores_np.ndim > 1:
            rpn_scores_np = rpn_scores_np.squeeze()  # 例如，如果它是 (N,1)

        if len(rpn_scores_np) == len(rpn_bboxes_np):  # 检查长度是否匹配
            sorted_indices = np.argsort(rpn_scores_np)[::-1]  # 按分数降序
            proposals_to_plot_indices = sorted_indices[:num_proposals_to_draw]
        else:
            print(
                f"Warning: Mismatch between rpn_bboxes ({len(rpn_bboxes_np)}) and rpn_scores ({len(rpn_scores_np)}). Drawing first N proposals.")
            proposals_to_plot_indices = np.arange(min(num_proposals_to_draw, len(rpn_bboxes_np)))
    else:  # 没有分数，随机或取前N个
        proposals_to_plot_indices = np.random.choice(
            rpn_bboxes_np.shape[0],
            num_proposals_to_draw,
            replace=False) if rpn_bboxes_np.shape[0] > num_proposals_to_draw else np.arange(rpn_bboxes_np.shape[0])

    selected_rpn_proposals = rpn_bboxes_np[proposals_to_plot_indices]

    for box in selected_rpn_proposals:
        b = box.astype(np.int32)
        # 绘制 RPN proposals (例如用绿色细线)
        cv2.rectangle(img_to_draw_rpn, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
        cv2.rectangle(img_to_draw_final_with_rpn_combined, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)

    rpn_output_filename = os.path.join(output_dir, f"rpn_proposals_{os.path.basename(img_path)}")
    cv2.imwrite(rpn_output_filename, img_to_draw_rpn)
    print(f"RPN proposals visualization saved to {rpn_output_filename}")

# 绘制最终预测结果 (使用 MMDetection Visualizer)
# 注意：我们之前可能修改了 result.pred_instances.masks
# Visualizer 会使用 result 中的 pred_instances
# ... (初始化 visualizer 的代码，确保 visualizer.dataset_meta 设置正确) ...
if 'visualizer' in cfg:
    visualizer_cfg = cfg.visualizer
elif ('default_hooks' in cfg
      and 'visualization' in cfg.default_hooks
      and 'visualizer' in cfg.default_hooks.visualization):
    visualizer_cfg = cfg.default_hooks.visualization.visualizer
else:
    print("Warning: Visualizer config not found. Using a default DetLocalVisualizer.")
    visualizer_cfg = dict(type='DetLocalVisualizer', name='visualizer')
visualizer = VISUALIZERS.build(visualizer_cfg)
if hasattr(model, 'dataset_meta'):
    visualizer.dataset_meta = model.dataset_meta
# ... (设置 dataset_meta 的后备逻辑) ...

# 可视化最终结果（可能已包含平滑掩码）
final_pred_output_filename = os.path.join(output_dir, f"final_pred_{os.path.basename(img_path)}")
visualizer.add_datasample(
    name=os.path.basename(img_path) + "_final",
    image=img_bgr,  # 使用原始 BGR 图像
    data_sample=result,  # result 现在包含 pred_instances (可能带有平滑掩码)
    draw_gt=False,
    pred_score_thr=score_thr,
    out_file=final_pred_output_filename,
    show=False
)

print(f"Final prediction visualization saved to {final_pred_output_filename}")

# （可选）在一张图上同时显示 RPN proposals 和最终预测的边界框
# Visualizer 主要绘制 pred_instances。要同时绘制 RPN，需要手动在 visualizer 绘制的图上再画，
# 或者自定义 visualizer。img_to_draw_final_with_rpn_combined 已经画了 RPN proposals。
# 现在我们把最终的 bbox 也画上去。

if final_bboxes_tensor is not None:
    final_bboxes_np = final_bboxes_tensor.cpu().numpy()
    final_scores_np = final_scores_tensor.cpu().numpy()
    final_labels_np = final_labels_tensor.cpu().numpy()

    for i in range(len(final_bboxes_np)):
        if final_scores_np[i] >= score_thr:
            box = final_bboxes_np[i].astype(np.int32)
            label_idx = final_labels_np[i]
            label_name = model.dataset_meta['classes'][label_idx] if hasattr(model,
                                                                             'dataset_meta') and model.dataset_meta and 'classes' in model.dataset_meta and label_idx < len(
                model.dataset_meta['classes']) else f'cls_{label_idx}'
            score = final_scores_np[i]
            # 绘制最终 bbox (例如用蓝色粗线)
            cv2.rectangle(img_to_draw_final_with_rpn_combined, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(img_to_draw_final_with_rpn_combined, f'{label_name}: {score:.2f}', (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

combined_output_filename = os.path.join(output_dir, f"combined_proposals_final_{os.path.basename(img_path)}")
cv2.imwrite(combined_output_filename, img_to_draw_final_with_rpn_combined)
print(f"Combined RPN proposals and final bboxes visualization saved to {combined_output_filename}")
