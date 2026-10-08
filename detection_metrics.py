import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms

from dataset import VesselDataset

# Keep the evaluation score threshold in sync with train.py (SD2_EVAL_SCORE).
EVAL_SCORE = float(os.environ.get('SD2_EVAL_SCORE', '0.5'))


def box_cxcywh_to_xyxy(boxes):
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)

    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - (w / 2)
    y1 = cy - (h / 2)
    x2 = cx + (w / 2)
    y2 = cy + (h / 2)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def canvas_to_orig_boxes(boxes_cxcywh, new_w, new_h, pad_l, pad_t, canvas=1024):
    """Map normalized cxcywh boxes from the padded model-input canvas back to
    original-image normalized space (for visualization on the raw image).

    The dataset letterboxes images: original -> (new_w, new_h) -> padded canvas
    of size `canvas` at offset (pad_l, pad_t).
    """
    boxes = boxes_cxcywh.clone()
    boxes[:, 0] = (boxes[:, 0] * canvas - pad_l) / new_w
    boxes[:, 1] = (boxes[:, 1] * canvas - pad_t) / new_h
    boxes[:, 2] = (boxes[:, 2] * canvas) / new_w
    boxes[:, 3] = (boxes[:, 3] * canvas) / new_h
    return boxes


def box_iou_xyxy(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter_area
    return inter_area / union.clamp(min=1e-6)


def filter_predictions(pred_boxes, pred_logits, score_thresh=0.1, nms_iou_thresh=0.5):
    probs = torch.softmax(pred_logits, dim=-1)
    scores = probs[:, 1]
    keep = scores >= score_thresh

    if not keep.any():
        return torch.empty((0, 4)), torch.empty((0,))

    boxes = box_cxcywh_to_xyxy(pred_boxes[keep]).clamp(0.0, 1.0)
    scores = scores[keep]
    keep_idx = nms(boxes, scores, nms_iou_thresh)
    return boxes[keep_idx], scores[keep_idx]


def compute_ap(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for idx in range(len(mpre) - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])

    change_idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[change_idx + 1] - mrec[change_idx]) * mpre[change_idx + 1])


def evaluate_detection(
    model,
    data_root='DSCA_new',
    split='val',
    score_thresh=EVAL_SCORE,
    nms_iou_thresh=0.5,
    match_iou_thresh=0.5,
    batch_size=1,
    num_workers=4,
):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    dataset = VesselDataset(data_root, split=split, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    gt_map = {}
    predictions = []
    total_gt = 0

    with torch.no_grad():
        for images, _, det_dict, img_names in loader:
            images = images.to(device)
            _, pred_boxes, pred_logits, det_outputs, *_ = model(images)

            for idx, img_name in enumerate(img_names):
                gt_boxes = det_dict['boxes'][idx][det_dict['labels'][idx] > 0]
                gt_boxes = box_cxcywh_to_xyxy(gt_boxes).clamp(0.0, 1.0)
                gt_map[img_name] = gt_boxes
                total_gt += gt_boxes.shape[0]

                boxes, scores = filter_predictions(
                    pred_boxes[idx].cpu(),
                    pred_logits[idx].cpu(),
                    score_thresh=score_thresh,
                    nms_iou_thresh=nms_iou_thresh,
                )
                for box, score in zip(boxes, scores):
                    predictions.append({
                        'image_id': img_name,
                        'score': float(score.item()),
                        'box': box,
                    })

    predictions.sort(key=lambda item: item['score'], reverse=True)

    matched = {
        image_id: torch.zeros(len(boxes), dtype=torch.bool)
        for image_id, boxes in gt_map.items()
    }

    tp = []
    fp = []
    matched_ious = []

    best_ious_per_gt = []
    for image_id, gt_boxes in gt_map.items():
        image_preds = [pred['box'] for pred in predictions if pred['image_id'] == image_id]
        if gt_boxes.numel() == 0:
            continue
        if image_preds:
            pred_boxes = torch.stack(image_preds, dim=0)
            ious = box_iou_xyxy(gt_boxes, pred_boxes)
            best_ious_per_gt.extend(ious.max(dim=1).values.tolist())
        else:
            best_ious_per_gt.extend([0.0] * gt_boxes.shape[0])

    for pred in predictions:
        gt_boxes = gt_map[pred['image_id']]
        if gt_boxes.numel() == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue

        ious = box_iou_xyxy(pred['box'].unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_idx = ious.max(dim=0)
        if best_iou.item() >= match_iou_thresh and not matched[pred['image_id']][best_idx]:
            matched[pred['image_id']][best_idx] = True
            tp.append(1.0)
            fp.append(0.0)
            matched_ious.append(float(best_iou.item()))
        else:
            tp.append(0.0)
            fp.append(1.0)

    if predictions:
        tp = np.cumsum(np.array(tp, dtype=np.float64))
        fp = np.cumsum(np.array(fp, dtype=np.float64))
        recalls = tp / max(total_gt, 1)
        precisions = tp / np.maximum(tp + fp, 1e-12)
        ap50 = compute_ap(recalls, precisions) if total_gt > 0 else 0.0
        final_tp = float(tp[-1])
        final_fp = float(fp[-1])
    else:
        recalls = np.array([], dtype=np.float64)
        precisions = np.array([], dtype=np.float64)
        ap50 = 0.0
        final_tp = 0.0
        final_fp = 0.0

    final_fn = float(total_gt - final_tp)
    precision = final_tp / max(final_tp + final_fp, 1e-12)
    recall = final_tp / max(total_gt, 1)
    mean_matched_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    mean_best_iou = float(np.mean(best_ious_per_gt)) if best_ious_per_gt else 0.0
    if was_training:
        model.train()

    return {
        'ap50': float(ap50),
        'mean_matched_iou': mean_matched_iou,
        'mean_best_iou': mean_best_iou,
        'precision': float(precision),
        'recall': float(recall),
        'num_gt': int(total_gt),
        'num_predictions': int(len(predictions)),
        'num_true_positive': int(final_tp),
        'num_false_positive': int(final_fp),
        'num_false_negative': int(final_fn),
        'score_thresh': score_thresh,
        'nms_iou_thresh': nms_iou_thresh,
        'match_iou_thresh': match_iou_thresh,
    }
