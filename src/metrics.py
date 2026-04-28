"""Shared detection metrics: IoU + precision / recall / F1 / mean-IoU."""

import numpy as np
import torch


def compute_iou(box_a, box_b):
    """IoU between two [x1, y1, x2, y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih   = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter    = iw * ih
    union    = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter
    return inter / union if union > 0 else 0.0


def compute_metrics(model, loader, threshold, device, iou_match=0.5):
    """
    Run a torchvision detector over `loader` and return precision, recall, F1, mean IoU.

    A predicted box is a true positive if its highest IoU with an unmatched ground-truth
    box is >= iou_match. Score threshold filters predictions before matching.
    """
    model.eval()
    tp = fp = fn = 0
    iou_scores = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(device) for img in imgs]
            preds = model(imgs)
            for pred, tgt in zip(preds, targets):
                keep = pred['scores'].cpu() >= threshold
                pred_boxes = pred['boxes'].cpu()[keep]
                pred_scores = pred['scores'].cpu()[keep]
                order = torch.argsort(pred_scores, descending=True)
                pred_boxes = [pred_boxes[i].tolist() for i in order]
                gt_boxes = tgt['boxes'].tolist()
                matched = set()
                for pb in pred_boxes:
                    best_iou, best_j = 0.0, -1
                    for j, gb in enumerate(gt_boxes):
                        if j in matched:
                            continue
                        iou = compute_iou(pb, gb)
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    if best_iou >= iou_match and best_j != -1:
                        tp += 1
                        matched.add(best_j)
                        iou_scores.append(best_iou)
                    else:
                        fp += 1
                fn += len(gt_boxes) - len(matched)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou  = float(np.mean(iou_scores)) if iou_scores else 0.0
    return precision, recall, f1, mean_iou
