"""Unified evaluation for YOLOv8s, Faster R-CNN, and RetinaNet.

This script evaluates all detectors through the same COCO-style AP evaluator and
also reports a shared IoU>=0.5 F1 protocol with thresholds tuned on validation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "yolo"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def device_from_arg(name: str | None = None) -> torch.device | str:
    if name:
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if name == "cpu":
            return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def yolo_label_to_xyxy(line: str, width: int, height: int) -> list[float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    _, cx, cy, bw, bh = map(float, parts[:5])
    x1 = max(0.0, (cx - bw / 2) * width)
    y1 = max(0.0, (cy - bh / 2) * height)
    x2 = min(float(width), (cx + bw / 2) * width)
    y2 = min(float(height), (cy + bh / 2) * height)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def load_split(split: str) -> list[dict]:
    img_dir = DATA_DIR / "images" / split
    lbl_dir = DATA_DIR / "labels" / split
    records = []
    image_id = 1

    for label_path in sorted(lbl_dir.glob("*.txt")):
        stem = label_path.stem
        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        boxes = []
        with open(label_path) as f:
            for line in f:
                box = yolo_label_to_xyxy(line, width, height)
                if box is not None:
                    boxes.append(box)

        records.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            }
        )
        image_id += 1

    return records


def coco_ground_truth(records: list[dict]) -> COCO:
    images, annotations = [], []
    ann_id = 1
    for record in records:
        images.append(
            {
                "id": record["id"],
                "file_name": record["file_name"],
                "width": record["width"],
                "height": record["height"],
            }
        )
        for x1, y1, x2, y2 in record["boxes"]:
            w, h = float(x2 - x1), float(y2 - y1)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": record["id"],
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = COCO()
    coco.dataset = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "licence"}],
    }
    coco.createIndex()
    return coco


def xyxy_to_coco(box: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_cached_f1(cache: list[dict], threshold: float, iou_match: float = 0.5) -> dict:
    tp = fp = fn = 0
    ious = []
    for item in cache:
        gt_boxes = item["gt_boxes"]
        pred_boxes = item["boxes"][item["scores"] >= threshold]
        matched = set()
        for pred_box in pred_boxes:
            best_iou, best_j = 0.0, -1
            for j, gt_box in enumerate(gt_boxes):
                if j in matched:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_match and best_j != -1:
                tp += 1
                matched.add(best_j)
                ious.append(best_iou)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def find_best_threshold(cache: list[dict], min_threshold: float = 0.001, max_threshold: float = 0.99) -> dict:
    score_arrays = [item["scores"] for item in cache if len(item["scores"])]
    if not score_arrays:
        return evaluate_cached_f1(cache, min_threshold)

    scores = np.concatenate(score_arrays)
    candidates = np.unique(scores[(scores >= min_threshold) & (scores <= max_threshold)])
    candidates = np.unique(np.concatenate(([min_threshold, max_threshold], candidates)))

    best = evaluate_cached_f1(cache, min_threshold)
    for threshold in candidates:
        metrics = evaluate_cached_f1(cache, float(threshold))
        if (metrics["f1"], metrics["precision"], metrics["threshold"]) > (
            best["f1"],
            best["precision"],
            best["threshold"],
        ):
            best = metrics
    return best


def bootstrap_f1_ci(cache: list[dict], threshold: float, n_boot: int = 1000, seed: int = 42) -> dict:
    if n_boot <= 0 or not cache:
        return {}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(cache), size=len(cache))
        sample = [cache[i] for i in indices]
        values.append(evaluate_cached_f1(sample, threshold)["f1"])
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"f1_ci95_low": float(lo), "f1_ci95_high": float(hi)}


def coco_eval(records: list[dict], cache: list[dict]) -> dict:
    coco_gt = coco_ground_truth(records)
    detections = []
    for record, item in zip(records, cache):
        for box, score in zip(item["boxes"], item["scores"]):
            detections.append(
                {
                    "image_id": record["id"],
                    "category_id": 1,
                    "bbox": xyxy_to_coco(box),
                    "score": float(score),
                }
            )

    if not detections:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0}

    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = [record["id"] for record in records]
    evaluator.params.catIds = [1]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AR100": float(evaluator.stats[8]),
    }


def collect_yolo(records: list[dict], weights: Path, imgsz: int) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    cache = []
    for record in tqdm(records, desc="YOLO predictions"):
        result = model.predict(record["image_path"], conf=0.001, imgsz=imgsz, verbose=False)[0]
        boxes, scores = [], []
        for box in result.boxes:
            boxes.append(box.xyxy[0].cpu().numpy())
            scores.append(float(box.conf[0]))
        order = np.argsort(-np.asarray(scores)) if scores else []
        cache.append(
            {
                "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4)[order],
                "scores": np.asarray(scores, dtype=np.float32)[order],
                "gt_boxes": record["boxes"],
            }
        )
    return cache


def build_fasterrcnn(device: torch.device | str):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(MODELS_DIR / "fasterrcnn_best.pth", map_location=device, weights_only=True))
    return model.to(device).eval()


def build_retinanet(device: torch.device | str):
    from torchvision.models.detection import retinanet_resnet50_fpn_v2
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    model = retinanet_resnet50_fpn_v2(weights=None)
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.head.classification_head.conv[0][0].in_channels
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=2,
        norm_layer=torch.nn.BatchNorm2d,
    )
    model.load_state_dict(torch.load(MODELS_DIR / "retinanet_best.pth", map_location=device, weights_only=True))
    return model.to(device).eval()


def collect_torchvision(records: list[dict], model_builder: Callable, device: torch.device | str, name: str) -> list[dict]:
    model = model_builder(device)
    cache = []
    with torch.no_grad():
        for record in tqdm(records, desc=f"{name} predictions"):
            img = Image.open(record["image_path"]).convert("RGB")
            tensor = TF.to_tensor(img).to(device)
            pred = model([tensor])[0]
            scores = pred["scores"].detach().cpu().numpy()
            order = np.argsort(-scores)
            cache.append(
                {
                    "boxes": pred["boxes"].detach().cpu().numpy()[order],
                    "scores": scores[order],
                    "gt_boxes": record["boxes"],
                }
            )
    return cache


def rounded(metrics: dict, digits: int = 4) -> dict:
    return {key: round(value, digits) if isinstance(value, float) else value for key, value in metrics.items()}


def evaluate_model(name: str, val_records: list[dict], test_records: list[dict], args, device) -> dict:
    if name == "yolo":
        weights = MODELS_DIR / "yolov8s_best.pt"
        val_cache = collect_yolo(val_records, weights, args.imgsz)
        test_cache = collect_yolo(test_records, weights, args.imgsz)
    elif name == "fasterrcnn":
        val_cache = collect_torchvision(val_records, build_fasterrcnn, device, "Faster R-CNN")
        test_cache = collect_torchvision(test_records, build_fasterrcnn, device, "Faster R-CNN")
    elif name == "retinanet":
        val_cache = collect_torchvision(val_records, build_retinanet, device, "RetinaNet")
        test_cache = collect_torchvision(test_records, build_retinanet, device, "RetinaNet")
    else:
        raise ValueError(f"Unknown model: {name}")

    val_f1 = find_best_threshold(val_cache)
    test_f1 = evaluate_cached_f1(test_cache, val_f1["threshold"])
    ci = bootstrap_f1_ci(test_cache, val_f1["threshold"], n_boot=args.bootstrap, seed=args.seed)
    coco = coco_eval(test_records, test_cache)

    return {
        "model": name,
        "validation_threshold": round(float(val_f1["threshold"]), 4),
        "validation_f1": rounded(val_f1),
        "test_f1_at_validation_threshold": rounded({**test_f1, **ci}),
        "test_coco": rounded(coco),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified license plate detector evaluation")
    parser.add_argument("--models", nargs="+", default=["yolo", "fasterrcnn", "retinanet"])
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference/evaluation image size")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples for F1 CI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--output", default=str(RESULTS_DIR / "unified_metrics.json"))
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = device_from_arg(args.device)
    val_records = load_split("val")
    test_records = load_split("test")
    print(f"Device: {device}")
    print(f"Validation images: {len(val_records)} | Test images: {len(test_records)}")

    results = {
        "protocol": {
            "coco": "COCO bbox AP on the held-out test split for all models",
            "f1": "Threshold selected on validation by max F1, applied once to test",
            "yolo_imgsz": args.imgsz,
            "bootstrap_samples": args.bootstrap,
        },
        "models": {},
    }

    for model_name in args.models:
        results["models"][model_name] = evaluate_model(model_name, val_records, test_records, args, device)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
