import os, json
import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw

# ── paths relative to this file ──────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, 'models')
RESULT = os.path.join(BASE, 'results')

YOLO_WEIGHTS    = os.path.join(MODELS, 'yolov8s_best.pt')
FRCNN_WEIGHTS   = os.path.join(MODELS, 'fasterrcnn_best.pth')
RETINA_WEIGHTS  = os.path.join(MODELS, 'retinanet_best.pth')

DEVICE = (torch.device('cuda') if torch.cuda.is_available() else
          torch.device('mps')  if torch.backends.mps.is_available() else
          torch.device('cpu'))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="License Plate Detector", layout="wide")
st.title("License Plate Detector")
st.caption("CMPS 261 — Machine Learning Project")

def _metric_threshold(filename, fallback):
    path = os.path.join(RESULT, filename)
    if not os.path.exists(path):
        return fallback
    with open(path) as f:
        return float(json.load(f).get("threshold", fallback))

def _unified_metric(model_key):
    path = os.path.join(RESULT, "unified_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("models", {}).get(model_key)

def _unified_threshold(model_key, fallback):
    metric = _unified_metric(model_key)
    if metric is None:
        return fallback
    return float(metric.get("validation_threshold", fallback))

# ── best thresholds found on validation data ──────────────────────────────────
BEST_THRESHOLD = {
    "YOLOv8s":      _unified_threshold("yolo", 0.25),
    "RetinaNet":    _unified_threshold("retinanet", _metric_threshold("retinanet_metrics.json", 0.45)),
    "Faster R-CNN": _unified_threshold("fasterrcnn", _metric_threshold("fasterrcnn_metrics.json", 0.80)),
}

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    model_choice = st.radio(
        "Model",
        ["YOLOv8s", "RetinaNet", "Faster R-CNN"],
        help="YOLOv8s is fastest and most accurate on this dataset."
    )

    auto = st.checkbox("Auto threshold", value=True,
                       help="Uses validation-F1 thresholds from unified_metrics.json when available.")

    if auto:
        threshold = BEST_THRESHOLD[model_choice]
        st.caption(f"Using validation-F1 threshold for {model_choice}: **{threshold:.4f}**")
    else:
        threshold = st.slider(
            "Confidence threshold", min_value=0.10, max_value=0.90,
            value=BEST_THRESHOLD[model_choice], step=0.05
        )

    st.divider()
    st.subheader("Model Results")

    def _load_metric(filename):
        path = os.path.join(RESULT, filename)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    metric_files = {
        "YOLOv8s":      "yolo_metrics.json",
        "RetinaNet":    "retinanet_metrics.json",
        "Faster R-CNN": "fasterrcnn_metrics.json",
    }
    for name, fname in metric_files.items():
        m = _load_metric(fname)
        unified_key = {"YOLOv8s": "yolo", "RetinaNet": "retinanet", "Faster R-CNN": "fasterrcnn"}[name]
        unified = _unified_metric(unified_key)
        if m is None and unified is None:
            continue
        if unified is not None:
            test_metrics = unified.get("test_f1_at_validation_threshold", {})
            coco_metrics = unified.get("test_coco", {})
            f1 = f"{test_metrics.get('f1', 0):.4f}"
            map5 = f"{coco_metrics.get('AP50', 0):.4f}"
        else:
            f1 = f"{m.get('f1', 0):.4f}"
            map5 = f"{m['map50']:.4f}" if 'map50' in m else "—"
        bold = "**" if name == model_choice else ""
        st.markdown(f"{bold}{name}{bold} — F1: `{f1}` | AP@0.5: `{map5}`")

# ── model loaders (cached so they only load once) ─────────────────────────────
@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO(YOLO_WEIGHTS)

@st.cache_resource
def load_frcnn():
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_f  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    model.load_state_dict(torch.load(FRCNN_WEIGHTS, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_retinanet():
    from torchvision.models.detection import retinanet_resnet50_fpn_v2
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
    model = retinanet_resnet50_fpn_v2(weights=None)
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.head.classification_head.conv[0][0].in_channels
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels, num_anchors=num_anchors, num_classes=2,
        norm_layer=torch.nn.BatchNorm2d,
    )
    model.load_state_dict(torch.load(RETINA_WEIGHTS, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model

# ── inference ─────────────────────────────────────────────────────────────────
def run_yolo(img: Image.Image, conf: float):
    model  = load_yolo()
    result = model.predict(img, conf=conf, verbose=False)[0]
    boxes  = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append((x1, y1, x2, y2, float(box.conf[0])))
    return boxes

def run_torchvision(img: Image.Image, conf: float, loader_fn):
    import torchvision.transforms.functional as TF
    model  = loader_fn()
    tensor = TF.to_tensor(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        preds = model(tensor)[0]
    boxes = []
    for box, score in zip(preds['boxes'], preds['scores']):
        if score.item() >= conf:
            x1, y1, x2, y2 = box.cpu().tolist()
            boxes.append((x1, y1, x2, y2, float(score)))
    return boxes

# ── drawing ───────────────────────────────────────────────────────────────────
def draw_boxes(img: Image.Image, boxes):
    out  = img.copy()
    draw = ImageDraw.Draw(out)
    for (x1, y1, x2, y2, conf) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 80), width=3)
        label = f"{conf:.0%}"
        tw, th = draw.textlength(label), 14
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 6, y1], fill=(0, 220, 80))
        draw.text((x1 + 3, y1 - th - 2), label, fill=(0, 0, 0))
    return out

# ── main area ─────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a car image", type=["jpg", "jpeg", "png"],
    help="Upload any photo of a car — the model will detect and box the licence plate."
)

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")

    # Check weights exist
    weight_map = {
        "YOLOv8s":      YOLO_WEIGHTS,
        "Faster R-CNN": FRCNN_WEIGHTS,
        "RetinaNet":    RETINA_WEIGHTS,
    }
    if not os.path.exists(weight_map[model_choice]):
        st.error(f"Weights not found: `{weight_map[model_choice]}`\n\nTrain the model first using the training notebooks.")
        st.stop()

    with st.spinner(f"Running {model_choice}..."):
        if model_choice == "YOLOv8s":
            boxes = run_yolo(img, threshold)
        elif model_choice == "Faster R-CNN":
            boxes = run_torchvision(img, threshold, load_frcnn)
        else:
            boxes = run_torchvision(img, threshold, load_retinanet)

    annotated = draw_boxes(img, boxes)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader(f"Detected ({len(boxes)} plate{'s' if len(boxes) != 1 else ''})")
        st.image(annotated, use_container_width=True)

    if boxes:
        st.success(f"Found {len(boxes)} licence plate{'s' if len(boxes) != 1 else ''} — "
                   f"confidence: {', '.join(f'{b[4]:.0%}' for b in boxes)}")
    else:
        st.warning("No licence plates detected. Try lowering the confidence threshold.")

else:
    st.info("Upload a car image above to get started.")
