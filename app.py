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

# ── best thresholds found during training (maximise F1 on val set) ────────────
BEST_THRESHOLD = {
    "YOLOv8s":      0.25,
    "RetinaNet":    0.45,
    "Faster R-CNN": 0.50,
}

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    model_choice = st.radio(
        "Model",
        ["YOLOv8s", "RetinaNet", "Faster R-CNN"],
        help="YOLOv8s is fastest and most accurate on this dataset."
    )

    auto = st.checkbox("Auto threshold (best F1)", value=True,
                       help="Uses the confidence threshold that maximised F1 on the validation set for each model.")

    if auto:
        threshold = BEST_THRESHOLD[model_choice]
        st.caption(f"Using best threshold for {model_choice}: **{threshold}**")
    else:
        threshold = st.slider(
            "Confidence threshold", min_value=0.10, max_value=0.90,
            value=BEST_THRESHOLD[model_choice], step=0.05
        )

    st.divider()
    st.subheader("Model Results")
    metrics_data = [
        {"Model": "YOLOv8s",       "F1": 0.9169, "mAP@0.5": 0.9467},
        {"Model": "RetinaNet",     "F1": 0.8904, "mAP@0.5": "—"},
        {"Model": "Faster R-CNN",  "F1": 0.8732, "mAP@0.5": "—"},
    ]
    for m in metrics_data:
        bold = "**" if m["Model"] == model_choice else ""
        st.markdown(f"{bold}{m['Model']}{bold} — F1: `{m['F1']}` | mAP@0.5: `{m['mAP@0.5']}`")

# ── model loaders (cached so they only load once) ─────────────────────────────
@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO(YOLO_WEIGHTS)

@st.cache_resource
def load_frcnn():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_f  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    model.load_state_dict(torch.load(FRCNN_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_retinanet():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    from torchvision.models.detection import retinanet_resnet50_fpn_v2
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
    model = retinanet_resnet50_fpn_v2(weights=None)
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.head.classification_head.conv[0][0].in_channels
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels, num_anchors=num_anchors, num_classes=2,
        norm_layer=torch.nn.BatchNorm2d,
    )
    model.load_state_dict(torch.load(RETINA_WEIGHTS, map_location=DEVICE))
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
