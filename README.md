# License Plate Detection
**CMPS 261 — Machine Learning Project**

Object detection system that detects and localizes license plates in vehicle images using three architectures: YOLOv8s, RetinaNet (ResNet50-FPN v2), and Faster R-CNN (ResNet50-FPN v2).

---

## Results

### Unified COCO-Style Evaluation

The strictest comparison path is `src/unified_evaluation.py`, which scores all three detectors through the same COCO-style AP evaluator. It also selects each model's confidence threshold on the validation set by maximizing F1, then applies that threshold once to the test set.

| Model | Val-Tuned Threshold | Test F1 | F1 95% CI | COCO AP | AP@0.5 | AP@0.75 | Mean IoU |
|-------|---------------------|---------|-----------|---------|--------|---------|----------|
| YOLOv8s | 0.4798 | **0.9265** | 0.8806-0.9655 | **0.5114** | 0.9385 | 0.5187 | 0.7862 |
| Faster R-CNN ResNet50-FPN v2 | 0.8529 | 0.9000 | 0.8467-0.9496 | 0.4995 | **0.9602** | 0.4446 | **0.7941** |
| RetinaNet ResNet50-FPN v2 | 0.4745 | 0.8828 | 0.8219-0.9315 | 0.5075 | 0.9054 | **0.5547** | 0.7889 |

Under this unified evaluator, YOLOv8s is the strongest overall model after retraining at image size 960, with the best validation-tuned test F1 and COCO AP. Faster R-CNN still has the best AP@0.5, and RetinaNet has the best AP@0.75 localization. Because the test set is small and the F1 confidence intervals overlap, the differences should still be interpreted cautiously.

### Ultralytics YOLO Reference Metric

YOLOv8s also reports its native Ultralytics test metrics:

| Model | Image Size | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|-------|------------|-----------|--------|----|---------|--------------|
| YOLOv8s | 960 | 0.9700 | 0.9103 | 0.9392 | 0.9418 | 0.5168 |

The YOLO image-size experiment is configured at `YOLO_IMGSZ = 960`; the reported result uses YOLOv8s retrained at this resolution.

---

## Dataset

- **Source**: [Car Plate Detection — Kaggle](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
- **Size**: 433 images with bounding box annotations
- **Format**: PASCAL VOC (XML)
- **Split**: 70% train (303) / 15% val (64) / 15% test (66)

---

## Project Structure

```
├── data/
│   ├── archive/               # Raw dataset — images + VOC XML annotations
│   └── yolo/                  # Auto-generated YOLO format (created by notebook 02)
├── notebooks/
│   ├── 00_full_pipeline.ipynb    # Comprehensive single-file pipeline (submission copy)
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_train_yolo.ipynb       # YOLOv8s training (runs locally and on Colab)
│   ├── 03_train_fasterrcnn.ipynb # Faster R-CNN training (runs locally and on Colab)
│   ├── 04_train_retinanet.ipynb  # RetinaNet training (runs locally and on Colab)
│   └── 05_evaluation.ipynb       # Load weights → compute metrics → compare models
├── src/
│   ├── prepare_data.py        # VOC → YOLO format conversion + train/val/test split
│   ├── dataset.py             # PyTorch Dataset for VOC license-plate annotations
│   ├── metrics.py             # Shared IoU + precision/recall/F1 evaluation
│   ├── unified_evaluation.py  # Unified COCO AP + validation-tuned F1 evaluator
│   └── check_split_duplicates.py # Near-duplicate checks across splits
├── models/                    # Trained weights — stored locally (too large for GitHub)
├── results/                   # Metrics JSON + comparison plots
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

> **macOS users:** if you hit `SSL: CERTIFICATE_VERIFY_FAILED` when torchvision downloads pretrained weights, run
> `/Applications/Python\ 3.x/Install\ Certificates.command` once. The notebooks also fall back to `certifi` automatically.

### Data Setup

The raw dataset is not included in this repo (433 images, too large).

1. Download from [Kaggle — Car Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
2. Extract into:
```
data/
└── archive/
    ├── images/         ← .jpg files
    └── annotations/    ← .xml files
```

---

## Run Order

There are two equivalent ways to run the project:

### Option A — single comprehensive notebook (recommended for submission)

| Notebook | What it does |
|----------|--------------|
| `00_full_pipeline.ipynb` | EDA → YOLOv8s → Faster R-CNN → RetinaNet → comparison, all in one file. Auto-skips training if weights already exist in `models/`. |

### Option B — separate notebooks (one model at a time)

| Step | Notebook | What it does |
|------|----------|-------------|
| 1 | `01_eda.ipynb` | Explore dataset — distributions, sample images |
| 2 | `02_train_yolo.ipynb` | Train YOLOv8s, saves `models/yolov8s_best.pt` |
| 3 | `03_train_fasterrcnn.ipynb` | Train Faster R-CNN, saves `models/fasterrcnn_best.pth` |
| 4 | `04_train_retinanet.ipynb` | Train RetinaNet, saves `models/retinanet_best.pth` |
| 5 | `05_evaluation.ipynb` | Load all weights → evaluate on test set → run unified COCO/F1 evaluation and split duplicate checks |

> **Tip:** All notebooks auto-detect their environment — run them locally on CPU/MPS or upload to Google Colab (T4 GPU) for faster training. No separate Colab files needed.

### Running on Google Colab

1. Upload `license_plate_data.zip` to your Google Drive root
2. Open any training notebook on Colab → Runtime → Change runtime type → T4 GPU → Run all
3. The notebook mounts Drive, extracts the data, trains, and downloads the weights automatically
4. Place the downloaded `.pt` / `.pth` files in `models/` and run `05_evaluation.ipynb` locally

### Extra Diagnostics

```bash
python src/unified_evaluation.py --imgsz 960 --bootstrap 1000
python src/check_split_duplicates.py --max-distance 5
```

`unified_evaluation.py` is the strictest comparison path because all three models are scored by the same COCO AP code. `check_split_duplicates.py` flags visually similar images across train/val/test so you can inspect possible split leakage.
