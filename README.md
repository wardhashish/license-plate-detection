# License Plate Detection
**CMPS 261 — Machine Learning Project**

Object detection system that detects and localizes license plates in vehicle images, comparing three architectures: YOLOv8s, RetinaNet (ResNet50-FPN v2), and Faster R-CNN (ResNet50-FPN v2).

---

## Results

| Model | Precision | Recall | F1 | mAP@0.5 |
|-------|-----------|--------|----|---------|
| YOLOv8s | 0.9182 | 0.9155 | **0.9169** | **0.9467** |
| RetinaNet ResNet50-FPN v2 | 0.8667 | 0.9155 | 0.8904 | — |
| Faster R-CNN ResNet50-FPN v2 | 0.8732 | 0.8732 | 0.8732 | — |

**YOLOv8s wins** — single-stage detectors with built-in augmentation outperform two-stage and anchor-based detectors on small datasets (433 images). RetinaNet (focal loss, two-phase fine-tuning) edges out Faster R-CNN by ~1.7 F1 points.

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
│   └── metrics.py             # Shared IoU + precision/recall/F1 evaluation
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
| 5 | `05_evaluation.ipynb` | Load all weights → evaluate on test set → compare models |

> **Tip:** All notebooks auto-detect their environment — run them locally on CPU/MPS or upload to Google Colab (T4 GPU) for faster training. No separate Colab files needed.

### Running on Google Colab

1. Upload `license_plate_data.zip` to your Google Drive root
2. Open any training notebook on Colab → Runtime → Change runtime type → T4 GPU → Run all
3. The notebook mounts Drive, extracts the data, trains, and downloads the weights automatically
4. Place the downloaded `.pt` / `.pth` files in `models/` and run `05_evaluation.ipynb` locally
