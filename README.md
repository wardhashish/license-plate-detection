# License Plate Detection
**CMPS 261 — Machine Learning Project**

Object detection system that detects and localizes license plates in vehicle images, comparing two architectures: YOLOv8s and Faster R-CNN (ResNet50-FPN v2).

---

## Results

| Model | Precision | Recall | F1 | mAP@0.5 |
|-------|-----------|--------|----|---------|
| YOLOv8s | 0.9182 | 0.9155 | 0.9169 | **0.9467** |
| Faster R-CNN | 0.8732 | 0.8732 | 0.8732 | — |

**YOLOv8s wins** — single-stage detectors with built-in augmentation outperform two-stage detectors on small datasets (433 images).

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
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_train_yolo.ipynb    # YOLOv8s training & evaluation (local)
│   ├── 03_train_fasterrcnn.ipynb  # Faster R-CNN training & evaluation (local)
│   ├── 04_evaluation.ipynb    # Load weights → compute metrics → compare models
│   ├── colab_yolo_training.ipynb       # YOLOv8s training on Google Colab (GPU)
│   └── colab_fasterrcnn_training.ipynb # Faster R-CNN training on Google Colab (GPU)
├── src/
│   ├── prepare_data.py        # VOC → YOLO format conversion + train/val/test split
│   └── fasterrcnn_dataset.py  # PyTorch Dataset class for Faster R-CNN
├── models/                    # Trained weights — stored locally (too large for GitHub)
├── results/                   # Metrics JSON + comparison plots
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

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

Run notebooks in this order:

| Step | Notebook | What it does |
|------|----------|-------------|
| 1 | `01_eda.ipynb` | Explore dataset — distributions, sample images |
| 2 | `02_train_yolo.ipynb` | Train YOLOv8s, saves `models/yolov8s_best.pt` |
| 3 | `03_train_fasterrcnn.ipynb` | Train Faster R-CNN, saves `models/fasterrcnn_best.pth` |
| 4 | `04_evaluation.ipynb` | Load weights → evaluate on test set → generate plots |

> **Tip:** Steps 2 and 3 are slow on CPU. Use the `colab_*.ipynb` notebooks on Google Colab (free T4 GPU) to train faster, then download the weights and place them in `models/`.

---

## Colab Training (Recommended)

If training locally is too slow:

1. Upload `license_plate_data.zip` to Google Drive root
2. Open `notebooks/colab_yolo_training.ipynb` on Colab → Runtime → T4 GPU → Run all
3. Open `notebooks/colab_fasterrcnn_training.ipynb` on Colab → Run all
4. Download the weights files and place in `models/`:
   - `models/yolov8s_best.pt`
   - `models/fasterrcnn_best.pth`
5. Run `04_evaluation.ipynb` locally
