# License Plate Detection
**CMPS 261 — Machine Learning Project**

Object detection system that detects and localizes license plates in vehicle images, comparing two architectures: YOLOv8 and Faster R-CNN.

---

## Dataset
- **Name**: Car Plate Detection (Kaggle)
- **Size**: 433 images with bounding box annotations
- **Format**: PASCAL VOC (XML)
- **Split**: 70% train / 15% val / 15% test

---

## Models
| Model | Type | Backbone |
|---|---|---|
| YOLOv8n | Single-stage | CSPDarknet |
| Faster R-CNN | Two-stage | ResNet-50 FPN |

---

## Project Structure
```
├── data/
│   ├── archive/          # Raw dataset (images + VOC annotations)
│   └── yolo/             # Converted YOLO format (auto-generated)
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_train_yolo.ipynb    # YOLOv8 training & evaluation
│   ├── 03_train_fasterrcnn.ipynb  # Faster R-CNN training & evaluation
│   └── 04_evaluation.ipynb    # Model comparison & analysis
├── src/
│   ├── prepare_data.py        # VOC → YOLO conversion + train/val/test split
│   └── fasterrcnn_dataset.py  # PyTorch Dataset for Faster R-CNN
├── models/                    # Saved weights (gitignored)
├── results/                   # Plots and metrics (gitignored)
└── requirements.txt
```

---

## Setup
```bash
pip install -r requirements.txt
```

## Run Order
1. `notebooks/01_eda.ipynb` — explore the dataset
2. `notebooks/02_train_yolo.ipynb` — train and evaluate YOLOv8
3. `notebooks/03_train_fasterrcnn.ipynb` — train and evaluate Faster R-CNN
4. `notebooks/04_evaluation.ipynb` — compare both models
