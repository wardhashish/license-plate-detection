# License Plate Detection
**CMPS 261 — Machine Learning Project**

Object detection system that detects and localizes license plates in vehicle images using three architectures: YOLOv8s, RetinaNet (ResNet50-FPN v2), and Faster R-CNN (ResNet50-FPN v2).

---

## Results

> **Numbers below are pending re-run on the deduplicated split** (see [Dataset](#dataset) — the raw Kaggle archive contains 131 byte-identical duplicate images that previously leaked across train/val/test). Re-run `notebooks/02_train_yolo.ipynb`, `03_train_fasterrcnn.ipynb`, `04_train_retinanet.ipynb`, then `src/unified_evaluation.py` to populate this section with honest numbers.

### Unified COCO-Style Evaluation

The strictest comparison path is `src/unified_evaluation.py`, which scores all three detectors through the same COCO-style AP evaluator. It also selects each model's confidence threshold on the validation set by maximizing F1, then applies that threshold once to the test set.

---

## Dataset

- **Source**: [Car Plate Detection — Kaggle](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
- **Raw archive**: 433 image files, but only **302 unique images** by MD5 — the archive ships 131 byte-identical copies under different `Cars###.png` filenames. `src/prepare_data.py` deduplicates by content hash before splitting, so no image appears in more than one split.
- **Format**: PASCAL VOC (XML)
- **Split (after dedup)**: 70% train (211) / 15% val (45) / 15% test (46)

---

## Project Structure

```
├── data/
│   ├── archive/               # Raw dataset — images + VOC XML annotations
│   └── yolo/                  # Auto-generated purified YOLO split
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

The raw dataset is not included in this repo (302 unique images after dedup, 433 file paths in the archive).

1. Download from [Kaggle — Car Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
2. Extract into:
```
data/
└── archive/
    ├── images/         ← .jpg files
    └── annotations/    ← .xml files
```
Each notebook prepares the purified `data/yolo` split automatically before doing its own work.

---

## Run Order

There are two equivalent ways to run the project:

### Option A — single comprehensive notebook (recommended for submission)

| Notebook | What it does |
|----------|--------------|
| `00_full_pipeline.ipynb` | Purified data preparation → EDA → YOLOv8s → Faster R-CNN → RetinaNet → comparison, all in one file. Auto-skips training if weights already exist in `models/`. |

### Option B — separate notebooks (one model at a time)

| Step | Notebook | What it does |
|------|----------|-------------|
| 1 | `01_eda.ipynb` | Prepare purified data, then explore dataset distributions and samples |
| 2 | `02_train_yolo.ipynb` | Prepare purified data, train YOLOv8s, saves `models/yolov8s_best.pt` |
| 3 | `03_train_fasterrcnn.ipynb` | Prepare purified data, train Faster R-CNN, saves `models/fasterrcnn_best.pth` |
| 4 | `04_train_retinanet.ipynb` | Prepare purified data, train RetinaNet, saves `models/retinanet_best.pth` |
| 5 | `05_evaluation.ipynb` | Prepare purified data, load all weights, evaluate on test set, and run unified COCO/F1 evaluation |

> **Tip:** All notebooks auto-detect their environment — run them locally on CPU/MPS or upload to Google Colab (T4 GPU) for faster training. No separate Colab files needed.

### Running on Google Colab

1. Upload `license_plate_data.zip` to your Google Drive root
2. Open any notebook on Colab → Runtime → Change runtime type → T4 GPU for training notebooks
3. Run all cells. The notebook mounts Drive, extracts the data if needed, rebuilds the purified split, and then continues automatically
4. Place the downloaded `.pt` / `.pth` files in `models/` and run `05_evaluation.ipynb` locally

### Extra Diagnostics

```bash
python src/unified_evaluation.py --imgsz 960 --bootstrap 1000
python src/check_split_duplicates.py --max-distance 5
```

`unified_evaluation.py` is the strictest comparison path because all three models are scored by the same COCO AP code. `check_split_duplicates.py` flags visually similar images across train/val/test so you can inspect possible split leakage.
