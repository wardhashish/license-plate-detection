"""
Convert PASCAL VOC annotations to YOLO format and split into train/val/test.

Output structure:
    data/
        yolo/
            images/
                train/  val/  test/
            labels/
                train/  val/  test/
            dataset.yaml
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

BASE_DIR   = Path(__file__).resolve().parent.parent
IMG_DIR    = BASE_DIR / 'data' / 'archive' / 'images'
ANN_DIR    = BASE_DIR / 'data' / 'archive' / 'annotations'
YOLO_DIR   = BASE_DIR / 'data' / 'yolo'
# ──────────────────────────────────────────────────────────────────────────────


def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert VOC bounding box to YOLO normalized format."""
    cx = (xmin + xmax) / 2 / img_w
    cy = (ymin + ymax) / 2 / img_h
    w  = (xmax - xmin) / img_w
    h  = (ymax - ymin) / img_h
    return cx, cy, w, h


def parse_xml(xml_path):
    """Return (filename, img_w, img_h, list_of_boxes)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find('filename').text
    img_w    = int(root.find('size/width').text)
    img_h    = int(root.find('size/height').text)
    boxes = []
    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
    return filename, img_w, img_h, boxes


def prepare(seed=SEED):
    random.seed(seed)

    # Gather all XML files
    xml_files = sorted(ANN_DIR.glob('*.xml'))
    random.shuffle(xml_files)

    n       = len(xml_files)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    splits = {
        'train': xml_files[:n_train],
        'val'  : xml_files[n_train:n_train + n_val],
        'test' : xml_files[n_train + n_val:],
    }

    # Create output dirs
    for split in splits:
        (YOLO_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    counts = {}
    for split, files in splits.items():
        for xml_path in files:
            filename, img_w, img_h, boxes = parse_xml(xml_path)

            # Copy image
            src_img  = IMG_DIR / filename
            dest_img = YOLO_DIR / 'images' / split / filename
            shutil.copy2(src_img, dest_img)

            # Write YOLO label
            stem      = Path(filename).stem
            label_path = YOLO_DIR / 'labels' / split / f'{stem}.txt'
            with open(label_path, 'w') as f:
                for (xmin, ymin, xmax, ymax) in boxes:
                    cx, cy, w, h = voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
                    f.write(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')

        counts[split] = len(files)

    # Write dataset.yaml
    yaml_path = YOLO_DIR / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"path: {YOLO_DIR}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/val\n")
        f.write(f"test:  images/test\n\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['licence']\n")

    print(f"Data prepared:")
    print(f"  Train : {counts['train']} images")
    print(f"  Val   : {counts['val']}   images")
    print(f"  Test  : {counts['test']}  images")
    print(f"  YAML  : {yaml_path}")
    return str(yaml_path)


if __name__ == '__main__':
    prepare()
