"""
PyTorch Dataset for Faster R-CNN using PASCAL VOC annotations.
"""

import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F


class LicensePlateDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir    = img_dir
        self.ann_dir    = ann_dir
        self.transforms = transforms
        self.samples    = sorted([
            f for f in os.listdir(ann_dir) if f.endswith('.xml')
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xml_file = self.samples[idx]
        tree = ET.parse(os.path.join(self.ann_dir, xml_file))
        root = tree.getroot()

        filename = root.find('filename').text
        img = Image.open(os.path.join(self.img_dir, filename)).convert('RGB')

        boxes = []
        for obj in root.findall('object'):
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            # Clamp to avoid degenerate boxes
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])

        boxes  = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones(len(boxes), dtype=torch.int64)  # class 1 = licence plate

        target = {
            'boxes' : boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) else torch.tensor([]),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
        }

        if self.transforms:
            img = self.transforms(img)
        else:
            img = F.to_tensor(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
