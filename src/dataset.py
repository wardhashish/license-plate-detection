"""PyTorch Dataset for license-plate VOC annotations (Faster R-CNN / RetinaNet)."""

import os
import xml.etree.ElementTree as ET

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


class LicensePlateDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir    = img_dir
        self.ann_dir    = ann_dir
        self.transforms = transforms
        self.samples    = sorted(f for f in os.listdir(ann_dir) if f.endswith('.xml'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xml_file = self.samples[idx]
        root = ET.parse(os.path.join(self.ann_dir, xml_file)).getroot()

        filename = root.find('filename').text
        img = Image.open(os.path.join(self.img_dir, filename)).convert('RGB')

        boxes = []
        for obj in root.findall('object'):
            xmin = float(obj.find('bndbox/xmin').text)
            ymin = float(obj.find('bndbox/ymin').text)
            xmax = float(obj.find('bndbox/xmax').text)
            ymax = float(obj.find('bndbox/ymax').text)
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            area    = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            area    = torch.zeros((0,),   dtype=torch.float32)

        target = {
            'boxes'   : boxes_t,
            'labels'  : torch.ones((len(boxes),), dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area'    : area,
            'iscrowd' : torch.zeros((len(boxes),), dtype=torch.int64),
        }

        img = self.transforms(img) if self.transforms else F.to_tensor(img)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
