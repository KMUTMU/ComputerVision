"""
데이터 로딩 유틸
- data/images/ 의 이미지와 data/labels.csv 의 라벨을 읽어
  (이미지 텐서, 정답 라벨) 배치를 제공합니다.

중요한 설계 포인트:
- 적대적 공격은 [0,1] 범위의 "정규화 전" 이미지에 섭동을 더합니다.
  (사람이 보는 픽셀 공간에서 노이즈를 다뤄야 ε 제약이 의미가 있음)
- 모델에 넣기 직전에만 ImageNet 정규화를 적용합니다. (attacks.py의 normalize)
"""

import csv
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")

# ImageNet 표준 정규화 상수 (모델 forward 직전에만 사용)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_normalizer(device):
    """[0,1] 이미지를 ImageNet 정규화하는 함수를 돌려준다."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return lambda x: (x - mean) / std


class ImageNetSample(Dataset):
    def __init__(self, size=224):
        # 정규화는 하지 않고 [0,1] 텐서로만 변환 (공격을 위해)
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),  # [0,1]
        ])
        self.items = []
        if not os.path.exists(LABELS_CSV):
            raise FileNotFoundError(
                f"'{LABELS_CSV}' 가 없습니다. 먼저 데이터를 준비하세요:\n"
                f"    python adversarial/download_data.py --num 200")
        with open(LABELS_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.items.append((row["filename"], int(row["label"])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, label = self.items[idx]
        img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
        return self.tf(img), label


def get_loader(batch_size=16, size=224):
    ds = ImageNetSample(size=size)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
