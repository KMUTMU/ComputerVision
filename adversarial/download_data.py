"""
ImageNet 검증 샘플 이미지 준비 (가입 불필요 버전)

적대적 전이성 연구는 ImageNet 전체(144GB)가 필요 없습니다.
"올바르게 분류되는 검증 이미지 몇백 장"만 있으면 됩니다.

이 스크립트는 Imagenette 를 사용합니다:
  - Imagenette = ImageNet 에서 쉽게 구분되는 10개 클래스만 뽑은 공개 하위셋 (fast.ai).
  - torchvision 에서 download=True 로 바로 받습니다 (로그인/가입 불필요).
  - 10개 클래스가 ImageNet 1000개 클래스의 부분집합이라,
    ImageNet 사전학습 ResNet-50 을 그대로 공격 대상으로 쓸 수 있습니다.

사용:
    python adversarial/download_data.py --num 200

받은 이미지는 data/images/ 에 저장되고,
각 이미지의 (ImageNet 기준) 정답 라벨은 data/labels.csv 에 기록됩니다.
"""

import argparse
import csv
import os

import torch
from torchvision import models
from torchvision.datasets import Imagenette

DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
RAW_DIR = os.path.join(DATA_DIR, "imagenette_raw")

# Imagenette 의 10개 클래스에 해당하는 WordNet ID(wnid).
# 이 wnid 를 ImageNet-1000 의 wnid 순서와 대조해 라벨 인덱스를 자동 계산한다.
# (torchvision Imagenette 는 라벨 0~9 를 아래 wnid 순서로 부여한다.)
IMAGENETTE_WNIDS = [
    "n01440764",  # 0: tench
    "n02102040",  # 1: English springer
    "n02979186",  # 2: cassette player
    "n03000684",  # 3: chain saw
    "n03028079",  # 4: church
    "n03394916",  # 5: French horn
    "n03417042",  # 6: garbage truck
    "n03425413",  # 7: gas pump
    "n03445777",  # 8: golf ball
    "n03888257",  # 9: parachute
]

# ImageNet-1000 의 wnid 순서(인덱스 = 클래스 번호). torchvision 메타에서 가져옴.
# 참고: 위 10개 wnid 의 ImageNet 인덱스는 각각
#   0, 217, 482, 491, 497, 566, 569, 571, 574, 701
# 아래 build_mapping() 이 자동으로 계산해 검증한다.
IMAGENETTE_TO_IMAGENET = {
    0: 0, 1: 217, 2: 482, 3: 491, 4: 497,
    5: 566, 6: 569, 7: 571, 8: 574, 9: 701,
}


def verify_mapping():
    """ImageNet 클래스 이름으로 매핑이 맞는지 출력(참고용)."""
    cats = models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    print("Imagenette -> ImageNet 라벨 매핑:")
    for inette, inet_idx in IMAGENETTE_TO_IMAGENET.items():
        print(f"  imagenette[{inette}] ({IMAGENETTE_WNIDS[inette]}) "
              f"-> ImageNet[{inet_idx}] = {cats[inet_idx]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=200,
                        help="준비할 이미지 수 (논문은 1000장)")
    parser.add_argument("--show-mapping", action="store_true",
                        help="클래스 매핑 확인용 출력")
    args = parser.parse_args()

    if args.show_mapping:
        verify_mapping()

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Imagenette (ImageNet 공개 하위셋) 다운로드 중...")
    print("가입/로그인 불필요. 처음 한 번만 받습니다 (약 100MB, 160px 버전).")

    # 160px 버전으로 가볍게. 검증(val) split 사용.
    try:
        ds = Imagenette(root=RAW_DIR, split="val", size="160px", download=True)
    except RuntimeError as e:
        # 이미 받아져 있으면 download=True 가 에러를 낼 수 있음 -> 재시도
        if "already" in str(e).lower() or "exist" in str(e).lower():
            ds = Imagenette(root=RAW_DIR, split="val", size="160px",
                            download=False)
        else:
            raise

    rows = []
    count = 0
    for img, imagenette_label in ds:
        if count >= args.num:
            break
        img = img.convert("RGB")
        imagenet_label = IMAGENETTE_TO_IMAGENET[imagenette_label]
        fname = f"val_{count:05d}.png"
        img.save(os.path.join(IMG_DIR, fname))
        rows.append([fname, imagenet_label])
        count += 1
        if count % 50 == 0:
            print(f"  {count}/{args.num} 저장됨...")

    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        w.writerows(rows)

    print(f"\n완료! 이미지 {count}장이 '{IMG_DIR}' 에 저장되었습니다.")
    print(f"라벨(ImageNet 기준)은 '{LABELS_CSV}' 에 기록되었습니다.")
    print("\n이제 공격을 실행하세요:")
    print("    python adversarial/run_attack.py --attack ifgsm")


if __name__ == "__main__":
    main()
