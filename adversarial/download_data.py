"""
ImageNet 검증 샘플 이미지 준비

적대적 전이성 연구는 ImageNet 전체(144GB)가 필요 없습니다.
"올바르게 분류되는 검증 이미지 몇백~1000장"만 있으면 됩니다.

이 스크립트는 두 가지 경로를 지원합니다:
  1) torchvision 의 ImageNet 샘플이 담긴 공개 데이터셋을 Hugging Face 에서 받기
  2) 사용자가 직접 넣은 이미지 폴더 사용 (data/images/*.jpg)

가장 간단한 사용:
    python adversarial/download_data.py --num 200

받은 이미지는 data/images/ 에 저장되고,
각 이미지의 정답 라벨은 data/labels.csv 에 기록됩니다.
"""

import argparse
import csv
import os

DATA_DIR = os.path.join("data")
IMG_DIR = os.path.join(DATA_DIR, "images")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=200,
                        help="준비할 이미지 수 (논문은 1000장)")
    args = parser.parse_args()

    os.makedirs(IMG_DIR, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("`datasets` 라이브러리가 필요합니다. 먼저 설치하세요:")
        print("    pip install datasets pillow")
        return

    print(f"ImageNet 검증 샘플 {args.num}장 다운로드 중...")
    print("(처음 한 번만 받습니다. 네트워크 상황에 따라 몇 분 걸릴 수 있어요.)")

    # Hugging Face의 공개 ImageNet-1k 검증 split 스트리밍 (전체를 받지 않음)
    ds = load_dataset("imagenet-1k", split="validation",
                      streaming=True, trust_remote_code=True)

    rows = []
    count = 0
    for sample in ds:
        if count >= args.num:
            break
        img = sample["image"].convert("RGB")
        label = sample["label"]  # 0~999 정수 라벨
        fname = f"val_{count:05d}.png"
        img.save(os.path.join(IMG_DIR, fname))
        rows.append([fname, label])
        count += 1
        if count % 50 == 0:
            print(f"  {count}/{args.num} 저장됨...")

    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        w.writerows(rows)

    print(f"\n완료! 이미지 {count}장이 '{IMG_DIR}' 에 저장되었습니다.")
    print(f"라벨은 '{LABELS_CSV}' 에 기록되었습니다.")


if __name__ == "__main__":
    main()
