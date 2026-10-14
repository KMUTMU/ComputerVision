"""
공식 ImageNet 검증셋(ILSVRC 2012 val) 준비  [논문과 동일 환경]

전제: 아래 두 파일을 data/imagenet/ 에 넣어두었다고 가정합니다.
    data/imagenet/ILSVRC2012_img_val.tar          (6.3GB, 검증 이미지 5만장)
    data/imagenet/ILSVRC2012_devkit_t12.tar.gz     (라벨 정보)

이 스크립트가 하는 일:
    1) torchvision.datasets.ImageNet 으로 val 셋을 로드 (tar 자동 압축해제 + 라벨 매칭)
    2) 논문 방식대로 '서로 다른 클래스에서' 이미지를 선별
       - 기본: 클래스당 1장씩, 최대 1000장 (논문 설정과 동일: 1000개 서로 다른 카테고리)
    3) 사전학습 ResNet-50 이 '올바르게 분류하는' 이미지만 채택
       (논문 4.1.1: 거의 모든 모델이 올바르게 분류하는 이미지를 사용)
    4) 선별 이미지를 data/images/ 에 저장하고 data/labels.csv 에 라벨 기록
       -> 기존 run_attack.py / visualize.py 가 그대로 사용 가능

사용 예:
    python adversarial/prepare_imagenet.py --per-class 1          # 논문식 1000장
    python adversarial/prepare_imagenet.py --per-class 1 --max 200  # 빠른 테스트
"""

import argparse
import csv
import os

import torch
from torchvision import models, transforms
from torchvision.datasets import ImageNet

IMAGENET_ROOT = os.path.join("data", "imagenet")
OUT_IMG_DIR = os.path.join("data", "images")
OUT_LABELS = os.path.join("data", "labels.csv")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=1,
                   help="각 클래스에서 뽑을 이미지 수 (논문=1)")
    p.add_argument("--max", type=int, default=1000,
                   help="최대 총 이미지 수 (논문=1000)")
    p.add_argument("--no-filter", action="store_true",
                   help="정답 필터링을 끄기 (더 빠르지만 덜 깨끗)")
    p.add_argument("--filter-models", type=str, nargs="+",
                   default=["resnet50"],
                   help="이 모델들이 '모두' 맞히는 이미지만 선별 "
                        "(논문 기준에 맞추려면 여러 개 지정, "
                        "예: resnet50 densenet121 vit_b)")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


# 필터링용 모델 로더
_FILTER_ZOO = {
    "resnet50": (models.resnet50, "ResNet50_Weights"),
    "densenet121": (models.densenet121, "DenseNet121_Weights"),
    "resnext50": (models.resnext50_32x4d, "ResNeXt50_32X4D_Weights"),
    "convnext_t": (models.convnext_tiny, "ConvNeXt_Tiny_Weights"),
    "vit_b": (models.vit_b_16, "ViT_B_16_Weights"),
    "swin_t": (models.swin_t, "Swin_T_Weights"),
}


def load_filter_models(names, device):
    loaded = []
    for name in names:
        if name not in _FILTER_ZOO:
            raise ValueError(f"알 수 없는 필터 모델: {name} "
                             f"(가능: {list(_FILTER_ZOO)})")
        builder, weights_attr = _FILTER_ZOO[name]
        weights = getattr(models, weights_attr).IMAGENET1K_V1
        m = builder(weights=weights).eval().to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        loaded.append((name, m))
    return loaded


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir(IMAGENET_ROOT):
        print(f"'{IMAGENET_ROOT}' 폴더가 없습니다.")
        print("다음 두 파일을 거기에 넣어주세요:")
        print("  - ILSVRC2012_img_val.tar")
        print("  - ILSVRC2012_devkit_t12.tar.gz")
        return

    print("ImageNet val 로드 중... (처음엔 tar 압축해제로 수 분 걸릴 수 있어요)")
    # 모델에 넣기 전 [0,1] 텐서로만 (정규화는 공격 파이프라인에서 따로)
    load_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    dataset = ImageNet(root=IMAGENET_ROOT, split="val", transform=load_tf)

    # 정답 필터링용 모델들 (여러 개면 '모두' 맞히는 이미지만 채택 = 논문 기준)
    filter_models = []
    normalize = None
    to_tensor = None
    if not args.no_filter:
        print(f"필터 모델 로드 중: {', '.join(args.filter_models)} "
              f"(처음이면 다운로드)...")
        filter_models = load_filter_models(args.filter_models, device)
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        normalize = lambda x: (x - mean) / std
        to_tensor = transforms.ToTensor()

    os.makedirs(OUT_IMG_DIR, exist_ok=True)

    per_class_count = {}
    rows = []
    saved = 0

    print(f"선별 시작: 클래스당 최대 {args.per_class}장, 총 최대 {args.max}장")
    if not args.no_filter:
        if len(filter_models) == 1:
            print(f"({filter_models[0][0]} 이 올바르게 분류하는 이미지만 채택)")
        else:
            print(f"({', '.join(args.filter_models)} 이 '모두' 올바르게 "
                  f"분류하는 이미지만 채택 — 논문 기준)")

    for img, label in dataset:
        if saved >= args.max:
            break
        if per_class_count.get(label, 0) >= args.per_class:
            continue

        # 정답 필터: 지정된 모델들이 '전부' 맞히는지 확인
        if not args.no_filter:
            x = to_tensor(img).unsqueeze(0).to(device)
            all_correct = True
            with torch.no_grad():
                xn = normalize(x)
                for _, m in filter_models:
                    if m(xn).argmax(1).item() != label:
                        all_correct = False
                        break
            if not all_correct:
                continue  # 하나라도 틀리면 건너뜀 (논문: 모든 모델이 맞히는 이미지)

        fname = f"val_{saved:05d}.png"
        img.convert("RGB").save(os.path.join(OUT_IMG_DIR, fname))
        rows.append([fname, label])
        per_class_count[label] = per_class_count.get(label, 0) + 1
        saved += 1
        if saved % 100 == 0:
            print(f"  {saved}/{args.max} 선별됨 "
                  f"({len(per_class_count)}개 클래스)")

    with open(OUT_LABELS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        w.writerows(rows)

    print(f"\n완료! {saved}장 선별 "
          f"({len(per_class_count)}개 서로 다른 클래스).")
    print(f"이미지: '{OUT_IMG_DIR}'   라벨: '{OUT_LABELS}'")
    print("\n이제 공격을 실행하세요:")
    print("    python adversarial/run_attack.py --attack ifgsm")


if __name__ == "__main__":
    main()
