"""
적대적 '전이성(transferability)' 평가  [레벨 2 - 논문의 핵심]

레벨 1(run_attack.py)은 '화이트박스' 였습니다:
    공격 만든 모델 = 평가 모델 (ResNet-50 -> ResNet-50)  => ASR 거의 100%

레벨 2는 '블랙박스 전이' 입니다 (논문 Table 1 의 설정):
    surrogate(공격 생성) 모델 = ResNet-50
    target(평가) 모델        = DenseNet-121, ResNeXt-50, ViT-B, ...  (서로 다른 구조)
    => ResNet-50 으로 만든 공격이 '다른 모델'에도 통하는가?

핵심 포인트:
  - 공격은 surrogate(ResNet-50) 에서 '한 번만' 생성한다.
  - 그 똑같은 공격 이미지를 여러 target 모델에 넣어 ASR 을 각각 측정한다.
  - 화이트박스(ResNet-50 자기 자신)는 높고, 전이(다른 모델)는 낮게 나올 것 —
    그 '떨어지는 폭'이 바로 전이성 문제이며, LPAA 논문이 풀려는 것.

사용 예:
    python adversarial/run_transfer.py --attack ifgsm
    python adversarial/run_transfer.py --attack mifgsm
"""

import argparse
import time

import torch
from torchvision import models

from attacks import build_attack
from data_utils import get_loader, get_normalizer


# ----------------------------------------------------------------------
# 모델 로더: surrogate 1개 + target 여러 개
# 논문 Table 1 의 CNN/ViT 중 torchvision 에서 바로 받을 수 있는 것들로 구성.
# ----------------------------------------------------------------------
def load_model(name, device):
    weights = {
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
        "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1,
        "resnext50": models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
        "vit_b": models.ViT_B_16_Weights.IMAGENET1K_V1,
        "convnext_t": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "swin_t": models.Swin_T_Weights.IMAGENET1K_V1,
    }
    builders = {
        "resnet50": models.resnet50,
        "densenet121": models.densenet121,
        "resnext50": models.resnext50_32x4d,
        "vit_b": models.vit_b_16,
        "convnext_t": models.convnext_tiny,
        "swin_t": models.swin_t,
    }
    if name not in builders:
        raise ValueError(f"알 수 없는 모델: {name} (가능: {list(builders)})")
    model = builders[name](weights=weights[name]).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_args():
    p = argparse.ArgumentParser(description="적대적 전이성 평가 (레벨 2)")
    p.add_argument("--attack", type=str, default="ifgsm",
                   choices=["fgsm", "ifgsm", "mifgsm"])
    p.add_argument("--surrogate", type=str, default="resnet50",
                   help="공격을 생성할 모델")
    p.add_argument("--targets", type=str, nargs="+",
                   default=["resnet50", "densenet121", "resnext50",
                            "vit_b", "convnext_t", "swin_t"],
                   help="공격이 통하는지 평가할 모델들")
    # 논문 4.1.4 기본값
    p.add_argument("--epsilon", type=float, default=16 / 255)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=1.6 / 255)
    p.add_argument("--momentum", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


@torch.no_grad()
def predict(model, normalize, x):
    return model(normalize(x)).argmax(dim=1)


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    normalize = get_normalizer(device)

    print("=" * 70)
    print(f"전이성 평가  |  공격: {args.attack.upper()}  "
          f"|  surrogate: {args.surrogate}")
    print(f"ε={args.epsilon:.4f} (~{args.epsilon*255:.0f}/255)  "
          f"T={args.steps}  α={args.alpha:.4f}  μ={args.momentum}")
    print(f"target 모델: {', '.join(args.targets)}")
    print(f"장치: {device}")
    print("=" * 70)

    # 1) surrogate 모델 로드 + 공격 함수
    print(f"\n[1/3] surrogate 모델({args.surrogate}) 로드 중...")
    surrogate = load_model(args.surrogate, device)
    attack = build_attack(args.attack, args.epsilon, args.steps,
                          args.alpha, args.momentum)

    # 2) target 모델들 로드 (처음이면 다운로드됨)
    print(f"[2/3] target 모델 {len(args.targets)}개 로드 중 "
          f"(처음이면 다운로드)...")
    targets = {}
    for name in args.targets:
        print(f"    - {name}")
        targets[name] = load_model(name, device)

    # 3) 공격 생성 + 각 target 에서 ASR 측정
    print(f"[3/3] 공격 생성 및 전이 평가 중...\n")
    loader = get_loader(batch_size=args.batch_size)

    # target 별 집계: (원래 맞힌 수, 공격 후 틀린 수)
    stats = {name: {"clean_correct": 0, "fooled": 0} for name in args.targets}
    start = time.time()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # surrogate 에서 공격 '한 번' 생성 (이게 전이성의 핵심)
        adv = attack(surrogate, images, labels, normalize)

        # 각 target 모델에서 평가
        for name, model in targets.items():
            clean_pred = predict(model, normalize, images)
            correct = clean_pred.eq(labels)  # 이 모델이 원래 맞힌 것
            adv_pred = predict(model, normalize, adv)
            fooled = correct & adv_pred.ne(labels)
            stats[name]["clean_correct"] += correct.sum().item()
            stats[name]["fooled"] += fooled.sum().item()

    elapsed = time.time() - start

    # 결과 표 출력
    # (한글은 폭이 넓어 정렬이 어긋나므로 헤더는 영어로, 숫자는 우측정렬 고정폭)
    header = (f"{'target model':<14}{'clean_ok':>10}{'adv_ok':>10}"
              f"{'fooled':>10}{'ASR(%)':>10}   note")
    print("-" * len(header.encode("ascii", "ignore")))
    print(header)
    print("-" * 68)
    for name in args.targets:
        cc = stats[name]["clean_correct"]      # 공격 전 맞힌 수 (clean_ok)
        fo = stats[name]["fooled"]             # 공격으로 틀리게 된 수 (fooled)
        still = cc - fo                        # 공격 후에도 맞힌 수 (adv_ok)
        asr = 100.0 * fo / max(cc, 1)          # 공격 성공률 = 틀린 비율
        note = "whitebox" if name == args.surrogate else "transfer"
        print(f"{name:<14}{cc:>10}{still:>10}{fo:>10}{asr:>9.2f}%   {note}")
    print("-" * 68)
    print("  clean_ok=공격전 맞힘  adv_ok=공격후 맞힘  "
          "fooled=공격후 틀림  ASR=공격성공률")
    print(f"소요 시간: {elapsed:.1f}초")
    print("\n※ ASR(공격 성공률) = 공격후 틀린 장수 / 공격전 맞힌 장수")
    print("  ASR이 높을수록 공격이 강함(= 더 많이 틀리게 만듦).")
    print("  화이트박스(surrogate 자기자신)는 높고, 전이(다른 모델)는 낮게 나옵니다.")
    print("  이 '떨어지는 폭'이 전이성 문제이며, LPAA가 줄이려는 대상입니다.")


if __name__ == "__main__":
    main()
