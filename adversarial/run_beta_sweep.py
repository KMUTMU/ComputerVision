"""
β(증강계수) 스케줄 비교 실험 (여러 설정을 한 번에)  [실험 9 후속]

LPAA 의 β 스케줄을 여러 가지로 바꿔가며, 하나의 target 모델에 대한
전이 ASR 을 비교한다. 모든 설정은:
  - 동일 데이터 (data/images)
  - 동일 시드 (공정 비교)
  - β 평균을 논문값(35)에 고정  (β 변화 '패턴'의 효과만 측정)

사용 예:
    python adversarial/run_beta_sweep.py --target vit_b_timm
    python adversarial/run_beta_sweep.py --target vit_b_timm --seed 0

주의: LPAA 는 느리다(설정당 ~14분). 설정 수 x 14분 ≈ 총 시간.
"""

import argparse
import time

import torch
from torchvision import models

from attacks import build_attack
from data_utils import get_loader, get_normalizer

# 비교할 β 스케줄 목록: (라벨, beta_start, beta_end)
#   beta_start==beta_end 이면 고정 β.
#   모두 평균 35 로 맞춰 공정 비교.
BETA_CONFIGS = [
    ("fixed-35",      35,   35),    # 기준 (논문값)
    ("rise 34to36",   34,   36),    # 아주 좁은 증가 (±1)
    ("rise 33to37",   33,   37),    # 좁은 증가 (±2)
    ("rise 32.5to37.5", 32.5, 37.5),# 좁은 증가 (±2.5)
    ("rise 30to40",   30,   40),    # 증가 (±5)
    ("fixed-30",      30,   30),    # 고정값 재탐색 (35보다 작게)
    ("fixed-40",      40,   40),    # 고정값 재탐색 (35보다 크게)
]

_TORCHVISION = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
    "resnext50": (models.resnext50_32x4d,
                  models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1),
    "convnext_t": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "vit_b": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
    "swin_t": (models.swin_t, models.Swin_T_Weights.IMAGENET1K_V1),
}
_TIMM = {
    "resnet50_timm": "resnet50",
    "vit_b_timm": "vit_base_patch16_224",
    "swin_t_timm": "swin_tiny_patch4_window7_224",
}


def load_model(name, device):
    if name in _TIMM:
        import timm
        m = timm.create_model(_TIMM[name], pretrained=True)
    elif name in _TORCHVISION:
        builder, weights = _TORCHVISION[name]
        m = builder(weights=weights)
    else:
        raise ValueError(f"알 수 없는 모델: {name}")
    m = m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def get_args():
    p = argparse.ArgumentParser(description="β 스케줄 비교 (실험 9 후속)")
    p.add_argument("--surrogate", type=str, default="resnet50")
    p.add_argument("--target", type=str, default="vit_b_timm",
                   help="전이 평가할 단일 target 모델")
    p.add_argument("--epsilon", type=float, default=16 / 255)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=1.6 / 255)
    p.add_argument("--momentum", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def predict(model, normalize, x):
    return model(normalize(x)).argmax(dim=1)


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    normalize = get_normalizer(device)

    print("=" * 66)
    print(f"β 스케줄 비교  |  surrogate: {args.surrogate}  "
          f"→ target: {args.target}")
    print(f"공통: ε={args.epsilon*255:.0f}/255  T={args.steps}  "
          f"seed={args.seed}  (β 평균=35 고정)")
    print(f"장치: {device}")
    print("=" * 66)

    print("\n모델 로드 중...")
    surrogate = load_model(args.surrogate, device)
    target = load_model(args.target, device)

    results = []  # (label, asr, cc, fo)
    for i, (label, b0, b1) in enumerate(BETA_CONFIGS, 1):
        # 매 설정마다 '같은' 시드로 리셋 → 완전히 공정한 비교
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        print(f"\n[{i}/{len(BETA_CONFIGS)}] β={label} 실행 중...")
        attack = build_attack("lpaa", args.epsilon, args.steps, args.alpha,
                              args.momentum, beta_start=b0, beta_end=b1)
        loader = get_loader(batch_size=args.batch_size)

        cc = fo = 0
        t0 = time.time()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            adv = attack(surrogate, images, labels, normalize)
            clean_pred = predict(target, normalize, images)
            correct = clean_pred.eq(labels)
            adv_pred = predict(target, normalize, adv)
            fooled = correct & adv_pred.ne(labels)
            cc += correct.sum().item()
            fo += fooled.sum().item()
        asr = 100.0 * fo / max(cc, 1)
        results.append((label, asr, cc, fo))
        print(f"    ASR={asr:.2f}%  ({time.time()-t0:.0f}초)")

    # 결과 표 (ASR 높은 순)
    print("\n" + "=" * 66)
    print(f" β 스케줄 비교 결과 ({args.target} 전이 ASR)")
    print("=" * 66)
    print(f"{'β schedule':<16}{'ASR(%)':>10}   {'vs 기준':>10}")
    print("-" * 66)
    base_asr = results[0][1]  # fixed-35 가 기준
    for label, asr, cc, fo in sorted(results, key=lambda r: r[1], reverse=True):
        diff = asr - base_asr
        mark = "" if label == "fixed-35" else f"{diff:+.2f}%p"
        best = "  <- 최고" if asr == max(r[1] for r in results) else ""
        print(f"{label:<16}{asr:>9.2f}%   {mark:>10}{best}")
    print("-" * 66)
    print("* 기준 = fixed-35 (논문값). 모든 설정 β 평균=35, 동일 시드.")

    # ---- 결과를 파일로 저장 (캡처 없이 git push로 공유 가능) ----
    save_results(args, results, base_asr)


def save_results(args, results, base_asr):
    """실험 결과를 CSV로 저장 (results/ 폴더). git push 하면 공유 가능."""
    import csv
    import os
    from datetime import datetime

    out_dir = os.path.join("adversarial", "results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"beta_sweep_{args.target}_{stamp}.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# 실험: β 스케줄 비교"])
        w.writerow(["# surrogate", args.surrogate, "target", args.target])
        w.writerow(["# seed", args.seed, "eps", f"{args.epsilon*255:.0f}/255",
                    "T", args.steps])
        w.writerow([])
        w.writerow(["beta_schedule", "ASR(%)", "clean_ok", "fooled",
                    "vs_fixed35(%p)"])
        for label, asr, cc, fo in results:
            w.writerow([label, f"{asr:.2f}", cc, fo, f"{asr-base_asr:+.2f}"])

    print(f"\n📁 결과 저장됨: {path}")
    print("   (git add/commit/push 하면 다른 곳에서도 확인 가능)")


if __name__ == "__main__":
    main()
