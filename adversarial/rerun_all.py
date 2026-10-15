"""
전체 핵심 실험을 '논문식 ASR'(전체 분모)로 한 번에 재실행

지금까지의 핵심 실험들을 현재 표준 데이터(data/) + 논문식 ASR 기준으로 다시 측정.
결과는 adversarial/results/rerun_all_<timestamp>.csv 로 저장되고,
각 실험 완료 시마다 중간 저장하므로 중단되어도 그때까지 결과는 남는다.

재실행 항목:
  1) I-FGSM 전이성 (6개 target 모델)
  2) MI-FGSM 전이성 (6개 target 모델)
  3) LPAA 전이성 (초기화 켬, ResNet-50 → ViT-B)
  4) LPAA 전이성 (초기화 끔, --no-init)
모두 surrogate=ResNet-50, 논문 파라미터(ε=16/255,T=10,α=1.6/255,μ=1.0), seed 고정.

사용:
    python adversarial/rerun_all.py
    python adversarial/rerun_all.py --seed 0

주의: 느림(특히 LPAA). 전체 수십 분~2시간. 걸어두고 기다릴 것.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import torch
from torchvision import models

from attacks import build_attack
from data_utils import get_loader, get_normalizer

# CNN=torchvision, ViT=timm (분야 표준)
_TORCHVISION = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
    "resnext50": (models.resnext50_32x4d,
                  models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1),
    "convnext_t": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "swin_t": (models.swin_t, models.Swin_T_Weights.IMAGENET1K_V1),
}
_TIMM = {"vit_b_timm": "vit_base_patch16_224"}

TARGETS = ["resnet50", "densenet121", "resnext50", "convnext_t",
           "swin_t", "vit_b_timm"]

# 재실행할 공격 설정: (실험라벨, attack, use_init)
RUNS = [
    ("ifgsm",        "ifgsm",  True),
    ("mifgsm",       "mifgsm", True),
    ("lpaa_init_on", "lpaa",   True),
    ("lpaa_init_off","lpaa",   False),
]


def load_model(name, device):
    if name in _TIMM:
        import timm
        m = timm.create_model(_TIMM[name], pretrained=True)
    else:
        builder, weights = _TORCHVISION[name]
        m = builder(weights=weights)
    m = m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def predict(model, normalize, x):
    return model(normalize(x)).argmax(dim=1)


def get_args():
    p = argparse.ArgumentParser(description="전체 실험 논문식 ASR 재실행")
    p.add_argument("--surrogate", type=str, default="resnet50")
    p.add_argument("--epsilon", type=float, default=16 / 255)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=1.6 / 255)
    p.add_argument("--momentum", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    normalize = get_normalizer(device)

    out_dir = os.path.join("adversarial", "results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"rerun_all_{stamp}.csv")

    print("=" * 72)
    print("전체 핵심 실험 재실행 (논문식 ASR = 공격후 틀린 전체 / 전체)")
    print(f"surrogate={args.surrogate}  targets={TARGETS}")
    print(f"ε={args.epsilon*255:.0f}/255 T={args.steps} seed={args.seed}")
    print(f"장치: {device}")
    print("=" * 72)

    print("\n모델 로드 중 (처음이면 다운로드)...")
    loaded = {args.surrogate: load_model(args.surrogate, device)}
    for t in TARGETS:
        if t not in loaded:
            print(f"  - {t}")
            loaded[t] = load_model(t, device)
    surrogate = loaded[args.surrogate]

    # CSV 헤더 작성
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# 전체 재실행 (논문식 ASR, 전체 분모)"])
        w.writerow(["# surrogate", args.surrogate, "seed", args.seed,
                    "eps", f"{args.epsilon*255:.0f}/255", "T", args.steps])
        w.writerow([])
        w.writerow(["run", "target", "total", "clean_ok", "adv_wrong",
                    "ASR_paper(%)", "whitebox"])

    overall_start = time.time()
    for ri, (label, attack_name, use_init) in enumerate(RUNS, 1):
        # 시드 고정 (공정 재현)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        print(f"\n{'#'*72}")
        print(f"# [{ri}/{len(RUNS)}] {label} "
              f"(attack={attack_name}, init={'on' if use_init else 'off'})")
        print(f"{'#'*72}")

        attack = build_attack(attack_name, args.epsilon, args.steps,
                              args.alpha, args.momentum, use_init=use_init)
        loader = get_loader(batch_size=args.batch_size)

        agg = {t: {"tot": 0, "cc": 0, "aw": 0} for t in TARGETS}
        t0 = time.time()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            adv = attack(surrogate, images, labels, normalize)
            for t in TARGETS:
                m = loaded[t]
                clean_pred = predict(m, normalize, images)
                adv_pred = predict(m, normalize, adv)
                agg[t]["tot"] += labels.size(0)
                agg[t]["cc"] += clean_pred.eq(labels).sum().item()
                agg[t]["aw"] += adv_pred.ne(labels).sum().item()

        # 이 실험 결과를 즉시 CSV에 추가 (중단 대비)
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            print(f"\n  {'target':<14}{'ASR(논문식)':>12}")
            for t in TARGETS:
                tot, cc, aw = agg[t]["tot"], agg[t]["cc"], agg[t]["aw"]
                asr = 100.0 * aw / max(tot, 1)
                wb = "yes" if t == args.surrogate else ""
                w.writerow([label, t, tot, cc, aw, f"{asr:.2f}", wb])
                tag = " (화이트박스)" if t == args.surrogate else ""
                print(f"  {t:<14}{asr:>11.2f}%{tag}")
        print(f"  ...{label} 완료 ({time.time()-t0:.0f}초, 중간 저장됨)")

    print(f"\n{'='*72}")
    print(f"전체 완료! 총 {(time.time()-overall_start)/60:.0f}분")
    print(f"결과 저장: {out_path}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
