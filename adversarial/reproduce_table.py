"""
논문 Table 1 재현 (RN-50 surrogate)  [레벨 4]

여러 공격 method 를 여러 target 모델에 대해 평가하여,
논문 Table 1 과 같은 '공격 x 모델' ASR 표를 만든다.

- surrogate: ResNet-50 (논문 RN-50 행)
- method: 우리가 구현한 것들 (ifgsm, mifgsm, lpaa). 필요시 추가 가능.
- target: torchvision 에서 바로 받을 수 있는 모델들
- 파라미터: 논문 4.1.4 기본값 (ε=16/255, T=10, α=1.6/255, μ=1.0, N=20, p=0.95, β=35)

결과:
  - 화면에 ASR 표 출력 (행=공격, 열=target 모델)
  - adversarial/table1_results.csv 저장

사용 예:
    python adversarial/reproduce_table.py
    python adversarial/reproduce_table.py --methods ifgsm mifgsm lpaa
    python adversarial/reproduce_table.py --targets resnet50 densenet121 vit_b
"""

import argparse
import csv
import os
import time
import unicodedata

import torch
from torchvision import models

from attacks import build_attack
from data_utils import get_loader, get_normalizer

# 논문 Table 1 의 모델 이름 -> torchvision 매핑 (받을 수 있는 것만)
MODEL_ZOO = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
    "resnext50": (models.resnext50_32x4d,
                  models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1),
    "convnext_t": (models.convnext_tiny,
                   models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "vit_b": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
    "swin_t": (models.swin_t, models.Swin_T_Weights.IMAGENET1K_V1),
}

# 논문 Table 1 의 표기 라벨 (출력 가독성용)
DISPLAY = {
    "resnet50": "RN-50", "densenet121": "DN-121", "resnext50": "RNX-50",
    "convnext_t": "CNX-T", "vit_b": "ViT-B", "swin_t": "Swin-T",
}


def disp_width(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s, width, align="right"):
    gap = width - disp_width(s)
    if gap <= 0:
        return s
    return (s + " " * gap) if align == "left" else (" " * gap + s)


def load_model(name, device):
    builder, weights = MODEL_ZOO[name]
    m = builder(weights=weights).eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def get_args():
    p = argparse.ArgumentParser(description="논문 Table 1 재현")
    p.add_argument("--surrogate", type=str, default="resnet50")
    p.add_argument("--methods", type=str, nargs="+",
                   default=["ifgsm", "mifgsm", "lpaa"],
                   help="비교할 공격들")
    p.add_argument("--targets", type=str, nargs="+",
                   default=["resnet50", "densenet121", "resnext50",
                            "convnext_t", "vit_b", "swin_t"])
    # 논문 4.1.4
    p.add_argument("--epsilon", type=float, default=16 / 255)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=1.6 / 255)
    p.add_argument("--momentum", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


@torch.no_grad()
def predict(model, normalize, x):
    return model(normalize(x)).argmax(dim=1)


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    normalize = get_normalizer(device)

    print("=" * 78)
    print(f"논문 Table 1 재현  |  surrogate: {DISPLAY.get(args.surrogate, args.surrogate)}")
    print(f"공격: {', '.join(args.methods)}")
    print(f"target: {', '.join(DISPLAY.get(t, t) for t in args.targets)}")
    print(f"ε={args.epsilon*255:.0f}/255  T={args.steps}  "
          f"α={args.alpha*255:.1f}/255  μ={args.momentum}")
    print(f"장치: {device}")
    print("=" * 78)

    # 모델 로드 (surrogate + 모든 target)
    print("\n모델 로드 중 (처음이면 다운로드)...")
    needed = set(args.targets) | {args.surrogate}
    loaded = {}
    for name in needed:
        print(f"  - {DISPLAY.get(name, name)}")
        loaded[name] = load_model(name, device)
    surrogate = loaded[args.surrogate]

    # 각 target 의 '공격 전 맞힌' 마스크를 미리 계산해 재사용
    loader = get_loader(batch_size=args.batch_size)

    # 결과: results[method][target] = {total, cc(공격전맞힘), aw(공격후틀린전체)}
    results = {m: {t: {"tot": 0, "cc": 0, "aw": 0} for t in args.targets}
               for m in args.methods}

    overall_start = time.time()
    for mi, method in enumerate(args.methods, 1):
        print(f"\n[{mi}/{len(args.methods)}] 공격 '{method}' 실행 중...")
        attack = build_attack(method, args.epsilon, args.steps,
                              args.alpha, args.momentum)
        m_start = time.time()
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # surrogate 에서 공격 생성 (한 번)
            adv = attack(surrogate, images, labels, normalize)

            # 각 target 평가 (논문식 ASR: 전체 분모)
            for t in args.targets:
                model = loaded[t]
                clean_pred = predict(model, normalize, images)
                adv_pred = predict(model, normalize, adv)
                results[method][t]["tot"] += labels.size(0)
                results[method][t]["cc"] += clean_pred.eq(labels).sum().item()
                results[method][t]["aw"] += adv_pred.ne(labels).sum().item()
        print(f"    완료 ({time.time()-m_start:.0f}초)")

    # ---- 표 출력 (논문 Table 1 형식: 행=method, 열=target) ----
    print("\n" + "=" * 78)
    col_w = 9
    name_w = 10
    header = pad("Method", name_w, "left")
    for t in args.targets:
        header += pad(DISPLAY.get(t, t), col_w)
    header += pad("Avg.", col_w)
    print(header)
    print("-" * disp_width(header))

    csv_rows = []
    for method in args.methods:
        line = pad(method, name_w, "left")
        asrs = []
        for t in args.targets:
            tot = results[method][t]["tot"]
            aw = results[method][t]["aw"]
            asr = 100.0 * aw / max(tot, 1)   # 논문식: 공격후 틀린 전체 / 전체
            # surrogate(화이트박스)는 * 표시 (논문 관례)
            star = "*" if t == args.surrogate else ""
            line += pad(f"{asr:.1f}{star}", col_w)
            if t != args.surrogate:
                asrs.append(asr)  # 전이성 평균은 화이트박스 제외
        avg = sum(asrs) / len(asrs) if asrs else 0.0
        line += pad(f"{avg:.1f}", col_w)
        print(line)
        csv_rows.append((method, results[method], avg))

    print("-" * disp_width(header))
    print("* = 화이트박스(surrogate 자기자신).  Avg.는 화이트박스 제외 전이 평균.")
    print(f"\n총 소요: {time.time()-overall_start:.0f}초")

    # ---- CSV 저장 ----
    out = os.path.join("adversarial", "table1_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [DISPLAY.get(t, t) for t in args.targets]
                   + ["transfer_avg"])
        for method, res, avg in csv_rows:
            row = [method]
            for t in args.targets:
                tot = res[t]["tot"]
                aw = res[t]["aw"]
                row.append(f"{100.0*aw/max(tot,1):.1f}")
            row.append(f"{avg:.1f}")
            w.writerow(row)
    print(f"결과 CSV 저장: {out}")


if __name__ == "__main__":
    main()
