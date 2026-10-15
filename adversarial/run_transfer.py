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
#
# TransferAttack(분야 표준) 방식을 따름:
#   - CNN 계열: torchvision 가중치("DEFAULT")
#   - ViT 계열: timm 가중치 (예: vit_base_patch16_224)
# "vit_b_timm" 처럼 _timm 접미사가 붙으면 timm 에서 로드.
# ----------------------------------------------------------------------
_TORCHVISION = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
    "resnext50": (models.resnext50_32x4d,
                  models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1),
    "vit_b": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
    "convnext_t": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "swin_t": (models.swin_t, models.Swin_T_Weights.IMAGENET1K_V1),
}

# timm 모델 이름 매핑 (분야 표준 가중치)
_TIMM = {
    "resnet50_timm": "resnet50",
    "vit_b_timm": "vit_base_patch16_224",
    "pit_b_timm": "pit_b_224",
    "swin_t_timm": "swin_tiny_patch4_window7_224",
}


def load_model(name, device):
    if name in _TIMM:
        import timm
        model = timm.create_model(_TIMM[name], pretrained=True)
    elif name in _TORCHVISION:
        builder, weights = _TORCHVISION[name]
        model = builder(weights=weights)
    else:
        raise ValueError(
            f"알 수 없는 모델: {name} "
            f"(torchvision: {list(_TORCHVISION)}, timm: {list(_TIMM)})")
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_args():
    p = argparse.ArgumentParser(description="적대적 전이성 평가 (레벨 2)")
    p.add_argument("--attack", type=str, default="ifgsm",
                   choices=["fgsm", "ifgsm", "mifgsm", "lpaa"])
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
    p.add_argument("--no-init", action="store_true",
                   help="LPAA의 초기화(PI)를 끄기 (검증 3: 초기화 기여 분리)")
    p.add_argument("--beta-start", type=float, default=None,
                   help="[개선] LPAA β 스케줄 시작값 (초반, 크게 탐색)")
    p.add_argument("--beta-end", type=float, default=None,
                   help="[개선] LPAA β 스케줄 끝값 (후반, 작게 수렴)")
    p.add_argument("--seed", type=int, default=0,
                   help="랜덤 시드 (공정 비교를 위해 고정)")
    return p.parse_args()


@torch.no_grad()
def predict(model, normalize, x):
    return model(normalize(x)).argmax(dim=1)


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    normalize = get_normalizer(device)

    # 공정 비교를 위한 시드 고정 (LPAA의 랜덤 마스크 등 재현성 확보)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print(f"전이성 평가  |  공격: {args.attack.upper()}  "
          f"|  surrogate: {args.surrogate}")
    print(f"ε={args.epsilon:.4f} (~{args.epsilon*255:.0f}/255)  "
          f"T={args.steps}  α={args.alpha:.4f}  μ={args.momentum}  "
          f"seed={args.seed}")
    if args.beta_start is not None and args.beta_end is not None:
        print(f"β 스케줄: {args.beta_start} → {args.beta_end} (선형 감소, "
              f"평균 {(args.beta_start+args.beta_end)/2:.1f})")
    print(f"target 모델: {', '.join(args.targets)}")
    print(f"장치: {device}")
    print("=" * 70)

    # 1) surrogate 모델 로드 + 공격 함수
    print(f"\n[1/3] surrogate 모델({args.surrogate}) 로드 중...")
    surrogate = load_model(args.surrogate, device)
    attack = build_attack(args.attack, args.epsilon, args.steps,
                          args.alpha, args.momentum, use_init=not args.no_init,
                          beta_start=args.beta_start, beta_end=args.beta_end)
    if args.attack == "lpaa":
        print(f"    LPAA 초기화(PI): {'끔(--no-init)' if args.no_init else '켬'}")

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

    # target 별 집계
    #   total        : 전체 이미지 수
    #   clean_correct: 공격 전 맞힌 수
    #   fooled       : 원래 맞혔는데 공격 후 틀린 수 (조건부 ASR 분자)
    #   adv_wrong    : 공격 후 틀린 전체 수 (논문식 ASR 분자)
    stats = {name: {"total": 0, "clean_correct": 0, "fooled": 0, "adv_wrong": 0}
             for name in args.targets}
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
            adv_wrong = adv_pred.ne(labels)         # 공격 후 틀린 전체
            fooled = correct & adv_wrong            # 원래 맞혔는데 틀린 것
            stats[name]["total"] += labels.size(0)
            stats[name]["clean_correct"] += correct.sum().item()
            stats[name]["fooled"] += fooled.sum().item()
            stats[name]["adv_wrong"] += adv_wrong.sum().item()

    elapsed = time.time() - start

    # 결과 표 출력 (한글 헤더 사용 + 폭 정확히 맞춤)
    # 한글/전각 문자는 터미널에서 2칸을 차지하므로, 그걸 고려해 패딩한다.
    def disp_width(s):
        import unicodedata
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)

    def pad(s, width, align="left"):
        gap = width - disp_width(s)
        if gap <= 0:
            return s
        return s + " " * gap if align == "left" else " " * gap + s

    # 컬럼 폭 정의 (표시 칸 기준)
    # ASR(논문식) = 공격후 틀린 전체 / 전체  ← 논문 Table 1 과 동일한 분모
    # ASR(조건부) = 원래 맞힌 것 중 틀린 비율 (참고용)
    cols = [("target 모델", 16, "left"), ("전체", 8, "right"),
            ("공격전 맞힘", 12, "right"), ("공격후 틀림(전체)", 18, "right"),
            ("ASR(논문식)", 13, "right"), ("ASR(조건부)", 13, "right"),
            ("비고", 16, "left")]
    total_w = sum(w for _, w, _ in cols) + len(cols) - 1

    header = " ".join(pad(name, w, a) for name, w, a in cols)
    print("-" * total_w)
    print(header)
    print("-" * total_w)
    for name in args.targets:
        tot = stats[name]["total"]             # 전체
        cc = stats[name]["clean_correct"]      # 공격 전 맞힌 수
        fo = stats[name]["fooled"]             # 원래 맞혔는데 틀린 수
        aw = stats[name]["adv_wrong"]          # 공격 후 틀린 전체 수
        asr_paper = 100.0 * aw / max(tot, 1)   # 논문식: 전체 분모
        asr_cond = 100.0 * fo / max(cc, 1)     # 조건부: 원래 맞힌 것 분모
        note = "<- 화이트박스" if name == args.surrogate else "전이(블랙박스)"
        row_vals = [
            (name, 16, "left"),
            (f"{tot}", 8, "right"),
            (f"{cc}장", 12, "right"),
            (f"{aw}장", 18, "right"),
            (f"{asr_paper:.2f}%", 13, "right"),
            (f"{asr_cond:.2f}%", 13, "right"),
            (note, 16, "left"),
        ]
        print(" ".join(pad(v, w, a) for v, w, a in row_vals))
    print("-" * total_w)
    print(f"소요 시간: {elapsed:.1f}초")
    print("\n※ ASR(논문식) = 공격후 틀린 전체 / 전체  ← 논문 Table 1 과 동일 분모 (이걸로 논문과 비교)")
    print("  ASR(조건부) = 원래 맞힌 것 중 틀린 비율 (참고용, 화이트박스서 ~100% 나옴)")
    print("  화이트박스는 논문처럼 ~94-98%, 전이(다른 모델)는 낮게 나옵니다.")


if __name__ == "__main__":
    main()
