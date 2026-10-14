"""
적대적 공격 실행 + 공격 성공률(ASR) 측정  [레벨 1]

논문과 동일한 설정:
  - surrogate(공격 생성) 모델: ResNet-50 (사전학습)
  - 공격 파라미터: ε=16/255, T=10, α=1.6/255, μ=1.0 (논문 4.1.4)

측정하는 것:
  - Clean accuracy : 공격 전, 모델이 원래 얼마나 맞히나
  - Attack Success Rate (ASR) : 공격 후, "원래 맞히던 것"을 틀리게 만든 비율
    (전이성 연구의 표준 지표)

사용 예:
    python adversarial/run_attack.py --attack ifgsm
    python adversarial/run_attack.py --attack fgsm
    python adversarial/run_attack.py --attack mifgsm
"""

import argparse
import time

import torch
from torchvision import models

from attacks import build_attack
from data_utils import get_loader, get_normalizer


def load_resnet50(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_args():
    p = argparse.ArgumentParser(description="적대적 공격 실행 (레벨 1)")
    p.add_argument("--attack", type=str, default="ifgsm",
                   choices=["fgsm", "ifgsm", "mifgsm"])
    # 논문 4.1.4 기본값
    p.add_argument("--epsilon", type=float, default=16 / 255)
    p.add_argument("--steps", type=int, default=10)       # T
    p.add_argument("--alpha", type=float, default=1.6 / 255)  # α
    p.add_argument("--momentum", type=float, default=1.0)  # μ (mifgsm용)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


@torch.no_grad()
def predict(model, normalize, x):
    return model(normalize(x)).argmax(dim=1)


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 64)
    print(f"공격: {args.attack.upper()}  |  surrogate: ResNet-50")
    print(f"ε={args.epsilon:.4f} (~{args.epsilon*255:.0f}/255)  "
          f"T={args.steps}  α={args.alpha:.4f}  μ={args.momentum}")
    print(f"장치: {device} "
          f"({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print("=" * 64)

    model = load_resnet50(device)
    normalize = get_normalizer(device)
    attack = build_attack(args.attack, args.epsilon, args.steps,
                          args.alpha, args.momentum)
    loader = get_loader(batch_size=args.batch_size)

    total = 0            # 전체 이미지 수
    clean_correct = 0    # 공격 전 맞힌 수
    fooled = 0           # 원래 맞혔는데 공격 후 틀린 수
    start = time.time()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        clean_pred = predict(model, normalize, images)
        correct_mask = clean_pred.eq(labels)  # 원래 맞힌 것만 공격 대상
        clean_correct += correct_mask.sum().item()
        total += labels.size(0)

        # 공격 생성 (원래 맞힌 이미지에 대해서만 성공률 계산)
        adv = attack(model, images, labels, normalize)
        adv_pred = predict(model, normalize, adv)

        # 원래 맞혔는데( correct_mask ) 공격 후 틀린( != labels ) 경우
        newly_fooled = correct_mask & adv_pred.ne(labels)
        fooled += newly_fooled.sum().item()

    elapsed = time.time() - start
    clean_acc = 100.0 * clean_correct / total
    still_correct = clean_correct - fooled      # 공격 후에도 맞힌 수
    asr = 100.0 * fooled / max(clean_correct, 1)

    print("-" * 64)
    print(f"전체 이미지         : {total}")
    print(f"공격 전 맞힌 장수   : {clean_correct}장  (정확도 {clean_acc:.2f}%)")
    print(f"공격 후 맞힌 장수   : {still_correct}장")
    print(f"공격 후 틀린 장수   : {fooled}장")
    print(f"공격 성공률 (ASR)   : {asr:.2f}%  "
          f"(= 틀린 {fooled} / 맞혔던 {clean_correct})")
    print(f"소요 시간           : {elapsed:.1f}초")
    print("-" * 64)
    print("※ 이것은 '화이트박스' 공격입니다 (공격 만든 모델 = 평가 모델).")
    print("  레벨 2에서는 '다른 모델'에 통하는지(전이성)를 측정합니다.")


if __name__ == "__main__":
    main()
