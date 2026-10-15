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
    fooled = 0           # 원래 맞혔는데 공격 후 틀린 수 (조건부 분자)
    adv_wrong = 0        # 공격 후 틀린 전체 수 (논문식 분자)
    start = time.time()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        clean_pred = predict(model, normalize, images)
        correct_mask = clean_pred.eq(labels)  # 원래 맞힌 것
        clean_correct += correct_mask.sum().item()
        total += labels.size(0)

        adv = attack(model, images, labels, normalize)
        adv_pred = predict(model, normalize, adv)

        adv_wrong += adv_pred.ne(labels).sum().item()               # 전체 중 틀린 수
        fooled += (correct_mask & adv_pred.ne(labels)).sum().item() # 원래 맞혔는데 틀린 수

    elapsed = time.time() - start
    clean_acc = 100.0 * clean_correct / total
    asr_paper = 100.0 * adv_wrong / max(total, 1)        # 논문식: 전체 분모
    asr_cond = 100.0 * fooled / max(clean_correct, 1)    # 조건부: 원래 맞힌 것 분모

    print("-" * 64)
    print(f"전체 이미지         : {total}")
    print(f"공격 전 맞힌 장수   : {clean_correct}장  (정확도 {clean_acc:.2f}%)")
    print(f"공격 후 틀린 장수   : {adv_wrong}장 (전체 기준)")
    print(f"ASR (논문식)        : {asr_paper:.2f}%  "
          f"(= 공격후 틀린 {adv_wrong} / 전체 {total})")
    print(f"ASR (조건부, 참고)  : {asr_cond:.2f}%  "
          f"(= 원래 맞힌 것 중 틀린 비율)")
    print(f"소요 시간           : {elapsed:.1f}초")
    print("-" * 64)
    print("※ 논문식 ASR = 전체 분모 → 논문 Table 과 동일 기준 (화이트박스서 ~94-98%).")
    print("  화이트박스 = 공격 만든 모델 = 평가 모델. 레벨 2에서 전이성 측정.")


if __name__ == "__main__":
    main()
