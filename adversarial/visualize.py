"""
적대적 공격 시각화  [레벨 1의 하이라이트]

"원본 이미지 / 추가된 노이즈 / 공격 이미지"를 나란히 그려서,
사람 눈엔 거의 똑같은데 모델은 다르게 분류하는 것을 눈으로 확인합니다.

결과는 adv_plots/ 폴더에 PNG로 저장됩니다.

사용 예:
    python adversarial/visualize.py --attack ifgsm --num 5
"""

import argparse
import os

import torch
from torchvision import models

from attacks import build_attack
from data_utils import get_loader, get_normalizer

OUT_DIR = "adv_plots"


def load_resnet50(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_imagenet_classes():
    """라벨 번호 -> 클래스 이름 (torchvision 메타데이터에서)"""
    return models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", type=str, default="ifgsm",
                   choices=["fgsm", "ifgsm", "mifgsm"])
    p.add_argument("--num", type=int, default=5, help="시각화할 이미지 수")
    p.add_argument("--epsilon", type=float, default=16 / 255)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=1.6 / 255)
    p.add_argument("--momentum", type=float, default=1.0)
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 가 필요합니다:  pip install matplotlib")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_resnet50(device)
    normalize = get_normalizer(device)
    classes = get_imagenet_classes()
    attack = build_attack(args.attack, args.epsilon, args.steps,
                          args.alpha, args.momentum)

    os.makedirs(OUT_DIR, exist_ok=True)
    loader = get_loader(batch_size=1)

    done = 0
    for images, labels in loader:
        if done >= args.num:
            break
        images = images.to(device)
        labels = labels.to(device)

        # 공격 전 예측
        with torch.no_grad():
            clean_logits = model(normalize(images))
        clean_pred = clean_logits.argmax(1).item()

        # 원래 맞히는 이미지만 시각화 대상으로
        if clean_pred != labels.item():
            continue

        adv = attack(model, images, labels, normalize)
        with torch.no_grad():
            adv_logits = model(normalize(adv))
        adv_pred = adv_logits.argmax(1).item()

        # 텐서 -> 넘파이 이미지
        orig = images[0].detach().cpu().permute(1, 2, 0).numpy()
        advimg = adv[0].detach().cpu().permute(1, 2, 0).numpy()
        # 노이즈를 보기 좋게 증폭 (실제 섭동은 아주 작음)
        noise = (adv[0] - images[0]).detach().cpu().permute(1, 2, 0).numpy()
        noise_vis = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        axes[0].imshow(orig)
        axes[0].set_title(f"Original\npred: {classes[clean_pred][:20]}",
                          fontsize=10)
        axes[1].imshow(noise_vis)
        axes[1].set_title(f"Perturbation (x amplified)\nmax |δ|={args.epsilon:.3f}",
                          fontsize=10)
        axes[2].imshow(advimg)
        color = "red" if adv_pred != labels.item() else "green"
        axes[2].set_title(f"Adversarial\npred: {classes[adv_pred][:20]}",
                          fontsize=10, color=color)
        for ax in axes:
            ax.axis("off")

        success = "FOOLED" if adv_pred != labels.item() else "robust"
        out = os.path.join(OUT_DIR, f"{args.attack}_{done:02d}_{success}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        plt.close()
        print(f"[{done+1}/{args.num}] 저장: {out}  "
              f"({classes[clean_pred][:15]} -> {classes[adv_pred][:15]})")
        done += 1

    print(f"\n완료! '{OUT_DIR}' 폴더의 PNG를 열어보세요.")
    print("사람 눈엔 Original 과 Adversarial 이 거의 같은데,")
    print("모델 예측(pred)은 달라진 걸 확인할 수 있습니다.")


if __name__ == "__main__":
    main()
