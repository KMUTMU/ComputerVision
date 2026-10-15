"""
데이터셋 분산 검증 실험: 독립 샘플 3개로 ASR 안정성 측정

목적: "ASR과 화이트박스 100% 여부가 데이터 샘플에 따라 변하는가?"를 검증.
      공식 ImageNet val(5만 장)에서 '서로 겹치지 않는' 1000장을 3번 뽑아(독립 샘플 A/B/C),
      각 샘플에서 I-FGSM / LPAA 의 전이 ASR 을 측정해 비교.

선별 조건 (논문 방식, 4.1.1):
  - "여러 모델(ResNet-50, DenseNet-121, ViT-B)이 '모두' 올바르게 분류하는" 이미지만 채택
  - 서로 다른 카테고리에서 (클래스당 최대 N장)
  - 샘플 A/B/C 는 겹치지 않게 (이미 쓴 이미지는 제외)

전제: data/imagenet/ 에 ILSVRC2012_img_val.tar + devkit 이 있어야 함.

사용:
    python adversarial/run_dataset_robustness.py
    python adversarial/run_dataset_robustness.py --per-sample 1000 --per-class 1

주의: 매우 느림 (3 샘플 x 2 공격 x 6 모델 x LPAA). 수 시간. 걸어두고 기다릴 것.
결과: adversarial/results/dataset_robustness_<timestamp>.csv (각 단계 중간 저장)
"""

import argparse
import csv
import os
import time
from datetime import datetime

import torch
from torchvision import models, transforms
from torchvision.datasets import ImageNet

from attacks import build_attack

IMAGENET_ROOT = os.path.join("data", "imagenet")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 평가할 6개 target 모델 (CNN=torchvision, ViT=timm)
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
SURROGATE = "resnet50"
# 선별 필터용 모델 (논문: 여러 모델이 다 맞히는 것) - 대표 3종
FILTER_MODELS = ["resnet50", "densenet121", "vit_b_timm"]
RUNS = [("ifgsm", "ifgsm"), ("lpaa", "lpaa")]


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


def get_normalizer(device):
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return lambda x: (x - mean) / std


@torch.no_grad()
def predict(model, normalize, x):
    return model(normalize(x)).argmax(dim=1)


def get_args():
    p = argparse.ArgumentParser(description="데이터셋 분산 검증 (독립 샘플 3개)")
    p.add_argument("--per-sample", type=int, default=1000,
                   help="샘플당 이미지 수 (논문=1000)")
    p.add_argument("--per-class", type=int, default=1,
                   help="클래스당 최대 장수")
    p.add_argument("--num-samples", type=int, default=3,
                   help="독립 샘플 수 (A/B/C)")
    p.add_argument("--epsilon", type=float, default=16 / 255)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=1.6 / 255)
    p.add_argument("--momentum", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_samples(args, filter_models, normalize, device):
    """
    ImageNet val 을 한 번 순회하며, 필터 모델이 모두 맞히는 이미지를
    겹치지 않게 3개 샘플로 나눠 담는다.
    반환: samples = [[(tensor, label), ...], ...]  (num_samples 개)
    """
    print("ImageNet val 로드 중 (tar 압축해제는 처음 한 번)...")
    load_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = ImageNet(root=IMAGENET_ROOT, split="val", transform=load_tf)

    n = args.num_samples
    samples = [[] for _ in range(n)]
    # 클래스별로 "필터 통과한 이미지"를 모아뒀다가, 샘플 A/B/C 에 라운드로빈 분배.
    # ImageNet val 은 클래스 정렬이므로, 각 클래스에서 (per_class * n)장을 모아
    # 샘플마다 per_class 장씩 겹치지 않게 나눈다.
    need_per_class = args.per_class * n   # 한 클래스에서 모을 총 장수
    class_buffer = {}                     # label -> [img, img, ...]

    print(f"선별 중: 샘플 {n}개 x {args.per_sample}장 "
          f"(필터: {FILTER_MODELS} 모두 맞히는 것)")

    def distribute(label, imgs):
        """한 클래스에서 모은 imgs 를 샘플 A/B/C 에 per_class 장씩 분배."""
        for i in range(n):
            chunk = imgs[i * args.per_class:(i + 1) * args.per_class]
            samples[i].extend((im, label) for im in chunk)

    processed = 0
    for img, label in dataset:
        # 모든 샘플이 목표치를 채웠으면 종료
        if all(len(s) >= args.per_sample for s in samples):
            break
        # 이 클래스가 이미 충분히 모였으면 skip
        if len(class_buffer.get(label, [])) >= need_per_class:
            continue
        # 필터: 모든 필터 모델이 맞히는지
        x = img.unsqueeze(0).to(device)
        ok = True
        with torch.no_grad():
            xn = normalize(x)
            for _, m in filter_models:
                if m(xn).argmax(1).item() != label:
                    ok = False
                    break
        if not ok:
            continue
        class_buffer.setdefault(label, []).append(img)
        processed += 1
        # 이 클래스가 need_per_class 만큼 모이면 즉시 분배
        if len(class_buffer[label]) == need_per_class:
            distribute(label, class_buffer[label])
        if processed % 300 == 0:
            filled = sum(len(s) for s in samples)
            print(f"  누적 통과 {processed}장, 분배됨 {filled}장")

    # 끝까지 돌았는데 need_per_class 를 못 채운 클래스도 가능한 만큼 분배
    for label, imgs in class_buffer.items():
        if len(imgs) < need_per_class:
            distribute(label, imgs)

    for i, s in enumerate(samples):
        print(f"  샘플 {chr(65+i)}: {len(s)}장")
    return samples


def eval_sample(sample, surrogate, targets, attack, normalize, device, bs):
    """한 샘플에 대해 각 target 의 논문식 ASR 측정."""
    agg = {t: {"tot": 0, "cc": 0, "aw": 0} for t in TARGETS}
    i = 0
    while i < len(sample):
        batch = sample[i:i + bs]
        i += bs
        images = torch.stack([b[0] for b in batch]).to(device)
        labels = torch.tensor([b[1] for b in batch], device=device)
        adv = attack(surrogate, images, labels, normalize)
        for t in TARGETS:
            m = targets[t]
            clean_pred = predict(m, normalize, images)
            adv_pred = predict(m, normalize, adv)
            agg[t]["tot"] += labels.size(0)
            agg[t]["cc"] += clean_pred.eq(labels).sum().item()
            agg[t]["aw"] += adv_pred.ne(labels).sum().item()
    return agg


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    normalize = get_normalizer(device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.isdir(IMAGENET_ROOT):
        print(f"'{IMAGENET_ROOT}' 폴더가 없습니다. "
              "ILSVRC2012_img_val.tar + devkit 을 넣어주세요.")
        return

    print("=" * 72)
    print("데이터셋 분산 검증: 독립 샘플 3개 x (I-FGSM, LPAA) x 6모델")
    print(f"선별=논문방식(필터 {FILTER_MODELS} 모두 맞힘), 논문식 ASR")
    print(f"장치: {device}")
    print("=" * 72)

    # 모델 로드 (필터용 + 평가용, 공유)
    print("\n모델 로드 중...")
    loaded = {}
    for name in set(TARGETS) | set(FILTER_MODELS) | {SURROGATE}:
        print(f"  - {name}")
        loaded[name] = load_model(name, device)
    filter_models = [(n, loaded[n]) for n in FILTER_MODELS]
    surrogate = loaded[SURROGATE]
    targets = {t: loaded[t] for t in TARGETS}

    # 독립 샘플 생성
    samples = build_samples(args, filter_models, normalize, device)

    # 결과 CSV 준비
    out_dir = os.path.join("adversarial", "results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"dataset_robustness_{stamp}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# 데이터셋 분산 검증 (논문식 ASR)"])
        w.writerow(["# 선별=논문방식", "필터", "+".join(FILTER_MODELS)])
        w.writerow([])
        w.writerow(["sample", "attack", "target", "total", "clean_ok",
                    "adv_wrong", "ASR(%)", "whitebox"])

    overall = time.time()
    for si, sample in enumerate(samples):
        sample_name = chr(65 + si)  # A, B, C
        for attack_name, _ in RUNS:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            print(f"\n### 샘플 {sample_name} / {attack_name} 실행 중...")
            attack = build_attack(attack_name, args.epsilon, args.steps,
                                  args.alpha, args.momentum)
            t0 = time.time()
            agg = eval_sample(sample, surrogate, targets, attack,
                              normalize, device, args.batch_size)
            with open(out_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for t in TARGETS:
                    tot, cc, aw = agg[t]["tot"], agg[t]["cc"], agg[t]["aw"]
                    asr = 100.0 * aw / max(tot, 1)
                    wb = "yes" if t == SURROGATE else ""
                    w.writerow([sample_name, attack_name, t, tot, cc, aw,
                                f"{asr:.2f}", wb])
                    tag = "(화이트박스)" if t == SURROGATE else ""
                    print(f"   {t:<14}{asr:>7.2f}% {tag}")
            print(f"   ...완료 ({time.time()-t0:.0f}초, 중간 저장됨)")

    print(f"\n{'='*72}")
    print(f"전체 완료! 총 {(time.time()-overall)/60:.0f}분  →  {out_path}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
