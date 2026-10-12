"""
CIFAR-10 전이학습(Transfer Learning) 스크립트
- 사전학습된 ResNet / EfficientNet 등을 가져와 CIFAR-10에 맞게 미세조정합니다.
- 하이퍼파라미터를 명령줄에서 바꿔가며 실험할 수 있습니다.

사용 예:
    python train_transfer.py --model resnet18 --lr 0.01 --optimizer sgd --epochs 15

결과:
    - results_transfer.csv : 실험별 최종 정확도/시간 등 (한 줄씩 누적)
    - logs/<실험이름>.csv   : epoch별 loss/정확도 (그래프용)
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models


# ----------------------------------------------------------------------
# 1. 명령줄 인자(바꿀 수 있는 파라미터들)
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="CIFAR-10 전이학습 실험")
    p.add_argument("--model", type=str, default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50",
                            "efficientnet_b0", "mobilenet_v3_small"],
                   help="사전학습 백본 모델")
    p.add_argument("--lr", type=float, default=0.01, help="학습률")
    p.add_argument("--optimizer", type=str, default="sgd",
                   choices=["sgd", "adam", "adamw"], help="옵티마이저")
    p.add_argument("--batch-size", type=int, default=128, help="배치 크기")
    p.add_argument("--epochs", type=int, default=15, help="학습 반복 횟수")
    p.add_argument("--weight-decay", type=float, default=5e-4, help="가중치 감쇠")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="백본을 얼리고 마지막 분류층만 학습 (빠름)")
    p.add_argument("--results-file", type=str, default="results_transfer.csv")
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--exp-name", type=str, default=None,
                   help="실험 이름 (로그 파일명). 미지정 시 자동 생성")
    return p.parse_args()


# ----------------------------------------------------------------------
# 2. 사전학습 모델 불러와서 CIFAR-10(10개 클래스)에 맞게 머리 교체
# ----------------------------------------------------------------------
def build_model(name, freeze_backbone):
    num_classes = 10

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"알 수 없는 모델: {name}")

    # 백본 얼리기 옵션 (마지막 분류층만 학습 -> 훨씬 빠름)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # 마지막 분류층만 다시 켜기
        if name.startswith("resnet"):
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.classifier.parameters():
                param.requires_grad = True

    return model


# ----------------------------------------------------------------------
# 3. 데이터 준비 (사전학습 모델은 224x224 입력 + ImageNet 정규화 필요)
# ----------------------------------------------------------------------
def get_loaders(batch_size):
    # ImageNet 사전학습 모델에 맞춘 전처리 + 학습용 데이터 증강
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=16),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def save_summary(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["시각", "exp_name", "model", "optimizer", "lr",
                        "batch_size", "epochs", "freeze", "best_accuracy(%)",
                        "final_accuracy(%)", "학습시간(초)"])
        w.writerow(row)


# ----------------------------------------------------------------------
# 4. 메인 학습 루프 (AMP 혼합정밀도로 5060에서 더 빠르게)
# ----------------------------------------------------------------------
def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = args.exp_name or (
        f"{args.model}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}"
        f"{'_frozen' if args.freeze_backbone else ''}")

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{exp_name}.csv")

    print("=" * 68)
    print(f"실험: {exp_name}")
    print(f"model={args.model}  opt={args.optimizer}  lr={args.lr}  "
          f"bs={args.batch_size}  epochs={args.epochs}  "
          f"freeze={args.freeze_backbone}")
    print(f"장치: {device} "
          f"({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print("=" * 68)

    train_loader, test_loader = get_loaders(args.batch_size)
    model = build_model(args.model, args.freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9,
                              weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(params, lr=args.lr,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(params, lr=args.lr,
                                weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP(혼합정밀도): 최신 GPU에서 메모리 절약 + 속도 향상
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # epoch별 로그 파일 헤더
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "test_accuracy(%)", "lr"])

    best_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), \
                             labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        avg_loss = running / len(train_loader)
        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        elapsed = time.time() - start
        print(f"  [epoch {epoch:2d}/{args.epochs}]  loss: {avg_loss:.4f}  "
              f"acc: {acc:5.2f}%  (best {best_acc:5.2f}%)  "
              f"[{elapsed:.0f}s]")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{avg_loss:.4f}",
                                    f"{acc:.2f}", f"{cur_lr:.6f}"])

    total_time = time.time() - start
    final_acc = acc
    print("-" * 68)
    print(f"✅ 완료  최고 정확도: {best_acc:.2f}%  "
          f"최종: {final_acc:.2f}%  (총 {total_time:.0f}초)")
    print("-" * 68)

    save_summary(args.results_file, [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        exp_name, args.model, args.optimizer, args.lr, args.batch_size,
        args.epochs, args.freeze_backbone,
        f"{best_acc:.2f}", f"{final_acc:.2f}", f"{total_time:.0f}",
    ])


if __name__ == "__main__":
    main()
