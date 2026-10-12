"""
CIFAR-10 이미지 분류 학습 스크립트
- 파라미터를 명령줄에서 바꿔가며 실험할 수 있게 만들었습니다.

사용 예:
    python train.py --lr 0.01
    python train.py --lr 0.1 --epochs 10 --batch-size 128

결과(정확도 등)는 results.csv 파일에 한 줄씩 자동으로 쌓입니다.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# ----------------------------------------------------------------------
# 1. 명령줄 인자(파라미터) 정의  <- 여기서 바꿀 수 있는 값들이 정해집니다
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="CIFAR-10 분류 실험")
    p.add_argument("--lr", type=float, default=0.01, help="학습률 (learning rate)")
    p.add_argument("--epochs", type=int, default=5, help="학습 반복 횟수")
    p.add_argument("--batch-size", type=int, default=128, help="배치 크기")
    p.add_argument("--optimizer", type=str, default="sgd",
                   choices=["sgd", "adam"], help="옵티마이저 종류")
    p.add_argument("--results-file", type=str, default="results.csv",
                   help="결과를 저장할 파일")
    return p.parse_args()


# ----------------------------------------------------------------------
# 2. 간단한 CNN 모델 (컴퓨터 비전 입문용)
# ----------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ----------------------------------------------------------------------
# 3. 데이터 준비 (CIFAR-10 자동 다운로드)
# ----------------------------------------------------------------------
def get_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ----------------------------------------------------------------------
# 4. 정확도 평가
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 5. 결과를 CSV에 한 줄 추가
# ----------------------------------------------------------------------
def save_result(path, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["시각", "optimizer", "lr", "batch_size",
                             "epochs", "test_accuracy(%)", "학습시간(초)"])
        writer.writerow(row)


# ----------------------------------------------------------------------
# 6. 메인 학습 루프
# ----------------------------------------------------------------------
def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"실험 시작  |  optimizer={args.optimizer}  lr={args.lr}  "
          f"batch_size={args.batch_size}  epochs={args.epochs}")
    print(f"장치: {device}  ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print("=" * 60)

    train_loader, test_loader = get_loaders(args.batch_size)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"  [epoch {epoch}/{args.epochs}]  평균 loss: {avg_loss:.4f}")

    elapsed = time.time() - start
    acc = evaluate(model, test_loader, device)

    print("-" * 60)
    print(f"✅ 최종 테스트 정확도: {acc:.2f}%   (학습시간 {elapsed:.1f}초)")
    print("-" * 60)

    save_result(args.results_file, [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        args.optimizer, args.lr, args.batch_size, args.epochs,
        f"{acc:.2f}", f"{elapsed:.1f}",
    ])
    print(f"결과가 '{args.results_file}'에 저장되었습니다.\n")


if __name__ == "__main__":
    main()
