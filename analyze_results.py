"""
밤샘 실험 결과 분석
- results_transfer.csv 를 정확도 순으로 순위표 출력
- logs/*.csv 를 읽어 각 실험의 정확도/loss 곡선 그래프를 PNG로 저장

사용법: (cv) 환경에서  ->  python analyze_results.py

그래프 저장을 쓰려면 matplotlib 필요:
    pip install matplotlib
(없어도 순위표는 출력됩니다.)
"""

import csv
import glob
import os

RESULTS_FILE = "results_transfer.csv"
LOG_DIR = "logs"
OUT_DIR = "plots"


def print_ranking():
    if not os.path.exists(RESULTS_FILE):
        print(f"'{RESULTS_FILE}' 가 없습니다. 먼저 실험을 돌려주세요.")
        return None

    with open(RESULTS_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("결과가 비어 있습니다.")
        return None

    rows.sort(key=lambda r: float(r["best_accuracy(%)"]), reverse=True)

    print("\n" + "=" * 92)
    print(" 밤샘 실험 결과 (최고 정확도 순)")
    print("=" * 92)
    print(f"{'순위':<4}{'model':<20}{'opt':<8}{'lr':<9}"
          f"{'batch':<7}{'epochs':<8}{'best(%)':<10}{'final(%)':<10}{'시간(s)':<8}")
    print("-" * 92)
    for rank, r in enumerate(rows, start=1):
        mark = "  <- 최고!" if rank == 1 else ""
        print(f"{rank:<4}{r['model']:<20}{r['optimizer']:<8}{r['lr']:<9}"
              f"{r['batch_size']:<7}{r['epochs']:<8}"
              f"{r['best_accuracy(%)']:<10}{r['final_accuracy(%)']:<10}"
              f"{r['학습시간(초)']:<8}{mark}")
    print("=" * 92)

    best = rows[0]
    print(f"\n🏆 최고 조합: {best['model']} + {best['optimizer']} "
          f"+ lr={best['lr']}  →  정확도 {best['best_accuracy(%)']}%\n")
    return rows


def plot_curves():
    try:
        import matplotlib
        matplotlib.use("Agg")  # 화면 없이 파일로 저장
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib 미설치 → 그래프는 건너뜁니다. "
              "그래프 보려면: pip install matplotlib)")
        return

    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.csv")))
    if not log_files:
        print("(logs 폴더에 로그가 없어 그래프를 건너뜁니다.)")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # 그래프 1: 모든 실험의 정확도 곡선 한 장에
    plt.figure(figsize=(10, 6))
    for lf in log_files:
        name = os.path.splitext(os.path.basename(lf))[0]
        epochs, accs = [], []
        with open(lf, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                accs.append(float(row["test_accuracy(%)"]))
        if epochs:
            plt.plot(epochs, accs, marker="o", markersize=3, label=name)
    plt.title("실험별 테스트 정확도 (epoch별)")
    plt.xlabel("epoch")
    plt.ylabel("test accuracy (%)")
    plt.legend(fontsize=7, loc="lower right")
    plt.grid(True, alpha=0.3)
    out1 = os.path.join(OUT_DIR, "accuracy_curves.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=120)
    plt.close()

    # 그래프 2: 모든 실험의 loss 곡선 한 장에
    plt.figure(figsize=(10, 6))
    for lf in log_files:
        name = os.path.splitext(os.path.basename(lf))[0]
        epochs, losses = [], []
        with open(lf, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                losses.append(float(row["train_loss"]))
        if epochs:
            plt.plot(epochs, losses, marker="o", markersize=3, label=name)
    plt.title("실험별 학습 loss (epoch별)")
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.legend(fontsize=7, loc="upper right")
    plt.grid(True, alpha=0.3)
    out2 = os.path.join(OUT_DIR, "loss_curves.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=120)
    plt.close()

    print(f"📊 그래프 저장됨:\n   - {out1}\n   - {out2}\n")


if __name__ == "__main__":
    print_ranking()
    plot_curves()
