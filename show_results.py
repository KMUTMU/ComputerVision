"""
results.csv 에 쌓인 실험 결과를 보기 좋게 정리해서 출력합니다.
- 정확도(test_accuracy) 높은 순으로 정렬해서 보여줍니다.

사용법: (cv) 환경에서  ->  python show_results.py
"""

import csv
import os

RESULTS_FILE = "results.csv"


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"'{RESULTS_FILE}' 가 없습니다. 먼저 실험을 돌려주세요.")
        print("예:  python run_experiments.py")
        return

    with open(RESULTS_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("결과가 비어 있습니다.")
        return

    # 정확도 기준 내림차순 정렬
    rows.sort(key=lambda r: float(r["test_accuracy(%)"]), reverse=True)

    print("\n" + "=" * 70)
    print(" 실험 결과 (정확도 높은 순)")
    print("=" * 70)
    print(f"{'순위':<4}{'optimizer':<11}{'lr':<9}{'batch':<8}"
          f"{'epochs':<8}{'정확도(%)':<11}{'시간(초)':<9}")
    print("-" * 70)
    for rank, r in enumerate(rows, start=1):
        marker = " <- 최고!" if rank == 1 else ""
        print(f"{rank:<4}{r['optimizer']:<11}{r['lr']:<9}{r['batch_size']:<8}"
              f"{r['epochs']:<8}{r['test_accuracy(%)']:<11}{r['학습시간(초)']:<9}{marker}")
    print("=" * 70)

    best = rows[0]
    print(f"\n🏆 가장 좋은 조합: lr={best['lr']}, optimizer={best['optimizer']} "
          f"→ 정확도 {best['test_accuracy(%)']}%\n")


if __name__ == "__main__":
    main()
