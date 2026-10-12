"""
밤샘 그리드 서치 러너 (3시간 목표)
- 여러 하이퍼파라미터 조합을 자동으로 순서대로 실행합니다.
- 중단되어도 이미 끝난 실험은 건너뛰고 이어서 진행합니다 (results_transfer.csv 기준).
- 각 실험 전후로 경과/예상 시간을 출력합니다.

사용법: (cv) 환경, ComputerVision 폴더에서
    python run_overnight.py

실행 전 체크:
    - 전원 어댑터 연결
    - 절전모드 "안 함"
    - 노트북 덮개 닫아도 안 꺼지게 설정
"""

import csv
import itertools
import os
import subprocess
import sys
import time

RESULTS_FILE = "results_transfer.csv"

# ----------------------------------------------------------------------
# 실험할 하이퍼파라미터 조합 (여기만 바꾸면 됩니다)
# ----------------------------------------------------------------------
# 전이학습 백본은 무거우므로, 5060(8GB) 기준 안전하게 구성.
# 조합 수 x 에포크로 총 시간이 결정됩니다.

GRID = {
    "model": ["resnet18", "mobilenet_v3_small"],   # 2종 모델 비교
    "optimizer": ["sgd", "adamw"],                 # 2종 옵티마이저 비교
    "lr": [0.01, 0.001],                           # 2종 학습률 비교
}

# 모든 실험 공통 설정
COMMON = {
    "epochs": 12,          # 각 실험 에포크 (시간 조절의 핵심 손잡이)
    "batch-size": 64,      # 8GB VRAM 안전값. OOM 나면 32로 줄이세요
    "freeze-backbone": False,
}

# 전체 목표 시간 안전장치 (초). 이 시간을 넘기면 더 시작하지 않고 종료.
TIME_BUDGET_SEC = 3 * 60 * 60 + 20 * 60   # 약 3시간 20분


def already_done():
    """results_transfer.csv에 이미 기록된 실험 이름 집합"""
    done = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(row["exp_name"])
    return done


def make_exp_name(combo):
    model = combo["model"]
    opt = combo["optimizer"]
    lr = combo["lr"]
    bs = COMMON["batch-size"]
    frozen = "_frozen" if COMMON.get("freeze-backbone") else ""
    return f"{model}_{opt}_lr{lr}_bs{bs}{frozen}"


def build_cmd(combo):
    cmd = [sys.executable, "train_transfer.py"]
    merged = {k: v for k, v in COMMON.items() if k != "freeze-backbone"}
    merged.update(combo)
    for key, value in merged.items():
        cmd += [f"--{key}", str(value)]
    if COMMON.get("freeze-backbone"):
        cmd.append("--freeze-backbone")
    cmd += ["--exp-name", make_exp_name(combo)]
    return cmd


def main():
    keys = list(GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]
    total = len(combos)

    done = already_done()
    print("=" * 68)
    print(f"총 {total}개 조합  |  각 {COMMON['epochs']} epochs  "
          f"|  batch {COMMON['batch-size']}")
    print(f"이미 완료된 실험: {len(done)}개 (건너뜀)")
    print(f"시간 예산: {TIME_BUDGET_SEC/3600:.1f}시간")
    print("=" * 68)

    overall_start = time.time()
    durations = []
    ran = 0

    for i, combo in enumerate(combos, start=1):
        name = make_exp_name(combo)

        if name in done:
            print(f"\n[{i}/{total}] 건너뜀 (이미 완료): {name}")
            continue

        # 시간 예산 체크 (평균 실험 시간으로 다음 실험이 예산 넘을지 예측)
        elapsed = time.time() - overall_start
        if durations:
            avg = sum(durations) / len(durations)
            if elapsed + avg > TIME_BUDGET_SEC:
                print(f"\n시간 예산 초과 예상 → 여기서 중단합니다.")
                print(f"(남은 조합은 다음 실행 시 자동으로 이어서 진행됩니다.)")
                break

        print(f"\n{'#'*68}")
        print(f"#  [{i}/{total}] 실행: {name}")
        if durations:
            avg = sum(durations) / len(durations)
            remaining = (total - i + 1) * avg
            print(f"#  평균 {avg/60:.1f}분/실험  |  남은 예상 ~{remaining/60:.0f}분")
        print(f"{'#'*68}")

        t0 = time.time()
        subprocess.run(build_cmd(combo))
        durations.append(time.time() - t0)
        ran += 1

    total_elapsed = time.time() - overall_start
    print(f"\n{'='*68}")
    print(f"밤샘 실행 종료  |  이번에 {ran}개 실험 실행  "
          f"|  총 {total_elapsed/60:.0f}분 소요")
    print(f"결과 분석:  python analyze_results.py")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
