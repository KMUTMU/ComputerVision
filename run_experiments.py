"""
여러 실험을 자동으로 돌리는 스크립트 (Windows/Mac/Linux 공통)
- train.py 를 파라미터만 바꿔가며 반복 실행합니다.
- .bat 파일보다 유연합니다. 조합(그리드 서치)도 쉽게 가능.

사용법: (cv) 환경에서  ->  python run_experiments.py
"""

import subprocess
import sys

# ----------------------------------------------------------------------
# 여기서 실험할 파라미터 조합을 정의합니다.  <- 이 부분만 바꾸면 됩니다!
# ----------------------------------------------------------------------

# [방법 1] 학습률만 바꿔가며 실험
experiments = [
    {"lr": 0.001},
    {"lr": 0.01},
    {"lr": 0.1},
    {"lr": 0.5},
]

# [방법 2] 여러 파라미터를 조합해서 실험하고 싶다면 아래처럼 (주석 해제):
# import itertools
# lrs = [0.01, 0.1]
# optimizers = ["sgd", "adam"]
# experiments = [
#     {"lr": lr, "optimizer": opt}
#     for lr, opt in itertools.product(lrs, optimizers)
# ]

# 모든 실험에 공통으로 적용할 값
COMMON = {"epochs": 5, "batch-size": 128}


def main():
    total = len(experiments)
    print(f"총 {total}개의 실험을 순서대로 실행합니다.\n")

    for i, params in enumerate(experiments, start=1):
        # train.py 에 넘길 명령어 조립
        cmd = [sys.executable, "train.py"]
        merged = {**COMMON, **params}
        for key, value in merged.items():
            cmd.append(f"--{key}")
            cmd.append(str(value))

        print(f"\n{'#' * 55}")
        print(f"#  실험 {i}/{total}:  {params}")
        print(f"{'#' * 55}")

        # 실제 실행 (출력은 그대로 화면에 보임)
        subprocess.run(cmd)

    print(f"\n{'=' * 55}")
    print("모든 실험 완료! results.csv 를 열어 결과를 비교하세요.")
    print("정리된 비교표는  ->  python show_results.py")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
