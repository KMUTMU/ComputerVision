# CIFAR-10 컴퓨터 비전 실험 (하이퍼파라미터 튜닝 입문)

RTX GPU가 있는 로컬 노트북에서 CIFAR-10 이미지 분류를 돌리며,
**파라미터(학습률 등)를 바꿔가며 여러 실험을 자동으로 실행**하고 결과를 비교하는 입문용 프로젝트입니다.

## 환경 준비 (Windows + NVIDIA GPU)

자세한 세팅 가이드는 [`miniconda_pytorch_setup.md`](miniconda_pytorch_setup.md) 참고.

```cmd
conda create -n cv python=3.11 -y
conda activate cv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132
```

> CUDA 버전 태그(`cu132`)는 본인 GPU/드라이버에 맞게 https://pytorch.org/get-started/locally/ 에서 확인하세요.

GPU가 제대로 잡혔는지 확인:
```cmd
python check_gpu.py
```

## 파일 설명

| 파일 | 역할 |
|------|------|
| `train.py` | CIFAR-10 분류 학습. `--lr`, `--epochs` 등 파라미터를 바꿔 실행 |
| `run_experiments.py` | 여러 실험을 자동 실행 (Python, 조합 실험 가능) |
| `run_experiments.bat` | 여러 실험을 자동 실행 (Windows 전용) |
| `show_results.py` | 결과를 정확도 높은 순으로 비교표 출력 |
| `check_gpu.py` | GPU 세팅 확인용 스크립트 |
| `miniconda_pytorch_setup.md` | 환경 세팅 전체 가이드 |

## 사용법

### 1. 하나만 테스트
```cmd
python train.py --lr 0.01
```
(처음 실행 시 CIFAR-10 데이터가 자동 다운로드됩니다.)

### 2. 여러 실험 자동 실행
```cmd
python run_experiments.py
```
학습률 `0.001, 0.01, 0.1, 0.5`를 순서대로 자동 실행합니다.

### 3. 결과 비교
```cmd
python show_results.py
```
정확도 높은 순으로 순위표를 출력합니다.

## 파라미터 바꾸기

`run_experiments.py` 상단의 `experiments` 리스트를 수정하면 원하는 실험 조합을 만들 수 있습니다.

```python
experiments = [
    {"lr": 0.001},
    {"lr": 0.01},
    {"lr": 0.1, "optimizer": "adam"},
]
```

바꿀 수 있는 파라미터: `--lr`, `--epochs`, `--batch-size`, `--optimizer` (sgd/adam)
