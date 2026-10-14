# Computer Vision 실험 저장소 (학부연구생 학습 기록)

로컬 노트북(RTX 5060 Laptop GPU, Windows)에서 컴퓨터 비전 실험을 단계적으로 수행한 기록입니다.
입문(이미지 분류)부터 CVPR 논문 재현(적대적 공격 전이성)까지 레벨을 올려가며 진행합니다.

## 프로젝트 구성

| 단계 | 주제 | 위치 |
|------|------|------|
| 레벨 0 | CIFAR-10 이미지 분류 + 하이퍼파라미터 튜닝 | 루트 (`train.py` 등) |
| 레벨 0+ | 전이학습 그리드 서치 (ResNet/MobileNet) | 루트 (`train_transfer.py` 등) |
| 레벨 1~2 | 적대적 공격 & 전이성 (논문 재현) | [`adversarial/`](adversarial/) |

재현 중인 논문: *Improving Adversarial Transferability with Local Perturbation Augmentation* (CVPR).
실험 결과와 해석은 아래 **"📊 실험 결과 및 해석"** 섹션에 정리되어 있습니다.

---

## CIFAR-10 이미지 분류 (레벨 0)

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

---

## 🌙 밤샘 전이학습 그리드 서치 (심화)

사전학습 모델(ResNet/MobileNet 등)을 CIFAR-10에 미세조정하면서, 여러 하이퍼파라미터 조합을 자동으로 실행하는 심화 실험입니다. (3시간 목표)

| 파일 | 역할 |
|------|------|
| `train_transfer.py` | 전이학습 1회 실험 (모델/lr/옵티마이저 등 지정) |
| `run_overnight.py` | 여러 조합 자동 실행 (시간 추정 + 중단 복구) |
| `analyze_results.py` | 결과 순위표 + loss/정확도 곡선 그래프(PNG) |

### 실행 전 체크 ⚠️
- 전원 어댑터 연결 🔌
- 절전모드 "안 함"
- 노트북 덮개 닫아도 안 꺼지게 설정

### 사용법
```cmd
conda activate cv
pip install matplotlib            # 그래프용 (한 번만)
python run_overnight.py           # 밤새 여러 실험 자동 실행
```

아침에 일어나서:
```cmd
python analyze_results.py         # 순위표 + plots/ 폴더에 그래프 PNG
```

### 시간 조절
`run_overnight.py` 상단에서:
- `GRID` : 실험할 조합 (모델/옵티마이저/학습률)
- `COMMON["epochs"]` : 각 실험 에포크 수 (시간의 핵심 손잡이)
- `COMMON["batch-size"]` : VRAM 부족(OOM) 시 64 → 32로 줄이기
- `TIME_BUDGET_SEC` : 전체 시간 예산 (초과 예상 시 자동 중단, 다음 실행 때 이어서)

중단되어도 `results_transfer.csv` 에 기록된 실험은 다시 실행 시 자동으로 건너뜁니다.

---

# 📊 실험 결과 및 해석 (연구 노트)

이 섹션은 실제로 수행한 실험의 결과와 해석을 기록합니다. (환경: RTX 5060 Laptop GPU, Windows)

## 실험 1: CIFAR-10 기본 CNN (레벨 0 — 입문)

처음 밑바닥 CNN으로 학습률 0.01, 5 epoch 학습.

| 지표 | 값 |
|------|-----|
| 최종 테스트 정확도 | **68.30%** |
| 학습 시간 | 63.6초 |
| loss 변화 | 1.89 → 0.90 (에포크마다 감소) |

**해석:** loss가 매 에포크 감소 = 모델이 정상적으로 학습(피드백받아 수정)됨.
10개 클래스 랜덤(10%) 대비 68%는 간단한 CNN + 5 epoch치고 양호. 이후 전이학습으로 크게 향상됨.

---

## 실험 2: 밤샘 전이학습 그리드 서치 (레벨 0 — 심화)

사전학습 ResNet18 / MobileNetV3-small을 CIFAR-10에 미세조정.
학습률 × 옵티마이저 × 모델 조합 8개를 자동 실행 (각 12 epoch).

| 순위 | model | optimizer | lr | best(%) | 시간(s) |
|------|-------|-----------|-----|---------|---------|
| 🥇 1 | resnet18 | sgd | 0.01 | **96.54** | 954 |
| 🥈 2 | mobilenet_v3_small | sgd | 0.01 | 96.01 | 714 |
| 🥉 3 | mobilenet_v3_small | adamw | 0.001 | 95.98 | 720 |
| 4 | resnet18 | sgd | 0.001 | 95.56 | 905 |
| 5 | resnet18 | adamw | 0.001 | 95.48 | 910 |
| 6 | mobilenet_v3_small | sgd | 0.001 | 94.25 | 715 |
| 7 | mobilenet_v3_small | adamw | 0.01 | 91.70 | 721 |
| 8 | resnet18 | adamw | 0.01 | **87.76** | 905 |

**핵심 해석:**
1. **전이학습의 위력**: 밑바닥 CNN(68%) → 전이학습(최고 96.54%), 약 28%p 향상.
2. **옵티마이저마다 최적 학습률이 다르다** ⭐
   - SGD는 lr=0.01에서 최고 (96.54%)
   - AdamW는 lr=0.01에서 꼴찌(87.76%), lr=0.001에서 양호(95.48%)
   - → AdamW는 작은 학습률을 선호. "lr은 옵티마이저에 맞춰 조정해야 한다"는 교훈을 직접 확인.
3. **정확도 vs 속도 트레이드오프**: MobileNet이 ResNet18보다 ~25% 빠르면서 정확도는 비슷 → 효율적.

---

## 실험 3: 적대적 공격 — 화이트박스 (레벨 1)

논문 *"Improving Adversarial Transferability with Local Perturbation Augmentation"* (CVPR) 재현.
- 데이터: 공식 ImageNet 검증셋에서 1000개 서로 다른 클래스, ResNet-50이 올바르게 분류하는 이미지 선별 (논문과 동일)
- surrogate = 평가 모델 = ResNet-50 (화이트박스)
- 파라미터: ε=16/255, T=10, α=1.6/255, μ=1.0 (논문 4.1.4)

| 공격 | 공격 전 맞힘 | ASR (공격 성공률) |
|------|-------------|------------------|
| I-FGSM | 916장 | **100.00%** (916장 전부 오분류) |

**해석:**
- ASR = (공격 후 틀린 장수) / (공격 전 맞힌 장수). 높을수록 공격이 강함.
- 화이트박스(공격 만든 모델 = 평가 모델)에서는 ASR이 거의 100% → 공격이 완벽히 작동.
- 공격 전 정확도가 100%가 아닌 이유: 선별 후 PNG 저장/재로딩 과정의 미세한 픽셀 손실로
  경계선 이미지 일부(84장)가 재분류 시 틀려짐. ASR은 "공격 직전 맞힌 것"만 분모로 쓰므로 계산은 공정함.

---

## 실험 4: 적대적 전이성 — 블랙박스 (레벨 2 — 논문의 핵심)

ResNet-50(surrogate)에서 I-FGSM 공격을 한 번 생성 → 서로 다른 구조의 target 모델들에 통하는지 평가.

| target 모델 | 구조 계열 | ASR (전이성) | 비고 |
|------------|----------|-------------|------|
| resnet50 | CNN (자기 자신) | **100.00%** | 화이트박스 |
| resnext50 | CNN (ResNet 유사) | 80.48% | 전이 잘됨 |
| densenet121 | CNN | 71.93% | 전이 꽤됨 |
| convnext_t | CNN (신식) | 38.26% | 중간 |
| swin_t | Transformer | 27.92% | 낮음 |
| vit_b | Transformer (순수) | **10.76%** | 가장 낮음! |

**핵심 해석:**
1. **구조가 비슷할수록 전이가 잘 된다**: ResNeXt(80%)·DenseNet(72%)는 ResNet과 같은 CNN 계열이라 공격이 잘 옮겨감.
2. **CNN → ViT 전이는 극도로 어렵다**: ViT는 10.76%로 처참. CNN과 구조가 완전히 달라
   ResNet-50에 특화된 노이즈가 거의 통하지 않음.
3. **이것이 논문의 문제의식**: 기본 반복 공격(I-FGSM)은 surrogate 모델에 과적합되어 전이성이 낮음.
   LPAA는 이 전이 ASR(특히 ViT)을 높이는 것이 목표.

**논문 Table 1과 비교 (참고):** 논문의 발전된 공격은 ViT-B 전이 ASR이 훨씬 높음
(예: LPAA ≈ 58%). 기본 I-FGSM(10.76%) 대비 큰 격차가 곧 논문의 기여.

---

## 다음 단계 (진행 예정)
- **레벨 3**: MI-FGSM(momentum) → LPAA 핵심(stochastic masking + augmentation) 구현
- **레벨 4**: LPAA vs baseline(GRA/PGN 등) 축소 비교 실험 (논문 Table 재현)

적대적 공격 실험의 코드/사용법은 [`adversarial/README.md`](adversarial/README.md) 참고.
