# 적대적 전이성 공격 재현 (레벨 1: FGSM / I-FGSM)

논문 *"Improving Adversarial Transferability with Local Perturbation Augmentation"* (CVPR)
의 기반이 되는 적대적 공격을 **논문과 동일한 환경**(ImageNet 사전학습 모델 + ε=16/255 등)으로 재현합니다.

> ⚠️ 이 코드는 **AI 모델의 강건성(robustness)을 연구·평가하기 위한 방어적 목적**의 학술 재현입니다.
> 공개 데이터셋과 공개 사전학습 모델만 사용합니다.

## 레벨 1 목표
- 적대적 예제(adversarial example)가 무엇인지 이해
- FGSM(1-step) / I-FGSM(iterative) / MI-FGSM(momentum) 직접 실행
- "사람 눈엔 똑같은데 모델은 틀리게 분류"하는 것을 눈으로 확인
- 공격 성공률(ASR) 측정

## 설치 (한 번만)
```cmd
conda activate cv
pip install timm datasets pillow matplotlib
```

## 사용법

### 1. 데이터 준비 (Imagenette = ImageNet 공개 하위셋)
```cmd
python adversarial/download_data.py --num 200
```
→ `data/images/` 에 이미지, `data/labels.csv` 에 정답 라벨(ImageNet 기준) 저장.

- 전체 ImageNet(144GB)이 아니라 **Imagenette**(10개 클래스, 약 100MB)를 받습니다.
- **가입/로그인 불필요** (torchvision `download=True`로 자동).
- Imagenette 10개 클래스는 ImageNet 1000개의 부분집합이라,
  ImageNet 사전학습 ResNet-50을 그대로 공격 대상으로 쓸 수 있습니다.

> 참고: Imagenette는 가입 없이 빠르게 체험하기 위한 대체재입니다.
> 논문과 **완전히 동일한 환경**을 원하면 아래 "공식 ImageNet" 방식을 쓰세요.

### 1-B. (논문과 동일) 공식 ImageNet 검증셋 사용
논문은 ILSVRC 2012 검증셋에서 서로 다른 클래스 1000장을 썼습니다. 이를 그대로 재현하려면:

1. image-net.org 가입 후 아래 두 파일을 받아 `data/imagenet/` 에 넣습니다 (압축 풀 필요 없음):
   - `ILSVRC2012_img_val.tar` (6.3GB, 검증 이미지 5만장)
   - `ILSVRC2012_devkit_t12.tar.gz` (라벨 정보, Task 1&2)
2. 선별 스크립트 실행:
   ```cmd
   python adversarial/prepare_imagenet.py --per-class 1
   ```
   → 1000개 서로 다른 클래스에서 1장씩, ResNet-50이 올바르게 분류하는 이미지만 골라
   `data/images/` + `data/labels.csv` 로 저장 (논문 설정과 동일).
   → 빠른 테스트는 `--max 200` 추가.
3. 이후 `run_attack.py` / `visualize.py` 를 그대로 사용.

### 2. 공격 실행 + 성공률 측정
```cmd
python adversarial/run_attack.py --attack ifgsm
```
공격 종류: `fgsm`(1-step), `ifgsm`(반복), `mifgsm`(momentum)

### 3. 시각화 (원본 vs 노이즈 vs 공격 이미지)
```cmd
python adversarial/visualize.py --attack ifgsm --num 5
```
→ `adv_plots/` 폴더에 PNG 저장.

## 논문과 동일한 파라미터 (4.1.4)
| 파라미터 | 값 |
|---------|-----|
| perturbation budget ε | 16/255 |
| iterations T | 10 |
| step size α | 1.6/255 |
| momentum μ | 1.0 |

## 레벨 2: 전이성 평가 (논문 Table 1 설정)
ResNet-50(surrogate)에서 공격을 한 번 생성해, 여러 target 모델에 통하는지 ASR을 측정합니다.
```cmd
python adversarial/run_transfer.py --attack ifgsm
python adversarial/run_transfer.py --attack mifgsm
```
→ 화이트박스(ResNet-50 자기자신)는 높고, 전이(DenseNet/ViT 등)는 낮게 나옵니다.
  그 '떨어지는 폭'이 전이성 문제이며, LPAA가 줄이려는 대상입니다.
  (target 모델은 처음 실행 시 자동 다운로드됩니다.)

## 파일 설명
| 파일 | 역할 |
|------|------|
| `attacks.py` | FGSM / I-FGSM / MI-FGSM 공격 구현 (논문 Eq.1) |
| `download_data.py` | Imagenette(간편) 데이터 준비 |
| `prepare_imagenet.py` | 공식 ImageNet val 선별 (논문 동일, 1000장) |
| `data_utils.py` | 이미지/라벨 로딩 ([0,1] 공간 + 정규화 분리) |
| `run_attack.py` | 공격 실행 + ASR 측정 (화이트박스, 레벨 1) |
| `run_transfer.py` | 전이성 평가 (여러 target 모델, 레벨 2) |
| `visualize.py` | 원본/노이즈/공격 이미지 시각화 |

## 다음 단계
- ~~레벨 2: 전이성~~ (완료: `run_transfer.py`)
- **레벨 3**: MI-FGSM → LPAA 핵심(masking + augmentation) 구현
- **레벨 4**: LPAA vs baseline 축소 비교 실험 (논문 Table 재현)
