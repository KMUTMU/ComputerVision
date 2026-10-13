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

### 1. 데이터 준비 (ImageNet 검증 샘플 소량)
```cmd
python adversarial/download_data.py --num 200
```
→ `data/images/` 에 이미지, `data/labels.csv` 에 정답 라벨 저장.
(전체 ImageNet 144GB가 아니라, 검증 샘플 수백 장만 받습니다.)

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

## 파일 설명
| 파일 | 역할 |
|------|------|
| `attacks.py` | FGSM / I-FGSM / MI-FGSM 공격 구현 (논문 Eq.1) |
| `download_data.py` | ImageNet 검증 샘플 준비 |
| `data_utils.py` | 이미지/라벨 로딩 ([0,1] 공간 + 정규화 분리) |
| `run_attack.py` | 공격 실행 + ASR 측정 (화이트박스) |
| `visualize.py` | 원본/노이즈/공격 이미지 시각화 |

## 다음 단계
- **레벨 2**: 전이성 — ResNet-50로 만든 공격이 다른 모델(DenseNet, ViT)에 통하는지
- **레벨 3**: MI-FGSM → LPAA 핵심(masking + augmentation) 구현
- **레벨 4**: LPAA vs baseline 축소 비교 실험 (논문 Table 재현)
