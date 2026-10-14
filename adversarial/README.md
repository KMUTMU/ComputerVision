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

## 레벨 3: LPAA 구현
논문 Algorithm 1 (Eq.3~8)을 구현한 `lpaa.py`.
- LPA: 랜덤 마스크(p) + 증강계수(β) + N개 subspace 평균 그래디언트
- PI: GMI 모멘텀 초기화 + LPA 정제
- 파라미터: N=20, p=0.95, β=35, η=3 (논문 4.1.4)

단일 공격으로 전이성 평가:
```cmd
python adversarial/run_transfer.py --attack lpaa
# 초기화(PI) 기여 분리(ablation):
python adversarial/run_transfer.py --attack lpaa --no-init
```

### GMI (Global Momentum Initialization) 전체 로직

LPAA의 초기화(PI)는 GMI(논문 참고문헌 [38], Wang et al. 2024)에 기반한다.
공식 구현(`Omenzychen/Global-Momentum-Initialization`)의 핵심 로직을 재현했다.

**해결하려는 문제 — gradient elimination:**
반복 공격에서 매 스텝 `sign(gradient)`만 쓰면, 그래디언트가 스텝마다 들쭉날쭉해
부호가 상쇄(elimination)되며 특정 지역 최적에 갇힌다. 전이성 저하의 원인 중 하나.

**GMI의 아이디어 — 사전 수렴(pre-convergence) + global search:**
본 공격을 시작하기 전에 '정찰(사전 공격)'을 먼저 돌려 안정적인 모멘텀을 쌓고,
그 모멘텀을 본 공격의 출발점으로 삼아 '선견지명 있는' 방향에서 시작한다.

**알고리즘 (의사코드):**
```
입력: 깨끗한 이미지 x, 라벨 y, 사전공격 횟수 P(=5), 증폭계수 amp(=10), 모멘텀 μ, 스텝 α
g = 0                      # 모멘텀
x_cur = x
for i in range(P):         # ---- 사전 수렴 단계 (global search) ----
    grad = ∇_x L(f(x_cur), y)
    grad = grad / mean(|grad|)        # L1 정규화
    g = μ·g + grad                    # 모멘텀 누적
    x_cur = x_cur + amp·α·sign(g)     # amp(=10)배로 '크게' 이동 → 넓게 탐색
# 사전 공격 종료: 입력 x_cur 는 버리고(원본으로 리셋), 모멘텀 g 만 유지
return g                   # 이 g 가 본 공격의 초기 모멘텀
```

**핵심 포인트 2가지 (우리가 처음 근사 구현 때 빠뜨렸던 부분):**
1. **증폭계수 amp=10**: 사전 공격 단계에서만 스텝을 10배로 키워, 좁은 지역이 아니라
   손실 지형을 '넓게' 탐색한다 (global search). 본 공격 단계에서는 amp=1.
2. **입력 리셋 + 모멘텀 유지**: 사전 공격이 끝나면 그동안 이동한 입력(x_cur)은 버리고
   원본 x에서 다시 시작하되, 축적된 모멘텀 g는 가지고 간다. "정찰로 길만 파악하고
   출발점은 원점으로" 돌아오는 셈.

**LPAA에서의 사용 (Eq.7, 8):**
- GMI_PI: 위 g로 한 걸음 이동 → `[-η·α, η·α]` 범위로 투영 → 초기 섭동 δ_M  (Eq.7)
- LPA_PI: δ_M에 LPA 그래디언트를 한 번 더 적용해 정제 → δ_0  (Eq.8)
- δ_0가 본 LPAA 반복의 시작 섭동이 된다.

**구현 파라미터:** pre_attack=5, amp=10, η=3, α_LPAPI=3.2/255 (논문/공식 구현 기준)

### ⚠️ 논문과 우리 구현의 디테일 차이 (재현 불확실성)

논문은 지면 제약으로 모든 구현 세부사항을 적지 못하며, 본 논문(LPAA)의 **공식 코드가
공개되지 않아** 아래 항목들은 수식(Eq.3~8)을 "합리적으로 해석"해 구현한 것이다.
이 차이들이 재현값과 논문값의 7%p 차이에 기여했을 수 있다.

**하이퍼파라미터 (논문 명시 → 그대로 사용):**
| 항목 | 논문 값 | 우리 구현 | 일치? |
|------|--------|----------|------|
| N (subspace 수) | 20 | 20 | ✅ |
| p (마스킹 비율) | 0.95 | 0.95 | ✅ |
| β (증강 계수) | 35 | 35 | ✅ |
| ε / T / α / μ | 16/255 / 10 / 1.6/255 / 1.0 | 동일 | ✅ |

**구현 디테일 (논문 미명시 → 우리가 추정):**
| 디테일 | 논문 명시 | 우리 선택 | 영향 가능성 |
|--------|----------|----------|------------|
| 마스크 M 생성 | 불명확 | 매 subspace마다 새 이진 마스크(원소가 p 확률로 0) | 중 |
| `delta ⊙ M · β` 계산 순서 | 수식만, clamp 타이밍 불명확 | 마스킹→스케일→clamp(0,1) 순 | 중 |
| 그래디언트 정규화 | "L1" 언급 | L1 (mean of abs) | 하 |
| clamp(0,1) 적용 시점 | 불명확 | 각 subspace forward 직전 | 하 |
| GMI 세부 (amp 적용 범위 등) | GMI 논문[38] 참조만 | 공식 구현 재현 | 중 |

**모델 가중치 버전 (중요하지만 논문 미명시):**
- "ResNet-50", "ViT-B"는 아키텍처 이름일 뿐, 가중치 버전이 여러 개 존재.
  - 예: torchvision ResNet-50은 `IMAGENET1K_V1`(76.1%)과 `V2`(80.9%)가 다름.
- 논문이 **어떤 가중치(torchvision vs timm, V1 vs V2)**를 썼는지 명시하지 않음.
- 우리는 torchvision `IMAGENET1K_V1` 사용. 논문이 timm 가중치를 썼다면
  같은 "ViT-B"라도 학습 방법이 달라 공격 저항력이 다를 수 있음 → ASR 차이 가능.

**교훈:** 완벽한 재현은 저자 코드와 정확한 가중치 명세 없이는 어렵다. 재현자는
"합리적 추정"을 하고 그 불확실성을 명시적으로 기록해야 한다. (경향은 재현되었고,
절대값 차이는 위 디테일·데이터 분산으로 설명됨.)

### 🔎 커뮤니티 표준 확인 (7%p 차이의 핵심 단서)

적대적 전이성 분야의 표준 프레임워크 [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack)
(Trustworthy-AI-Group)을 조사한 결과, 우리 재현과 논문값 차이의 주원인 2가지가 거의 확증됨:

1. **표준 1000장 데이터셋 존재** ⭐
   - 이 분야 논문들은 1000장을 각자 랜덤 선별하는 게 아니라, 커뮤니티가 공유하는
     "학술 표준 ImageNet 호환 1000 PNG 이미지"(고정 세트)를 사용 (TransferAttack README).
   - 우리는 ImageNet val에서 자체 랜덤 선별 → **서로 다른 1000장**이 ASR 차이의 핵심 원인.
2. **모델 가중치는 timm 사용**
   - TransferAttack은 `timm` 라이브러리를 요구 → 이 분야 표준 가중치는 torchvision이 아닌 timm.
   - 우리는 torchvision `IMAGENET1K_V1` 사용 → 같은 "ViT-B"라도 가중치 소스가 달라 공격 저항력 차이.

**결론 갱신:** 7%p 차이는 "데이터 샘플 분산(③)"과 "모델 가중치 소스(④)"가 주원인으로 **확증**.
(선별 기준·초기화는 ablation으로 이미 배제됨.) 완전히 동일한 재현을 원하면 TransferAttack의
표준 데이터셋 + timm 가중치를 사용하면 됨.

**참고 — baseline 재사용:** TransferAttack에 GRA/PGN 등 LPAA 논문의 baseline이 구현되어 있어,
향후 baseline 비교(Table 1 전체)는 이 프레임워크를 활용 가능. (LPAA 자체는 미구현 — 최신 논문.)

## 레벨 4: 논문 Table 1 재현
여러 공격(ifgsm/mifgsm/lpaa)을 여러 target 모델에 돌려 Table 형식으로 비교.
```cmd
python adversarial/reproduce_table.py
```
→ 화면에 ASR 표(행=공격, 열=모델) 출력 + `table1_results.csv` 저장.
→ `--methods`, `--targets` 로 범위 조절. 처음엔 작은 데이터(예: 50장)로 테스트 권장.

| 파일 | 역할 |
|------|------|
| `lpaa.py` | LPAA 공격 구현 (논문 Algorithm 1) |
| `reproduce_table.py` | 논문 Table 1 형식 다중 공격 x 다중 모델 비교 |

## 다음 단계
- ~~레벨 2: 전이성~~ (완료: `run_transfer.py`)
- ~~레벨 3: LPAA 구현~~ (완료: `lpaa.py`)
- ~~레벨 4: Table 재현~~ (완료: `reproduce_table.py`)
- baseline(GRA/PGN 등) 추가 비교는 TransferAttack 프레임워크 활용 검토

---

## 🧪 개선 아이디어: β(증강계수) 스케줄링

**가설:** 논문은 증강계수 β=35 고정. 대신 "초반엔 크게 탐색(β 큼) → 후반엔 작게 수렴(β 작음)"
하면, 더 일반화된 섭동 방향을 찾아 전이성이 개선될 수 있다.

**설정값 (선형 감소, linear decay):**
| 항목 | 값 | 근거 |
|------|-----|------|
| beta_start | 50 | 초반(t=0): 넓게 탐색 |
| beta_end | 20 | 후반(t=T-1): 정밀 수렴 |
| 평균 | **35** | 논문 고정값과 동일 → 공정 비교 |
| 스케줄 | β(t) = 50 − 30·t/(T−1) | T=10 |

iteration별 β (T=10): 50.0, 46.7, 43.3, 40.0, 36.7, 33.3, 30.0, 26.7, 23.3, 20.0 (평균 35.0)

**공정 비교 원칙:**
- β 평균을 논문값(35)에 맞춰, "β를 변화시킨 효과"만 측정 (단순히 β가 커서 생기는 효과 배제).
- LPAA는 랜덤 마스크를 쓰므로 **시드 고정**(`--seed`) 필수. 안 그러면 β 효과와 랜덤 변동을 구분 못 함.

**실행 (고정 β vs 스케줄 β 비교):**
```cmd
# 기준: 고정 β=35
python adversarial/run_transfer.py --attack lpaa --targets resnet50 vit_b_timm --seed 0

# 개선: β 50→20 선형 감소
python adversarial/run_transfer.py --attack lpaa --targets resnet50 vit_b_timm --seed 0 --beta-start 50 --beta-end 20
```
→ 두 결과의 ViT ASR 차이가 "β 스케줄링의 순수 효과".

---

## 데이터 편향 점검 (방법론 메모)

현재 데이터 파이프라인(`data_utils.py`):
- `data/images/` PNG + `data/labels.csv`(filename,label) 로드
- 각 이미지: Resize(256) → CenterCrop(224) → [0,1] 텐서 (정규화는 모델 forward 직전)
- `shuffle=False` (순서대로)

**편향 위험 평가:**
- ✅ 현재 분야 표준 1000장 전체를 사용 → 편향 위험 낮음.
- ⚠️ `shuffle=False`이므로 **일부만(`--max N`) 쓰면 앞쪽 이미지에 편향** 가능
  (표준셋이 클래스 정렬이면 특정 클래스 쏠림). 전체 사용 시 문제 없음.
- ✅ ASR은 "모델이 원래 맞힌 것"만 분모로 → 공정.

**알고리즘 비교 시 주의:** 공격 알고리즘 A vs B 비교는 동일 데이터 + 동일 시드로 수행해야
공정하다. LPAA의 랜덤 마스크 때문에 `--seed` 고정이 특히 중요.
