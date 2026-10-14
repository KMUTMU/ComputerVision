# 완전 재현 세팅 (TransferAttack 표준 데이터셋 + timm 가중치)

논문과 수치까지 최대한 일치시키기 위해, 적대적 전이성 분야의 표준 세팅을 따른다.
(출처: Trustworthy-AI-Group/TransferAttack)

## 1. 표준 1000장 데이터셋 받기

이 분야 논문들이 공통으로 쓰는 "학술 표준 ImageNet 호환 1000 PNG" 데이터셋.

1. TransferAttack README의 다운로드 링크(Google Drive)에서 데이터 받기:
   https://github.com/Trustworthy-AI-Group/TransferAttack
   (README의 "Download the data from GoogleDrive" 링크)
2. 압축을 풀면 보통 아래 구조:
   ```
   images/            # 1000장 PNG
   labels.csv         # filename,label (+ 때로 targeted label)
   ```
3. 이것을 우리 파이프라인 위치로 복사:
   ```
   data/images/       <- 표준 PNG 1000장
   data/labels.csv    <- filename,label
   ```
   (기존 ImageNet 선별본을 덮어쓰거나, 백업 후 교체)

> labels.csv 컬럼이 `filename,label,...` 형태면 우리 data_utils.py 가 그대로 읽는다.
> (첫 두 컬럼 filename,label 만 사용)

## 2. timm 가중치 사용

- CNN(ResNet-50 등): torchvision 기본 가중치 (그대로)
- ViT: **timm** 가중치 사용 → 모델 이름에 `_timm` 접미사
  - `vit_b_timm`  = timm `vit_base_patch16_224`
  - `pit_b_timm`  = timm `pit_b_224`
  - `swin_t_timm` = timm `swin_tiny_patch4_window7_224`

timm 설치:
```cmd
pip install timm
```

## 3. 완전 재현 실행

표준 데이터 + timm ViT 로 LPAA 전이성 평가:
```cmd
python adversarial/run_transfer.py --attack lpaa --surrogate resnet50 --targets resnet50 vit_b_timm
```

→ 논문 Table 1 의 LPAA+ViT-B(≈58.3%)에 더 가까워지는지 확인.

## 참고
- 이렇게 "표준 데이터 + timm 가중치"로 맞추면, 앞서 분석한 7%p 차이의
  두 주원인(데이터 샘플 분산, 모델 가중치 소스)이 제거된다.
- 완전히 동일하지 않더라도(구현 디테일 차이 잔존), 논문값에 더 근접하면
  "차이의 원인이 데이터/가중치였다"는 가설이 확증된다.
