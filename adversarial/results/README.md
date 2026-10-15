# 실험 결과 로그

실험 결과를 파일로 보관하는 폴더. 캡처 없이 git push로 어디서든 확인/분석 가능.

## 파일
- `all_experiments_log.csv` — 지금까지 모든 실험의 핵심 수치 (대화 중 확인된 값들 정리)
- `beta_sweep_*.csv` — β 스케줄 비교 실행 시 자동 생성 (run_beta_sweep.py)

## 핵심 결과 요약 (ViT-B 전이 ASR 중심)

| 실험 | 설정 | ViT ASR | 메모 |
|------|------|---------|------|
| exp4 | I-FGSM | 10.76% | 기본 공격은 CNN→ViT 전이 처참 |
| exp5 | LPAA (자체데이터+torchvision) | 65.34% | 전이성 폭등 (논문 핵심 재현) |
| exp8 | LPAA (표준데이터+timm) | 62.63% | 논문 58.3%에 근접 |
| exp9A | LPAA fixed β=35 (seed고정) | 63.61% | β비교 기준 |
| exp9B | LPAA β 50→20 감소 | 62.53% | 가설 기각(-1.08%p) |
| **exp10** | **β sweep (5종 비교)** | **대기 중** | run_beta_sweep.py 결과 pending |
| 논문 | LPAA (Table 1) | 58.30% | 공식 수치 |

## 7%p 차이 수사 결론
- 선별기준·초기화: 무죄 (ablation으로 배제)
- 데이터 종류: 맞춤 (NIPS 2017 ImageNet-Compatible 확인)
- 데이터 전처리(224 vs 299) + 모델 가중치(timm) + LPA 구현 디테일: 잔존 원인
- 표준 세팅 적용으로 gap 7.0%p → 4.3%p 축소

## exp10 (β sweep) 결과 — 집에서 돌린 뒤 여기 채울 예정
run_beta_sweep.py 가 생성한 CSV를 이 폴더에 두고 git push 하면 자동 보관됨.
(fixed-35 / decay50to20 / rise20to50 / decay60to10 / rise10to60 비교)
