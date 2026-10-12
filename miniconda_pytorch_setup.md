# 윈도우 + RTX 4060 Laptop 딥러닝 환경 세팅 (미니콘다 + PyTorch)

> 목표: 내 노트북(Windows, RTX 4060 Laptop)에서 GPU로 컴퓨터 비전 실험 돌리기
> 핵심 원칙: **CUDA Toolkit 전체 설치는 필요 없다.** NVIDIA 드라이버 + CUDA 버전 PyTorch만 있으면 됨.

---

## 0단계: 꼭 알아야 할 사실

- 딥러닝용 PyTorch만 쓸 거면 **CUDA Toolkit을 따로 설치할 필요가 없다.**
- PyTorch GPU 버전을 설치하면 필요한 CUDA 런타임이 **함께 설치**된다.
- 그래서 필요한 건 딱 2가지:
  1. NVIDIA 그래픽 드라이버 (보통 이미 설치돼 있음)
  2. CUDA 버전 PyTorch

---

## 1단계: NVIDIA 드라이버 확인

명령 프롬프트(cmd)를 열고 입력:

```cmd
nvidia-smi
```

### 성공 시 이렇게 나옴
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 5xx.xx    Driver Version: 5xx.xx    CUDA Version: 12.x           |
|   NVIDIA GeForce RTX 4060 Laptop GPU ...                                    |
+-----------------------------------------------------------------------------+
```
- 오른쪽 위 `CUDA Version: 12.x` = 이 드라이버가 지원하는 **최대 CUDA 버전** (실제 설치된 게 아니라 "지원 가능" 표시)
- 이 숫자를 기억 → 나중에 PyTorch 버전 고를 때 사용

### `'nvidia-smi'은(는) 인식할 수 없는...` 에러가 나면
드라이버가 없거나 오래된 것. 아래에서 받아 설치 후 재부팅:
- https://www.nvidia.com/download/index.aspx
- Product Series: RTX 40 Series (Notebooks)
- Product: RTX 4060 Laptop GPU
- OS: Windows 11 (또는 10)

---

## 2단계: 미니콘다 설치

1. 다운로드: https://docs.conda.io/en/latest/miniconda.html (Windows 64-bit)
2. 설치 (옵션은 기본값대로 진행)
3. 설치 후 시작 메뉴에서 **"Anaconda Prompt"** 실행 (일반 cmd 아님!)

---

## 3단계: 전용 가상환경 만들기

Anaconda Prompt에서:

```cmd
conda create -n cv python=3.11 -y
conda activate cv
```

- `cv` = 환경 이름 (원하는 대로 가능)
- 성공하면 프롬프트 앞에 `(cv)`가 붙음: `(cv) C:\Users\...>`
- **실험할 때마다 항상 `conda activate cv` 먼저 실행!**

---

## 4단계: PyTorch (GPU 버전) 설치 ⭐

### 가장 정확한 방법: 공식 사이트에서 명령어 받기
https://pytorch.org/get-started/locally/ 에서 아래처럼 선택:

| 항목 | 선택값 |
|------|--------|
| PyTorch Build | Stable |
| Your OS | Windows |
| Package | Pip |
| Language | Python |
| Compute Platform | CUDA 12.x (1단계에서 본 숫자 이하로) |

→ 자동 생성된 명령어를 복사해서 실행 (버전이 계속 업데이트되므로 이게 제일 정확)

### 예시 명령어 (CUDA 12.6 기준)
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
- 드라이버가 CUDA 12.8을 지원하면 `cu126` → `cu128`

### ⚠️ 가장 흔한 실수
- 절대 그냥 `pip install torch` 만 치지 말 것! → **CPU 버전**이 깔려서 GPU를 못 씀
- 꼭 `--index-url https://download.pytorch.org/whl/cuXXX` 부분이 있어야 함

---

## 5단계: GPU 인식 확인 (제일 중요)

Anaconda Prompt에서 (환경 활성화된 상태로):

```cmd
python check_gpu.py
```

또는 직접 파이썬 셸에서:
```python
import torch
print("PyTorch 버전:", torch.__version__)
print("GPU 사용 가능?:", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0))
```

### 성공 시
```
PyTorch 버전: 2.x.x+cu126
GPU 사용 가능?: True          ← 이게 True여야 함!
GPU 이름: NVIDIA GeForce RTX 4060 Laptop GPU
```

### False가 나오면 (CPU 버전이 깔린 경우)
```cmd
pip uninstall torch torchvision torchaudio -y
```
→ 그다음 4단계의 `--index-url` 붙은 명령어로 재설치

---

## 전체 흐름 요약

```
1. nvidia-smi 로 드라이버 + 지원 CUDA 버전 확인
2. 미니콘다 설치
3. conda create -n cv python=3.11  →  conda activate cv
4. 공식 사이트 명령어로 PyTorch 설치 (--index-url 필수!)
5. torch.cuda.is_available() == True 확인
```

---

## 자주 쓰는 conda 명령어 치트시트

```cmd
conda create -n 이름 python=3.11 -y   # 환경 생성
conda activate 이름                    # 환경 켜기
conda deactivate                       # 환경 끄기
conda env list                         # 환경 목록 보기
conda remove -n 이름 --all             # 환경 삭제
pip list                               # 설치된 패키지 보기
```
