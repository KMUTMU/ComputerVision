"""
GPU 세팅 확인용 스크립트
사용법: Anaconda Prompt에서 환경 활성화 후 ->  python check_gpu.py
"""

import torch


def main():
    print("=" * 50)
    print("PyTorch GPU 환경 확인")
    print("=" * 50)

    print(f"PyTorch 버전       : {torch.__version__}")

    cuda_ok = torch.cuda.is_available()
    print(f"GPU 사용 가능?      : {cuda_ok}")

    if cuda_ok:
        print(f"GPU 개수           : {torch.cuda.device_count()}")
        print(f"GPU 이름           : {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전(PyTorch) : {torch.version.cuda}")

        # VRAM 정보
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024 ** 3)
        print(f"VRAM 총 용량       : {total_vram:.1f} GB")

        # 간단한 GPU 연산 테스트
        print("-" * 50)
        print("GPU 연산 테스트 중...")
        x = torch.rand(1000, 1000, device="cuda")
        y = torch.rand(1000, 1000, device="cuda")
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        print(f"행렬곱 결과 shape  : {tuple(z.shape)}  (device: {z.device})")
        print("✅ GPU가 정상 작동합니다! 실험 준비 완료 🎉")
    else:
        print("-" * 50)
        print("❌ GPU를 사용할 수 없습니다.")
        print("   원인: 대부분 CPU 버전 PyTorch가 설치된 경우입니다.")
        print("   해결:")
        print("     1) pip uninstall torch torchvision torchaudio -y")
        print("     2) 공식 사이트의 --index-url 붙은 명령어로 재설치")
        print("        예: pip install torch torchvision torchaudio \\")
        print("            --index-url https://download.pytorch.org/whl/cu126")

    print("=" * 50)


if __name__ == "__main__":
    main()
