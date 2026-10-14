"""
적대적 공격(Adversarial Attack) 구현
논문 "Improving Adversarial Transferability with Local Perturbation Augmentation"
의 가장 기본이 되는 FGSM / I-FGSM / MI-FGSM 공격.

- 레벨 1 목표: 적대적 예제가 무엇인지, 어떻게 만드는지 이해.
- 모든 공격은 L-infinity 제약 (||δ||∞ ≤ ε) 을 만족하도록 clamp 합니다.
"""

import torch
import torch.nn as nn


def _clamp_perturbation(adv, x, epsilon):
    """
    1) 섭동(perturbation)이 ε 범위를 넘지 않게 자르고
    2) 이미지가 유효 범위[0,1]를 벗어나지 않게 자른다.
    (논문의 ||δ||∞ ≤ ε 제약을 코드로 구현한 부분)
    """
    delta = torch.clamp(adv - x, min=-epsilon, max=epsilon)
    return torch.clamp(x + delta, min=0.0, max=1.0)


def fgsm(model, x, y, epsilon, normalize):
    """
    FGSM (Fast Gradient Sign Method) - 논문 참고문헌 [14]
    한 번(single step)만에 공격을 만든다.

        x_adv = x + ε · sign( ∇ₓ L(f(x), y) )

    가장 기본적인 공격. "손실을 키우는 방향으로 한 걸음" 간다.
    """
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)

    # 모델은 정규화된 입력을 기대하므로 forward 전에 normalize 적용
    logits = model(normalize(x_adv))
    loss = nn.functional.cross_entropy(logits, y)

    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv.detach() + epsilon * grad.sign()
    return _clamp_perturbation(x_adv, x, epsilon)


def i_fgsm(model, x, y, epsilon, steps, alpha, normalize, momentum=0.0):
    """
    I-FGSM (Iterative FGSM) - 논문 참고문헌 [19]
    FGSM 을 여러 번(iterative) 반복한다. 논문 Eq.(1):

        δ_t = δ_{t-1} + α · sign( ∇ₓ L(f(x+δ_{t-1}), y) )

    momentum > 0 이면 MI-FGSM (논문 참고문헌 [7]) 이 된다:
    기울기에 관성(momentum)을 더해 업데이트 방향을 안정화한다.
    (이게 논문 baseline들의 공통 토대이고, LPAA도 이 위에 쌓임)

    매개변수:
        epsilon : 전체 섭동 예산 (논문 ε = 16/255)
        steps   : 반복 횟수 T    (논문 T = 10)
        alpha   : 스텝 크기 α    (논문 α = 1.6/255)
        momentum: 관성 계수 μ    (MI-FGSM, 논문 μ = 1.0)
    """
    model.eval()
    x_adv = x.clone().detach()
    g = torch.zeros_like(x)  # 누적 모멘텀

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(normalize(x_adv))
        loss = nn.functional.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        if momentum > 0:
            # 기울기를 L1 정규화 후 모멘텀 누적 (MI-FGSM 방식)
            grad = grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-12)
            g = momentum * g + grad
            step_dir = g.sign()
        else:
            step_dir = grad.sign()

        x_adv = x_adv.detach() + alpha * step_dir
        x_adv = _clamp_perturbation(x_adv, x, epsilon)

    return x_adv.detach()


# 공격 이름 -> 함수 매핑 (run_attack.py 에서 사용)
# use_init: LPAA의 초기화(PI) 사용 여부 (검증 3용). LPAA 외 공격엔 영향 없음.
def build_attack(name, epsilon, steps, alpha, momentum, use_init=True):
    name = name.lower()
    if name == "fgsm":
        return lambda model, x, y, normalize: fgsm(model, x, y, epsilon, normalize)
    elif name == "ifgsm":
        return lambda model, x, y, normalize: i_fgsm(
            model, x, y, epsilon, steps, alpha, normalize, momentum=0.0)
    elif name == "mifgsm":
        return lambda model, x, y, normalize: i_fgsm(
            model, x, y, epsilon, steps, alpha, normalize, momentum=momentum)
    elif name == "lpaa":
        from lpaa import lpaa_attack
        return lambda model, x, y, normalize: lpaa_attack(
            model, x, y, normalize,
            epsilon=epsilon, steps=steps, alpha=alpha, mu=momentum,
            use_init=use_init)
    else:
        raise ValueError(
            f"알 수 없는 공격: {name} (fgsm/ifgsm/mifgsm/lpaa 중 선택)")
