"""
LPAA (Local Perturbation Augmentation Attack) 구현
논문 "Improving Adversarial Transferability with Local Perturbation Augmentation"
(CVPR) 의 Algorithm 1 을 재현합니다.

구성 요소:
  1) LPA (Local Perturbation Augmentation) — Eq.(3)~(6)
     - 랜덤 이진 마스크 M (각 원소가 확률 p로 0) 으로 섭동의 일부만 사용
     - 증강계수 β 로 마스킹된 섭동을 스케일해 탐색 범위 확장
     - N 개의 독립 subspace 에서 구한 그래디언트를 평균
  2) PI (Perturbation Initialization) — Eq.(7),(8)
     - GMI_PI: GMI(Global Momentum Initialization, [38]) 로 선견지명 있는 모멘텀 확보
       · pre_attack 번의 사전 수렴 + 증폭계수 amp(=10)로 global search
       · 사전 공격 후 입력은 원본으로 리셋하되 누적 모멘텀은 유지
       (Omenzychen/Global-Momentum-Initialization 공식 구현 방식 재현)
     - LPA_PI: 그 위에 LPA 연산을 한 번 적용해 초기 섭동 정제
  3) 모멘텀 업데이트 (MI-FGSM 토대) — Algorithm 1 line 13~14

논문 파라미터(4.1.4): T=10, ε=16/255, α=1.6/255, μ=1.0,
                      N=20, p=0.95, β=35, η=3, LPA_PI iters=5, α_LPAPI=3.2/255
"""

import torch
import torch.nn as nn


def _loss_grad(model, x_input, y, normalize):
    """정규화 적용 후 손실의 입력에 대한 그래디언트."""
    x_input = x_input.clone().detach().requires_grad_(True)
    logits = model(normalize(x_input))
    loss = nn.functional.cross_entropy(logits, y)
    return torch.autograd.grad(loss, x_input)[0]


def _beta_schedule(t, steps, beta, beta_start, beta_end):
    """
    iteration t 에서 사용할 증강계수 β 를 돌려준다.

    - beta_start/beta_end 가 None 이면 고정 β (논문 방식).
    - 둘 다 주어지면 선형 감소(linear decay):
        초반(t=0)엔 beta_start 로 크게 탐색,
        후반(t=steps-1)엔 beta_end 로 작게 수렴.
      β(t) = beta_start - (beta_start - beta_end) * t / (steps - 1)
    """
    if beta_start is None or beta_end is None:
        return beta
    if steps <= 1:
        return beta_start
    frac = t / (steps - 1)
    return beta_start - (beta_start - beta_end) * frac


def _lpa_gradient(model, x, delta, y, normalize, N, p, beta):
    """
    LPA 평균 그래디언트 — Eq.(5),(6).
    N개의 랜덤 마스크 subspace 에서 그래디언트를 구해 평균.

      g_i = ∇_x L( f( x + (delta ⊙ M_i) · β ), y )
      g_bar = (1/N) Σ g_i
    """
    g_sum = torch.zeros_like(x)
    for _ in range(N):
        # 각 원소가 확률 p 로 0, (1-p) 로 1 인 이진 마스크
        mask = (torch.rand_like(delta) > p).float()
        perturbed = x + (delta * mask) * beta
        perturbed = torch.clamp(perturbed, 0.0, 1.0)
        g_sum += _loss_grad(model, perturbed, y, normalize)
    return g_sum / N


def _gmi_momentum(model, x, y, normalize, alpha, mu, pre_attack, amp):
    """
    GMI (Global Momentum Initialization) — 논문 [38], 공식 구현 방식 재현.
    (Omenzychen/Global-Momentum-Initialization 의 핵심 로직)

    핵심 아이디어:
      - 본 공격 전에 'pre_attack' 번의 사전 수렴(pre-convergence)을 수행
      - 사전 공격 단계에서는 증폭계수 amp(=10)로 '크게' 움직이며 global search
        (약점 방향을 넓게 탐색해 gradient elimination 완화)
      - 사전 공격이 끝나면 입력을 '원본으로 리셋'하되, 누적된 모멘텀(grad)은 유지
      - 이렇게 얻은 '선견지명 있는 모멘텀'을 본 공격의 출발 모멘텀으로 사용

    반환: 사전 수렴으로 얻은 모멘텀 g (본 공격의 초기 모멘텀)
    """
    g = torch.zeros_like(x)
    x_cur = x.clone()
    for i in range(pre_attack):
        grad = _loss_grad(model, torch.clamp(x_cur, 0, 1), y, normalize)
        grad = grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-12)
        g = mu * g + grad
        # pre_attack 단계에서는 증폭계수 amp 로 크게 이동 (global search)
        x_cur = x_cur + amp * alpha * g.sign()
    # 사전 공격 후 입력은 버리고(원본으로 리셋됨) 모멘텀 g 만 반환
    return g.detach()


def _perturbation_init(model, x, y, normalize, epsilon, alpha, eta,
                       N, p, beta, mu, lpapi_iters, lpapi_alpha,
                       pre_attack=5, amp=10.0):
    """
    PI (Perturbation Initialization) — Eq.(7),(8) + GMI([38]).

    1) GMI_PI: GMI 로 '선견지명 있는 모멘텀'을 얻어, 그 방향으로 한 걸음 이동한 뒤
               [-η·α, η·α] 범위로 투영 -> 초기 섭동 delta_M (Eq.7)
    2) LPA_PI: delta_M 에서 LPA 그래디언트로 한 번 더 정제 -> delta_0 (Eq.8)
    """
    proj = eta * alpha

    # ---- GMI_PI: 사전 수렴으로 모멘텀 확보 후 초기 섭동 생성 ----
    g_mom = _gmi_momentum(model, x, y, normalize, alpha, mu, pre_attack, amp)
    delta_M = lpapi_alpha * g_mom.sign()
    delta_M = torch.clamp(delta_M, -proj, proj)  # [-η·α, η·α] 투영 (Eq.7)

    # ---- LPA_PI: delta_M 에서 LPA 그래디언트로 정제 (Eq.8) ----
    g_lpa = _lpa_gradient(model, x, delta_M, y, normalize, N, p, beta)
    delta_0 = delta_M + lpapi_alpha * g_lpa.sign()
    delta_0 = torch.clamp(delta_0, -proj, proj)
    return delta_0.detach()


def lpaa_attack(model, x, y, normalize, *,
                epsilon=16 / 255, steps=10, alpha=1.6 / 255, mu=1.0,
                N=20, p=0.95, beta=35.0, eta=3,
                lpapi_iters=5, lpapi_alpha=3.2 / 255,
                pre_attack=5, amp=10.0,
                beta_start=None, beta_end=None,
                use_init=True):
    """
    LPAA 전체 공격 — Algorithm 1.

    매개변수:
        epsilon : 섭동 예산 (||δ||∞ ≤ ε)
        steps   : 반복 횟수 T
        alpha   : 스텝 크기 α
        mu      : 모멘텀 감쇠 μ
        N       : subspace 개수
        p       : 마스킹 비율 (원소가 0이 될 확률)
        beta    : 증강 계수 (고정값, 논문 방식)
        beta_start/beta_end: 둘 다 지정하면 β를 선형 감소시킴
                  (초반 beta_start로 크게 탐색 → 후반 beta_end로 작게 수렴).
                  [개선 아이디어] 공정 비교를 위해 평균을 논문값(35)에 맞추길 권장.
        eta     : 초기화 투영 범위 상한 계수
        use_init: PI(초기화 전략) 사용 여부
    """
    model.eval()
    x = x.clone().detach()

    # 초기 섭동 (PI 전략)
    if use_init:
        delta = _perturbation_init(
            model, x, y, normalize, epsilon, alpha, eta,
            N, p, beta, mu, lpapi_iters, lpapi_alpha,
            pre_attack=pre_attack, amp=amp)
    else:
        delta = torch.zeros_like(x)

    g = torch.zeros_like(x)  # 모멘텀

    for t in range(steps):
        # Algorithm 1 line 10~12: 첫 반복에서는 delta 를 0 으로 리셋
        delta_for_grad = torch.zeros_like(delta) if t == 0 else delta

        # 이번 iteration 의 β (스케줄링 적용 시 t에 따라 변함)
        beta_t = _beta_schedule(t, steps, beta, beta_start, beta_end)

        # LPA 평균 그래디언트 — Eq.(6)
        g_lpa = _lpa_gradient(model, x, delta_for_grad, y, normalize, N, p, beta_t)

        # 모멘텀 업데이트 (line 13): m_t = μ·m_{t-1} + g/||g||_1
        g_lpa = g_lpa / g_lpa.abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-12)
        g = mu * g + g_lpa

        # 섭동 업데이트 (line 14): δ_t = δ_{t-1} + α·sign(m_t)
        delta = delta + alpha * g.sign()
        delta = torch.clamp(delta, -epsilon, epsilon)  # ε 제약

    x_adv = torch.clamp(x + delta, 0.0, 1.0)
    return x_adv.detach()
