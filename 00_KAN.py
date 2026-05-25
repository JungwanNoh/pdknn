"""
00_KAN.py
KAN (Kolmogorov-Arnold Network) 구조 시각화

01_hypernetwork_kan.py 의 KANHyperPDKConv2d 와 동일한 구조:
  (x, y) 좌표 → sinusoidal encoding (16D) → KAN([16, 32, 1]) → 출력

  * 실제 모델은 KAN([16, 32, 32]) + Linear decoder → kernel weight
  * 여기서는 시각화를 위해 출력을 1D로 단순화 (32D plot은 너무 커서 안 보임)

학습 함수: f(x, y) = sin(π·x) · cos(π·y)
  → 위치에 따라 smooth하게 변하는 함수 (실제 kernel 생성과 유사한 성질)

설치:
  pip install pykan matplotlib scikit-learn tqdm sympy pandas
"""

import torch
import math
import matplotlib.pyplot as plt
from kan import KAN


# ---------------------------------------------------------------------------
# 1. Sinusoidal Position Encoding  (01_hypernetwork_kan.py 와 동일)
# ---------------------------------------------------------------------------

def sinusoidal_pos_encoding(coords: torch.Tensor, dim: int) -> torch.Tensor:
    """
    coords : (N, 2)  — 정규화된 (i/H, j/W) ∈ [0, 1]
    dim    : 출력 차원 (4의 배수)
    return : (N, dim)

    dim=16 기준:
      y좌표: [sin(y·f0), cos(y·f0), sin(y·f1), cos(y·f1), ... ] → 8D
      x좌표: [sin(x·f0), cos(x·f0), sin(x·f1), cos(x·f1), ... ] → 8D
      freqs: [1/10000^(0/4), 1/10000^(2/4), 1/10000^(4/4), 1/10000^(6/4)]
    """
    assert dim % 4 == 0, "dim must be divisible by 4 (2 coords × 2 sin/cos)"
    half  = dim // 2
    freqs = torch.arange(half // 2, device=coords.device).float()
    freqs = 1.0 / (10000 ** (2 * freqs / half))   # (half//2,)

    enc_list = []
    for c in range(2):                              # y coord, x coord
        v = coords[:, c:c+1] * freqs.unsqueeze(0)  # (N, half//2)
        enc_list += [torch.sin(v), torch.cos(v)]

    return torch.cat(enc_list, dim=-1)              # (N, dim)


# ---------------------------------------------------------------------------
# 2. 학습 데이터 생성
#    입력  : sinusoidal_pos_encoding(x, y) → 16D   (01과 동일한 입력 파이프라인)
#    출력  : f(x, y) = sin(π·x) · cos(π·y) → 1D   (kernel 대표값 단순화)
# ---------------------------------------------------------------------------

def make_dataset(n_train: int = 1000, n_test: int = 200,
                 pos_enc_dim: int = 16, seed: int = 42):
    torch.manual_seed(seed)

    def f(xy):
        x, y = xy[:, 0], xy[:, 1]
        return (torch.sin(torch.pi * x) * torch.cos(torch.pi * y)).unsqueeze(-1)

    # 랜덤 위치 좌표 샘플링 ∈ [0, 1]²
    train_coords = torch.rand(n_train, 2)
    test_coords  = torch.rand(n_test,  2)

    # sinusoidal encoding  (01_hypernetwork_kan.py 와 동일한 전처리)
    train_enc = sinusoidal_pos_encoding(train_coords, pos_enc_dim)
    test_enc  = sinusoidal_pos_encoding(test_coords,  pos_enc_dim)

    # KAN grid 범위 [-1, 1] 에 맞게 정규화  (01과 동일)
    mean = train_enc.mean()
    std  = train_enc.std() + 1e-6
    train_enc = (train_enc - mean) / std
    test_enc  = (test_enc  - mean) / std

    return {
        'train_input': train_enc,             # (N, 16)
        'train_label': f(train_coords),       # (N, 1)
        'test_input' : test_enc,              # (M, 16)
        'test_label' : f(test_coords),        # (M, 1)
    }


# ---------------------------------------------------------------------------
# 3. KAN 모델 정의 및 학습
# ---------------------------------------------------------------------------

POS_ENC_DIM = 16   # 01_hypernetwork_kan.py: pos_enc_dim=16
KAN_HIDDEN  = 32   # 01_hypernetwork_kan.py: kan_hidden=32
# 실제 모델 출력: kan_latent=32  →  여기선 시각화를 위해 1로 단순화
KAN_OUT     = 1


def run():
    dataset = make_dataset(pos_enc_dim=POS_ENC_DIM)

    # ── 01_hypernetwork_kan.py 와 동일한 KAN 구조 ──
    # 실제: KAN([16, 32, 32]) + Linear(32 → C_out*C_in*kH*kW)
    # 데모: KAN([16, 32,  1])          ← 출력만 1D로 단순화 (시각화용)
    model = KAN(width=[POS_ENC_DIM, KAN_HIDDEN, KAN_OUT], grid=5, k=3, seed=42)

    n_splines_l1 = POS_ENC_DIM * KAN_HIDDEN   # 16 × 32 = 512
    n_splines_l2 = KAN_HIDDEN  * KAN_OUT      # 32 ×  1 =  32
    n_basis      = 5 + 3                       # grid_size + spline_order

    print("=" * 60)
    print("KAN 구조  (01_hypernetwork_kan.py 와 동일한 입력 파이프라인)")
    print(f"  입력 파이프라인 : (x,y) 2D → sinusoidal enc → {POS_ENC_DIM}D")
    print(f"  Layer 0→1 : {POS_ENC_DIM} 입력 × {KAN_HIDDEN} 노드 "
          f"= {n_splines_l1}개의 독립 B-spline 함수")
    print(f"  Layer 1→2 : {KAN_HIDDEN} 노드 × {KAN_OUT} 출력  "
          f"= {n_splines_l2}개의 독립 B-spline 함수")
    print(f"  총 spline 수    : {n_splines_l1 + n_splines_l2}개 "
          f"(각 {n_basis}개 계수 → 총 {(n_splines_l1+n_splines_l2)*n_basis:,}개)")
    print(f"  총 파라미터     : {sum(p.numel() for p in model.parameters()):,}")
    print()
    print("  ※ 실제 모델과의 차이")
    print(f"     실제 KAN 출력 : {KAN_HIDDEN}D  → Linear({KAN_HIDDEN}→C_out×C_in×k×k)")
    print(f"     데모 KAN 출력 : {KAN_OUT}D     (시각화 가능 크기로 단순화)")
    print("=" * 60)

    # --- [1단계] 5 step 만 학습 후 초기 구조 시각화 ---
    print("\n[1단계] 초기 구조 시각화 (5 steps)...")
    model.fit(dataset, opt="LBFGS", steps=5)
    model.plot(beta=3, title=f"KAN([{POS_ENC_DIM}, {KAN_HIDDEN}, {KAN_OUT}]) - Early (5 steps)")
    plt.savefig("kan_before_training.png", dpi=150, bbox_inches='tight')
    plt.close('all')
    print("  → kan_before_training.png 저장 완료")

    # --- [2단계] 추가 학습 (총 55 steps) ---
    print(f"\n[2단계] KAN 추가 학습 중 (LBFGS, 50 steps)...")
    results = model.fit(
        dataset,
        opt="LBFGS",
        steps=50,
        lamb=0.001,        # L1 spline sparsity
        lamb_entropy=2.0,  # 활성 spline 간소화
    )

    # --- [3단계] 학습 후 시각화 ---
    print("\n[3단계] 학습 후 KAN 구조 시각화...")
    model.plot(beta=3, title=f"KAN([{POS_ENC_DIM}, {KAN_HIDDEN}, {KAN_OUT}]) - After Training")
    plt.savefig("kan_after_training.png", dpi=150, bbox_inches='tight')
    plt.close('all')
    print("  → kan_after_training.png 저장 완료")

    # --- [4단계] Loss curve ---
    print("\n[4단계] Loss curve 저장...")
    plt.figure(figsize=(6, 4))
    plt.plot(results['train_loss'], label='train loss')
    plt.plot(results['test_loss'],  label='test loss')
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('RMSE')
    plt.title(f'KAN([{POS_ENC_DIM}, {KAN_HIDDEN}, {KAN_OUT}]) Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("kan_loss_curve.png", dpi=150)
    plt.close('all')
    print("  → kan_loss_curve.png 저장 완료")

    # --- [5단계] 심볼릭 추론 ---
    print("\n[5단계] 심볼릭 함수 추론 (auto_symbolic)...")
    try:
        model.auto_symbolic(lib=['sin', 'cos', 'exp', 'x^2', 'x'])
        model.plot(beta=3, title="KAN - Symbolic")
        plt.savefig("kan_symbolic.png", dpi=150, bbox_inches='tight')
        plt.close('all')
        print("  → kan_symbolic.png 저장 완료")
        print("  추론된 수식:", model.symbolic_formula()[0][0])
    except Exception as e:
        print(f"  심볼릭 추론 실패 (무시 가능): {e}")

    # --- 최종 성능 ---
    print("\n" + "=" * 60)
    print("최종 결과")
    print(f"  train RMSE : {results['train_loss'][-1]:.6f}")
    print(f"  test  RMSE : {results['test_loss'][-1]:.6f}")
    print("=" * 60)


if __name__ == '__main__':
    run()
