"""
10_vignetting_example.py
========================
3×3 Position-Dependent Kernel을 이용한 Vignetting 시뮬레이션

구성:
  1. Raw (clean) 이미지 생성
  2. Vignetting degradation 적용
  3. 3×3 PD kernel로 보정 (두 가지 방식)
     A. 해석적: alpha를 알 때 (상한선)
     B. 학습 기반: alpha를 모를 때 (실용적)
  4. PSNR / SSIM 비교
  5. 결과 저장 (comparison.png, kernel_map.png)

핵심 아이디어:
  Vignetting:  I_deg(x,y) = I_raw(x,y) × V(r),    V(r) = 1/(1 + α·r²)
  3×3 kernel:  center weight = 1/V(r) = 1 + α·r²  (위치마다 다름)
               나머지 weight = 0
  → PSF blur와 동시 발생 시 3×3 전체를 활용 가능 (확장 용이)

Run:
  python 10_vignetting_example.py
  python 10_vignetting_example.py --alpha 3.0 --pattern checker
  python 10_vignetting_example.py --img-path /path/to/image.png
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================================
# 1. 유틸
# ============================================================================

def radial_map(H, W, device='cpu'):
    """
    각 픽셀의 정규화 반경 r을 반환.
    중심=0, 코너=1 (정규화)
    returns: (H, W)
    """
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    r = (gy**2 + gx**2).sqrt() / (2**0.5)  # [0, 1]
    return r


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR (dB). a, b: [0,1] float tensor."""
    mse = F.mse_loss(a.float(), b.float()).item()
    if mse < 1e-10:
        return 99.9
    return 10 * np.log10(1.0 / mse)


def to_np(t: torch.Tensor) -> np.ndarray:
    """(1,H,W) or (H,W) → (H,W) numpy [0,1]"""
    return t.squeeze().detach().cpu().float().numpy()


# ============================================================================
# 2. 이미지 로드 / 합성
# ============================================================================

def load_image(path: str, size: int) -> torch.Tensor:
    """이미지 파일 → (1,1,H,W) float [0,1]"""
    img = Image.open(path).convert('L')
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def make_pattern(pattern: str, size: int) -> torch.Tensor:
    """합성 패턴 생성 → (1,1,H,W) float [0,1]"""
    H = W = size
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')

    if pattern == 'checker':
        v = (0.5 + 0.5 * torch.sign(
            torch.sin(gy * 8) * torch.sin(gx * 8))).clamp(0, 1)
    elif pattern == 'gradient':
        v = ((gx + 1) / 2).clamp(0, 1)
    elif pattern == 'sine':
        v = (0.5 + 0.25 * torch.sin(gy * 12)
                 + 0.25 * torch.sin(gx * 12)).clamp(0, 1)
    elif pattern == 'circle':
        v = (0.5 + 0.5 * torch.cos(
            (gy**2 + gx**2).sqrt() * 10)).clamp(0, 1)
    else:
        raise ValueError(f'Unknown pattern: {pattern}')

    return v.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


# ============================================================================
# 3. Vignetting Degradation
# ============================================================================

def apply_vignetting(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Vignetting: I_deg = I_raw × V(r),  V(r) = 1/(1 + α·r²)

    x: (B,1,H,W)
    returns: (B,1,H,W)
    """
    B, C, H, W = x.shape
    r = radial_map(H, W, x.device)          # (H,W)
    r2 = r ** 2
    V = 1.0 / (1.0 + alpha * r2)            # (H,W) ∈ (0,1]
    return (x * V.unsqueeze(0).unsqueeze(0)).clamp(0, 1)


# ============================================================================
# 4. 3×3 PD Kernel 생성 및 적용
# ============================================================================

def make_3x3_vignetting_kernel(H: int, W: int,
                                alpha: float,
                                device='cpu') -> torch.Tensor:
    """
    Vignetting 보정용 3×3 PD kernel 생성.

    각 위치 (x,y)에서:
      center weight = 1/V(r) = 1 + α·r²   (vignetting 역보정)
      나머지 8개   = 0

    returns: (H, W, 9)  [kernel 순서: row-major 3×3 → 9개 flatten]
    """
    r = radial_map(H, W, device)
    r2 = r ** 2
    gain = 1.0 + alpha * r2                  # (H,W): 역vignetting gain

    # 3×3 kernel: center(index=4)만 gain, 나머지 0
    kernels = torch.zeros(H, W, 9, device=device)
    kernels[:, :, 4] = gain                  # center weight
    return kernels                            # (H, W, 9)


def spatially_varying_conv(x: torch.Tensor,
                            kernels: torch.Tensor) -> torch.Tensor:
    """
    Position-dependent 3×3 convolution.

    x:       (B, C, H, W)
    kernels: (H, W, 9)
    returns: (B, C, H, W)

    원리:
      F.unfold로 각 픽셀의 3×3 이웃을 추출 → (B, C×9, H×W)
      위치별 kernel weight와 내적 → (B, C, H, W)
    """
    B, C, H, W = x.shape

    # 3×3 이웃 추출: (B, C*9, H*W)
    patches = F.unfold(x, kernel_size=3, padding=1)  # (B, C*9, H*W)
    patches = patches.view(B, C, 9, H * W)           # (B, C, 9, H*W)

    # kernels: (H, W, 9) → (1, 1, 9, H*W)
    k = kernels.view(H * W, 9).T                      # (9, H*W)
    k = k.unsqueeze(0).unsqueeze(0)                   # (1, 1, 9, H*W)

    # weighted sum
    out = (patches * k).sum(dim=2)                    # (B, C, H*W)
    return out.view(B, C, H, W).clamp(0, 1)


# ============================================================================
# 5. Alpha 추정 (학습 기반 대용: 해석적 추정)
# ============================================================================

def estimate_alpha(deg: torch.Tensor) -> float:
    """
    Degraded 이미지에서 alpha를 추정.

    원리:
      V(r) = I_deg / I_raw ≈ I_deg / mean(I_deg_center)  (중심≈원본)
      V(r_edge) = 1/(1 + α·r²_edge)
      → α = (1/V_edge - 1) / r²_edge

    실제 학습 기반에서는 이 alpha를 네트워크가 찾음.
    """
    B, C, H, W = deg.shape
    r = radial_map(H, W, deg.device)

    # 중심 영역 (r < 0.15) 평균
    center_mask = (r < 0.15).float().unsqueeze(0).unsqueeze(0)
    I_center = (deg * center_mask).sum() / center_mask.sum().clamp(1)

    # 가장자리 영역 (r > 0.7) 평균
    edge_mask = (r > 0.7).float().unsqueeze(0).unsqueeze(0)
    I_edge = (deg * edge_mask).sum() / edge_mask.sum().clamp(1)
    r2_edge = (r * edge_mask).sum() / edge_mask.sum().clamp(1)
    r2_edge = r2_edge ** 2

    # V_edge = I_edge / I_center
    V_edge = (I_edge / I_center.clamp(1e-5)).clamp(1e-5, 1.0)

    # α = (1/V_edge - 1) / r²_edge
    alpha_est = ((1.0 / V_edge) - 1.0) / r2_edge.clamp(1e-5)
    return alpha_est.item()


# ============================================================================
# 6. 학습 기반 PD Kernel (Alpha를 파라미터로 학습)
# ============================================================================

class VignettingPDK(nn.Module):
    """
    3×3 PD kernel로 vignetting 보정.
    alpha를 학습 파라미터로 두고 gradient descent로 최적화.

    이것이 '학습 기반 PD kernel'의 가장 단순한 형태:
      - 파라미터: alpha 1개
      - kernel: center weight = 1 + alpha * r²  (differentiable)
      - loss: L1(output, clean)

    핵심: make_3x3_vignetting_kernel에 alpha.item()을 넘기면
          gradient가 끊김. forward에서 직접 differentiable하게 계산.
    """
    def __init__(self, H: int, W: int, alpha_init: float = 1.0):
        super().__init__()
        self.H = H
        self.W = W
        # alpha를 학습 파라미터로 (양수 제약: softplus)
        self._alpha_raw = nn.Parameter(
            torch.tensor(np.log(np.exp(alpha_init) - 1.0), dtype=torch.float32)
        )
        # r² map은 고정 (학습 대상 아님) → buffer로 등록
        r = radial_map(H, W)
        self.register_buffer('r2', r ** 2)   # (H, W)

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self._alpha_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # gain = 1 + alpha * r²  → differentiable w.r.t. alpha
        gain = 1.0 + self.alpha * self.r2          # (H, W)

        # 3×3 kernel: center(index=4) = gain, 나머지 = 0
        # zeros_like로 만들면 gradient 안 흐름 → gain으로 직접 구성
        zeros = torch.zeros(H, W, 9, device=x.device)
        # scatter로 center에만 gain 할당 (differentiable)
        center = gain.unsqueeze(-1)                # (H, W, 1)
        idx = torch.full((H, W, 1), 4,
                         dtype=torch.long, device=x.device)
        kernels = zeros.scatter(2, idx, center)    # (H, W, 9)

        return spatially_varying_conv(x, kernels)


def train_pdk(deg: torch.Tensor,
              clean: torch.Tensor,
              n_iter: int = 300,
              lr: float = 0.05) -> tuple:
    """
    Degraded → clean 복원을 목표로 alpha 학습.

    Loss: L1(output, clean)
    returns: (corrected, learned_alpha, loss_history)
    """
    B, C, H, W = deg.shape
    model = VignettingPDK(H, W, alpha_init=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    for i in range(n_iter):
        optimizer.zero_grad()
        out = model(deg)
        loss = F.l1_loss(out, clean)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if (i + 1) % 50 == 0:
            print(f'  iter {i+1:4d}/{n_iter}  '
                  f'loss={loss.item():.5f}  '
                  f'alpha={model.alpha.item():.4f}')

    with torch.no_grad():
        corrected = model(deg).clamp(0, 1)

    return corrected, model.alpha.item(), loss_history


# ============================================================================
# 7. 시각화
# ============================================================================

def save_comparison(raw, deg, cor_ana, cor_learn,
                    true_alpha, est_alpha, learn_alpha,
                    loss_history, res_dir, pattern):
    """
    comparison.png: raw / degraded / corrected(해석) / corrected(학습) 비교
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    psnr_deg   = psnr(raw, deg)
    psnr_ana   = psnr(raw, cor_ana)
    psnr_learn = psnr(raw, cor_learn)

    imgs = [
        (to_np(raw),       '1. Raw (clean)',           ''),
        (to_np(deg),       '2. Degraded',
         f'PSNR = {psnr_deg:.1f} dB   α = {true_alpha:.2f}'),
        (to_np(cor_ana),   '3. Corrected (analytic)',
         f'PSNR = {psnr_ana:.1f} dB   α_est = {est_alpha:.2f}'),
        (to_np(cor_learn), '4. Corrected (learned)',
         f'PSNR = {psnr_learn:.1f} dB   α_lrn = {learn_alpha:.2f}'),
    ]

    for ax, (img, title, subtitle) in zip(axes, imgs):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        if subtitle:
            ax.text(0.5, -0.06, subtitle,
                    transform=ax.transAxes,
                    ha='center', va='top',
                    fontsize=9, color='dimgray')
        ax.axis('off')

    fig.suptitle(f'Vignetting — 3×3 PD Kernel  [pattern={pattern}]',
                 fontsize=13, y=1.03)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(res_dir, 'comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {path}')


def _draw_3x3_kernel(ax, kernel_9, title, alpha_val, r_val):
    """
    3×3 kernel 값을 heatmap + 숫자로 시각화.
    kernel_9: (9,) array  [row-major: TL,TC,TR, ML,MC,MR, BL,BC,BR]
    """
    k = kernel_9.reshape(3, 3)
    vmax = max(abs(k.max()), abs(k.min()), 1e-6)
    im = ax.imshow(k, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='equal')
    for i in range(3):
        for j in range(3):
            val = k[i, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['-1', '0', '+1'], fontsize=9)
    ax.set_yticklabels(['-1', '0', '+1'], fontsize=9)
    ax.set_title(f'{title}', fontsize=10, pad=6)
    return im


def save_kernel_map(true_alpha, est_alpha, learn_alpha,
                    loss_history, size, res_dir):
    """
    kernel_map.png:
      행 1: 위치별 gain map (heatmap) + loss curve
      행 2: 주요 구역(center / mid / edge)별 3×3 kernel 값
    """
    H = W = size
    r_map = radial_map(H, W).numpy()   # (H, W)
    r2    = r_map ** 2

    # Gain map: 1 + α·r²
    gain_true  = 1.0 + true_alpha  * r2
    gain_est   = 1.0 + est_alpha   * r2
    gain_learn = 1.0 + learn_alpha * r2

    # ── 주요 구역 좌표 정의 ──
    # center(r≈0), mid(r≈0.5), edge(r≈0.85)
    zones = [
        ('center', 0.00),
        ('mid',    0.50),
        ('edge',   0.85),
    ]

    def kernel_at_r(alpha, r_val):
        """r 위치의 3×3 kernel (9개 값, center=gain 나머지=0)"""
        k = np.zeros(9)
        k[4] = 1.0 + alpha * r_val**2   # center weight
        return k

    # ── 레이아웃: 2행 × 4열 ──
    # 행1: gain_true / gain_est / gain_learn / loss
    # 행2: zone별 3×3 (true / analytic / learned) × 3 zones
    #       → 3zones × 3alphas = 9 subplots → 3열에 true/ana/lrn 묶기
    #   구체적으로: col0=center, col1=mid, col2=edge, col3=빈칸(legend)

    # ── 레이아웃: 4행 × 4열 ──
    # 행0: gain map × 3 + loss curve
    # 행1: zone center  — true / analytic / learned 3×3 kernel
    # 행2: zone mid     — true / analytic / learned 3×3 kernel
    # 행3: zone edge    — true / analytic / learned 3×3 kernel
    # (col3는 행1~3 병합해서 legend)

    fig = plt.figure(figsize=(18, 20))
    gs  = fig.add_gridspec(4, 4, hspace=0.6, wspace=0.4,
                           height_ratios=[1.4, 1, 1, 1])

    # ── 행 0: gain map ──
    vmax = gain_true.max()
    for col, (gain, label) in enumerate([
        (gain_true,  f'Gain map\ntrue a={true_alpha:.2f}'),
        (gain_est,   f'Gain map\nanalytic a={est_alpha:.2f}'),
        (gain_learn, f'Gain map\nlearned a={learn_alpha:.2f}'),
    ]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(gain, cmap='hot', vmin=1, vmax=vmax)
        ax.set_title(label, fontsize=11, pad=6)
        ax.axis('off')
        cx, cy = W / 2, H / 2
        for zname, zr in zones:
            px = cx + zr * (W / 2) * 0.72
            py = cy - zr * (H / 2) * 0.72
            ax.plot(px, py, 'o', markersize=8,
                    markerfacecolor='none', markeredgecolor='cyan',
                    markeredgewidth=2.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_loss = fig.add_subplot(gs[0, 3])
    ax_loss.plot(loss_history, color='steelblue', linewidth=1.5)
    ax_loss.set_title('Training loss (L1)', fontsize=11)
    ax_loss.set_xlabel('Iteration', fontsize=9)
    ax_loss.set_ylabel('L1 loss',   fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    # ── 행 1~3: zone별 3×3 kernel (각 zone이 한 행 차지) ──
    alphas = [
        (true_alpha,  'true'),
        (est_alpha,   'analytic'),
        (learn_alpha, 'learned'),
    ]

    for row, (zname, zr) in enumerate(zones):
        for col, (alpha_val, alabel) in enumerate(alphas):
            inner = gs[row + 1, col].subgridspec(1, 1)
            ax = fig.add_subplot(inner[0, 0])
            k9 = kernel_at_r(alpha_val, zr)
            _draw_3x3_kernel(ax, k9,
                             title=f'{alabel}  (a={alpha_val:.2f}, r={zr:.2f})',
                             alpha_val=alpha_val,
                             r_val=zr)
            if col == 0:
                ax.set_ylabel(f'zone: {zname}', fontsize=11,
                              labelpad=6, color='dimgray',
                              fontweight='bold')

    # col3 행1~3 병합: legend
    ax_leg = fig.add_subplot(gs[1:, 3])
    ax_leg.axis('off')
    legend_text = (
        '3x3 kernel structure\n\n'
        '[ 0    ][ 0    ][ 0    ]\n'
        '[ 0    ][ gain ][ 0    ]\n'
        '[ 0    ][ 0    ][ 0    ]\n\n'
        'gain = 1 + a * r^2\n\n'
        'Zone definition\n'
        '  center : r = 0.00\n'
        '  mid    : r = 0.50\n'
        '  edge   : r = 0.85\n\n'
        'cyan o = zone position\n\n'
        'Note:\n'
        '  vignetting-only case\n'
        '  -> only center(4)\n'
        '     weight is nonzero\n'
        '  PSF blur added\n'
        '  -> all 9 weights\n'
        '     become nonzero'
    )
    ax_leg.text(0.08, 0.95, legend_text,
                transform=ax_leg.transAxes,
                va='top', ha='left',
                fontsize=11, fontfamily='monospace',
                color='dimgray',
                bbox=dict(boxstyle='round,pad=0.7',
                          facecolor='#f5f5f5', edgecolor='#cccccc'))

    fig.suptitle('3x3 PD Kernel — gain map & zone kernel values',
                 fontsize=15, y=1.01)

    path = os.path.join(res_dir, 'kernel_map.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {path}')


def save_kernel_profile(true_alpha, est_alpha, learn_alpha, res_dir):
    """
    kernel_profile.png: r에 따른 center weight (gain) 1D 프로파일
    """
    r = np.linspace(0, 1, 200)
    r2 = r ** 2

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r, 1 + true_alpha  * r2, label=f'true  α={true_alpha:.2f}',
            color='black', linewidth=2)
    ax.plot(r, 1 + est_alpha   * r2, label=f'analytic α={est_alpha:.2f}',
            color='steelblue', linewidth=1.5, linestyle='--')
    ax.plot(r, 1 + learn_alpha * r2, label=f'learned α={learn_alpha:.2f}',
            color='tomato', linewidth=1.5, linestyle=':')
    ax.set_xlabel('r (normalized radius)')
    ax.set_ylabel('center weight = 1 + α·r²')
    ax.set_title('3×3 PD Kernel — center weight profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(res_dir, 'kernel_profile.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {path}')


# ============================================================================
# 8. Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='3×3 PD Kernel Vignetting 보정 예제')
    p.add_argument('--img-path',  type=str,   default='',
                   help='입력 이미지 경로 (비워두면 합성 패턴 사용)')
    p.add_argument('--pattern',   type=str,   default='checker',
                   choices=['checker', 'gradient', 'sine', 'circle'],
                   help='합성 패턴 종류 (default: checker)')
    p.add_argument('--img-size',  type=int,   default=128,
                   help='이미지 크기 (default: 128)')
    p.add_argument('--alpha',     type=float, default=2.5,
                   help='Vignetting 강도 α (default: 2.5)')
    p.add_argument('--n-iter',    type=int,   default=300,
                   help='학습 반복 횟수 (default: 300)')
    p.add_argument('--lr',        type=float, default=0.05,
                   help='학습률 (default: 0.05)')
    p.add_argument('--res-dir',   type=str,   default='./res/vignetting',
                   help='결과 저장 폴더 (default: ./res/vignetting)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.res_dir, exist_ok=True)

    # ── 이미지 로드 ──
    if args.img_path and os.path.isfile(args.img_path):
        raw = load_image(args.img_path, args.img_size)
        print(f'[Load] {args.img_path}')
    else:
        raw = make_pattern(args.pattern, args.img_size)
        print(f'[Load] synthetic pattern: {args.pattern}')

    print(f'[Config] size={args.img_size}  α={args.alpha}  '
          f'n_iter={args.n_iter}  lr={args.lr}')

    true_alpha = args.alpha

    # ── Step 1: Vignetting 적용 ──
    deg = apply_vignetting(raw, true_alpha)
    print(f'[Degrade] PSNR = {psnr(raw, deg):.2f} dB')

    # ── Step 2A: 해석적 보정 (alpha 추정 → 3×3 PD kernel) ──
    est_alpha = estimate_alpha(deg)
    print(f'[Analytic] estimated α = {est_alpha:.4f}  (true={true_alpha:.4f})')
    kernels_ana = make_3x3_vignetting_kernel(
        args.img_size, args.img_size, est_alpha)
    cor_ana = spatially_varying_conv(deg, kernels_ana)
    print(f'[Analytic] PSNR = {psnr(raw, cor_ana):.2f} dB')

    # ── Step 2B: 학습 기반 보정 (alpha를 gradient descent로 학습) ──
    print('[Learn] training PD kernel...')
    cor_learn, learn_alpha, loss_history = train_pdk(
        deg, raw, n_iter=args.n_iter, lr=args.lr)
    print(f'[Learn] learned α = {learn_alpha:.4f}  (true={true_alpha:.4f})')
    print(f'[Learn] PSNR = {psnr(raw, cor_learn):.2f} dB')

    # ── 저장 ──
    save_comparison(raw, deg, cor_ana, cor_learn,
                    true_alpha, est_alpha, learn_alpha,
                    loss_history, args.res_dir, args.pattern)
    save_kernel_map(true_alpha, est_alpha, learn_alpha,
                    loss_history, args.img_size, args.res_dir)
    save_kernel_profile(true_alpha, est_alpha, learn_alpha, args.res_dir)

    print(f'\n[Done] 결과 저장: {os.path.abspath(args.res_dir)}')
    print('  comparison.png    — raw / degraded / corrected 비교')
    print('  kernel_map.png    — gain map 시각화 + loss curve')
    print('  kernel_profile.png — r에 따른 center weight 프로파일')


if __name__ == '__main__':
    main()