"""
24_kernel_analysis.py
==========================================
23번에서 학습된 ComaPDK 모델의 kernel 값을 수치화.

출력:
  - 위치별 center weight (w5) heatmap
  - 샘플 위치별 full 3x3 kernel 수치 테이블
  - kernel weight 분포 (9개 position별)
  - 수치 CSV 저장

Run:
  python 24_kernel_analysis.py
  python 24_kernel_analysis.py --res-dir ./res/paper_fig
"""

import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── ComaPDK 모델 (23번과 동일) ─────────────────────────────────

def radial_map(H, W, device="cpu"):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r = (gy**2 + gx**2).sqrt() / (2**0.5)
    return r, gy, gx

class ComaPDK(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._s0  = nn.Parameter(torch.tensor(_sp(0.40)))
        self._aps = nn.Parameter(torch.tensor(_sp(1.0)))
        self._ck  = nn.Parameter(torch.tensor(_sp(0.20)))
        self._l0  = nn.Parameter(torch.tensor(_sp(0.50)))
        self._al  = nn.Parameter(torch.tensor(_sp(0.30)))
        self._av  = nn.Parameter(torch.tensor(_sp(2.0)))
        r, gy, gx = radial_map(H, W)
        self.register_buffer("r",  r.contiguous())
        self.register_buffer("r2", r.pow(2).contiguous())
        self.register_buffer("gy", gy.contiguous())
        self.register_buffer("gx", gx.contiguous())
        self.register_buffer("coords", torch.tensor([
            [-1.,-1.],[-1.,0.],[-1.,1.],
            [ 0.,-1.],[ 0.,0.],[ 0.,1.],
            [ 1.,-1.],[ 1.,0.],[ 1.,1.]]))

    @property
    def sigma0(self):    return F.softplus(self._s0)
    @property
    def alpha_psf(self): return F.softplus(self._aps)
    @property
    def coma_k(self):    return F.softplus(self._ck)
    @property
    def lambda0(self):   return F.softplus(self._l0)
    @property
    def alpha_lam(self): return F.softplus(self._al)
    @property
    def alpha_vig(self): return F.softplus(self._av)

    def get_kernels(self):
        with torch.no_grad():
            vig_gain = 1.0 + self.alpha_vig * self.r2
            sigma = (self.sigma0 + self.alpha_psf * self.r2
                     ).unsqueeze(-1).clamp(0.1, 2.0)
            rs = self.r.clamp(1e-6)
            sy = (-self.coma_k * self.r2 * self.gy / rs).unsqueeze(-1)
            sx = (-self.coma_k * self.r2 * self.gx / rs).unsqueeze(-1)
            d2 = (self.coords**2).sum(-1)
            g1 = torch.exp(-d2/(2*sigma**2))
            g1 = g1/g1.sum(-1,keepdim=True)
            dy = self.coords[:,0].unsqueeze(0).unsqueeze(0) - sy
            dx = self.coords[:,1].unsqueeze(0).unsqueeze(0) - sx
            g2 = torch.exp(-(dy**2+dx**2)/(2*sigma**2))
            g2 = g2/g2.sum(-1,keepdim=True)
            w = (self.r*0.8).clamp(0,0.8).unsqueeze(-1)
            g = (1-w)*g1 + w*g2
            g = g/g.sum(-1,keepdim=True)
            delta = torch.zeros_like(g); delta[:,:,4]=1.0
            lam = (self.lambda0 + self.alpha_lam*self.r2
                   ).unsqueeze(-1).clamp(0.05,3.0)
            k = (delta + lam*(delta-g)) * vig_gain.unsqueeze(-1)
            return k.detach().cpu().numpy()  # (H,W,9)

    def forward(self, x):
        vig_gain = 1.0 + self.alpha_vig * self.r2
        sigma = (self.sigma0 + self.alpha_psf * self.r2
                 ).unsqueeze(-1).clamp(0.1, 2.0)
        rs = self.r.clamp(1e-6)
        sy = (-self.coma_k * self.r2 * self.gy / rs).unsqueeze(-1)
        sx = (-self.coma_k * self.r2 * self.gx / rs).unsqueeze(-1)
        d2 = (self.coords**2).sum(-1)
        g1 = torch.exp(-d2/(2*sigma**2))
        g1 = g1/g1.sum(-1,keepdim=True)
        dy = self.coords[:,0].unsqueeze(0).unsqueeze(0) - sy
        dx = self.coords[:,1].unsqueeze(0).unsqueeze(0) - sx
        g2 = torch.exp(-(dy**2+dx**2)/(2*sigma**2))
        g2 = g2/g2.sum(-1,keepdim=True)
        w = (self.r*0.8).clamp(0,0.8).unsqueeze(-1)
        g = (1-w)*g1 + w*g2
        g = g/g.sum(-1,keepdim=True)
        delta = torch.zeros_like(g); delta[:,:,4]=1.0
        lam = (self.lambda0 + self.alpha_lam*self.r2
               ).unsqueeze(-1).clamp(0.05,3.0)
        k = (delta + lam*(delta-g)) * vig_gain.unsqueeze(-1)
        return torch.nn.functional.conv2d(
            x, torch.zeros(1,1,3,3))  # placeholder


# ── 분석 함수 ──────────────────────────────────────────────────

def analyze_kernels(kernels_hwk, res_dir, H, W):
    """
    kernels_hwk: (H,W,9) numpy
    w 인덱스: 0=TL 1=TC 2=TR
              3=ML 4=MC 5=MR
              6=BL 7=BC 8=BR
              (4 = center)
    """
    os.makedirs(res_dir, exist_ok=True)

    # ── 1. 9개 weight 각각의 2D map ─────────────────────────────
    w_names = ['w₁(TL)','w₂(TC)','w₃(TR)',
               'w₄(ML)','w₅(C)','w₆(MR)',
               'w₇(BL)','w₈(BC)','w₉(BR)']

    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    for i in range(9):
        r, c = i//3, i%3
        ax = axes[r, c]
        wmap = kernels_hwk[:,:,i]
        vmax = max(abs(wmap).max(), 0.01)
        im = ax.imshow(wmap, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(w_names[i], fontsize=11)
        ax.set_xlabel('x (col)', fontsize=8)
        ax.set_ylabel('y (row)', fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    plt.suptitle('PDK correction kernel: weight maps (H×W)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(res_dir, '24_weight_maps_9ch.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {path}')

    # ── 2. Center weight (w5) 단독 heatmap + radial profile ─────
    w5 = kernels_hwk[:,:,4]
    fig = plt.figure(figsize=(12, 4.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.2, 1],
                           wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(w5, cmap='hot', aspect='auto')
    ax0.set_title('Center weight w₅ (heatmap)', fontsize=11)
    ax0.set_xlabel('x (col)', fontsize=9)
    ax0.set_ylabel('y (row)', fontsize=9)
    plt.colorbar(im0, ax=ax0, shrink=0.85, label='w₅ value')

    ax1 = fig.add_subplot(gs[1])
    center_row = w5[H//2, :]
    center_col = w5[:, W//2]
    ax1.plot(center_row, label='Horizontal (y=center)', color='#D85A30')
    ax1.plot(center_col, label='Vertical (x=center)',   color='#185FA5')
    ax1.set_title('w₅ radial profile', fontsize=11)
    ax1.set_xlabel('pixel position', fontsize=9)
    ax1.set_ylabel('w₅ value', fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[2])
    r_vals = np.linspace(-1, 1, H)
    r_map  = np.sqrt(r_vals[:,None]**2 + r_vals[None,:]**2) / np.sqrt(2)
    r_flat = r_map.ravel()
    w5_flat = w5.ravel()
    ax2.scatter(r_flat, w5_flat, s=0.3, alpha=0.2,
                color='#D85A30', rasterized=True)
    ax2.set_title('w₅ vs radial dist.', fontsize=11)
    ax2.set_xlabel('r (normalized)', fontsize=9)
    ax2.set_ylabel('w₅ value', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Center weight w₅ analysis', fontsize=12, y=1.02)
    path = os.path.join(res_dir, '24_center_weight_analysis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {path}')

    # ── 3. 샘플 위치별 full 3x3 수치 테이블 ─────────────────────
    sample_pts = [
        (0,       0,       'top-left (0,0)'),
        (0,       W//2,    'top-center (W/2,0)'),
        (0,       W-1,     'top-right (W-1,0)'),
        (H//2,    0,       'mid-left (0,H/2)'),
        (H//2,    W//2,    'center (W/2,H/2)'),
        (H//2,    W-1,     'mid-right (W-1,H/2)'),
        (H-1,     0,       'bot-left (0,H-1)'),
        (H-1,     W//2,    'bot-center (W/2,H-1)'),
        (H-1,     W-1,     'bot-right (W-1,H-1)'),
    ]

    n = len(sample_pts)
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    axes = axes.ravel()

    all_k = np.array([kernels_hwk[iy,ix] for (iy,ix,_) in sample_pts])
    vmax = max(abs(all_k).max(), 0.01)

    for idx, (iy, ix, label) in enumerate(sample_pts):
        ax = axes[idx]
        k3 = kernels_hwk[iy, ix].reshape(3, 3)
        im = ax.imshow(k3, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       interpolation='nearest', aspect='equal')
        # cell 테두리
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.tick_params(which='minor', length=0)
        ax.grid(which='minor', color='black', linewidth=1.0)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor('black'); s.set_linewidth(0.8)

        # 각 cell에 수치 표시
        for r in range(3):
            for c in range(3):
                val = k3[r, c]
                color = 'white' if abs(val) > vmax*0.5 else 'black'
                ax.text(c, r, f'{val:.3f}',
                        ha='center', va='center',
                        fontsize=8, color=color, fontweight='500')

        ax.set_title(label, fontsize=8, pad=4)
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)

    plt.suptitle('PDK kernel values at sampled positions',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(res_dir, '24_kernel_values_table.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {path}')

    # ── 4. 수치 CSV 저장 ─────────────────────────────────────────
    rows = []
    for iy in np.linspace(0, H-1, 9, dtype=int):
        for ix in np.linspace(0, W-1, 9, dtype=int):
            k = kernels_hwk[iy, ix]
            rows.append([iy, ix] + k.tolist())

    header = 'iy,ix,w1,w2,w3,w4,w5,w6,w7,w8,w9'
    arr = np.array(rows)
    csv_path = os.path.join(res_dir, '24_kernel_values.csv')
    np.savetxt(csv_path, arr, delimiter=',',
               header=header, comments='', fmt='%.6f')
    print(f'[Saved] {csv_path}')

    # ── 5. 터미널 출력 ────────────────────────────────────────────
    print('\n' + '='*60)
    print('PDK kernel summary')
    print('='*60)
    print(f'{"Position":<24} {"w5(center)":>10} {"sum":>8} {"max":>8} {"min":>8}')
    print('-'*60)
    for (iy, ix, label) in sample_pts:
        k = kernels_hwk[iy, ix]
        print(f'{label:<24} {k[4]:>10.4f} {k.sum():>8.4f}'
              f' {k.max():>8.4f} {k.min():>8.4f}')
    print('='*60)

    print('\nFull 3x3 at key positions:')
    for (iy, ix, label) in [
        sample_pts[0], sample_pts[4], sample_pts[8]]:
        k3 = kernels_hwk[iy, ix].reshape(3,3)
        print(f'\n  [{label}]')
        for row in k3:
            print('  ' + '  '.join(f'{v:7.4f}' for v in row))


# ── Main ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',     type=str,
                   default='./res/paper_fig/pdk_model.pt',
                   help='saved model state dict path')
    p.add_argument('--img-size', type=int, default=256)
    p.add_argument('--res-dir',  type=str,
                   default='./res/paper_fig')
    return p.parse_args()


def main():
    args = parse_args()
    H = W = args.img_size
    out_dir = os.path.join(args.res_dir, 'kernel_analysis')
    os.makedirs(out_dir, exist_ok=True)

    # 모델 로드 or 기본 파라미터로 재현
    model = ComaPDK(H, W)
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
        print(f'[Load] checkpoint: {args.ckpt}')
    else:
        print(f'[Warning] checkpoint not found: {args.ckpt}')
        print('  Using default-initialized model parameters.')
        print('  To use trained weights, save the model in 23_paper_figure_coma_vig.py:')
        print('  torch.save(pdk_model.state_dict(), "./res/paper_fig/pdk_model.pt")')

    model.eval()
    print(f'\n[Params]')
    print(f'  sigma0    = {model.sigma0.item():.4f}')
    print(f'  alpha_psf = {model.alpha_psf.item():.4f}')
    print(f'  coma_k    = {model.coma_k.item():.4f}')
    print(f'  lambda0   = {model.lambda0.item():.4f}')
    print(f'  alpha_lam = {model.alpha_lam.item():.4f}')
    print(f'  alpha_vig = {model.alpha_vig.item():.4f}')

    print('\n[Computing kernel map...]')
    kernels = model.get_kernels()  # (H,W,9)
    print(f'  shape: {kernels.shape}')
    print(f'  global min: {kernels.min():.4f}  max: {kernels.max():.4f}')

    analyze_kernels(kernels, out_dir, H, W)

    print(f'\n[Done] {os.path.abspath(out_dir)}')
    print('  24_weight_maps_9ch.png       -- 9개 weight 각각 heatmap')
    print('  24_center_weight_analysis.png -- w5 분석 (heatmap+profile)')
    print('  24_kernel_values_table.png   -- 9개 위치 full 3x3 수치')
    print('  24_kernel_values.csv         -- 수치 CSV')


if __name__ == '__main__':
    main()