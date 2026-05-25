"""
10_vignetting_bsd500.py
========================================
3x3 Position-Dependent Kernel Vignetting correction — BSD500

Dataset: BSD500
  CIFAR-10 : torchvision auto-download (32x32)
  BSD500   : standard restoration benchmark (256x256)
             https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/
             fallback: skimage built-in images
  DIV2K    : high-resolution SR benchmark (512x512 crop)
             https://data.vision.ee.ethz.ch/cvl/DIV2K/
             fallback: synthetic patterns

Run:
  python 10_vignetting_bsd500.py
  python 10_vignetting_bsd500.py --n-images 4 --alpha 2.5
  python 10_vignetting_bsd500.py --data-dir /path/to/BSD500
"""
import argparse
import os
import glob
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
# Utilities
# ============================================================================

def radial_map(H, W, device='cpu'):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return (gy**2 + gx**2).sqrt() / (2**0.5)


def psnr(a, b):
    mse = F.mse_loss(a.float(), b.float()).item()
    return 99.9 if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def to_np(t):
    return t.squeeze().detach().cpu().float().numpy()


def load_single_image(path, size):
    img = Image.open(path).convert('L')
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def load_images_from_dir(img_dir, n, size):
    exts = ('*.png','*.jpg','*.jpeg','*.PNG','*.JPG','*.JPEG')
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(img_dir, '**', ext), recursive=True)
        paths += glob.glob(os.path.join(img_dir, ext))
    paths = sorted(set(paths))[:n*5]
    imgs = []
    for p in paths:
        try:
            imgs.append(load_single_image(p, size))
        except Exception:
            continue
        if len(imgs) == n:
            break
    return imgs

# ============================================================================
# Degradation
# ============================================================================

def apply_vignetting(x, alpha):
    B, C, H, W = x.shape
    r2 = radial_map(H, W, x.device) ** 2
    V  = 1.0 / (1.0 + alpha * r2)
    return (x * V.unsqueeze(0).unsqueeze(0)).clamp(0, 1)


# ============================================================================
# 3x3 PD Kernel
# ============================================================================

def make_3x3_vignetting_kernel(H, W, alpha, device='cpu'):
    r2   = radial_map(H, W, device) ** 2
    gain = 1.0 + alpha * r2
    k    = torch.zeros(H, W, 9, device=device)
    k[:, :, 4] = gain
    return k


def spatially_varying_conv(x, kernels):
    B, C, H, W = x.shape
    patches = F.unfold(x, kernel_size=3, padding=1).view(B, C, 9, H*W)
    k = kernels.view(H*W, 9).T.unsqueeze(0).unsqueeze(0)
    return (patches * k).sum(dim=2).view(B, C, H, W).clamp(0, 1)


# ============================================================================
# Alpha estimation
# ============================================================================

def estimate_alpha(deg):
    B, C, H, W = deg.shape
    r = radial_map(H, W, deg.device)
    cm = (r < 0.15).float().unsqueeze(0).unsqueeze(0)
    em = (r > 0.70).float().unsqueeze(0).unsqueeze(0)
    Ic = (deg * cm).sum() / cm.sum().clamp(1)
    Ie = (deg * em).sum() / em.sum().clamp(1)
    r2e = ((r * em).sum() / em.sum().clamp(1)) ** 2
    Ve  = (Ie / Ic.clamp(1e-5)).clamp(1e-5, 1.0)
    return ((1.0 / Ve) - 1.0).item() / r2e.clamp(1e-5).item()


# ============================================================================
# Learning-based PDK
# ============================================================================

class VignettingPDK(nn.Module):
    def __init__(self, H, W, alpha_init=1.0):
        super().__init__()
        self.H, self.W = H, W
        self._alpha_raw = nn.Parameter(
            torch.tensor(np.log(np.exp(alpha_init) - 1.0), dtype=torch.float32))
        self.register_buffer('r2', radial_map(H, W) ** 2)

    @property
    def alpha(self):
        return F.softplus(self._alpha_raw)

    def forward(self, x):
        B, C, H, W = x.shape
        gain   = 1.0 + self.alpha * self.r2
        zeros  = torch.zeros(H, W, 9, device=x.device)
        idx    = torch.full((H, W, 1), 4, dtype=torch.long, device=x.device)
        kernels = zeros.scatter(2, idx, gain.unsqueeze(-1))
        return spatially_varying_conv(x, kernels)


def train_pdk(deg, clean, n_iter=300, lr=0.05):
    B, C, H, W = deg.shape
    model = VignettingPDK(H, W, alpha_init=1.0)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for i in range(n_iter):
        opt.zero_grad()
        loss = F.l1_loss(model(deg), clean)
        loss.backward()
        opt.step()
        history.append(loss.item())
        if (i+1) % 50 == 0:
            print(f'  iter {i+1:4d}/{n_iter}  loss={loss.item():.5f}  ' +
                  f'alpha={model.alpha.item():.4f}')
    with torch.no_grad():
        cor = model(deg).clamp(0, 1)
    return cor, model.alpha.item(), history


# ============================================================================
# Visualization helpers
# ============================================================================

def _draw_3x3_kernel(ax, k9, title):
    k    = k9.reshape(3, 3)
    vmax = max(abs(k.max()), abs(k.min()), 1e-6)
    ax.imshow(k, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    for i in range(3):
        for j in range(3):
            val   = k[i, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['-1','0','+1'], fontsize=9)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(['-1','0','+1'], fontsize=9)
    ax.set_title(title, fontsize=10, pad=6)


def save_comparison(raw_list, deg_list, cor_ana_list, cor_learn_list,
                    true_alpha, est_alpha, learn_alpha, res_dir, tag):
    n = len(raw_list)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n + 1))
    if n == 1:
        axes = axes[np.newaxis, :]

    cols = [
        (raw_list,       '1. Raw (clean)',         ''),
        (deg_list,       '2. Degraded',
         f'PSNR={psnr(raw_list[0], deg_list[0]):.1f}dB  a={true_alpha:.2f}'),
        (cor_ana_list,   '3. Corrected (analytic)',
         f'PSNR={psnr(raw_list[0], cor_ana_list[0]):.1f}dB  a_est={est_alpha:.2f}'),
        (cor_learn_list, '4. Corrected (learned)',
         f'PSNR={psnr(raw_list[0], cor_learn_list[0]):.1f}dB  a_lrn={learn_alpha:.2f}'),
    ]
    for c, (lst, title, subtitle) in enumerate(cols):
        axes[0, c].set_title(title, fontsize=11, fontweight='bold', pad=8)
        for r in range(n):
            axes[r, c].imshow(to_np(lst[r]), cmap='gray', vmin=0, vmax=1)
            axes[r, c].axis('off')
        if subtitle:
            axes[0, c].text(0.5, -0.04, subtitle,
                            transform=axes[0, c].transAxes,
                            ha='center', va='top', fontsize=9, color='dimgray')

    fig.suptitle(f'Vignetting 3x3 PD Kernel [{tag}]', fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(res_dir, 'comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Saved] {path}')


def save_kernel_map(true_alpha, est_alpha, learn_alpha,
                    loss_history, size, res_dir):
    H = W = size
    r2   = radial_map(H, W).numpy() ** 2
    zones = [('center', 0.00), ('mid', 0.50), ('edge', 0.85)]

    def k_at(a, r): k=np.zeros(9); k[4]=1.0+a*r**2; return k

    fig = plt.figure(figsize=(18, 20))
    gs  = fig.add_gridspec(4, 4, hspace=0.6, wspace=0.4,
                           height_ratios=[1.4, 1, 1, 1])

    gain_t = 1.0 + true_alpha  * r2
    gain_e = 1.0 + est_alpha   * r2
    gain_l = 1.0 + learn_alpha * r2
    vmax   = gain_t.max()

    for col, (gain, lbl) in enumerate([
        (gain_t, f'Gain map\ntrue a={true_alpha:.2f}'),
        (gain_e, f'Gain map\nanalytic a={est_alpha:.2f}'),
        (gain_l, f'Gain map\nlearned a={learn_alpha:.2f}'),
    ]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(gain, cmap='hot', vmin=1, vmax=vmax)
        ax.set_title(lbl, fontsize=11, pad=6); ax.axis('off')
        cx, cy = W/2, H/2
        for _, zr in zones:
            ax.plot(cx+zr*(W/2)*0.72, cy-zr*(H/2)*0.72, 'o',
                    markersize=8, markerfacecolor='none',
                    markeredgecolor='cyan', markeredgewidth=2.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_l = fig.add_subplot(gs[0, 3])
    ax_l.plot(loss_history, color='steelblue', linewidth=1.5)
    ax_l.set_title('Training loss (L1)', fontsize=11)
    ax_l.set_xlabel('Iteration', fontsize=9)
    ax_l.set_ylabel('L1 loss', fontsize=9)
    ax_l.grid(True, alpha=0.3)

    alphas = [(true_alpha,'true'),(est_alpha,'analytic'),(learn_alpha,'learned')]
    for row, (zn, zr) in enumerate(zones):
        for col, (av, al) in enumerate(alphas):
            ax = fig.add_subplot(gs[row+1, col].subgridspec(1,1)[0,0])
            _draw_3x3_kernel(ax, k_at(av, zr),
                             f'{al}  (a={av:.2f}, r={zr:.2f})')
            if col == 0:
                ax.set_ylabel(f'zone: {zn}', fontsize=11,
                              labelpad=6, color='dimgray', fontweight='bold')

    ax_leg = fig.add_subplot(gs[1:, 3]); ax_leg.axis('off')
    ax_leg.text(0.08, 0.95,
        '3x3 kernel structure\n\n'
        '[ 0    ][ 0    ][ 0    ]\n'
        '[ 0    ][ gain ][ 0    ]\n'
        '[ 0    ][ 0    ][ 0    ]\n\n'
        'gain = 1 + a * r^2\n\n'
        'Zone definition\n'
        '  center : r = 0.00\n'
        '  mid    : r = 0.50\n'
        '  edge   : r = 0.85\n\n'
        'cyan o = zone position',
        transform=ax_leg.transAxes, va='top', ha='left',
        fontsize=11, fontfamily='monospace', color='dimgray',
        bbox=dict(boxstyle='round,pad=0.7',
                  facecolor='#f5f5f5', edgecolor='#cccccc'))

    fig.suptitle('3x3 PD Kernel — gain map & zone kernel values',
                 fontsize=15, y=1.01)
    path = os.path.join(res_dir, 'kernel_map.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Saved] {path}')


def save_kernel_profile(true_alpha, est_alpha, learn_alpha, res_dir):
    r  = np.linspace(0, 1, 200)
    r2 = r ** 2
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r, 1+true_alpha*r2,  label=f'true  a={true_alpha:.2f}',
            color='black', linewidth=2)
    ax.plot(r, 1+est_alpha*r2,   label=f'analytic a={est_alpha:.2f}',
            color='steelblue', linewidth=1.5, linestyle='--')
    ax.plot(r, 1+learn_alpha*r2, label=f'learned a={learn_alpha:.2f}',
            color='tomato', linewidth=1.5, linestyle=':')
    ax.set_xlabel('r (normalized radius)')
    ax.set_ylabel('center weight = 1 + a*r^2')
    ax.set_title('3x3 PD Kernel — center weight profile')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(res_dir, 'kernel_profile.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Saved] {path}')

# ============================================================================
# Dataset: BSD500 (Berkeley Segmentation Dataset)
# image restoration 분야 표준 벤치마크.
# 다운로드: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/
# 또는:     pip install scikit-image  (BSD68 포함)
# 경로 예시: --data-dir ./data/BSR/BSDS500/data/images/test
# ============================================================================

def load_dataset(args):
    # 1) 지정 경로
    if args.data_dir and os.path.isdir(args.data_dir):
        imgs = load_images_from_dir(args.data_dir, args.n_images, args.img_size)
        if imgs:
            print(f'[Load] BSD500: {len(imgs)} images from {args.data_dir}')
            return imgs

    # 2) scikit-image BSD68 fallback
    try:
        from skimage import data as skdata
        import skimage.io as skio
        import skimage.color as skcolor
        # skimage에 번들된 이미지로 대체
        test_imgs = [skdata.camera(), skdata.astronaut(),
                     skdata.chelsea(), skdata.coffee(),
                     skdata.horse(), skdata.hubble_deep_field()]
        imgs = []
        for arr in test_imgs[:args.n_images]:
            if arr.ndim == 3:
                arr = skcolor.rgb2gray(arr)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            from PIL import Image as PILImage
            pil = PILImage.fromarray((arr*255).astype(np.uint8))
            s   = min(pil.size)
            w,h = pil.size
            pil = pil.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
            pil = pil.resize((args.img_size, args.img_size), PILImage.BILINEAR)
            t   = torch.from_numpy(
                np.array(pil, dtype=np.float32)/255.0).unsqueeze(0).unsqueeze(0)
            imgs.append(t)
        print(f'[Load] BSD500 fallback: skimage built-in images ({len(imgs)})')
        return imgs
    except Exception as e:
        print(f'[Warning] skimage fallback failed: {e}')
        print('[Error] Please specify --data-dir with BSD500 images')
        return []


def parse_args():
    p = argparse.ArgumentParser(description='Vignetting PDK — BSD500')
    p.add_argument('--data-dir',  type=str,   default='',
                   help='BSD500 이미지 폴더 (비워두면 skimage fallback)')
    p.add_argument('--n-images',  type=int,   default=4)
    p.add_argument('--img-size',  type=int,   default=256)
    p.add_argument('--alpha',     type=float, default=2.5)
    p.add_argument('--n-iter',    type=int,   default=300)
    p.add_argument('--lr',        type=float, default=0.05)
    p.add_argument('--res-dir',   type=str,   default='./res/vignetting_bsd500')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.res_dir, exist_ok=True)
    imgs = load_dataset(args)
    if not imgs: return

    true_alpha = args.alpha
    H = W = imgs[0].shape[-1]

    raw_list, deg_list, cor_ana_list, cor_learn_list = [], [], [], []
    est_alphas, learn_alphas, all_loss = [], [], []

    for i, raw in enumerate(imgs):
        deg = apply_vignetting(raw, true_alpha)
        ea  = estimate_alpha(deg)
        ka  = make_3x3_vignetting_kernel(H, W, ea)
        cor_ana = spatially_varying_conv(deg, ka)

        print(f'[Image {i+1}/{len(imgs)}] Learning PDK...')
        cor_learn, la, loss_h = train_pdk(deg, raw,
                                          n_iter=args.n_iter, lr=args.lr)
        raw_list.append(raw); deg_list.append(deg)
        cor_ana_list.append(cor_ana); cor_learn_list.append(cor_learn)
        est_alphas.append(ea); learn_alphas.append(la); all_loss.append(loss_h)

    est_alpha   = float(np.mean(est_alphas))
    learn_alpha = float(np.mean(learn_alphas))
    loss_history = all_loss[0]

    print(f'[Analytic] mean a_est={est_alpha:.4f}')
    print(f'[Learn]    mean a_lrn={learn_alpha:.4f}')
    print(f'[Analytic] PSNR={psnr(raw_list[0], cor_ana_list[0]):.2f}dB')
    print(f'[Learn]    PSNR={psnr(raw_list[0], cor_learn_list[0]):.2f}dB')

    save_comparison(raw_list, deg_list, cor_ana_list, cor_learn_list,
                    true_alpha, est_alpha, learn_alpha, args.res_dir, 'BSD500')
    save_kernel_map(true_alpha, est_alpha, learn_alpha,
                    loss_history, H, args.res_dir)
    save_kernel_profile(true_alpha, est_alpha, learn_alpha, args.res_dir)
    print(f'\n[Done] {os.path.abspath(args.res_dir)}')


if __name__ == '__main__':
    main()
