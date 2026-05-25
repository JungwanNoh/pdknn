"""
20_nonparametric_pdk_rf.py
==========================================
Non-Parametric PDK for Real Degradation [RF Dataset]

기존 Parametric PDK의 한계:
  kernel(r) = f(alpha, sigma, ...) → 수식 가정 필요
  → real degradation에서 가정 불일치 → 성능 저하

NonParametric PDK:
  kernel(x,y) = 위치별 3x3 weights 직접 학습
  → 수식 가정 없음 → real degradation 적응
  → H x W x 9 파라미터 직접 최적화

Regularization:
  1. TV (Total Variation): 인접 위치 kernel이 smooth하게
  2. Radial smoothness: r이 같으면 비슷한 kernel
  3. Center bias: center weight가 dominant하게

비교:
  ParametricPDK (VignettingPDK, NonUnifPDK)  vs
  NonParametricPDK                             vs
  Oracle (각 이미지 직접 학습, upper bound)

Run:
  python 20_nonparametric_pdk_rf.py \
    --train-dir ./data/rf/train \
    --inf-dir   ./data/rf/test

  python 20_nonparametric_pdk_rf.py \
    --train-dir ./data/rf/train \
    --inf-dir   ./data/rf/test \
    --tv-weight 0.01 --radial-weight 0.1
"""

import argparse, os, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    pass

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ============================================================================
# Utilities
# ============================================================================

def radial_map(H, W, device="cpu"):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs)
    r = (gy**2 + gx**2).sqrt() / (2**0.5)
    return r, gy, gx

def psnr(a, b):
    mse = F.mse_loss(a.float(), b.float()).item()
    return 99.9 if mse < 1e-10 else 10 * np.log10(1.0 / mse)

def to_np(t):
    return t.squeeze().detach().cpu().float().numpy()

def load_single_image(path, size):
    img = Image.open(path).convert("L")
    w, h = img.size; s = min(w, h)
    img = img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
    img = img.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(
        np.array(img, dtype=np.float32)/255.).unsqueeze(0).unsqueeze(0)

def spatially_varying_conv(x, kernels):
    B, C, H, W = x.shape
    patches = F.unfold(x, kernel_size=3, padding=1).view(B, C, 9, H*W)
    k = kernels.view(H*W, 9).T.unsqueeze(0).unsqueeze(0)
    return (patches * k).sum(dim=2).view(B, C, H, W).clamp(0, 1)

def loss_fn(out, clean):
    l1 = F.l1_loss(out, clean)
    def gmap(t): return t[:,:,:,1:]-t[:,:,:,:-1], t[:,:,1:,:]-t[:,:,:-1,:]
    ox,oy = gmap(out); cx,cy = gmap(clean)
    return l1 + 0.1*(F.l1_loss(ox,cx)+F.l1_loss(oy,cy))


# ============================================================================
# Dataset loader
# ============================================================================

def find_paired_images(deg_dir, clean_dir):
    exts = ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG","*.bmp")
    def get_paths(d):
        paths = []
        for ext in exts:
            paths += glob.glob(os.path.join(d,"**",ext), recursive=True)
            paths += glob.glob(os.path.join(d, ext))
        return sorted(set(paths))

    deg_paths   = get_paths(deg_dir)
    clean_paths = get_paths(clean_dir)
    if not deg_paths or not clean_paths:
        return []

    deg_names   = {os.path.splitext(os.path.basename(p))[0]: p for p in deg_paths}
    clean_names = {os.path.splitext(os.path.basename(p))[0]: p for p in clean_paths}
    common = sorted(set(deg_names.keys()) & set(clean_names.keys()))
    if common:
        pairs = [(deg_names[n], clean_names[n]) for n in common]
        print(f"  [Paired] {len(pairs)} matched by filename")
    else:
        n = min(len(deg_paths), len(clean_paths))
        pairs = list(zip(deg_paths[:n], clean_paths[:n]))
        print(f"  [Paired] {len(pairs)} matched by sort order")
    return pairs


def infer_dirs(base_dir):
    candidates_deg   = ["input","low","blur","degraded","blurry","lq"]
    candidates_clean = ["gt","high","sharp","clean","target","hq"]
    if not os.path.isdir(base_dir):
        return None, None
    sub = [d for d in os.listdir(base_dir)
           if os.path.isdir(os.path.join(base_dir,d))]
    sub_lower = [s.lower() for s in sub]
    deg_dir = clean_dir = None
    for c in candidates_deg:
        if c in sub_lower:
            deg_dir = os.path.join(base_dir, sub[sub_lower.index(c)]); break
    for c in candidates_clean:
        if c in sub_lower:
            clean_dir = os.path.join(base_dir, sub[sub_lower.index(c)]); break
    if deg_dir and clean_dir:
        print(f"  [Auto] deg={deg_dir}, clean={clean_dir}")
    return deg_dir, clean_dir


def load_paired(deg_dir, clean_dir, n, size):
    pairs = find_paired_images(deg_dir, clean_dir)
    if not pairs:
        return [], []
    random.shuffle(pairs)
    degs, cleans = [], []
    for dp, cp in pairs[:n]:
        try:
            degs.append(load_single_image(dp, size))
            cleans.append(load_single_image(cp, size))
        except Exception as e:
            print(f"  [skip] {e}")
            if len(degs) > len(cleans): degs.pop()
    print(f"  [Load] {len(degs)} pairs (size={size})")
    return degs, cleans


# ============================================================================
# Non-Parametric PDK
# ============================================================================

class NonParametricPDK(nn.Module):
    """
    위치별 3x3 kernel weights를 직접 학습.
    파라미터: H x W x 9

    kernel(x,y)는 수식 가정 없이 gradient로 직접 최적화.
    softmax로 normalization → sum=1 유지.

    파라미터 수 비교:
      ParametricPDK (Vignetting): 1개
      ParametricPDK (PSF):        4개
      NonParametricPDK (128x128): 128*128*9 = 147,456개
      NonParametricPDK (256x256): 256*256*9 = 589,824개
    """
    def __init__(self, H, W, init_mode="delta"):
        super().__init__()
        # kernel logits: (H, W, 9)
        # init_mode:
        #   "delta": center weight=3, else=0 (identity 초기화)
        #   "zero":  모든 weight=0
        #   "random": 작은 random noise
        if init_mode == "delta":
            logits = torch.zeros(H, W, 9)
            logits[:, :, 4] = 3.0   # center weight dominant
        elif init_mode == "zero":
            logits = torch.zeros(H, W, 9)
        else:
            logits = torch.randn(H, W, 9) * 0.01

        self.kernel_logits = nn.Parameter(logits)
        self.H = H; self.W = W

        # radial map (regularization용)
        r, _, _ = radial_map(H, W)
        self.register_buffer("r", r)

    def get_kernels(self):
        """(H, W, 9) normalized kernel map"""
        return torch.softmax(self.kernel_logits, dim=-1)

    def forward(self, x):
        return spatially_varying_conv(x, self.get_kernels())

    def tv_loss(self):
        """
        Total Variation regularization:
        인접 위치 kernel이 급격히 변하지 않도록.
        """
        k = self.get_kernels()   # (H, W, 9)
        diff_h = (k[1:,:,:] - k[:-1,:,:]).pow(2).mean()
        diff_w = (k[:,1:,:] - k[:,:-1,:]).pow(2).mean()
        return diff_h + diff_w

    def radial_loss(self):
        """
        Radial smoothness: 같은 r 값이면 비슷한 kernel.
        r을 quantize해서 같은 bin의 kernel 분산 최소화.
        """
        k = self.get_kernels()              # (H, W, 9)
        r_flat = self.r.view(-1)            # (H*W,)
        k_flat = k.view(-1, 9)             # (H*W, 9)

        # r을 16개 bin으로 quantize
        n_bins = 16
        r_bin = (r_flat * n_bins).long().clamp(0, n_bins-1)

        loss = torch.tensor(0.0, device=k.device)
        for b in range(n_bins):
            mask = (r_bin == b)
            if mask.sum() < 2: continue
            k_bin = k_flat[mask]           # (N_b, 9)
            mean_k = k_bin.mean(0, keepdim=True)
            loss = loss + (k_bin - mean_k).pow(2).mean()
        return loss / n_bins

    def center_bias_loss(self):
        """
        Center weight (idx=4)가 dominant하도록.
        너무 균일한 kernel 방지.
        """
        k = self.get_kernels()
        center = k[:,:,4]                   # (H, W)
        return F.relu(0.5 - center).mean()  # center < 0.5이면 penalty


class ParametricVignettingPDK(nn.Module):
    """비교용: 기존 VignettingPDK"""
    def __init__(self, H, W):
        super().__init__()
        self._alpha = nn.Parameter(torch.tensor(float(np.log(np.exp(1.0)-1.0))))
        r, _, _ = radial_map(H, W)
        self.register_buffer("r2", r**2)

    @property
    def alpha(self): return F.softplus(self._alpha)

    def forward(self, x):
        gain = 1.0 + self.alpha * self.r2
        zeros = torch.zeros(*self.r2.shape, 9, device=x.device)
        idx = torch.full((*self.r2.shape,1), 4, dtype=torch.long, device=x.device)
        k = zeros.scatter(2, idx, gain.unsqueeze(-1))
        return spatially_varying_conv(x, k)


class ParametricNonUnifPDK(nn.Module):
    """비교용: 기존 NonUnifPDK"""
    def __init__(self, H, W):
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._cx  = nn.Parameter(torch.tensor(0.0))
        self._cy  = nn.Parameter(torch.tensor(0.0))
        self._sig = nn.Parameter(torch.tensor(_sp(0.6)))
        self._ell = nn.Parameter(torch.tensor(_sp(1.0)))
        self._av  = nn.Parameter(torch.tensor(_sp(1.0)))
        r, gy, gx = radial_map(H, W)
        self.register_buffer("r2", r**2)
        self.register_buffer("gy", gy)
        self.register_buffer("gx", gx)

    @property
    def cx(self):  return torch.tanh(self._cx)*0.5
    @property
    def cy(self):  return torch.tanh(self._cy)*0.5
    @property
    def sig(self): return F.softplus(self._sig).clamp(0.2, 2.0)
    @property
    def ell(self): return F.softplus(self._ell).clamp(0.3, 3.0)
    @property
    def av(self):  return F.softplus(self._av)

    def forward(self, x):
        B,C,H,W = x.shape
        V_rad = 1.0/(1.0+self.av*self.r2)
        dy = self.gy-self.cy; dx = self.gx-self.cx
        d2 = dx**2+(dy*self.ell)**2
        ill = torch.exp(-d2/(2*self.sig**2))
        ill = ill/ill.max().clamp(1e-5)
        V = (V_rad*(0.3+0.7*ill)).clamp(1e-3,1.0)
        gain = (1.0/V).clamp(1.0,10.0)
        zeros = torch.zeros(H,W,9,device=x.device)
        idx = torch.full((H,W,1),4,dtype=torch.long,device=x.device)
        k = zeros.scatter(2,idx,gain.unsqueeze(-1))
        return spatially_varying_conv(x,k)


# ============================================================================
# Training
# ============================================================================

def train_nonparam(model, deg_imgs, clean_imgs, n_iter, lr,
                   tv_weight=0.01, radial_weight=0.1,
                   center_weight=0.0, verbose=True):
    """
    NonParametricPDK 학습.
    reconstruction loss + regularization.
    """
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    history = {"total":[], "rec":[], "tv":[], "radial":[]}
    pairs = list(zip(deg_imgs, clean_imgs))

    for i in range(n_iter):
        opt.zero_grad()

        # reconstruction loss
        l_rec = torch.tensor(0.0)
        for deg, clean in pairs:
            l_rec = l_rec + loss_fn(model(deg), clean)
        l_rec = l_rec / len(pairs)

        # regularization
        l_tv     = model.tv_loss()     * tv_weight
        l_radial = model.radial_loss() * radial_weight
        l_center = model.center_bias_loss() * center_weight

        total = l_rec + l_tv + l_radial + l_center
        total.backward(); opt.step(); sched.step()

        history["total"].append(total.item())
        history["rec"].append(l_rec.item())
        history["tv"].append(l_tv.item())
        history["radial"].append(l_radial.item())

        if verbose and (i+1) % 100 == 0:
            print(f"    iter {i+1:4d}/{n_iter}"
                  f"  rec={l_rec.item():.5f}"
                  f"  tv={l_tv.item():.5f}"
                  f"  radial={l_radial.item():.5f}")
    return history


def train_parametric(model, deg_imgs, clean_imgs, n_iter, lr, verbose=False):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    history = []
    pairs = list(zip(deg_imgs, clean_imgs))
    for i in range(n_iter):
        opt.zero_grad()
        total = torch.tensor(0.0)
        for deg, clean in pairs:
            total = total + loss_fn(model(deg), clean)
        total = total / len(pairs)
        total.backward(); opt.step(); sched.step()
        history.append(total.item())
    return history


def train_oracle(model, deg, clean, n_iter, lr, is_nonparam=False,
                 tv_weight=0.01, radial_weight=0.1):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    for i in range(n_iter):
        opt.zero_grad()
        l = loss_fn(model(deg), clean)
        if is_nonparam:
            l = l + model.tv_loss()*tv_weight + model.radial_loss()*radial_weight
        l.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return model(deg).clamp(0,1)


def inference_fixed(model, deg):
    model.eval()
    with torch.no_grad():
        return model(deg).clamp(0,1)


# ============================================================================
# Visualization
# ============================================================================

def save_comparison(inf_results, res_dir):
    """
    N rows x 6 cols:
    [clean | degraded | VigPDK | NonUnifPDK | NonParam-PDK | Oracle(NonParam)]
    """
    n = len(inf_results)
    fig, axes = plt.subplots(n, 6, figsize=(22, 4*n+0.5))
    if n == 1: axes = axes[np.newaxis,:]

    col_titles = ["1. Clean (GT)", "2. Degraded",
                  "3. VignettingPDK", "4. NonUnifPDK",
                  "5. NonParam-PDK", "6. Oracle (NonParam)"]
    for c, t in enumerate(col_titles):
        axes[0,c].set_title(t, fontsize=9, fontweight="bold", pad=5)

    colors = ["dimgray","steelblue","tomato","darkorange","purple"]

    for r, res in enumerate(inf_results):
        imgs = [to_np(res["clean"]),   to_np(res["deg"]),
                to_np(res["vig_cor"]), to_np(res["nu_cor"]),
                to_np(res["np_cor"]),  to_np(res["ora_cor"])]
        for c, img in enumerate(imgs):
            axes[r,c].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[r,c].axis("off")

        axes[r,0].set_ylabel(f"img {r+1}", fontsize=9, fontweight="bold",
                             rotation=0, labelpad=40, va="center")

        psnr_vals = [res["psnr_deg"], res["psnr_vig"],
                     res["psnr_nu"],  res["psnr_np"], res["psnr_ora"]]
        for c, (pv, col) in enumerate(zip(psnr_vals, colors), start=1):
            axes[r,c].text(0.5,-0.07, f"PSNR={pv:.1f}dB",
                           transform=axes[r,c].transAxes,
                           ha="center", va="top", fontsize=8, color=col)

        gap = res["psnr_np"] - res["psnr_ora"]
        axes[r,4].text(0.5,-0.13,
                       f"vs Oracle: {gap:+.1f}dB",
                       transform=axes[r,4].transAxes,
                       ha="center", va="top", fontsize=7.5, color="gray")

    fig.suptitle("NonParametric PDK vs Parametric PDK [RF — Unseen test]",
                 fontsize=13, y=1.01)
    plt.tight_layout(rect=[0,0.05,1,1])
    path = os.path.join(res_dir, "20_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_psnr_summary(inf_results, res_dir):
    keys  = ["psnr_deg","psnr_vig","psnr_nu","psnr_np","psnr_ora"]
    labels= ["Degraded","VignettingPDK","NonUnifPDK","NonParam-PDK","Oracle(NP)"]
    colors= ["#aaaaaa","steelblue","tomato","darkorange","purple"]

    means = [np.mean([r[k] for r in inf_results]) for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), means, color=colors, width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("PSNR (dB)", fontsize=10)
    ax.set_title("Parametric vs NonParametric PDK [RF Dataset]", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (bar, v) in enumerate(zip(bars, means)):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.1,
                f"{v:.2f}", ha="center", fontsize=9, color=colors[i])

    plt.tight_layout()
    path = os.path.join(res_dir, "20_psnr_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_kernel_map(model, res_dir, tag="nonparam"):
    """NonParametricPDK의 kernel map 시각화"""
    k = model.get_kernels().detach().cpu().numpy()  # (H, W, 9)
    H, W, _ = k.shape

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    names = [f"k[{i//3-1},{i%3-1}]" for i in range(9)]
    names[4] = "k[0,0] (center)"

    for idx in range(9):
        ax = axes[idx//3, idx%3]
        im = ax.imshow(k[:,:,idx], cmap="RdBu_r",
                       vmin=-k[:,:,idx].max(), vmax=k[:,:,idx].max())
        ax.set_title(names[idx], fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"NonParametric PDK — kernel weight map ({H}x{W}x9)",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(res_dir, f"20_kernel_map_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_radial_profile(model, res_dir):
    """kernel center weight의 radial profile"""
    k = model.get_kernels().detach().cpu()  # (H, W, 9)
    H, W, _ = k.shape
    r, _, _ = radial_map(H, W)
    r_np = r.numpy().flatten()
    center = k[:,:,4].numpy().flatten()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(r_np, center, alpha=0.1, s=2, color="darkorange")

    # radial bin 평균
    bins = np.linspace(0, 1, 32)
    bin_idx = np.digitize(r_np, bins)
    bin_mean = [center[bin_idx==b].mean() if (bin_idx==b).sum()>0 else np.nan
                for b in range(1, len(bins))]
    bin_centers = (bins[:-1]+bins[1:])/2
    ax.plot(bin_centers, bin_mean, color="red", linewidth=2, label="bin mean")
    ax.set_xlabel("r (radial position)", fontsize=10)
    ax.set_ylabel("center kernel weight", fontsize=10)
    ax.set_title("NonParam-PDK: center weight radial profile", fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(res_dir, "20_radial_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_loss_curves(np_history, vig_history, nu_history, res_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # NonParam loss 분해
    axes[0].plot(np_history["rec"],    color="darkorange", label="rec loss")
    axes[0].plot(np_history["tv"],     color="steelblue",  label="TV reg")
    axes[0].plot(np_history["radial"], color="tomato",     label="radial reg")
    axes[0].set_title("NonParam-PDK training loss", fontsize=10)
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # 전체 비교
    axes[1].plot(np_history["total"], color="darkorange", linewidth=1.5,
                 label="NonParam-PDK")
    axes[1].plot(vig_history, color="steelblue", linewidth=1.5,
                 label="VignettingPDK")
    axes[1].plot(nu_history,  color="tomato",    linewidth=1.5,
                 label="NonUnifPDK")
    axes[1].set_title("Loss comparison", fontsize=10)
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Training loss: NonParam vs Parametric PDK", fontsize=12)
    plt.tight_layout()
    path = os.path.join(res_dir, "20_loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


# ============================================================================
# Args & Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="NonParametric PDK [RF Dataset]")
    p.add_argument("--train-dir",     type=str, default="./data/rf/train")
    p.add_argument("--inf-dir",       type=str, default="./data/rf/test")
    p.add_argument("--train-deg-dir", type=str, default="")
    p.add_argument("--train-cln-dir", type=str, default="")
    p.add_argument("--inf-deg-dir",   type=str, default="")
    p.add_argument("--inf-cln-dir",   type=str, default="")
    p.add_argument("--n-train",    type=int,   default=10)
    p.add_argument("--n-inf",      type=int,   default=4)
    p.add_argument("--img-size",   type=int,   default=128)
    p.add_argument("--n-iter",     type=int,   default=500)
    p.add_argument("--n-ora-iter", type=int,   default=500)
    p.add_argument("--lr",         type=float, default=0.02)
    p.add_argument("--tv-weight",      type=float, default=0.01,
                   help="TV regularization weight (default: 0.01)")
    p.add_argument("--radial-weight",  type=float, default=0.1,
                   help="Radial smoothness weight (default: 0.1)")
    p.add_argument("--center-weight",  type=float, default=0.0,
                   help="Center bias weight (default: 0.0)")
    p.add_argument("--init-mode",  type=str,   default="delta",
                   choices=["delta","zero","random"],
                   help="kernel initialization (delta=identity)")
    p.add_argument("--res-dir",    type=str,   default="./res/nonparam_pdk")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.res_dir, exist_ok=True)
    H = W = args.img_size

    # 폴더 경로 결정
    train_deg = args.train_deg_dir or None
    train_cln = args.train_cln_dir or None
    inf_deg   = args.inf_deg_dir   or None
    inf_cln   = args.inf_cln_dir   or None

    if not train_deg:
        train_deg, train_cln = infer_dirs(args.train_dir)
    if not inf_deg:
        inf_deg, inf_cln = infer_dirs(args.inf_dir)

    print(f"[Dirs] train deg={train_deg}")
    print(f"       train cln={train_cln}")
    print(f"       inf   deg={inf_deg}")
    print(f"       inf   cln={inf_cln}")

    # 이미지 로드
    print("\n[Load] Train pairs...")
    tr_degs, tr_clns = load_paired(train_deg, train_cln, args.n_train, H)
    print("[Load] Inference pairs...")
    inf_degs, inf_clns = load_paired(inf_deg, inf_cln, args.n_inf, H)

    if not tr_degs or not inf_degs:
        print("[Error] No images loaded"); return

    print(f"\n  Train: {len(tr_degs)} pairs")
    print(f"  Inference: {len(inf_degs)} pairs (unseen)")
    print(f"  NonParam params: {H}x{W}x9 = {H*W*9:,}")
    print(f"  TV weight={args.tv_weight}  "
          f"Radial weight={args.radial_weight}")

    # ── 학습 ──────────────────────────────────────────────────────
    print("\n[Train] NonParametric PDK...")
    np_model = NonParametricPDK(H, W, init_mode=args.init_mode)
    np_hist  = train_nonparam(np_model, tr_degs, tr_clns,
                               args.n_iter, args.lr,
                               tv_weight=args.tv_weight,
                               radial_weight=args.radial_weight,
                               center_weight=args.center_weight)

    print("\n[Train] VignettingPDK (parametric baseline)...")
    vig_model = ParametricVignettingPDK(H, W)
    vig_hist  = train_parametric(vig_model, tr_degs, tr_clns,
                                  args.n_iter, args.lr)

    print("\n[Train] NonUnifPDK (parametric baseline)...")
    nu_model = ParametricNonUnifPDK(H, W)
    nu_hist  = train_parametric(nu_model, tr_degs, tr_clns,
                                 args.n_iter, args.lr)

    # ── Inference ─────────────────────────────────────────────────
    print(f"\n[Inference] {len(inf_degs)} unseen pairs...")
    inf_results = []

    for i, (deg, clean) in enumerate(zip(inf_degs, inf_clns)):
        np_cor  = inference_fixed(np_model,  deg)
        vig_cor = inference_fixed(vig_model, deg)
        nu_cor  = inference_fixed(nu_model,  deg)

        print(f"  Oracle {i+1}/{len(inf_degs)}...")
        ora_model = NonParametricPDK(H, W, init_mode=args.init_mode)
        ora_cor   = train_oracle(ora_model, deg, clean,
                                  args.n_ora_iter, args.lr,
                                  is_nonparam=True,
                                  tv_weight=args.tv_weight,
                                  radial_weight=args.radial_weight)

        inf_results.append({
            "clean":    clean,
            "deg":      deg,
            "np_cor":   np_cor,
            "vig_cor":  vig_cor,
            "nu_cor":   nu_cor,
            "ora_cor":  ora_cor,
            "psnr_deg": psnr(clean, deg),
            "psnr_np":  psnr(clean, np_cor),
            "psnr_vig": psnr(clean, vig_cor),
            "psnr_nu":  psnr(clean, nu_cor),
            "psnr_ora": psnr(clean, ora_cor),
        })

        r = inf_results[-1]
        print(f"    Deg={r['psnr_deg']:.1f}  Vig={r['psnr_vig']:.1f}"
              f"  NU={r['psnr_nu']:.1f}  NP={r['psnr_np']:.1f}"
              f"  Ora={r['psnr_ora']:.1f}")

    # 요약
    print("\n[Summary]")
    for k, label in [("psnr_deg","Degraded"), ("psnr_vig","VigPDK"),
                      ("psnr_nu","NonUnifPDK"), ("psnr_np","NonParam"),
                      ("psnr_ora","Oracle")]:
        mean = np.mean([r[k] for r in inf_results])
        print(f"  {label:<14}: {mean:.2f} dB")

    gap = np.mean([r["psnr_np"]-r["psnr_ora"] for r in inf_results])
    print(f"  NonParam vs Oracle: {gap:+.2f} dB")

    # ── 시각화 ────────────────────────────────────────────────────
    print("\n[Saving results...]")
    save_comparison(inf_results, args.res_dir)
    save_psnr_summary(inf_results, args.res_dir)
    save_loss_curves(np_hist, vig_hist, nu_hist, args.res_dir)
    save_kernel_map(np_model, args.res_dir, tag="train")
    save_radial_profile(np_model, args.res_dir)

    print(f"\n[Done] {os.path.abspath(args.res_dir)}")
    print("  20_comparison.png    -- Param vs NonParam vs Oracle")
    print("  20_psnr_summary.png  -- PSNR bar chart")
    print("  20_loss_curves.png   -- loss 분해")
    print("  20_kernel_map_train.png -- 학습된 kernel weight map")
    print("  20_radial_profile.png   -- center weight radial profile")


if __name__ == "__main__":
    main()