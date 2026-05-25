"""
23_paper_figure_coma_vig.py
==========================================
논문용 Figure: Coma + Vignetting combined degradation

- Degradation: Coma aberration + Vignetting (강도 강하게)
- Train:  BSDS300 train set  (dataset-based PDK)
- Infer:  BSDS300 test  set  (unseen images)
- Models: GlobalKernel, ComaPDK
- Output:
    figures/raw_XX.png          (title 없음, 개별 저장)
    figures/degraded_XX.png
    figures/global_XX.png
    figures/pdk_XX.png
    figures/kernel_map_XX.png   (위치별 커널 시각화)
    figures/metrics_table.png   (PSNR / SSIM 비교표)
    23_summary.png              (4열 overview)

Run:
  python 23_paper_figure_coma_vig.py
  python 23_paper_figure_coma_vig.py \\
      --train-dir ./data/BSD300/images/train \\
      --inf-dir   ./data/BSD300/images/test  \\
      --n-train 20 --n-inf 4 --img-size 256  \\
      --n-iter 800
"""

import argparse, os, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

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
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r = (gy**2 + gx**2).sqrt() / (2**0.5)
    return r, gy, gx

def psnr(a, b):
    mse = F.mse_loss(a.float(), b.float()).item()
    return 99.9 if mse < 1e-10 else 10 * np.log10(1.0 / mse)

def ssim_simple(a, b):
    a = a.float(); b = b.float()
    mu_a = F.avg_pool2d(a, 11, stride=1, padding=5)
    mu_b = F.avg_pool2d(b, 11, stride=1, padding=5)
    mu_a2 = mu_a**2; mu_b2 = mu_b**2; mu_ab = mu_a*mu_b
    sig_a2 = F.avg_pool2d(a**2, 11, stride=1, padding=5) - mu_a2
    sig_b2 = F.avg_pool2d(b**2, 11, stride=1, padding=5) - mu_b2
    sig_ab = F.avg_pool2d(a*b,  11, stride=1, padding=5) - mu_ab
    c1, c2 = 0.01**2, 0.03**2
    num = (2*mu_ab+c1)*(2*sig_ab+c2)
    den = (mu_a2+mu_b2+c1)*(sig_a2+sig_b2+c2)
    return (num/den).mean().item()

def to_np(t):
    return t.squeeze().detach().cpu().float().numpy()

def load_images(data_dir, n, size):
    exts = ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG")
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(data_dir,"**",ext), recursive=True)
        paths += glob.glob(os.path.join(data_dir, ext))
    paths = sorted(set(paths))
    random.shuffle(paths)
    imgs = []
    for p in paths:
        try:
            img = Image.open(p).convert("L")
            w, h = img.size; s = min(w, h)
            img = img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
            img = img.resize((size, size), Image.BILINEAR)
            t = torch.from_numpy(
                np.array(img, dtype=np.float32)/255.
            ).unsqueeze(0).unsqueeze(0)
            imgs.append(t)
        except Exception:
            continue
        if len(imgs) == n:
            break
    print(f"  [Load] {len(imgs)} images from {data_dir}")
    return imgs

def load_images_by_names(data_dir, names, size):
    """특정 파일명(확장자 제외)으로 이미지 로드"""
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    imgs = []
    for name in names:
        found = None
        for ext in exts:
            candidate = os.path.join(data_dir, name + ext)
            if os.path.exists(candidate):
                found = candidate
                break
        if found is None:
            for ext in exts:
                hits = glob.glob(
                    os.path.join(data_dir, "**", name + ext),
                    recursive=True)
                if hits:
                    found = hits[0]
                    break
        if found is None:
            print(f"  [Warning] {name} not found in {data_dir}")
            continue
        try:
            img = Image.open(found).convert("L")
            w, h = img.size; s = min(w, h)
            img = img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
            img = img.resize((size, size), Image.BILINEAR)
            t = torch.from_numpy(
                np.array(img, dtype=np.float32)/255.
            ).unsqueeze(0).unsqueeze(0)
            imgs.append(t)
            print(f"  [Load] {found}")
        except Exception as e:
            print(f"  [Error] {found}: {e}")
    return imgs


# ============================================================================
# Degradation: Coma + Vignetting (강도 강하게)
# ============================================================================

def degrade_coma_vig(x,
                     sigma0=0.40,
                     alpha_psf=2.0,
                     coma_k=0.60,
                     alpha_vig=4.0):
    """
    Coma aberration (PSF shift + blur) + Vignetting 합성.
    강도:
      sigma0=0.40, alpha_psf=2.0  -> 가장자리 심한 blur
      coma_k=0.60                 -> 강한 방향성 shift
      alpha_vig=4.0               -> 강한 밝기 감소
    """
    r, gy, gx = radial_map(*x.shape[2:], x.device)
    r2 = r**2
    rs = r.clamp(1e-6)

    sigma = (sigma0 + alpha_psf * r2).clamp(0.1, 2.0)
    shift_y = coma_k * r2 * (gy / rs)
    shift_x = coma_k * r2 * (gx / rs)

    coords = torch.tensor([
        [-1.,-1.],[-1.,0.],[-1.,1.],
        [ 0.,-1.],[ 0.,0.],[ 0.,1.],
        [ 1.,-1.],[ 1.,0.],[ 1.,1.]], device=x.device)
    d2 = (coords**2).sum(-1)

    s = sigma.unsqueeze(-1)
    g1 = torch.exp(-d2 / (2*s**2))
    g1 = g1 / g1.sum(-1, keepdim=True)

    sy = shift_y.unsqueeze(-1)
    sx = shift_x.unsqueeze(-1)
    dy = coords[:,0].unsqueeze(0).unsqueeze(0) - sy
    dx = coords[:,1].unsqueeze(0).unsqueeze(0) - sx
    g2 = torch.exp(-(dy**2+dx**2) / (2*s**2))
    g2 = g2 / g2.sum(-1, keepdim=True)

    w = (r * 0.9).clamp(0, 0.9).unsqueeze(-1)
    k = (1-w)*g1 + w*g2
    k = k / k.sum(-1, keepdim=True)

    x = spatially_varying_conv(x, k)

    V = 1.0 / (1.0 + alpha_vig * r2)
    return (x * V.unsqueeze(0).unsqueeze(0)).clamp(0, 1)


# ============================================================================
# Conv helpers
# ============================================================================

def spatially_varying_conv(x, kernels):
    B, C, H, W = x.shape
    patches = F.unfold(x, kernel_size=3, padding=1).view(B, C, 9, H*W)
    k = kernels.view(H*W, 9).T.unsqueeze(0).unsqueeze(0)
    return (patches * k).sum(dim=2).view(B, C, H, W).clamp(0, 1)

def global_conv(x, kernel_9):
    k = kernel_9.view(1, 1, 3, 3)
    return F.conv2d(x, k.expand(x.shape[1],1,3,3),
                    padding=1, groups=x.shape[1]).clamp(0, 1)

def loss_fn(out, clean):
    l1 = F.l1_loss(out, clean)
    def gmap(t): return t[:,:,:,1:]-t[:,:,:,:-1], t[:,:,1:,:]-t[:,:,:-1,:]
    ox,oy = gmap(out); cx,cy = gmap(clean)
    return l1 + 0.1*(F.l1_loss(ox,cx)+F.l1_loss(oy,cy))


# ============================================================================
# Models
# ============================================================================

class GlobalKernel(nn.Module):
    def __init__(self):
        super().__init__()
        init = torch.zeros(9); init[4] = 3.0
        self.kernel_logits = nn.Parameter(init)

    def forward(self, x):
        k = torch.softmax(self.kernel_logits, dim=0)
        return global_conv(x, k)

    def get_kernel_3x3(self):
        return torch.softmax(self.kernel_logits, dim=0
                             ).detach().cpu().numpy().reshape(3,3)


class ComaPDK(nn.Module):
    """
    Coma + Vignetting 합성 degradation 보정 PDK.
    파라미터: sigma0, alpha_psf, coma_k, lambda0, alpha_lam, alpha_vig
    """
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
        """(H,W,9) correction kernel map 반환"""
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
        return spatially_varying_conv(x, k)


# ============================================================================
# Training (dataset-based: multiple images)
# ============================================================================

def train_dataset(model, train_imgs, degrade_fn, n_iter, lr, verbose=True):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    pairs = [(img, degrade_fn(img)) for img in train_imgs]
    for i in range(n_iter):
        opt.zero_grad()
        total = torch.tensor(0.0)
        for clean, deg in pairs:
            total = total + loss_fn(model(deg), clean)
        total = total / len(pairs)
        total.backward(); opt.step(); sched.step()
        if verbose and (i+1) % 200 == 0:
            print(f"    iter {i+1}/{n_iter}  loss={total.item():.5f}")

def infer(model, x):
    model.eval()
    with torch.no_grad():
        return model(x).clamp(0, 1)


# ============================================================================
# Saving individual images (no title, no axis)
# ============================================================================

def save_img(arr, path, cmap="gray"):
    """타이틀/axis 없이 이미지 한 장 저장 (300dpi)"""
    arr = np.clip(arr, 0, 1)
    h, w = arr.shape
    fig, ax = plt.subplots(1, 1,
                           figsize=(w/100, h/100), dpi=300)
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_kernel_map(kernels_hwk, img_size, path, n_sample=5):
    """
    위치별 커널 시각화.
    kernels_hwk: (H,W,9) numpy array
    n_sample: 각 축 방향 샘플 수 (n_sample x n_sample 격자)
    - colorbar를 그리드 바깥 오른쪽에 별도 배치 (겹침 없음)
    - 각 kernel cell에 테두리(grid) 표시
    """
    H, W, _ = kernels_hwk.shape
    positions = []
    for iy in np.linspace(0, H-1, n_sample, dtype=int):
        for ix in np.linspace(0, W-1, n_sample, dtype=int):
            positions.append((iy, ix))

    cols = n_sample
    rows = n_sample

    # 오른쪽에 colorbar용 공간 확보: width_ratios
    fig = plt.figure(figsize=(cols*1.8 + 0.8, rows*1.8))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(rows, cols + 1,
                  figure=fig,
                  width_ratios=[1]*cols + [0.08],
                  hspace=0.35, wspace=0.15)

    k_all = kernels_hwk.reshape(-1, 9)
    vmax = np.abs(k_all).max()
    vmax = max(vmax, 0.01)

    im = None
    for idx, (iy, ix) in enumerate(positions):
        r = idx // cols
        c = idx % cols
        ax = fig.add_subplot(gs[r, c])
        k = kernels_hwk[iy, ix].reshape(3, 3)
        im = ax.imshow(k, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       interpolation="nearest",
                       aspect="equal")
        # 각 pixel 경계 테두리
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.tick_params(which="minor", length=0)
        ax.grid(which="minor", color="black", linewidth=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"({ix},{iy})", fontsize=7, pad=3)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.8)

    # colorbar 전용 axes (마지막 열)
    cbar_ax = fig.add_subplot(gs[:, cols])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("kernel weight", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle("PDK kernel map (sampled positions)",
                 fontsize=10, y=1.01)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")


def save_metrics_table(results, path):
    """
    PSNR / SSIM 비교표 이미지로 저장
    results: list of dict per inference image
    """
    headers = ["Image", "Degraded", "Global", "PDK (ours)"]
    rows_psnr = []
    rows_ssim = []
    for i, r in enumerate(results):
        rows_psnr.append([
            f"img {i+1}",
            f"{r['psnr_deg']:.2f}",
            f"{r['psnr_glb']:.2f}",
            f"{r['psnr_pdk']:.2f}",
        ])
        rows_ssim.append([
            f"img {i+1}",
            f"{r['ssim_deg']:.4f}",
            f"{r['ssim_glb']:.4f}",
            f"{r['ssim_pdk']:.4f}",
        ])

    # mean row
    def mean_col(rows, col):
        return np.mean([float(r[col]) for r in rows])
    rows_psnr.append([
        "Mean",
        f"{mean_col(rows_psnr,1):.2f}",
        f"{mean_col(rows_psnr,2):.2f}",
        f"{mean_col(rows_psnr,3):.2f}",
    ])
    rows_ssim.append([
        "Mean",
        f"{mean_col(rows_ssim,1):.4f}",
        f"{mean_col(rows_ssim,2):.4f}",
        f"{mean_col(rows_ssim,3):.4f}",
    ])

    fig, axes = plt.subplots(1, 2, figsize=(10, 0.5+0.45*len(rows_psnr)))

    for ax, rows, title in [
        (axes[0], rows_psnr, "PSNR (dB)"),
        (axes[1], rows_ssim, "SSIM"),
    ]:
        ax.axis("off")
        tbl = ax.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.5)

        # header 색상
        for j in range(len(headers)):
            tbl[0, j].set_facecolor("#2C2C2A")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # PDK 열 강조
        for i in range(1, len(rows)+1):
            tbl[i, 3].set_facecolor("#E1F5EE")
            tbl[i, 3].set_text_props(color="#085041", fontweight="bold")

        # mean 행 강조
        for j in range(len(headers)):
            tbl[len(rows), j].set_facecolor("#F1EFE8")
            tbl[len(rows), j].set_text_props(fontweight="bold")

        ax.set_title(title, fontsize=11, pad=10)

    plt.suptitle("Coma + Vignetting correction: metric comparison",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")


def save_summary(results_list, path):
    """4열 overview: raw / degraded / global / pdk"""
    n = len(results_list)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.6*n))
    if n == 1: axes = axes[np.newaxis, :]

    col_titles = ["Clean", "Degraded", "Global kernel", "PDK (ours)"]
    for c, t in enumerate(col_titles):
        axes[0,c].set_title(t, fontsize=11, fontweight="bold", pad=6)

    for r, res in enumerate(results_list):
        imgs = [res["raw"], res["deg"], res["glb"], res["pdk"]]
        for c, img in enumerate(imgs):
            axes[r,c].imshow(to_np(img), cmap="gray", vmin=0, vmax=1)
            axes[r,c].axis("off")
        axes[r,1].text(0.5,-0.04,
                       f"PSNR={res['psnr_deg']:.1f}  SSIM={res['ssim_deg']:.3f}",
                       transform=axes[r,1].transAxes,
                       ha="center",fontsize=8,color="dimgray")
        axes[r,2].text(0.5,-0.04,
                       f"PSNR={res['psnr_glb']:.1f}  SSIM={res['ssim_glb']:.3f}",
                       transform=axes[r,2].transAxes,
                       ha="center",fontsize=8,color="steelblue")
        axes[r,3].text(0.5,-0.04,
                       f"PSNR={res['psnr_pdk']:.1f}  SSIM={res['ssim_pdk']:.3f}",
                       transform=axes[r,3].transAxes,
                       ha="center",fontsize=8,color="#085041")

    plt.tight_layout(rect=[0,0.03,1,1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")


# ============================================================================
# Args
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=str,
                   default="./data/BSD300/images/train")
    p.add_argument("--inf-dir",   type=str,
                   default="./data/BSD300/images/test")
    p.add_argument("--n-train",   type=int, default=20)
    p.add_argument("--n-inf",     type=int, default=4)
    p.add_argument("--img-size",  type=int, default=256)
    p.add_argument("--n-iter",    type=int, default=800)
    p.add_argument("--lr",        type=float, default=0.02)
    p.add_argument("--sigma0",    type=float, default=0.40)
    p.add_argument("--alpha-psf", type=float, default=2.0)
    p.add_argument("--coma-k",    type=float, default=0.60)
    p.add_argument("--alpha-vig", type=float, default=4.0)
    p.add_argument("--res-dir",   type=str,  default="./res/paper_fig")
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    fig_dir = os.path.join(args.res_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    H = W = args.img_size

    degrade = lambda x: degrade_coma_vig(
        x,
        sigma0    = args.sigma0,
        alpha_psf = args.alpha_psf,
        coma_k    = args.coma_k,
        alpha_vig = args.alpha_vig,
    )

    # ── 데이터 로드 ───────────────────────────────────────────────
    print("[Load] Train images (BSDS300 train)...")
    train_imgs = load_images(args.train_dir, args.n_train, H)
    print("[Load] Inference images (BSDS300 test, fixed)...")
    INF_NAMES = ["102061", "143090", "103070", "145086"]
    inf_imgs = load_images_by_names(args.inf_dir, INF_NAMES, H)

    if not train_imgs or not inf_imgs:
        print("[Error] No images found."); return

    print(f"  Train: {len(train_imgs)}  Inference: {len(inf_imgs)} (unseen)")

    # ── 모델 학습 (dataset-based) ─────────────────────────────────
    print("\n[Train] Global kernel (dataset-based)...")
    glb_model = GlobalKernel()
    train_dataset(glb_model, train_imgs, degrade, args.n_iter, args.lr)

    print("\n[Train] ComaPDK (dataset-based)...")
    pdk_model = ComaPDK(H, W)
    train_dataset(pdk_model, train_imgs, degrade, args.n_iter, args.lr)

    # ── Inference + 개별 이미지 저장 ──────────────────────────────
    print("\n[Inference] unseen images...")
    results = []

    for i, clean in enumerate(inf_imgs):
        deg = degrade(clean)
        glb = infer(glb_model, deg)
        pdk = infer(pdk_model, deg)

        p_deg = psnr(clean, deg);  s_deg = ssim_simple(clean, deg)
        p_glb = psnr(clean, glb);  s_glb = ssim_simple(clean, glb)
        p_pdk = psnr(clean, pdk);  s_pdk = ssim_simple(clean, pdk)

        print(f"  img {i+1}:  Deg={p_deg:.2f}dB  "
              f"Glb={p_glb:.2f}dB  PDK={p_pdk:.2f}dB")

        # 개별 이미지 저장 (title 없음)
        tag = f"{i+1:02d}"
        save_img(to_np(clean), os.path.join(fig_dir, f"raw_{tag}.png"))
        save_img(to_np(deg),   os.path.join(fig_dir, f"degraded_{tag}.png"))
        save_img(to_np(glb),   os.path.join(fig_dir, f"global_{tag}.png"))
        save_img(to_np(pdk),   os.path.join(fig_dir, f"pdk_{tag}.png"))
        print(f"    [Saved] raw/degraded/global/pdk images for img {i+1}")

        results.append(dict(
            raw=clean, deg=deg, glb=glb, pdk=pdk,
            psnr_deg=p_deg, psnr_glb=p_glb, psnr_pdk=p_pdk,
            ssim_deg=s_deg, ssim_glb=s_glb, ssim_pdk=s_pdk,
        ))

    # ── 모델 저장 ────────────────────────────────────────────────
    ckpt_path = os.path.join(args.res_dir, "pdk_model.pt")
    torch.save(pdk_model.state_dict(), ckpt_path)
    print(f"  [Saved] {ckpt_path}")

    # ── 커널 맵 시각화 ────────────────────────────────────────────
    print("\n[Kernel] Extracting PDK kernel map...")
    kernels_hwk = pdk_model.get_kernels()  # (H,W,9)
    glb_k3x3   = glb_model.get_kernel_3x3()

    # Global kernel 출력
    print(f"  Global kernel (3x3):\n{glb_k3x3}")

    # PDK 중심/가장자리 커널 샘플 출력
    sample_pts = [
        (H//2,  W//2,  "center"),
        (0,     0,     "top-left"),
        (0,     W-1,   "top-right"),
        (H-1,   0,     "bottom-left"),
        (H//4,  W//4,  "quarter"),
    ]
    print("  PDK kernel samples:")
    for (iy, ix, name) in sample_pts:
        k = kernels_hwk[iy, ix].reshape(3,3)
        print(f"    [{name}] ({ix},{iy}):\n{np.round(k,4)}")

    save_kernel_map(
        kernels_hwk,
        args.img_size,
        os.path.join(fig_dir, "kernel_map.png"),
        n_sample=5,
    )

    # 글로벌 커널 저장
    fig, ax = plt.subplots(1,1,figsize=(2,2))
    im = ax.imshow(glb_k3x3, cmap="RdBu_r",
                   vmin=-abs(glb_k3x3).max(),
                   vmax=abs(glb_k3x3).max(),
                   interpolation="nearest")
    ax.set_title("Global kernel", fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(fig_dir,"global_kernel_3x3.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── PSNR/SSIM 표 ──────────────────────────────────────────────
    print("\n[Metrics] Saving comparison table...")
    save_metrics_table(results, os.path.join(fig_dir,"metrics_table.png"))

    # ── Summary overview ──────────────────────────────────────────
    print("\n[Summary] Saving overview figure...")
    save_summary(results, os.path.join(args.res_dir, "23_summary.png"))

    # ── 최종 평균 출력 ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("[Result] Mean metrics over unseen images:")
    print(f"  Degraded:  PSNR={np.mean([r['psnr_deg'] for r in results]):.2f}dB"
          f"  SSIM={np.mean([r['ssim_deg'] for r in results]):.4f}")
    print(f"  Global:    PSNR={np.mean([r['psnr_glb'] for r in results]):.2f}dB"
          f"  SSIM={np.mean([r['ssim_glb'] for r in results]):.4f}")
    print(f"  PDK:       PSNR={np.mean([r['psnr_pdk'] for r in results]):.2f}dB"
          f"  SSIM={np.mean([r['ssim_pdk'] for r in results]):.4f}")
    print(f"\n[Done] {os.path.abspath(args.res_dir)}")
    print("  figures/raw_XX.png         -- clean images")
    print("  figures/degraded_XX.png    -- degraded images")
    print("  figures/global_XX.png      -- global kernel corrected")
    print("  figures/pdk_XX.png         -- PDK corrected")
    print("  figures/kernel_map.png     -- PDK kernel visualization")
    print("  figures/metrics_table.png  -- PSNR/SSIM table")
    print("  23_summary.png             -- 4-col overview")


if __name__ == "__main__":
    main()