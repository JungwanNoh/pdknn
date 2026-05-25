"""
22_edge_detection_pdk.py
==========================================
Feature Extraction (Laplacian Edge Detection) under Degradation

동기:
  PDK로 degradation 보정 후 downstream task 성능이 얼마나 복원되는지 검증.
  edge detection은 PSF blur/vignetting의 영향이 직접적으로 나타나는 task.
  Laplacian은 단일 3x3 kernel → PDK와 동일한 구조로 일관성 있는 비교.

시나리오:
  1. Clean image    → Laplacian → GT edge map
  2. Degraded image → Laplacian → degraded edge map
  3. Global kernel correction → Laplacian → global edge map
  4. PDK correction           → Laplacian → PDK edge map

  GT edge map 대비 각각 PSNR/SSIM 비교
  → degradation이 feature extraction에 미치는 영향
  → PDK 보정이 feature 품질을 얼마나 복원하는지

Degradation:
  Vignetting (강한 alpha=3.0)
  PSF blur   (강한 sigma0=0.5, alpha=1.5)

Dataset: BSD300 test set (unseen images)

Laplacian kernel (3x3):
  [[ 0,  1,  0],
   [ 1, -4,  1],
   [ 0,  1,  0]]
  → 2차 미분 → edge 위치 검출
  → 단일 3x3 kernel → PDK와 동일 구조

Run:
  python 22_edge_detection_pdk.py
  python 22_edge_detection_pdk.py --data-dir ./data/BSD300/images/test
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
# Laplacian edge detection
# ============================================================================

LAPLACIAN_KERNEL = torch.tensor([
    [ 0., -1.,  0.],
    [-1.,  4., -1.],
    [ 0., -1.,  0.]
]).view(1, 1, 3, 3)


def laplacian_edge(x):
    """
    단일 3x3 Laplacian kernel로 edge detection.
    출력: abs(response)를 [0,1]로 normalize.
    """
    k = LAPLACIAN_KERNEL.to(x.device)
    resp = F.conv2d(x, k, padding=1)
    resp = resp.abs()
    # 이미지별 normalize
    B, C, H, W = resp.shape
    mn = resp.view(B, -1).min(dim=1)[0].view(B,1,1,1)
    mx = resp.view(B, -1).max(dim=1)[0].view(B,1,1,1)
    return ((resp - mn) / (mx - mn + 1e-8)).clamp(0, 1)


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

def ssim_simple(a, b):
    a = a.float(); b = b.float()
    mu_a = F.avg_pool2d(a, 11, stride=1, padding=5)
    mu_b = F.avg_pool2d(b, 11, stride=1, padding=5)
    mu_a2 = mu_a**2; mu_b2 = mu_b**2; mu_ab = mu_a*mu_b
    sig_a2 = F.avg_pool2d(a**2, 11, stride=1, padding=5) - mu_a2
    sig_b2 = F.avg_pool2d(b**2, 11, stride=1, padding=5) - mu_b2
    sig_ab = F.avg_pool2d(a*b,  11, stride=1, padding=5) - mu_ab
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_ab+c1)*(2*sig_ab+c2)) /                ((mu_a2+mu_b2+c1)*(sig_a2+sig_b2+c2))
    return ssim_map.mean().item()

def to_np(t):
    return t.squeeze().detach().cpu().float().numpy()

def load_single_image(path, size):
    img = Image.open(path).convert("L")
    w, h = img.size; s = min(w, h)
    img = img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
    img = img.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(
        np.array(img, dtype=np.float32)/255.).unsqueeze(0).unsqueeze(0)

def load_images(data_dir, n, size):
    exts = ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG")
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(data_dir,"**",ext), recursive=True)
        paths += glob.glob(os.path.join(data_dir, ext))
    paths = sorted(set(paths))
    if not paths:
        print(f"  [Error] No images in {data_dir}")
        return []
    random.shuffle(paths)
    imgs = []
    for p in paths[:n]:
        try: imgs.append(load_single_image(p, size))
        except: continue
    print(f"  [Load] {len(imgs)} images from {data_dir}")
    return imgs

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
# Degradation
# ============================================================================

def degrade_vignetting(x, alpha=3.0):
    _, _, H, W = x.shape
    r, _, _ = radial_map(H, W)
    V = 1.0 / (1.0 + alpha * r**2)
    return (x * V).clamp(0, 1)

def degrade_psf(x, sigma0=0.5, alpha_psf=1.5):
    _, _, H, W = x.shape
    r, _, _ = radial_map(H, W)
    coords = torch.tensor([
        [-1.,-1.],[-1.,0.],[-1.,1.],
        [ 0.,-1.],[ 0.,0.],[ 0.,1.],
        [ 1.,-1.],[ 1.,0.],[ 1.,1.]])
    sigma = (sigma0 + alpha_psf * r**2).unsqueeze(-1).clamp(0.1, 2.0)
    d2 = (coords**2).sum(-1)
    g  = torch.exp(-d2 / (2*sigma**2))
    g  = g / g.sum(-1, keepdim=True)
    return spatially_varying_conv(x, g)


def degrade_combined(x, alpha=3.0, sigma0=0.5, alpha_psf=1.5):
    """Vignetting + PSF blur 합성 degradation"""
    x = degrade_psf(x, sigma0, alpha_psf)       # PSF blur 먼저
    x = degrade_vignetting(x, alpha)             # vignetting 후
    return x


# ============================================================================
# PDK Models
# ============================================================================

class VignettingPDK(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self._alpha = nn.Parameter(
            torch.tensor(float(np.log(np.exp(1.0)-1.0))))
        r, _, _ = radial_map(H, W)
        self.register_buffer("r2", r**2)

    @property
    def alpha(self): return F.softplus(self._alpha)

    def forward(self, x):
        gain = 1.0 + self.alpha * self.r2
        zeros = torch.zeros(*self.r2.shape, 9, device=x.device)
        idx = torch.full((*self.r2.shape,1), 4,
                         dtype=torch.long, device=x.device)
        k = zeros.scatter(2, idx, gain.unsqueeze(-1))
        return spatially_varying_conv(x, k)


class PSFPDK(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._s0  = nn.Parameter(torch.tensor(_sp(0.5)))
        self._aps = nn.Parameter(torch.tensor(_sp(0.5)))
        self._l0  = nn.Parameter(torch.tensor(_sp(0.5)))
        self._al  = nn.Parameter(torch.tensor(_sp(0.3)))
        r, _, _ = radial_map(H, W)
        self.register_buffer("r2", r**2)
        self.register_buffer("coords", torch.tensor([
            [-1.,-1.],[-1.,0.],[-1.,1.],
            [ 0.,-1.],[ 0.,0.],[ 0.,1.],
            [ 1.,-1.],[ 1.,0.],[ 1.,1.]]))

    @property
    def sigma0(self):    return F.softplus(self._s0)
    @property
    def alpha_psf(self): return F.softplus(self._aps)
    @property
    def lambda0(self):   return F.softplus(self._l0)
    @property
    def alpha_lam(self): return F.softplus(self._al)

    def forward(self, x):
        sigma = (self.sigma0 + self.alpha_psf*self.r2
                 ).unsqueeze(-1).clamp(0.1, 1.0)
        d2 = (self.coords**2).sum(-1)
        g  = torch.exp(-d2/(2*sigma**2))
        g  = g / g.sum(-1, keepdim=True)
        delta = torch.zeros_like(g); delta[:,:,4] = 1.0
        lam = (self.lambda0 + self.alpha_lam*self.r2
               ).unsqueeze(-1).clamp(0.05, 3.0)
        return spatially_varying_conv(x, delta + lam*(delta - g))


class GlobalKernel(nn.Module):
    def __init__(self):
        super().__init__()
        init = torch.zeros(9); init[4] = 3.0
        self.kernel_logits = nn.Parameter(init)

    def forward(self, x):
        k = torch.softmax(self.kernel_logits, dim=0)
        B, C, H, W = x.shape
        kmap = k.view(1,1,9).expand(H,W,9)
        return spatially_varying_conv(x, kmap)


class CombinedPDK(nn.Module):
    """
    Vignetting + PSF blur 합성 degradation 보정 PDK.
    두 모델을 순차 적용: PSF 보정 → Vignetting 보정.
    """
    def __init__(self, H, W):
        super().__init__()
        self.psf_pdk = PSFPDK(H, W)
        self.vig_pdk = VignettingPDK(H, W)

    def forward(self, x):
        x = self.psf_pdk(x)   # PSF blur 보정
        x = self.vig_pdk(x)   # Vignetting 보정
        return x.clamp(0, 1)


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_imgs, degrade_fn, n_iter, lr, verbose=False):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    pairs = [(img, degrade_fn(img)) for img in train_imgs]
    history = []
    for i in range(n_iter):
        opt.zero_grad()
        total = torch.tensor(0.0)
        for clean, deg in pairs:
            total = total + loss_fn(model(deg), clean)
        total = total / len(pairs)
        total.backward(); opt.step(); sched.step()
        history.append(total.item())
        if verbose and (i+1) % 100 == 0:
            print(f"    iter {i+1}/{n_iter}  loss={total.item():.6f}")
    return history

def inference_fixed(model, x):
    model.eval()
    with torch.no_grad():
        return model(x).clamp(0, 1)


# ============================================================================
# Visualization
# ============================================================================

def save_edge_comparison(results, res_dir, case_name):
    """
    각 이미지에 대해:
    상단: clean / degraded / global_cor / pdk_cor  (이미지)
    하단: GT edge / deg edge / global edge / pdk edge (edge map)
    """
    n = len(results)
    fig, axes = plt.subplots(n*2, 4, figsize=(16, 5*n))
    if n == 1:
        axes = axes.reshape(2, 4)

    col_titles = ["Clean", "Degraded", "Global corr.", "PDK corr."]
    row_labels  = ["Image", "Laplacian edge"]

    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=11, fontweight="bold", pad=6)

    for r, res in enumerate(results):
        img_row  = r * 2
        edge_row = r * 2 + 1

        # 이미지 행
        for c, key in enumerate(["clean","deg","glb_cor","pdk_cor"]):
            axes[img_row, c].imshow(to_np(res[key]),
                                    cmap="gray", vmin=0, vmax=1)
            axes[img_row, c].axis("off")
        axes[img_row, 0].set_ylabel(f"img {r+1}\nImage",
                                     fontsize=9, rotation=0,
                                     labelpad=55, va="center")

        # edge map 행
        edge_keys = ["gt_edge","deg_edge","glb_edge","pdk_edge"]
        psnr_keys = [None, "psnr_deg_e","psnr_glb_e","psnr_pdk_e"]
        ssim_keys = [None, "ssim_deg_e","ssim_glb_e","ssim_pdk_e"]
        colors    = [None, "dimgray","steelblue","darkorange"]

        for c, (ekey, pkey, skey, col) in enumerate(
                zip(edge_keys, psnr_keys, ssim_keys, colors)):
            axes[edge_row, c].imshow(to_np(res[ekey]),
                                     cmap="hot", vmin=0, vmax=1)
            axes[edge_row, c].axis("off")
            if pkey:
                axes[edge_row, c].text(
                    0.5, -0.06,
                    f"PSNR={res[pkey]:.1f}  SSIM={res[skey]:.3f}",
                    transform=axes[edge_row, c].transAxes,
                    ha="center", va="top", fontsize=8, color=col)

        axes[edge_row, 0].set_ylabel("Edge map", fontsize=9,
                                      rotation=0, labelpad=55, va="center")

    fig.suptitle(f"Edge Detection Quality [{case_name}]",
                 fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    tag = case_name.lower().replace(" ","_").replace("+","_")
    path = os.path.join(res_dir, f"22_edge_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_summary_bar(all_case_results, res_dir):
    """
    case별, method별 edge map PSNR/SSIM 요약 bar chart
    """
    cases   = list(all_case_results.keys())
    methods = ["Degraded","Global","PDK"]
    psnr_keys = ["psnr_deg_e","psnr_glb_e","psnr_pdk_e"]
    ssim_keys = ["ssim_deg_e","ssim_glb_e","ssim_pdk_e"]
    colors  = ["dimgray","steelblue","darkorange"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(cases)); w = 0.22
    offsets = [-w, 0, w]

    for ax, metric_keys, ylabel, title in [
        (axes[0], psnr_keys, "PSNR (dB)",
         "Edge map PSNR vs GT edges"),
        (axes[1], ssim_keys, "SSIM",
         "Edge map SSIM vs GT edges"),
    ]:
        for off, mkey, label, col in zip(
                offsets, metric_keys, methods, colors):
            vals = [np.mean([r[mkey] for r in all_case_results[c]])
                    for c in cases]
            bars = ax.bar(x+off, vals, w,
                          label=label, color=col, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2,
                        v + (0.2 if "PSNR" in ylabel else 0.002),
                        f"{v:.2f}" if "SSIM" in ylabel else f"{v:.1f}",
                        ha="center", fontsize=7, color=col)

        ax.set_xticks(x); ax.set_xticklabels(cases, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Feature Extraction Quality: "
                 "Degraded vs Global vs PDK correction",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(res_dir, "22_edge_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_figure_images(all_case_results, res_dir):
    """
    논문 figure용 개별 이미지 저장.
    각 case의 첫 번째 이미지에서
    clean_image / degraded_image / pdk_corrected_image /
    gt_edge / degraded_edge / pdk_edge
    각각 개별 파일로 저장 (타이틀/axis/여백 없음).
    """
    fig_dir = os.path.join(res_dir, "figure_images")
    os.makedirs(fig_dir, exist_ok=True)

    for case_name, results in all_case_results.items():
        res = results[2]  # 세 번째 이미지
        tag = case_name.lower().replace(" ","_").replace("+","_")

        # similarity map: 1 - |GT - method|
        # GT와 동일할수록 밝음(1), 다를수록 어두움(0)
        gt_np  = to_np(res["gt_edge"]).clip(0, 1)
        deg_sim = (1.0 - np.abs(gt_np - to_np(res["deg_edge"]).clip(0,1))).clip(0,1)
        glb_sim = (1.0 - np.abs(gt_np - to_np(res["glb_edge"]).clip(0,1))).clip(0,1)
        pdk_sim = (1.0 - np.abs(gt_np - to_np(res["pdk_edge"]).clip(0,1))).clip(0,1)

        def t(arr):
            return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

        save_items = {
            f"{tag}_clean":    (res["clean"],   "gray"),
            f"{tag}_degraded": (res["deg"],     "gray"),
            f"{tag}_pdk_corr": (res["pdk_cor"], "gray"),
            f"{tag}_gt_edge":  (res["gt_edge"], "hot"),
            f"{tag}_deg_edge": (res["deg_edge"],"hot"),
            f"{tag}_pdk_edge": (res["pdk_edge"],"hot"),
            f"{tag}_deg_sim":  (t(deg_sim),     "gray"),
            f"{tag}_glb_sim":  (t(glb_sim),     "gray"),
            f"{tag}_pdk_sim":  (t(pdk_sim),     "gray"),
        }

        for fkey, (tensor, cmap) in save_items.items():
            arr = to_np(tensor).clip(0, 1)
            fig, ax = plt.subplots(
                1, 1,
                figsize=(arr.shape[1]/100, arr.shape[0]/100),
                dpi=300)
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fpath = os.path.join(fig_dir, f"{fkey}.png")
            fig.savefig(fpath, dpi=300,
                        bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print(f"  [Figure] {fpath}")


# ============================================================================
# Args & Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Edge detection quality under degradation [BSD300]")
    p.add_argument("--train-dir", type=str,
                   default="./data/BSD300/images/train")
    p.add_argument("--inf-dir",   type=str,
                   default="./data/BSD300/images/test")
    p.add_argument("--n-train",   type=int, default=10)
    p.add_argument("--n-inf",     type=int, default=4)
    p.add_argument("--img-size",  type=int, default=256)
    p.add_argument("--n-iter",    type=int, default=500)
    p.add_argument("--lr",        type=float, default=0.02)
    p.add_argument("--vig-alpha", type=float, default=3.0,
                   help="Vignetting strength (default: 3.0, strong)")
    p.add_argument("--psf-sigma0",type=float, default=0.5,
                   help="PSF base sigma (default: 0.5, strong)")
    p.add_argument("--psf-alpha", type=float, default=1.5,
                   help="PSF radial alpha (default: 1.5, strong)")
    p.add_argument("--res-dir",   type=str,
                   default="./res/edge_pdk")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.res_dir, exist_ok=True)
    H = W = args.img_size

    # ── 이미지 로드 ───────────────────────────────────────────────
    print("[Load] Train images...")
    train_imgs = load_images(args.train_dir, args.n_train, H)
    print("[Load] Inference images...")
    inf_imgs   = load_images(args.inf_dir,   args.n_inf,   H)

    if not train_imgs or not inf_imgs:
        print("[Error] No images"); return

    print(f"  Train: {len(train_imgs)}  Inference: {len(inf_imgs)} (unseen)")

    # ── Case 정의 ─────────────────────────────────────────────────
    cases = [
        {
            "name":    "Vignetting+PSF",
            "degrade": lambda x: degrade_combined(
                           x, args.vig_alpha,
                           args.psf_sigma0, args.psf_alpha),
            "PDKModel": CombinedPDK,
            "desc":    (f"vig_alpha={args.vig_alpha} "
                        f"psf_sigma0={args.psf_sigma0} "
                        f"psf_alpha={args.psf_alpha}"),
        },
    ]

    all_case_results = {}

    for case in cases:
        cname    = case["name"]
        degrade  = case["degrade"]
        PDKModel = case["PDKModel"]
        print(f"\n{'='*55}")
        print(f"[Case] {cname} ({case['desc']})")

        # ── 모델 학습 ─────────────────────────────────────────────
        print("  [Train] PDK...")
        pdk_model = PDKModel(H, W)
        train_model(pdk_model, train_imgs, degrade,
                    args.n_iter, args.lr, verbose=True)

        print("  [Train] Global kernel...")
        glb_model = GlobalKernel()
        train_model(glb_model, train_imgs, degrade,
                    args.n_iter, args.lr, verbose=False)

        # ── Inference + edge detection ────────────────────────────
        print(f"  [Inference] {len(inf_imgs)} unseen images...")
        results = []

        for i, clean in enumerate(inf_imgs):
            deg     = degrade(clean)
            pdk_cor = inference_fixed(pdk_model, deg)
            glb_cor = inference_fixed(glb_model, deg)

            # Laplacian edge maps
            gt_edge  = laplacian_edge(clean)
            deg_edge = laplacian_edge(deg)
            glb_edge = laplacian_edge(glb_cor)
            pdk_edge = laplacian_edge(pdk_cor)

            # edge map 품질 평가 (GT edge 대비)
            p_deg = psnr(gt_edge, deg_edge)
            p_glb = psnr(gt_edge, glb_edge)
            p_pdk = psnr(gt_edge, pdk_edge)
            s_deg = ssim_simple(gt_edge, deg_edge)
            s_glb = ssim_simple(gt_edge, glb_edge)
            s_pdk = ssim_simple(gt_edge, pdk_edge)

            print(f"    img {i+1}  edge PSNR:"
                  f"  Deg={p_deg:.1f}"
                  f"  Glb={p_glb:.1f}"
                  f"  PDK={p_pdk:.1f}")

            results.append({
                "clean":      clean,
                "deg":        deg,
                "glb_cor":    glb_cor,
                "pdk_cor":    pdk_cor,
                "gt_edge":    gt_edge,
                "deg_edge":   deg_edge,
                "glb_edge":   glb_edge,
                "pdk_edge":   pdk_edge,
                "psnr_deg_e": p_deg,
                "psnr_glb_e": p_glb,
                "psnr_pdk_e": p_pdk,
                "ssim_deg_e": s_deg,
                "ssim_glb_e": s_glb,
                "ssim_pdk_e": s_pdk,
            })

        all_case_results[cname] = results

        # 평균
        mean_p = {k: np.mean([r[k] for r in results])
                  for k in ["psnr_deg_e","psnr_glb_e","psnr_pdk_e"]}
        mean_s = {k: np.mean([r[k] for r in results])
                  for k in ["ssim_deg_e","ssim_glb_e","ssim_pdk_e"]}
        print(f"  Edge PSNR  Deg={mean_p['psnr_deg_e']:.1f}"
              f"  Global={mean_p['psnr_glb_e']:.1f}"
              f"  PDK={mean_p['psnr_pdk_e']:.1f}")
        print(f"  Edge SSIM  Deg={mean_s['ssim_deg_e']:.3f}"
              f"  Global={mean_s['ssim_glb_e']:.3f}"
              f"  PDK={mean_s['ssim_pdk_e']:.3f}")
        print(f"  PDK vs Deg:    PSNR {mean_p['psnr_pdk_e']-mean_p['psnr_deg_e']:+.1f}dB"
              f"  SSIM {mean_s['ssim_pdk_e']-mean_s['ssim_deg_e']:+.3f}")
        print(f"  PDK vs Global: PSNR {mean_p['psnr_pdk_e']-mean_p['psnr_glb_e']:+.1f}dB"
              f"  SSIM {mean_s['ssim_pdk_e']-mean_s['ssim_glb_e']:+.3f}")

    # ── 시각화 ────────────────────────────────────────────────────
    print("\n[Saving results...]")
    for cname, results in all_case_results.items():
        save_edge_comparison(results, args.res_dir, cname)
    save_summary_bar(all_case_results, args.res_dir)
    save_figure_images(all_case_results, args.res_dir)

    print(f"\n[Done] {os.path.abspath(args.res_dir)}")
    print("  22_edge_vignetting.png  -- Vignetting edge comparison")
    print("  22_edge_psf_blur.png    -- PSF blur edge comparison")
    print("  22_edge_summary.png     -- PSNR/SSIM summary")
    print("  figure_images/          -- 논문용 개별 이미지")


if __name__ == "__main__":
    main()