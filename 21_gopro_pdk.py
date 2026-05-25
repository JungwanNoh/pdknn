"""
21_gopro_pdk.py
==========================================
GOPRO_Large 데이터셋으로 PDK 검증

데이터셋 특성 (RF와의 차이):
  GOPRO_Large (CVPR 2017, Nah et al.):
    - 같은 GoPro Hero4 Black 카메라로 촬영
    - sharp 연속 프레임을 평균내어 blur 생성
    - 동일 소자 → single-device 가정 성립
    - degradation: spatially variant motion blur
    - scene별 폴더 구조: GOPRO{n1}_{n2}_{n3}/blur/, sharp/

  RF와 다른 점:
    - RF: multi-device (환자마다 다른 장비) → 일반화 불가
    - GOPRO: single-device (같은 카메라) → 일반화 가능 기대

폴더 구조:
  ./data/GOPRO_Large/
    train/
      GOPRO_11_01_001/
        blur/         ← degraded (motion blur)
        blur_gamma/   ← gamma corrected blur
        sharp/        ← clean GT
      GOPRO_11_01_002/
        ...
    test/
      ...

구성 (19번과 동일):
  Train : N개 scene의 blur/sharp 쌍으로 kernel 학습
  Test  : unseen scene의 blur에 고정 kernel 적용
  Oracle: unseen 이미지 각각 직접 학습 (upper bound)

PDK 모델:
  PSFPDK    - spatially variant PSF (motion blur에 가장 적합)
  VignettingPDK - 비교용

Run:
  python 21_gopro_pdk.py
  python 21_gopro_pdk.py --n-train 10 --n-inf 4 --use-gamma
  python 21_gopro_pdk.py --n-scenes-train 5 --imgs-per-scene 3
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

def ssim_simple(a, b):
    a = a.float(); b = b.float()
    mu_a = F.avg_pool2d(a, 11, stride=1, padding=5)
    mu_b = F.avg_pool2d(b, 11, stride=1, padding=5)
    mu_a2 = mu_a**2; mu_b2 = mu_b**2; mu_ab = mu_a*mu_b
    sig_a2 = F.avg_pool2d(a**2, 11, stride=1, padding=5) - mu_a2
    sig_b2 = F.avg_pool2d(b**2, 11, stride=1, padding=5) - mu_b2
    sig_ab = F.avg_pool2d(a*b,  11, stride=1, padding=5) - mu_ab
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_ab+c1)*(2*sig_ab+c2)) / ((mu_a2+mu_b2+c1)*(sig_a2+sig_b2+c2))
    return ssim_map.mean().item()

def to_np(t):
    return t.squeeze().detach().cpu().float().numpy()

def load_single_image(path, size):
    img = Image.open(path).convert("L")
    w, h = img.size
    # center crop to square
    s = min(w, h)
    img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
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
# GOPRO Dataset Loader
# ============================================================================

def get_gopro_scenes(base_dir):
    """
    base_dir (train or test) 아래 GOPRO{n1}_{n2}_{n3} 폴더 목록 반환.
    각 scene은 blur/, sharp/ (또는 blur_gamma/) 포함.
    """
    scenes = []
    if not os.path.isdir(base_dir):
        print(f"  [Error] Not found: {base_dir}")
        return scenes

    for name in sorted(os.listdir(base_dir)):
        scene_dir = os.path.join(base_dir, name)
        if not os.path.isdir(scene_dir):
            continue
        blur_dir  = os.path.join(scene_dir, "blur")
        sharp_dir = os.path.join(scene_dir, "sharp")
        blur_gamma_dir = os.path.join(scene_dir, "blur_gamma")
        if os.path.isdir(sharp_dir) and (
                os.path.isdir(blur_dir) or os.path.isdir(blur_gamma_dir)):
            scenes.append({
                "name":       name,
                "blur":       blur_dir,
                "blur_gamma": blur_gamma_dir,
                "sharp":      sharp_dir,
            })
    print(f"  [GOPRO] Found {len(scenes)} scenes in {base_dir}")
    return scenes


def load_gopro_pairs(scenes, n_pairs, size, use_gamma=False):
    """
    여러 scene에서 blur-sharp 쌍을 로드.
    scene당 균등하게 샘플링.

    use_gamma: blur_gamma 사용 여부 (False = linear blur 사용)
    """
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG")

    # scene별 이미지 목록 수집
    all_pairs = []
    for sc in scenes:
        blur_key = "blur_gamma" if use_gamma else "blur"
        blur_dir = sc.get(blur_key, sc["blur"])
        if not os.path.isdir(blur_dir):
            blur_dir = sc["blur"]

        blur_paths  = []
        sharp_paths = []
        for ext in exts:
            blur_paths  += glob.glob(os.path.join(blur_dir,  ext))
            sharp_paths += glob.glob(os.path.join(sc["sharp"], ext))

        blur_paths  = sorted(set(blur_paths))
        sharp_paths = sorted(set(sharp_paths))

        # 파일명 기준 매칭
        blur_names  = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in blur_paths}
        sharp_names = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in sharp_paths}
        common = sorted(set(blur_names.keys()) & set(sharp_names.keys()))
        for n in common:
            all_pairs.append((blur_names[n], sharp_names[n], sc["name"]))

    if not all_pairs:
        print("  [Error] No pairs found")
        return [], []

    # 랜덤 샘플링
    random.shuffle(all_pairs)
    selected = all_pairs[:n_pairs]

    degs, cleans = [], []
    for bp, sp, scene_name in selected:
        try:
            degs.append(load_single_image(bp, size))
            cleans.append(load_single_image(sp, size))
        except Exception as e:
            print(f"  [skip] {e}")
            if len(degs) > len(cleans): degs.pop()

    print(f"  [Load] {len(degs)} pairs loaded "
          f"({'gamma' if use_gamma else 'linear'} blur, size={size})")
    return degs, cleans


def load_gopro_by_scene(scenes, n_scenes, imgs_per_scene, size, use_gamma=False):
    """
    scene 단위로 로드: n_scenes개 scene, 각 scene에서 imgs_per_scene장.
    scene 간 blur 특성 다양성 확보.
    """
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG")
    random.shuffle(scenes)
    selected_scenes = scenes[:n_scenes]

    degs, cleans = [], []
    for sc in selected_scenes:
        blur_key = "blur_gamma" if use_gamma else "blur"
        blur_dir = sc.get(blur_key, sc["blur"])
        if not os.path.isdir(blur_dir):
            blur_dir = sc["blur"]

        blur_paths  = []
        sharp_paths = []
        for ext in exts:
            blur_paths  += glob.glob(os.path.join(blur_dir,  ext))
            sharp_paths += glob.glob(os.path.join(sc["sharp"], ext))

        blur_paths  = sorted(set(blur_paths))
        sharp_paths = sorted(set(sharp_paths))

        blur_names  = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in blur_paths}
        sharp_names = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in sharp_paths}
        common = sorted(set(blur_names.keys()) & set(sharp_names.keys()))
        random.shuffle(common)

        for n in common[:imgs_per_scene]:
            try:
                degs.append(load_single_image(blur_names[n], size))
                cleans.append(load_single_image(sharp_names[n], size))
            except Exception as e:
                print(f"  [skip] {sc['name']}/{n}: {e}")

    print(f"  [Load] {len(degs)} pairs from {len(selected_scenes)} scenes")
    return degs, cleans


# ============================================================================
# PDK Models
# ============================================================================

class PSFPDK(nn.Module):
    """
    PSF blur 보정 PDK.
    GOPRO motion blur: spatially variant → radial PSF로 근사.
    unsharp masking 방식: delta + lambda*(delta - gaussian)
    """
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
        sigma = (self.sigma0 + self.alpha_psf*self.r2).unsqueeze(-1).clamp(0.1, 1.0)
        d2 = (self.coords**2).sum(-1)
        g  = torch.exp(-d2/(2*sigma**2))
        g  = g / g.sum(-1, keepdim=True)
        delta = torch.zeros_like(g); delta[:,:,4] = 1.0
        lam = (self.lambda0 + self.alpha_lam*self.r2).unsqueeze(-1).clamp(0.05, 3.0)
        return spatially_varying_conv(x, delta + lam*(delta - g))

    def get_params(self):
        return {"sigma0":    self.sigma0.item(),
                "alpha_psf": self.alpha_psf.item(),
                "lambda0":   self.lambda0.item(),
                "alpha_lam": self.alpha_lam.item()}


class VignettingPDK(nn.Module):
    """비교용: Vignetting PDK"""
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

    def get_params(self):
        return {"alpha": self.alpha.item()}


class GlobalKernel(nn.Module):
    """비교용: Global (spatially invariant) kernel"""
    def __init__(self):
        super().__init__()
        init = torch.zeros(9); init[4] = 3.0
        self.kernel_logits = nn.Parameter(init)

    def forward(self, x):
        k = torch.softmax(self.kernel_logits, dim=0)
        B, C, H, W = x.shape
        kmap = k.view(1,1,9).expand(H,W,9)
        return spatially_varying_conv(x, kmap)


# ============================================================================
# Training
# ============================================================================

def train_on_pairs(model, deg_imgs, clean_imgs, n_iter, lr, verbose=True):
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
        if verbose and (i+1) % 100 == 0:
            print(f"    iter {i+1:4d}/{n_iter}  loss={total.item():.6f}")
    return history


def train_oracle(model, deg, clean, n_iter, lr):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    for i in range(n_iter):
        opt.zero_grad()
        loss_fn(model(deg), clean).backward()
        opt.step(); sched.step()
    with torch.no_grad():
        return model(deg).clamp(0, 1)


def inference_fixed(model, deg):
    model.eval()
    with torch.no_grad():
        return model(deg).clamp(0, 1)


# ============================================================================
# N-sweep
# ============================================================================

def run_n_sweep(train_degs, train_clns, inf_degs, inf_clns,
                Model, H, W, n_iter, lr, n_list):
    sweep = {}
    for n in n_list:
        m = Model(H, W)
        train_on_pairs(m, train_degs[:n], train_clns[:n],
                       n_iter, lr, verbose=False)
        ps = [psnr(c, inference_fixed(m, d))
              for d, c in zip(inf_degs, inf_clns)]
        sweep[n] = np.mean(ps)
        print(f"    N={n:3d}  PSNR={sweep[n]:.2f}dB")
    return sweep


# ============================================================================
# Visualization
# ============================================================================

def save_comparison(inf_results, res_dir):
    """
    N rows x 6 cols:
    [sharp | blur | Global | VignettingPDK | PSFPDK | Oracle(PSF)]
    """
    n = len(inf_results)
    fig, axes = plt.subplots(n, 6, figsize=(22, 4*n+0.5))
    if n == 1: axes = axes[np.newaxis,:]

    col_titles = ["1. Sharp (GT)", "2. Blur (real)",
                  "3. Global kernel", "4. VignettingPDK",
                  "5. PSFPDK (ours)", "6. Oracle (PSF)"]
    for c, t in enumerate(col_titles):
        axes[0,c].set_title(t, fontsize=9, fontweight="bold", pad=5)

    colors = ["dimgray", "steelblue", "tomato", "darkorange", "green"]

    for r, res in enumerate(inf_results):
        imgs = [to_np(res["sharp"]),   to_np(res["blur"]),
                to_np(res["glb_cor"]), to_np(res["vig_cor"]),
                to_np(res["psf_cor"]), to_np(res["ora_cor"])]

        for c, img in enumerate(imgs):
            axes[r,c].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[r,c].axis("off")

        axes[r,0].set_ylabel(res.get("scene",""), fontsize=7,
                             rotation=0, labelpad=50, va="center")

        psnr_vals = [res["psnr_blur"], res["psnr_glb"],
                     res["psnr_vig"],  res["psnr_psf"], res["psnr_ora"]]
        for c, (pv, col) in enumerate(zip(psnr_vals, colors), start=1):
            axes[r,c].text(0.5,-0.07, f"PSNR={pv:.1f}dB",
                           transform=axes[r,c].transAxes,
                           ha="center", va="top", fontsize=8, color=col)

        gap_deg = res["psnr_psf"] - res["psnr_blur"]
        gap_ora = res["psnr_psf"] - res["psnr_ora"]
        axes[r,4].text(0.5,-0.14,
                       f"vs Blur: {gap_deg:+.1f}  vs Oracle: {gap_ora:+.1f}",
                       transform=axes[r,4].transAxes,
                       ha="center", va="top", fontsize=7, color="gray")

    fig.suptitle("GOPRO_Large — Dataset-PDK vs Oracle (unseen scenes)",
                 fontsize=13, y=1.01)
    plt.tight_layout(rect=[0,0.05,1,1])
    path = os.path.join(res_dir, "21_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_psnr_summary(inf_results, res_dir):
    keys   = ["psnr_blur","psnr_glb","psnr_vig","psnr_psf","psnr_ora"]
    labels = ["Blur","Global","VignettingPDK","PSFPDK","Oracle(PSF)"]
    colors = ["#aaaaaa","steelblue","tomato","darkorange","green"]
    means  = [np.mean([r[k] for r in inf_results]) for k in keys]

    x = np.arange(len(labels)); w = 0.55
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(x, means, width=w, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("PSNR (dB)", fontsize=10)
    ax.set_title("GOPRO_Large — PSNR: Global vs PDK vs Oracle",
                 fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v, col in zip(bars, means, colors):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.1,
                f"{v:.2f}", ha="center", fontsize=9, color=col)
    plt.tight_layout()
    path = os.path.join(res_dir, "21_psnr_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_loss_curves(psf_hist, vig_hist, glb_hist, res_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(psf_hist, color="darkorange", linewidth=1.5, label="PSFPDK")
    ax.plot(vig_hist, color="tomato",     linewidth=1.5, label="VignettingPDK")
    ax.plot(glb_hist, color="steelblue",  linewidth=1.5, label="Global kernel")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
    ax.set_title("Training loss: GOPRO_Large", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(res_dir, "21_loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_n_sweep(sweep_psf, sweep_vig, oracle_psnr, res_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, sweep, label, col in [
        (axes[0], sweep_psf, "PSFPDK",       "darkorange"),
        (axes[1], sweep_vig, "VignettingPDK", "tomato"),
    ]:
        ns   = sorted(sweep.keys())
        vals = [sweep[n] for n in ns]
        ax.plot(ns, vals, "o-", color=col, linewidth=2,
                markersize=7, label=label)
        ax.axhline(oracle_psnr, color="green", linewidth=1.5,
                   linestyle=":", label=f"Oracle ({oracle_psnr:.1f}dB)")
        ax.set_xlabel("# training pairs (N)", fontsize=9)
        ax.set_ylabel("PSNR (dB)", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xticks(ns)

    fig.suptitle("GOPRO_Large: PSNR vs # training pairs", fontsize=12)
    plt.tight_layout()
    path = os.path.join(res_dir, "21_n_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_scene_breakdown(inf_results, res_dir):
    """scene별 PSNR 막대 (GOPRO 특성: scene마다 blur 정도 다름)"""
    scenes  = [r.get("scene","") for r in inf_results]
    p_blur  = [r["psnr_blur"] for r in inf_results]
    p_psf   = [r["psnr_psf"]  for r in inf_results]
    p_ora   = [r["psnr_ora"]  for r in inf_results]

    x = np.arange(len(scenes)); w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, 2.5*len(scenes)), 4.5))
    ax.bar(x-w,  p_blur, w, label="Blur",        color="#aaaaaa")
    ax.bar(x,    p_psf,  w, label="PSFPDK",      color="darkorange")
    ax.bar(x+w,  p_ora,  w, label="Oracle(PSF)", color="green", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([s[:20] for s in scenes], rotation=20,
                       ha="right", fontsize=7)
    ax.set_ylabel("PSNR (dB)", fontsize=10)
    ax.set_title("GOPRO: per-scene PSNR breakdown", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(res_dir, "21_scene_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


# ============================================================================
# Args & Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="GOPRO_Large PDK experiment")
    p.add_argument("--train-dir",  type=str,
                   default="./data/GOPRO_Large/train",
                   help="GOPRO train 루트 (scene 폴더들 포함)")
    p.add_argument("--inf-dir",    type=str,
                   default="./data/GOPRO_Large/test",
                   help="GOPRO test 루트")
    p.add_argument("--n-train",    type=int, default=10,
                   help="train 이미지 수 (scene에서 랜덤 샘플)")
    p.add_argument("--n-inf",      type=int, default=4,
                   help="inference 이미지 수")
    p.add_argument("--n-scenes-train", type=int, default=0,
                   help="scene 단위 샘플 시 사용할 scene 수 (0=비활성)")
    p.add_argument("--imgs-per-scene", type=int, default=3,
                   help="scene당 이미지 수 (n-scenes-train 활성 시)")
    p.add_argument("--use-gamma",  action="store_true",
                   help="blur_gamma 사용 (기본: linear blur)")
    p.add_argument("--img-size",   type=int, default=256)
    p.add_argument("--n-iter",     type=int, default=500)
    p.add_argument("--n-ora-iter", type=int, default=500)
    p.add_argument("--lr",         type=float, default=0.02)
    p.add_argument("--n-sweep",    type=str, default="5,10,20",
                   help="N-sweep 값 (쉼표 구분)")
    p.add_argument("--res-dir",    type=str, default="./res/gopro_pdk")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.res_dir, exist_ok=True)
    H = W = args.img_size
    n_sweep_list = [int(x) for x in args.n_sweep.split(",")]

    # ── Scene 목록 수집 ───────────────────────────────────────────
    print("[GOPRO] Scanning scenes...")
    train_scenes = get_gopro_scenes(args.train_dir)
    inf_scenes   = get_gopro_scenes(args.inf_dir)

    if not train_scenes or not inf_scenes:
        print("[Error] No scenes found"); return

    # ── 이미지 로드 ───────────────────────────────────────────────
    max_n_train = max(n_sweep_list + [args.n_train])

    if args.n_scenes_train > 0:
        # scene 단위 샘플링
        print(f"\n[Load] Train: {args.n_scenes_train} scenes x "
              f"{args.imgs_per_scene} imgs/scene...")
        tr_degs, tr_clns = load_gopro_by_scene(
            train_scenes, args.n_scenes_train,
            args.imgs_per_scene, H, args.use_gamma)
    else:
        # 전체에서 랜덤 샘플링
        print(f"\n[Load] Train: {max_n_train} pairs (random)...")
        tr_degs, tr_clns = load_gopro_pairs(
            train_scenes, max_n_train, H, args.use_gamma)

    print(f"[Load] Inference: {args.n_inf} pairs (unseen scenes)...")
    inf_degs, inf_clns = load_gopro_pairs(
        inf_scenes, args.n_inf, H, args.use_gamma)

    # scene 이름 추적 (breakdown용)
    inf_scene_names = []
    for sc in inf_scenes:
        exts = ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG")
        paths = []
        for ext in exts:
            paths += glob.glob(os.path.join(
                sc["blur_gamma"] if args.use_gamma else sc["blur"], ext))
        if paths:
            inf_scene_names.extend([sc["name"]] * min(1, len(paths)))

    if not tr_degs or not inf_degs:
        print("[Error] No images loaded"); return

    n_sweep_list = sorted(set(min(n, len(tr_degs)) for n in n_sweep_list))
    print(f"\n  Train: {len(tr_degs)} pairs")
    print(f"  Inference: {len(inf_degs)} pairs (unseen)")
    print(f"  img_size: {H}x{W}  n_sweep: {n_sweep_list}")
    print(f"  blur_type: {'gamma' if args.use_gamma else 'linear'}")

    # ── 학습 ──────────────────────────────────────────────────────
    print("\n[Train] PSFPDK...")
    psf_model = PSFPDK(H, W)
    psf_hist  = train_on_pairs(psf_model, tr_degs[:args.n_train],
                                tr_clns[:args.n_train],
                                args.n_iter, args.lr)

    print("\n[Train] VignettingPDK...")
    vig_model = VignettingPDK(H, W)
    vig_hist  = train_on_pairs(vig_model, tr_degs[:args.n_train],
                                tr_clns[:args.n_train],
                                args.n_iter, args.lr, verbose=False)

    print("\n[Train] Global kernel...")
    glb_model = GlobalKernel()
    glb_hist  = train_on_pairs(glb_model, tr_degs[:args.n_train],
                                tr_clns[:args.n_train],
                                args.n_iter, args.lr, verbose=False)

    # ── Inference ─────────────────────────────────────────────────
    print(f"\n[Inference] {len(inf_degs)} unseen pairs...")
    inf_results = []

    for i, (deg, clean) in enumerate(zip(inf_degs, inf_clns)):
        psf_cor = inference_fixed(psf_model, deg)
        vig_cor = inference_fixed(vig_model, deg)
        glb_cor = inference_fixed(glb_model, deg)

        print(f"  Oracle {i+1}/{len(inf_degs)}...")
        ora_model = PSFPDK(H, W)
        ora_cor   = train_oracle(ora_model, deg, clean,
                                  args.n_ora_iter, args.lr)

        scene_name = (inf_scene_names[i]
                      if i < len(inf_scene_names) else f"scene_{i}")

        r = {
            "sharp":    clean,
            "blur":     deg,
            "psf_cor":  psf_cor,
            "vig_cor":  vig_cor,
            "glb_cor":  glb_cor,
            "ora_cor":  ora_cor,
            "psnr_blur": psnr(clean, deg),
            "psnr_psf":  psnr(clean, psf_cor),
            "psnr_vig":  psnr(clean, vig_cor),
            "psnr_glb":  psnr(clean, glb_cor),
            "psnr_ora":  psnr(clean, ora_cor),
            "scene":     scene_name,
        }
        inf_results.append(r)
        print(f"    Blur={r['psnr_blur']:.1f}  Glb={r['psnr_glb']:.1f}"
              f"  Vig={r['psnr_vig']:.1f}  PSF={r['psnr_psf']:.1f}"
              f"  Ora={r['psnr_ora']:.1f}")

    # 평균 PSNR
    mean = {k: np.mean([r[k] for r in inf_results])
            for k in ["psnr_blur","psnr_glb","psnr_vig","psnr_psf","psnr_ora"]}
    oracle_mean = mean["psnr_ora"]

    print("\n[Summary]")
    print(f"  {'Blur':<14}: {mean['psnr_blur']:.2f} dB")
    print(f"  {'Global':<14}: {mean['psnr_glb']:.2f} dB  "
          f"({mean['psnr_glb']-mean['psnr_blur']:+.2f})")
    print(f"  {'VignettingPDK':<14}: {mean['psnr_vig']:.2f} dB  "
          f"({mean['psnr_vig']-mean['psnr_blur']:+.2f})")
    print(f"  {'PSFPDK':<14}: {mean['psnr_psf']:.2f} dB  "
          f"({mean['psnr_psf']-mean['psnr_blur']:+.2f})")
    print(f"  {'Oracle(PSF)':<14}: {mean['psnr_ora']:.2f} dB  "
          f"({mean['psnr_ora']-mean['psnr_blur']:+.2f})")
    print(f"  PSFPDK vs Oracle: {mean['psnr_psf']-mean['psnr_ora']:+.2f} dB")

    # ── N-sweep ───────────────────────────────────────────────────
    print(f"\n[N-sweep] PSF: {n_sweep_list}...")
    sweep_psf = run_n_sweep(tr_degs, tr_clns, inf_degs, inf_clns,
                             PSFPDK, H, W, args.n_iter, args.lr, n_sweep_list)

    print(f"[N-sweep] Vignetting: {n_sweep_list}...")
    sweep_vig = run_n_sweep(tr_degs, tr_clns, inf_degs, inf_clns,
                             VignettingPDK, H, W,
                             args.n_iter, args.lr, n_sweep_list)

    # ── 시각화 ────────────────────────────────────────────────────
    print("\n[Saving results...]")
    save_comparison(inf_results, args.res_dir)
    save_psnr_summary(inf_results, args.res_dir)
    save_loss_curves(psf_hist, vig_hist, glb_hist, args.res_dir)
    save_n_sweep(sweep_psf, sweep_vig, oracle_mean, args.res_dir)
    save_scene_breakdown(inf_results, args.res_dir)

    print(f"\n[Done] {os.path.abspath(args.res_dir)}")
    print("  21_comparison.png      -- unseen scene 비교")
    print("  21_psnr_summary.png    -- 모델별 PSNR")
    print("  21_loss_curves.png     -- 학습 곡선")
    print("  21_n_sweep.png         -- N에 따른 PSNR")
    print("  21_scene_breakdown.png -- scene별 PSNR")


import glob
if __name__ == "__main__":
    main()