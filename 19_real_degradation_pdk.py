"""
19_real_degradation_pdk.py
==========================================
Real Degradation Dataset으로 PDK 검증

데이터셋:
  RF  (Real Fundus)  : 실제 안저 카메라 degradation
                       low-quality (degraded) <-> high-quality (clean) 쌍
                       주요 degradation: vignetting, non-uniform illumination, PSF blur

  RealBlur           : 실제 렌즈/카메라 motion blur
                       blurry <-> sharp 쌍
                       주요 degradation: spatially variant PSF blur

구성 (18번과 동일):
  Train : dataset 이미지 N장으로 kernel 학습
  Test  : unseen 이미지에 고정 kernel 적용
  Oracle: unseen 이미지 각각 직접 학습 (upper bound)

차이점 (시뮬레이션 대비):
  degradation 파라미터를 모름 (실제 소자/렌즈)
  → 파라미터 비교 없음, PSNR + 이미지 품질만 평가
  → PDK가 실제 degradation도 학습할 수 있는지 검증

실행:
  # RF 데이터셋
  python 19_real_degradation_pdk.py \
    --dataset rf \
    --train-dir ./data/RF/train \
    --inf-dir   ./data/RF/test

  # RealBlur 데이터셋
  python 19_real_degradation_pdk.py \
    --dataset realblur \
    --train-dir ./data/RealBlur/train \
    --inf-dir   ./data/RealBlur/test

데이터셋 구조 가정:
  RF:
    train/low/  : 저품질 (degraded)
    train/high/ : 고품질 (clean)
    test/low/   : 저품질
    test/high/  : 고품질

  RealBlur:
    train/blur/  : blurry
    train/sharp/ : sharp
    test/blur/   : blurry
    test/sharp/  : sharp
"""

import argparse, os, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r = (gy**2 + gx**2).sqrt() / (2**0.5)
    return r, gy, gx

def psnr(a, b):
    mse = F.mse_loss(a.float(), b.float()).item()
    return 99.9 if mse < 1e-10 else 10 * np.log10(1.0 / mse)

def ssim_simple(a, b):
    """간단한 SSIM 근사 (window=11)"""
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
# Dataset loader: paired (degraded, clean)
# ============================================================================

def find_paired_images(deg_dir, clean_dir):
    """
    deg_dir / clean_dir에서 같은 파일명의 paired 이미지 찾기.
    파일명이 다른 경우 정렬 순서로 매핑.
    """
    exts = ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG","*.bmp","*.BMP")
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

    # 파일명 기준으로 매칭 시도
    deg_names   = {os.path.splitext(os.path.basename(p))[0]: p for p in deg_paths}
    clean_names = {os.path.splitext(os.path.basename(p))[0]: p for p in clean_paths}
    common = sorted(set(deg_names.keys()) & set(clean_names.keys()))

    if common:
        pairs = [(deg_names[n], clean_names[n]) for n in common]
        print(f"  [Paired] {len(pairs)} matched by filename")
    else:
        # 정렬 순서로 매핑
        n = min(len(deg_paths), len(clean_paths))
        pairs = list(zip(deg_paths[:n], clean_paths[:n]))
        print(f"  [Paired] {len(pairs)} matched by sort order")

    return pairs


def load_paired_dataset(deg_dir, clean_dir, n, size):
    """degraded / clean 쌍 로드"""
    pairs = find_paired_images(deg_dir, clean_dir)
    if not pairs:
        print(f"  [Error] No paired images: deg={deg_dir}, clean={clean_dir}")
        return [], []

    random.shuffle(pairs)
    deg_imgs, clean_imgs = [], []
    for dp, cp in pairs[:n]:
        try:
            deg_imgs.append(load_single_image(dp, size))
            clean_imgs.append(load_single_image(cp, size))
        except Exception as e:
            print(f"  [Warning] skip {dp}: {e}")
            if len(deg_imgs) > len(clean_imgs):
                deg_imgs.pop()

    print(f"  [Load] {len(deg_imgs)} pairs loaded (size={size})")
    return deg_imgs, clean_imgs


def infer_deg_clean_dirs(base_dir, dataset):
    """
    base_dir 하위에서 degraded/clean 폴더 자동 감지.
    RF:       low/ , high/
    RealBlur: blur/, sharp/
    generic:  degraded/ or blur/ or low/ -> deg
              sharp/ or clean/ or high/  -> clean
    """
    candidates_deg   = ["input","low","blur","degraded","blurry","lq"]
    candidates_clean = ["gt","high","sharp","clean","target","hq"]

    sub = [d for d in os.listdir(base_dir)
           if os.path.isdir(os.path.join(base_dir, d))]
    sub_lower = [s.lower() for s in sub]

    deg_dir = clean_dir = None
    for c in candidates_deg:
        if c in sub_lower:
            deg_dir = os.path.join(base_dir, sub[sub_lower.index(c)])
            break
    for c in candidates_clean:
        if c in sub_lower:
            clean_dir = os.path.join(base_dir, sub[sub_lower.index(c)])
            break

    if deg_dir and clean_dir:
        print(f"  [Auto-detect] deg={deg_dir}, clean={clean_dir}")
    return deg_dir, clean_dir


# ============================================================================
# Calibration patterns (18번과 동일)
# ============================================================================

def make_cal_patterns(size, device="cpu"):
    H = W = size
    ys = torch.linspace(-1,1,H,device=device)
    xs = torch.linspace(-1,1,W,device=device)
    gy,gx = torch.meshgrid(ys,xs,indexing="ij")
    gen=torch.Generator(); gen.manual_seed(42)
    raw=torch.rand(1,1,H,W,generator=gen,device=device)
    k_size=15; sig=5.0
    c1d=torch.arange(k_size,device=device).float()-k_size//2
    g1d=torch.exp(-c1d**2/(2*sig**2)); g1d=g1d/g1d.sum()
    gk=(g1d.unsqueeze(0)*g1d.unsqueeze(1)).view(1,1,k_size,k_size)
    sr=F.conv2d(raw,gk,padding=k_size//2).squeeze()
    sr=(sr-sr.min())/(sr.max()-sr.min()+1e-5)
    pats={
        "checker_fine":   (0.5+0.5*torch.sign(torch.sin(gy*12)*torch.sin(gx*12))).clamp(0,1),
        "checker_coarse": (0.5+0.5*torch.sign(torch.sin(gy*6)*torch.sin(gx*6))).clamp(0,1),
        "sine_h":         (0.5+0.5*torch.sin(gy*15)).clamp(0,1),
        "sine_v":         (0.5+0.5*torch.sin(gx*15)).clamp(0,1),
        "sine_diag":      (0.5+0.5*torch.sin((gy+gx)*10)).clamp(0,1),
        "concentric":     (0.5+0.5*torch.cos((gy**2+gx**2).sqrt()*20)).clamp(0,1),
        "gradient_x":     ((gx+1)/2).clamp(0,1),
        "gradient_y":     ((gy+1)/2).clamp(0,1),
        "gradient_radial":(1.0-(gy**2+gx**2).sqrt()/(2**0.5)).clamp(0,1),
        "smooth_random":  sr,
    }
    return {k: v.unsqueeze(0).unsqueeze(0) for k,v in pats.items()}


# ============================================================================
# PDK Models
# ============================================================================

class VignettingPDK(nn.Module):
    """Vignetting 보정: 1x1 PD gain"""
    def __init__(self, H, W):
        super().__init__()
        self._alpha=nn.Parameter(torch.tensor(float(np.log(np.exp(1.0)-1.0))))
        r,_,_=radial_map(H,W); self.register_buffer("r2",r**2)
    @property
    def alpha(self): return F.softplus(self._alpha)
    def forward(self, x):
        gain=1.0+self.alpha*self.r2
        zeros=torch.zeros(*self.r2.shape,9,device=x.device)
        idx=torch.full((*self.r2.shape,1),4,dtype=torch.long,device=x.device)
        k=zeros.scatter(2,idx,gain.unsqueeze(-1))
        return spatially_varying_conv(x,k)
    def get_params(self): return {"alpha":self.alpha.item()}


class PSFPDK(nn.Module):
    """PSF blur 보정: unsharp mask PD kernel"""
    def __init__(self, H, W):
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._s0=nn.Parameter(torch.tensor(_sp(0.5)))
        self._aps=nn.Parameter(torch.tensor(_sp(0.5)))
        self._l0=nn.Parameter(torch.tensor(_sp(0.5)))
        self._al=nn.Parameter(torch.tensor(_sp(0.3)))
        r,_,_=radial_map(H,W)
        self.register_buffer("r2",r**2)
        self.register_buffer("coords",torch.tensor([
            [-1.,-1.],[-1.,0.],[-1.,1.],
            [ 0.,-1.],[ 0.,0.],[ 0.,1.],
            [ 1.,-1.],[ 1.,0.],[ 1.,1.]]))
    @property
    def sigma0(self): return F.softplus(self._s0)
    @property
    def alpha_psf(self): return F.softplus(self._aps)
    @property
    def lambda0(self): return F.softplus(self._l0)
    @property
    def alpha_lam(self): return F.softplus(self._al)
    def forward(self, x):
        sigma=(self.sigma0+self.alpha_psf*self.r2).unsqueeze(-1).clamp(0.1,1.0)
        d2=(self.coords**2).sum(-1)
        g=torch.exp(-d2/(2*sigma**2)); g=g/g.sum(-1,keepdim=True)
        delta=torch.zeros_like(g); delta[:,:,4]=1.0
        lam=(self.lambda0+self.alpha_lam*self.r2).unsqueeze(-1).clamp(0.05,3.0)
        return spatially_varying_conv(x,delta+lam*(delta-g))
    def get_params(self):
        return {"sigma0":self.sigma0.item(),"alpha_psf":self.alpha_psf.item()}


class NonUnifPDK(nn.Module):
    """Non-uniform illumination 보정"""
    def __init__(self, H, W):
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._cx=nn.Parameter(torch.tensor(0.0))
        self._cy=nn.Parameter(torch.tensor(0.0))
        self._sig=nn.Parameter(torch.tensor(_sp(0.6)))
        self._ell=nn.Parameter(torch.tensor(_sp(1.0)))
        self._av=nn.Parameter(torch.tensor(_sp(1.0)))
        r,gy,gx=radial_map(H,W)
        self.register_buffer("r2",r**2)
        self.register_buffer("gy",gy)
        self.register_buffer("gx",gx)
    @property
    def cx(self): return torch.tanh(self._cx)*0.5
    @property
    def cy(self): return torch.tanh(self._cy)*0.5
    @property
    def sig(self): return F.softplus(self._sig).clamp(0.2,2.0)
    @property
    def ell(self): return F.softplus(self._ell).clamp(0.3,3.0)
    @property
    def av(self): return F.softplus(self._av)
    def forward(self, x):
        B,C,H,W=x.shape
        V_rad=1.0/(1.0+self.av*self.r2)
        dy=self.gy-self.cy; dx=self.gx-self.cx
        d2=dx**2+(dy*self.ell)**2
        ill=torch.exp(-d2/(2*self.sig**2)); ill=ill/ill.max().clamp(1e-5)
        V=(V_rad*(0.3+0.7*ill)).clamp(1e-3,1.0)
        gain=(1.0/V).clamp(1.0,10.0)
        zeros=torch.zeros(H,W,9,device=x.device)
        idx=torch.full((H,W,1),4,dtype=torch.long,device=x.device)
        k=zeros.scatter(2,idx,gain.unsqueeze(-1))
        return spatially_varying_conv(x,k)
    def get_params(self):
        return {"cx":self.cx.item(),"cy":self.cy.item(),
                "sig":self.sig.item(),"ell":self.ell.item()}


class GlobalKernel(nn.Module):
    def __init__(self):
        super().__init__()
        init=torch.zeros(9); init[4]=3.0
        self.kernel_logits=nn.Parameter(init)
    def forward(self, x):
        k=torch.softmax(self.kernel_logits,dim=0)
        return global_conv(x,k)


# ============================================================================
# Training
# ============================================================================

def train_on_pairs(model, deg_imgs, clean_imgs, n_iter, lr, verbose=True):
    """
    실제 paired 이미지로 학습.
    degraded -> clean 방향으로 kernel 학습.
    """
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


def train_on_patterns(model, cal_clean, deg_fn_approx,
                      n_iter, lr, verbose=False):
    """
    Cal-PDK: calibration pattern으로 학습.
    real dataset에서는 degradation fn을 모르므로
    학습된 degradation 근사 (평균 degradation) 사용.

    실제로는: 패턴으로 소자 특성을 역추적하는 시나리오.
    여기서는 train 이미지로 학습한 degradation 통계를 활용.
    """
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    history = []
    pairs = [(img, deg_fn_approx(img)) for img in cal_clean.values()]

    for i in range(n_iter):
        opt.zero_grad()
        total = torch.tensor(0.0)
        for clean, deg in pairs:
            total = total + loss_fn(model(deg), clean)
        total = total / len(pairs)
        total.backward(); opt.step(); sched.step()
        history.append(total.item())
    return history


def train_oracle(model, deg, clean, n_iter, lr):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    for i in range(n_iter):
        opt.zero_grad()
        loss_fn(model(deg), clean).backward()
        opt.step(); sched.step()
    with torch.no_grad():
        return model(deg).clamp(0,1)


def inference_fixed(model, deg):
    model.eval()
    with torch.no_grad():
        return model(deg).clamp(0,1)


# ============================================================================
# Visualization
# ============================================================================

def save_comparison(results, res_dir, dataset_name):
    """
    N case x 5 col:
    [clean | degraded | dataset-pdk | oracle | diff(dataset-oracle)]
    """
    n = len(results)
    fig, axes = plt.subplots(n, 5, figsize=(20, 4.5*n+0.5))
    if n == 1: axes = axes[np.newaxis,:]

    col_titles = ["1. Clean (GT)", "2. Degraded (real)",
                  "3. Dataset-PDK", "4. Oracle PDK",
                  "5. |Dataset - Oracle|"]
    for c, t in enumerate(col_titles):
        axes[0,c].set_title(t, fontsize=10, fontweight="bold", pad=6)

    for r, res in enumerate(results):
        clean_np = to_np(res["clean"])
        deg_np   = to_np(res["deg"])
        ds_np    = to_np(res["ds_cor"])
        ora_np   = to_np(res["ora_cor"])
        diff_np  = np.abs(ds_np - ora_np)
        vmax_d   = max(diff_np.max(), 0.01)

        for c, (img, cmap, vmin, vmax) in enumerate([
            (clean_np, "gray", 0, 1),
            (deg_np,   "gray", 0, 1),
            (ds_np,    "gray", 0, 1),
            (ora_np,   "gray", 0, 1),
            (diff_np,  "hot",  0, vmax_d),
        ]):
            axes[r,c].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[r,c].axis("off")

        axes[r,0].set_ylabel(f"img {r+1}", fontsize=9, fontweight="bold",
                             rotation=0, labelpad=40, va="center")

        for c, (pv, col) in enumerate(zip(
            [res["psnr_deg"], res["psnr_ds"], res["psnr_ora"]],
            ["dimgray", "darkorange", "green"]), start=1):
            axes[r,c].text(0.5,-0.06, f"PSNR={pv:.1f}dB",
                           transform=axes[r,c].transAxes,
                           ha="center", va="top", fontsize=8.5, color=col)

        gap = res["psnr_ds"] - res["psnr_ora"]
        axes[r,2].text(0.5,-0.12,
                       f"vs Oracle: {gap:+.1f}dB",
                       transform=axes[r,2].transAxes,
                       ha="center", va="top", fontsize=7.5, color="gray")

    fig.suptitle(f"Real Degradation PDK [{dataset_name}] — Unseen test images",
                 fontsize=13, y=1.01)
    plt.tight_layout(rect=[0,0.04,1,1])
    path = os.path.join(res_dir, "19_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_psnr_summary(all_model_results, res_dir, dataset_name):
    """
    Model별 평균 PSNR 비교.
    bar chart: Degraded / Dataset-PDK / Oracle
    """
    models  = list(all_model_results.keys())
    psnr_d  = [np.mean(all_model_results[m]["psnr_degs"]) for m in models]
    psnr_ds = [np.mean(all_model_results[m]["psnr_dss"])  for m in models]
    psnr_o  = [np.mean(all_model_results[m]["psnr_oras"]) for m in models]

    x = np.arange(len(models)); w = 0.22
    fig, ax = plt.subplots(figsize=(max(8, 3*len(models)), 5))
    ax.bar(x-w,  psnr_d,  w, label="Degraded",    color="#aaaaaa")
    ax.bar(x,    psnr_ds, w, label="Dataset-PDK",  color="darkorange")
    ax.bar(x+w,  psnr_o,  w, label="Oracle PDK",   color="green", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("PSNR (dB)", fontsize=10)
    ax.set_title(f"Real Degradation PSNR [{dataset_name}]", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    for xi, (d, ds, o) in enumerate(zip(psnr_d, psnr_ds, psnr_o)):
        ax.text(xi-w, d+0.3,  f"{d:.1f}",  ha="center", fontsize=7, color="#666")
        ax.text(xi,   ds+0.3, f"{ds:.1f}", ha="center", fontsize=7, color="darkorange")
        ax.text(xi+w, o+0.3,  f"{o:.1f}",  ha="center", fontsize=7, color="green")

    plt.tight_layout()
    path = os.path.join(res_dir, "19_psnr_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_training_detail(model_hists, res_dir):
    """학습 곡선 + N-sweep"""
    n = len(model_hists)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]
    for ax, (name, hist) in zip(axes, model_hists.items()):
        ax.plot(hist, color="darkorange", linewidth=1.5)
        ax.set_title(f"Dataset-PDK: {name}", fontsize=10)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Training loss: Dataset-PDK on real paired data", fontsize=12)
    plt.tight_layout()
    path = os.path.join(res_dir, "19_training_loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_n_sweep(sweep_results, oracle_psnrs, res_dir, dataset_name):
    """N 이미지로 학습 시 성능 변화"""
    n_models = len(sweep_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4.5))
    if n_models == 1: axes = [axes]
    colors = ["darkorange", "steelblue", "green", "tomato"]

    for ax, ((model_name, sweep), ora_p, col) in zip(
            axes, zip(sweep_results.items(), oracle_psnrs, colors)):
        ns   = sorted(sweep.keys())
        vals = [sweep[n] for n in ns]
        ax.plot(ns, vals, "o-", color=col, linewidth=2,
                markersize=7, label="Dataset-PDK")
        ax.axhline(ora_p, color="green", linewidth=1.5,
                   linestyle=":", label=f"Oracle ({ora_p:.1f}dB)")
        ax.set_xlabel("# training pairs (N)", fontsize=9)
        ax.set_ylabel("PSNR (dB)", fontsize=9)
        ax.set_title(model_name, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xticks(ns)

    fig.suptitle(f"Dataset-PDK: PSNR vs # training pairs [{dataset_name}]",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(res_dir, "19_n_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


# ============================================================================
# Args & Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Real Degradation PDK [RF / RealBlur]")
    p.add_argument("--dataset",    type=str, default="rf",
                   choices=["rf","realblur","generic"],
                   help="데이터셋 종류 (rf/realblur/generic)")

    # 폴더 직접 지정
    p.add_argument("--train-deg-dir",   type=str, default="",
                   help="train degraded 폴더 (RF: low/, RealBlur: blur/)")
    p.add_argument("--train-clean-dir", type=str, default="",
                   help="train clean 폴더   (RF: high/, RealBlur: sharp/)")
    p.add_argument("--inf-deg-dir",     type=str, default="",
                   help="test degraded 폴더")
    p.add_argument("--inf-clean-dir",   type=str, default="",
                   help="test clean 폴더")

    # 루트 폴더만 지정 (자동 감지)
    p.add_argument("--train-dir",  type=str, default="",
                   help="train 루트 폴더 (하위 low/high 또는 blur/sharp 자동 감지)")
    p.add_argument("--inf-dir",    type=str, default="",
                   help="test 루트 폴더")

    p.add_argument("--n-train",    type=int, default=10)
    p.add_argument("--n-inf",      type=int, default=4)
    p.add_argument("--img-size",   type=int, default=256)
    p.add_argument("--n-iter",     type=int, default=500)
    p.add_argument("--n-ora-iter", type=int, default=500)
    p.add_argument("--lr",         type=float, default=0.02)
    p.add_argument("--n-sweep",    type=str, default="5,10,20")
    p.add_argument("--res-dir",    type=str, default="./res/real_pdk")

    # PDK 모델 선택
    p.add_argument("--pdk-model",  type=str, default="auto",
                   choices=["auto","vignetting","psf","nonunif"],
                   help="PDK 모델 (auto: 데이터셋에 따라 자동 선택)")
    return p.parse_args()


def get_deg_clean_dirs(args):
    """폴더 경로 자동 감지"""
    train_deg = args.train_deg_dir
    train_clean = args.train_clean_dir
    inf_deg = args.inf_deg_dir
    inf_clean = args.inf_clean_dir

    if args.train_dir and (not train_deg or not train_clean):
        d, c = infer_deg_clean_dirs(args.train_dir, args.dataset)
        if not train_deg:   train_deg   = d or ""
        if not train_clean: train_clean = c or ""

    if args.inf_dir and (not inf_deg or not inf_clean):
        d, c = infer_deg_clean_dirs(args.inf_dir, args.dataset)
        if not inf_deg:   inf_deg   = d or ""
        if not inf_clean: inf_clean = c or ""

    return train_deg, train_clean, inf_deg, inf_clean


def select_pdk_models(dataset, pdk_model_arg):
    """데이터셋에 따라 적합한 PDK 모델 선택"""
    if pdk_model_arg == "auto":
        if dataset == "rf":
            # RF: vignetting + non-uniform illumination
            return [
                {"name": "VignettingPDK", "Model": VignettingPDK},
                {"name": "NonUnifPDK",    "Model": NonUnifPDK},
            ]
        elif dataset == "realblur":
            # RealBlur: spatially variant PSF blur
            return [
                {"name": "PSFPDK", "Model": PSFPDK},
            ]
        else:
            return [
                {"name": "VignettingPDK", "Model": VignettingPDK},
                {"name": "PSFPDK",        "Model": PSFPDK},
                {"name": "NonUnifPDK",    "Model": NonUnifPDK},
            ]
    else:
        model_map = {
            "vignetting": VignettingPDK,
            "psf":        PSFPDK,
            "nonunif":    NonUnifPDK,
        }
        M = model_map[pdk_model_arg]
        return [{"name": pdk_model_arg, "Model": M}]


def main():
    args = parse_args()
    os.makedirs(args.res_dir, exist_ok=True)
    H = W = args.img_size
    n_sweep_list = [int(x) for x in args.n_sweep.split(",")]

    # 폴더 경로 결정
    train_deg, train_clean, inf_deg, inf_clean = get_deg_clean_dirs(args)

    print(f"[Dataset] {args.dataset.upper()}")
    print(f"  Train: deg={train_deg}")
    print(f"         clean={train_clean}")
    print(f"  Test:  deg={inf_deg}")
    print(f"         clean={inf_clean}")

    # 이미지 로드
    print("\n[Load] Train pairs...")
    max_n_train = max(n_sweep_list)
    train_degs, train_cleans = load_paired_dataset(
        train_deg, train_clean, max_n_train, H)

    print("[Load] Inference pairs...")
    inf_degs, inf_cleans = load_paired_dataset(
        inf_deg, inf_clean, args.n_inf, H)

    if not train_degs:
        print("[Error] No training pairs"); return
    if not inf_degs:
        print("[Error] No inference pairs"); return

    # n_sweep를 실제 로드 수로 클램핑
    n_sweep_list = sorted(set(min(n, len(train_degs)) for n in n_sweep_list))
    print(f"\n  Train pairs: {len(train_degs)}")
    print(f"  Inference pairs: {len(inf_degs)} (unseen)")
    print(f"  N-sweep: {n_sweep_list}")

    # PDK 모델 선택
    pdk_models = select_pdk_models(args.dataset, args.pdk_model)
    print(f"  PDK models: {[m['name'] for m in pdk_models]}")

    # ── 학습 및 inference ─────────────────────────────────────────
    all_model_results = {}
    model_hists = {}
    sweep_results = {}
    oracle_psnrs = []

    for pdk_cfg in pdk_models:
        mname  = pdk_cfg["name"]
        Model  = pdk_cfg["Model"]
        print(f"\n{'='*55}")
        print(f"[Model] {mname}")

        # Dataset-PDK 학습
        print(f"  [Dataset-PDK] training on {args.n_train} pairs...")
        ds_model = Model(H, W)
        ds_hist  = train_on_pairs(
            ds_model,
            train_degs[:args.n_train],
            train_cleans[:args.n_train],
            args.n_iter, args.lr)
        model_hists[mname] = ds_hist

        # Inference
        print(f"  [Inference] {len(inf_degs)} unseen pairs...")
        psnr_degs, psnr_dss, psnr_oras = [], [], []
        inf_results = []

        for i, (deg, clean) in enumerate(zip(inf_degs, inf_cleans)):
            ds_cor  = inference_fixed(ds_model, deg)

            print(f"    Oracle {i+1}/{len(inf_degs)}...")
            ora_model = Model(H, W)
            ora_cor   = train_oracle(ora_model, deg, clean,
                                     args.n_ora_iter, args.lr)

            pd_ = psnr(clean, deg)
            pds = psnr(clean, ds_cor)
            po_ = psnr(clean, ora_cor)

            psnr_degs.append(pd_)
            psnr_dss.append(pds)
            psnr_oras.append(po_)

            inf_results.append({
                "clean":    clean,
                "deg":      deg,
                "ds_cor":   ds_cor,
                "ora_cor":  ora_cor,
                "psnr_deg": pd_,
                "psnr_ds":  pds,
                "psnr_ora": po_,
            })

        mean_pd  = np.mean(psnr_degs)
        mean_pds = np.mean(psnr_dss)
        mean_po  = np.mean(psnr_oras)
        oracle_psnrs.append(mean_po)

        print(f"  PSNR  Degraded={mean_pd:.1f}  "
              f"Dataset-PDK={mean_pds:.1f}  Oracle={mean_po:.1f}")
        print(f"  Dataset-PDK vs Oracle: {mean_pds-mean_po:+.1f}dB")

        all_model_results[mname] = {
            "psnr_degs": psnr_degs,
            "psnr_dss":  psnr_dss,
            "psnr_oras": psnr_oras,
        }

        # N-sweep
        print(f"  [N-sweep] {n_sweep_list}...")
        sweep = {}
        for n in n_sweep_list:
            m = Model(H, W)
            train_on_pairs(m,
                           train_degs[:n], train_cleans[:n],
                           args.n_iter, args.lr, verbose=False)
            ps = [psnr(c, inference_fixed(m,d))
                  for d,c in zip(inf_degs, inf_cleans)]
            sweep[n] = np.mean(ps)
            print(f"    N={n}  PSNR={sweep[n]:.2f}dB")
        sweep_results[mname] = sweep

        # 첫 번째 모델 결과로 comparison 저장
        if pdk_cfg == pdk_models[0]:
            save_comparison(inf_results, args.res_dir, args.dataset.upper())

    # 시각화
    print("\n[Saving results...]")
    save_psnr_summary(all_model_results, args.res_dir, args.dataset.upper())
    save_training_detail(model_hists, args.res_dir)
    save_n_sweep(sweep_results, oracle_psnrs, args.res_dir, args.dataset.upper())

    print(f"\n[Done] {os.path.abspath(args.res_dir)}")
    print("  19_comparison.png    -- unseen test 이미지 비교")
    print("  19_psnr_summary.png  -- 모델별 PSNR")
    print("  19_training_loss.png -- 학습 곡선")
    print("  19_n_sweep.png       -- N에 따른 PSNR 변화")

    # 요약
    print("\n[Summary]")
    print(f"{'Model':<16} {'Degraded':>8} {'Dataset-PDK':>12} {'Oracle':>8} {'gap':>6}")
    print("-"*54)
    for mname in all_model_results:
        r = all_model_results[mname]
        pd_ = np.mean(r["psnr_degs"])
        pds = np.mean(r["psnr_dss"])
        po_ = np.mean(r["psnr_oras"])
        print(f"{mname:<16} {pd_:>8.1f} {pds:>12.1f} {po_:>8.1f} {pds-po_:>+6.1f}")


if __name__ == "__main__":
    main()

# RF
# python 19_real_degradation_pdk.py \
#   --dataset rf \
#   --train-dir ./data/rf/train \
#   --inf-dir   ./data/rf/test \
#   --n-train 10 --n-inf 4 \
#   --img-size 128 \
#   --res-dir ./res/rf_pdk
