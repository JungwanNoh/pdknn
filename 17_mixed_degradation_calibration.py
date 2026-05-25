"""
17_mixed_degradation_calibration.py
==========================================
Mixed Degradation Calibration-based PDK [BSD500]

혼합 degradation에서 calibration method가 통하는지 검증.

Case A: PSF blur + Vignetting
Case B: Coma + Vignetting

각 case마다:
  1. calibration pattern으로 kernel 학습 (offline)
  2. unseen BSD500 이미지에 적용 (no retraining)
  3. Global / Cal-PDK / Oracle 비교

핵심 질문:
  혼합 degradation에서도 calibration이 일반화되는가?
  파라미터 증가에 따라 identifiability 문제가 생기는가?

Run:
  python 17_mixed_degradation_calibration.py
"""

import argparse, os, glob
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


# ============================================================================
# Utilities (16번과 동일)
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
# Calibration patterns
# ============================================================================

def make_calibration_patterns(size, device="cpu"):
    H = W = size
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    gen = torch.Generator(); gen.manual_seed(42)
    raw_rand = torch.rand(1,1,H,W, generator=gen, device=device)
    k_size = 15; sigma = 5.0
    coords = torch.arange(k_size, device=device).float() - k_size//2
    g1d = torch.exp(-coords**2/(2*sigma**2)); g1d = g1d/g1d.sum()
    gk = (g1d.unsqueeze(0)*g1d.unsqueeze(1)).view(1,1,k_size,k_size)
    smooth_rand = F.conv2d(raw_rand, gk, padding=k_size//2).squeeze()
    smooth_rand = (smooth_rand-smooth_rand.min())/(smooth_rand.max()-smooth_rand.min()+1e-5)

    patterns = {
        "checker_fine":    (0.5+0.5*torch.sign(torch.sin(gy*12)*torch.sin(gx*12))).clamp(0,1),
        "checker_coarse":  (0.5+0.5*torch.sign(torch.sin(gy*6)*torch.sin(gx*6))).clamp(0,1),
        "sine_h":          (0.5+0.5*torch.sin(gy*15)).clamp(0,1),
        "sine_v":          (0.5+0.5*torch.sin(gx*15)).clamp(0,1),
        "sine_diag":       (0.5+0.5*torch.sin((gy+gx)*10)).clamp(0,1),
        "concentric":      (0.5+0.5*torch.cos((gy**2+gx**2).sqrt()*20)).clamp(0,1),
        "gradient_x":      ((gx+1)/2).clamp(0,1),
        "gradient_y":      ((gy+1)/2).clamp(0,1),
        "gradient_radial": (1.0-(gy**2+gx**2).sqrt()/(2**0.5)).clamp(0,1),
        "smooth_random":   smooth_rand,
    }
    return {k: v.unsqueeze(0).unsqueeze(0) for k,v in patterns.items()}


# ============================================================================
# Degradation: Mixed cases
# ============================================================================

def degrade_psf_vig(x, sigma0=0.30, alpha_psf=1.20, alpha_vig=2.50):
    """Case A: PSF blur -> Vignetting"""
    B, C, H, W = x.shape
    r, _, _ = radial_map(H, W, x.device)
    r2 = r**2

    # PSF blur (position-dependent Gaussian)
    sigma = sigma0 + alpha_psf * r2
    coords = torch.tensor([
        [-1.,-1.],[-1.,0.],[-1.,1.],
        [ 0.,-1.],[ 0.,0.],[ 0.,1.],
        [ 1.,-1.],[ 1.,0.],[ 1.,1.]], device=x.device)
    d2 = (coords**2).sum(-1)
    s  = sigma.unsqueeze(-1).clamp(0.1, 1.0)
    k  = torch.exp(-d2/(2*s**2)); k = k/k.sum(-1,keepdim=True)
    x  = spatially_varying_conv(x, k)

    # Vignetting
    V = 1.0/(1.0 + alpha_vig * r2)
    return (x * V.unsqueeze(0).unsqueeze(0)).clamp(0, 1)


def degrade_coma_vig(x, sigma0=0.30, alpha_psf=1.20,
                     coma_k=0.30, alpha_vig=2.00):
    """Case B: Coma aberration -> Vignetting"""
    B, C, H, W = x.shape
    r, gy, gx = radial_map(H, W, x.device)
    r2 = r**2; rs = r.clamp(1e-6)

    # Coma PSF
    sigma = sigma0 + alpha_psf * r2
    coords = torch.tensor([
        [-1.,-1.],[-1.,0.],[-1.,1.],
        [ 0.,-1.],[ 0.,0.],[ 0.,1.],
        [ 1.,-1.],[ 1.,0.],[ 1.,1.]], device=x.device)
    d2 = (coords**2).sum(-1)
    s  = sigma.unsqueeze(-1).clamp(0.1, 1.0)
    shift_y = coma_k * r2 * (gy/rs)
    shift_x = coma_k * r2 * (gx/rs)
    g1 = torch.exp(-d2/(2*s**2)); g1 = g1/g1.sum(-1,keepdim=True)
    sy = shift_y.unsqueeze(-1); sx = shift_x.unsqueeze(-1)
    dy = coords[:,0].unsqueeze(0).unsqueeze(0)-sy
    dx = coords[:,1].unsqueeze(0).unsqueeze(0)-sx
    g2 = torch.exp(-(dy**2+dx**2)/(2*s**2)); g2 = g2/g2.sum(-1,keepdim=True)
    w  = (r*0.8).clamp(0,0.8).unsqueeze(-1)
    k  = (1-w)*g1 + w*g2; k = k/k.sum(-1,keepdim=True)
    x  = spatially_varying_conv(x, k)

    # Vignetting
    V = 1.0/(1.0 + alpha_vig * r2)
    return (x * V.unsqueeze(0).unsqueeze(0)).clamp(0, 1)


# ============================================================================
# PDK Models: Mixed
# ============================================================================

class PSFVigPDK(nn.Module):
    """
    PSF + Vignetting 동시 보정.
    파라미터 5개: sigma0, alpha_psf, lambda0, alpha_lam, alpha_vig
    k(r) = vig_gain(r) * (delta + lambda(r) * (delta - G(sigma(r))))
    """
    def __init__(self, H, W):
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._s0  = nn.Parameter(torch.tensor(_sp(0.5)))
        self._aps = nn.Parameter(torch.tensor(_sp(0.5)))
        self._l0  = nn.Parameter(torch.tensor(_sp(0.5)))
        self._al  = nn.Parameter(torch.tensor(_sp(0.3)))
        self._av  = nn.Parameter(torch.tensor(_sp(1.0)))
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
    @property
    def alpha_vig(self): return F.softplus(self._av)

    def forward(self, x):
        vig_gain = 1.0 + self.alpha_vig * self.r2
        sigma = (self.sigma0 + self.alpha_psf * self.r2).unsqueeze(-1).clamp(0.1,1.0)
        d2    = (self.coords**2).sum(-1)
        g     = torch.exp(-d2/(2*sigma**2)); g = g/g.sum(-1,keepdim=True)
        delta = torch.zeros_like(g); delta[:,:,4] = 1.0
        lam   = (self.lambda0 + self.alpha_lam*self.r2).unsqueeze(-1).clamp(0.05,3.0)
        k     = (delta + lam*(delta-g)) * vig_gain.unsqueeze(-1)
        return spatially_varying_conv(x, k)

    def get_params(self):
        return {"sigma0":   self.sigma0.item(),
                "alpha_psf":self.alpha_psf.item(),
                "lambda0":  self.lambda0.item(),
                "alpha_lam":self.alpha_lam.item(),
                "alpha_vig":self.alpha_vig.item()}


class ComaVigPDK(nn.Module):
    """
    Coma + Vignetting 동시 보정.
    파라미터 6개: sigma0, alpha_psf, coma_k, lambda0, alpha_lam, alpha_vig
    """
    def __init__(self, H, W):
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._s0  = nn.Parameter(torch.tensor(_sp(0.5)))
        self._aps = nn.Parameter(torch.tensor(_sp(0.5)))
        self._ck  = nn.Parameter(torch.tensor(_sp(0.1)))
        self._l0  = nn.Parameter(torch.tensor(_sp(0.5)))
        self._al  = nn.Parameter(torch.tensor(_sp(0.3)))
        self._av  = nn.Parameter(torch.tensor(_sp(1.0)))
        r, gy, gx = radial_map(H, W)
        self.register_buffer("r",  r)
        self.register_buffer("r2", r**2)
        self.register_buffer("gy", gy)
        self.register_buffer("gx", gx)
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

    def forward(self, x):
        vig_gain = 1.0 + self.alpha_vig * self.r2
        sigma = (self.sigma0 + self.alpha_psf * self.r2).unsqueeze(-1).clamp(0.1,1.0)
        rs    = self.r.clamp(1e-6)
        sy    = (-self.coma_k * self.r2 * self.gy/rs).unsqueeze(-1)
        sx    = (-self.coma_k * self.r2 * self.gx/rs).unsqueeze(-1)
        d2    = (self.coords**2).sum(-1)
        g1    = torch.exp(-d2/(2*sigma**2)); g1 = g1/g1.sum(-1,keepdim=True)
        dy    = self.coords[:,0].unsqueeze(0).unsqueeze(0)-sy
        dx    = self.coords[:,1].unsqueeze(0).unsqueeze(0)-sx
        g2    = torch.exp(-(dy**2+dx**2)/(2*sigma**2)); g2 = g2/g2.sum(-1,keepdim=True)
        w     = (self.r*0.8).clamp(0,0.8).unsqueeze(-1)
        g     = (1-w)*g1 + w*g2; g = g/g.sum(-1,keepdim=True)
        delta = torch.zeros_like(g); delta[:,:,4] = 1.0
        lam   = (self.lambda0 + self.alpha_lam*self.r2).unsqueeze(-1).clamp(0.05,3.0)
        k     = (delta + lam*(delta-g)) * vig_gain.unsqueeze(-1)
        return spatially_varying_conv(x, k)

    def get_params(self):
        return {"sigma0":   self.sigma0.item(),
                "alpha_psf":self.alpha_psf.item(),
                "coma_k":   self.coma_k.item(),
                "lambda0":  self.lambda0.item(),
                "alpha_lam":self.alpha_lam.item(),
                "alpha_vig":self.alpha_vig.item()}


class GlobalKernel(nn.Module):
    def __init__(self):
        super().__init__()
        init = torch.zeros(9); init[4] = 3.0
        self.kernel_logits = nn.Parameter(init)

    def forward(self, x):
        k = torch.softmax(self.kernel_logits, dim=0)
        return global_conv(x, k)


# ============================================================================
# Training
# ============================================================================

def train_on_patterns(model, patterns_deg, patterns_clean,
                      n_iter, lr, verbose=True):
    """모든 패턴 배치 학습 → content bias 상쇄"""
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    history = []
    pat_names = list(patterns_deg.keys())
    for i in range(n_iter):
        opt.zero_grad()
        total = torch.tensor(0.0)
        for name in pat_names:
            out   = model(patterns_deg[name])
            total = total + loss_fn(out, patterns_clean[name])
        total = total / len(pat_names)
        total.backward(); opt.step(); sched.step()
        history.append(total.item())
        if verbose and (i+1) % 100 == 0:
            print(f"    iter {i+1:4d}/{n_iter}  loss={total.item():.6f}")
    return history


def train_oracle(model, deg, clean, n_iter, lr):
    """Oracle: 해당 이미지로 직접 학습"""
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    for i in range(n_iter):
        opt.zero_grad()
        loss = loss_fn(model(deg), clean)
        loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return model(deg).clamp(0, 1)


# ============================================================================
# Load images
# ============================================================================

def load_images(data_dir, n, size):
    if data_dir and os.path.isdir(data_dir):
        exts = ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG")
        paths = []
        for ext in exts:
            paths += glob.glob(os.path.join(data_dir,"**",ext), recursive=True)
            paths += glob.glob(os.path.join(data_dir, ext))
        paths = sorted(set(paths)); imgs = []
        for p in paths:
            try: imgs.append(load_single_image(p, size))
            except Exception: continue
            if len(imgs) == n: break
        if imgs:
            print(f"[Load] {len(imgs)} images from {data_dir}"); return imgs
    try:
        from skimage import data as skd; import skimage.color as skc
        bl = [skd.camera(),skd.astronaut(),skd.chelsea(),
              skd.coffee(),skd.horse(),skd.hubble_deep_field()]
        imgs = []
        for arr in bl[:n]:
            if arr.ndim == 3: arr = skc.rgb2gray(arr)
            arr = (arr-arr.min())/(arr.max()-arr.min()+1e-8)
            pil = Image.fromarray((arr*255).astype(np.uint8))
            s = min(pil.size); w,h = pil.size
            pil = pil.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
            pil = pil.resize((size,size), Image.BILINEAR)
            imgs.append(torch.from_numpy(
                np.array(pil,dtype=np.float32)/255.).unsqueeze(0).unsqueeze(0))
        print(f"[Load] BSD500 fallback: {len(imgs)} images"); return imgs
    except Exception as e:
        print(f"[Warning] {e}"); return []


# ============================================================================
# Visualization
# ============================================================================

def save_comparison(case_results, res_dir):
    """
    2 case x 5 col:
    [raw | degraded | global | cal-pdk | oracle]
    + per-degradation component maps
    """
    n_cases = len(case_results)
    fig, axes = plt.subplots(n_cases, 5, figsize=(18, 4.5*n_cases+0.5))
    if n_cases == 1: axes = axes[np.newaxis,:]

    col_titles = ["1. Raw (clean)", "2. Degraded (mixed)",
                  "3. Global kernel", "4. Cal-PDK (ours)", "5. Oracle PDK"]
    for c, t in enumerate(col_titles):
        axes[0,c].set_title(t, fontsize=10, fontweight="bold", pad=6)

    for r, res in enumerate(case_results):
        imgs = [to_np(res["raw"]),  to_np(res["deg"]),
                to_np(res["g_cor"]),to_np(res["cal_cor"]),
                to_np(res["ora_cor"])]
        for c, img in enumerate(imgs):
            axes[r,c].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[r,c].axis("off")

        axes[r,0].set_ylabel(res["name"], fontsize=9, fontweight="bold",
                             rotation=0, labelpad=90, va="center")

        for c, (pv, color) in enumerate(zip(
            [res["psnr_deg"], res["psnr_g"],
             res["psnr_cal"], res["psnr_ora"]],
            ["dimgray","steelblue","tomato","green"]), start=1):
            axes[r,c].text(0.5,-0.06,f"PSNR={pv:.1f}dB",
                           transform=axes[r,c].transAxes,
                           ha="center",va="top",fontsize=8.5,color=color)

        gap = res["psnr_cal"] - res["psnr_ora"]
        axes[r,4].text(0.5,-0.12,
                       f"gap vs oracle: {gap:+.1f}dB",
                       transform=axes[r,4].transAxes,
                       ha="center",va="top",fontsize=7.5,color="gray")

    fig.suptitle("Mixed Degradation: Calibration-based PDK [BSD500]",
                 fontsize=13, y=1.01)
    plt.tight_layout(rect=[0,0.04,1,1])
    path = os.path.join(res_dir, "17_comparison.png")
    fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_psnr_summary(case_results, res_dir):
    names  = [r["name"]     for r in case_results]
    psnr_d = [r["psnr_deg"] for r in case_results]
    psnr_g = [r["psnr_g"]   for r in case_results]
    psnr_c = [r["psnr_cal"] for r in case_results]
    psnr_o = [r["psnr_ora"] for r in case_results]

    x = np.arange(len(names)); w = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x-1.5*w, psnr_d, w, label="Degraded",       color="#aaaaaa")
    ax.bar(x-0.5*w, psnr_g, w, label="Global kernel",   color="steelblue")
    ax.bar(x+0.5*w, psnr_c, w, label="Cal-PDK (ours)",  color="tomato")
    ax.bar(x+1.5*w, psnr_o, w, label="Oracle PDK",      color="green", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("PSNR (dB)", fontsize=10)
    ax.set_title("Mixed Degradation PSNR: Cal-PDK vs Oracle vs Global",
                 fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    for xi,(d,g,c,o) in enumerate(zip(psnr_d,psnr_g,psnr_c,psnr_o)):
        for xoff,v,color in [(-1.5*w,d,"#666"),(-0.5*w,g,"steelblue"),
                              (0.5*w,c,"tomato"),(1.5*w,o,"green")]:
            ax.text(xi+xoff, v+0.4, f"{v:.1f}",
                    ha="center", fontsize=7.5, color=color)

    plt.tight_layout()
    path = os.path.join(res_dir, "17_psnr_summary.png")
    fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_params_analysis(case_results, res_dir):
    """
    true / cal / oracle 파라미터 비교 + loss curve
    """
    n_cases = len(case_results)
    fig = plt.figure(figsize=(7*n_cases, 8))
    gs  = gridspec.GridSpec(2, n_cases, hspace=0.45, wspace=0.35)

    for c, res in enumerate(case_results):
        true_p = res["true_params"]
        cal_p  = res["cal_params"]
        ora_p  = res["ora_params"]
        params = list(true_p.keys())
        x = np.arange(len(params)); w = 0.25

        ax = fig.add_subplot(gs[0, c])
        ax.bar(x-w, [true_p[p] for p in params], w,
               label="True",    color="#aaaaaa")
        ax.bar(x,   [cal_p.get(p,0) for p in params], w,
               label="Cal-PDK", color="tomato")
        ax.bar(x+w, [ora_p.get(p,0) for p in params], w,
               label="Oracle",  color="green", alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(params, fontsize=8, rotation=20)
        ax.set_title(f"Params: {res['name']}", fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

        # loss curve
        ax2 = fig.add_subplot(gs[1, c])
        ax2.plot(res["cal_hist"], color="tomato",    linewidth=1.5,
                 label="Cal-PDK")
        ax2.plot(res["g_hist"],   color="steelblue", linewidth=1.5,
                 label="Global")
        ax2.set_title(f"Cal loss: {res['name']}", fontsize=10)
        ax2.set_xlabel("Iteration", fontsize=8)
        ax2.set_ylabel("Loss", fontsize=8)
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.suptitle("Params comparison & calibration loss: mixed degradation",
                 fontsize=13)
    path = os.path.join(res_dir, "17_params_analysis.png")
    fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_degradation_maps(cases_cfg, res_dir, size):
    """각 mixed degradation의 구성 요소 map 시각화"""
    H = W = size
    r, gy, gx = radial_map(H, W)
    r  = r.numpy(); r2 = r**2
    gy = gy.numpy(); gx = gx.numpy()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for row, (case_name, params) in enumerate(cases_cfg.items()):
        s0  = params["sigma0"];    ap = params["alpha_psf"]
        av  = params["alpha_vig"]
        ck  = params.get("coma_k", 0.0)

        sigma_map = s0 + ap * r2
        vig_map   = 1.0/(1.0 + av * r2)
        coma_map  = ck * r2  # shift magnitude

        # col0: PSF sigma map
        im0 = axes[row,0].imshow(sigma_map, cmap="hot")
        axes[row,0].set_title(f"PSF sigma{case_name}", fontsize=9)
        axes[row,0].axis("off")
        plt.colorbar(im0, ax=axes[row,0], fraction=0.046)

        # col1: vignetting map or coma map
        if ck > 0:
            im1 = axes[row,1].imshow(coma_map, cmap="hot")
            axes[row,1].set_title(f"Coma shift magcoma_k={ck:.2f}", fontsize=9)
        else:
            im1 = axes[row,1].imshow(vig_map, cmap="hot", vmin=0, vmax=1)
            axes[row,1].set_title(f"Vignetting V(r)alpha_vig={av:.2f}", fontsize=9)
        axes[row,1].axis("off"); plt.colorbar(im1, ax=axes[row,1], fraction=0.046)

        # col2: vignetting map (항상)
        im2 = axes[row,2].imshow(vig_map, cmap="hot", vmin=0, vmax=1)
        axes[row,2].set_title(f"Vignetting V(r)alpha_vig={av:.2f}", fontsize=9)
        axes[row,2].axis("off"); plt.colorbar(im2, ax=axes[row,2], fraction=0.046)

    fig.suptitle("Mixed degradation component maps", fontsize=12)
    plt.tight_layout()
    path = os.path.join(res_dir, "17_degradation_maps.png")
    fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


# ============================================================================
# Args
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Mixed Degradation Calibration PDK [BSD500]")
    p.add_argument("--data-dir",     type=str,   default="")
    p.add_argument("--n-inf-images", type=int,   default=4)
    p.add_argument("--img-size",     type=int,   default=256)
    p.add_argument("--n-cal-iter",   type=int,   default=500)
    p.add_argument("--n-ora-iter",   type=int,   default=500)
    p.add_argument("--lr",           type=float, default=0.02)
    p.add_argument("--res-dir",      type=str,   default="./res/mixed_calibration")
    # Case A: PSF + Vignetting
    p.add_argument("--psf-sigma0",   type=float, default=0.30)
    p.add_argument("--psf-alpha",    type=float, default=1.20)
    p.add_argument("--vig-alpha",    type=float, default=2.50)
    # Case B: Coma + Vignetting
    p.add_argument("--coma-k",       type=float, default=0.30)
    p.add_argument("--coma-vig",     type=float, default=2.00)
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    os.makedirs(args.res_dir, exist_ok=True)
    H = W = args.img_size

    # calibration patterns
    print("[Phase 1] Calibration patterns...")
    cal_clean = make_calibration_patterns(H)
    print(f"  {len(cal_clean)} patterns: {list(cal_clean.keys())}")

    # unseen images
    print("[Load] Unseen images...")
    unseen = load_images(args.data_dir, args.n_inf_images, H)
    if not unseen: return

    # case 정의
    cases = [
        {
            "name":        "PSF + Vignetting",
            "degrade_fn":  lambda x: degrade_psf_vig(
                x, args.psf_sigma0, args.psf_alpha, args.vig_alpha),
            "true_params": {"sigma0":   args.psf_sigma0,
                            "alpha_psf":args.psf_alpha,
                            "alpha_vig":args.vig_alpha},
            "PDKModel":    PSFVigPDK,
        },
        {
            "name":        "Coma + Vignetting",
            "degrade_fn":  lambda x: degrade_coma_vig(
                x, args.psf_sigma0, args.psf_alpha, args.coma_k, args.coma_vig),
            "true_params": {"sigma0":   args.psf_sigma0,
                            "coma_k":   args.coma_k,
                            "alpha_vig":args.coma_vig},
            "PDKModel":    ComaVigPDK,
        },
    ]

    all_results = []

    for case in cases:
        print(f"\n{'='*55}")
        print(f"[Case] {case['name']}")
        degrade = case["degrade_fn"]

        # calibration pattern degradation
        cal_deg = {name: degrade(img) for name,img in cal_clean.items()}

        # calibration training
        print(f"  [Cal] PDK training on {len(cal_clean)} patterns "
              f"(n_iter={args.n_cal_iter})...")
        cal_model = case["PDKModel"](H, W)
        cal_hist  = train_on_patterns(cal_model, cal_deg, cal_clean,
                                      args.n_cal_iter, args.lr)

        # global calibration
        print("  [Global] training...")
        g_model = GlobalKernel()
        g_hist  = train_on_patterns(g_model, cal_deg, cal_clean,
                                    args.n_cal_iter, args.lr, verbose=False)

        print(f"  [Cal] learned: {cal_model.get_params()}")

        # inference on unseen images
        print(f"  [Inference] {len(unseen)} unseen images...")
        psnr_degs,psnr_gs,psnr_cals,psnr_oras = [],[],[],[]
        ora_cor0 = None; ora_params0 = None

        for i, raw in enumerate(unseen):
            deg     = degrade(raw)
            cal_cor = cal_model.eval(); cal_cor = cal_model(deg).detach().clamp(0,1)
            g_cor   = g_model.eval();   g_cor   = g_model(deg).detach().clamp(0,1)

            ora_model = case["PDKModel"](H, W)
            print(f"    Oracle img {i+1}/{len(unseen)}...")
            ora_cor   = train_oracle(ora_model, deg, raw,
                                     args.n_ora_iter, args.lr)

            psnr_degs.append(psnr(raw,deg))
            psnr_gs.append(psnr(raw,g_cor))
            psnr_cals.append(psnr(raw,cal_cor))
            psnr_oras.append(psnr(raw,ora_cor))

            if i == 0:
                ora_cor0    = ora_cor
                ora_params0 = ora_model.get_params()
                deg0        = deg
                g_cor0      = g_cor
                cal_cor0    = cal_cor

        pd_=np.mean(psnr_degs); pg_=np.mean(psnr_gs)
        pc_=np.mean(psnr_cals); po_=np.mean(psnr_oras)
        print(f"  PSNR  Deg={pd_:.1f}  Global={pg_:.1f}"
              f"  Cal-PDK={pc_:.1f}  Oracle={po_:.1f}"
              f"  gap={pc_-po_:+.1f}dB")

        all_results.append({
            "name":        case["name"],
            "raw":         unseen[0],
            "deg":         deg0,
            "g_cor":       g_cor0,
            "cal_cor":     cal_cor0,
            "ora_cor":     ora_cor0,
            "psnr_deg":    pd_,
            "psnr_g":      pg_,
            "psnr_cal":    pc_,
            "psnr_ora":    po_,
            "true_params": case["true_params"],
            "cal_params":  cal_model.get_params(),
            "ora_params":  ora_params0,
            "cal_hist":    cal_hist,
            "g_hist":      g_hist,
        })

    # 시각화
    print("\n[Saving results...]")
    save_comparison(all_results, args.res_dir)
    save_psnr_summary(all_results, args.res_dir)
    save_params_analysis(all_results, args.res_dir)
    save_degradation_maps(
        {r["name"]: {**r["true_params"],
                     "sigma0":   args.psf_sigma0,
                     "alpha_psf":args.psf_alpha}
         for r in all_results},
        args.res_dir, H)

    print(f"\n[Done] {os.path.abspath(args.res_dir)}")
    print("  17_comparison.png       -- 2 cases x 5 cols")
    print("  17_psnr_summary.png     -- PSNR bar chart")
    print("  17_params_analysis.png  -- params + loss curves")
    print("  17_degradation_maps.png -- component maps")

    print("\n[Summary]")
    print(f"{'Case':<25} {'Deg':>6} {'Global':>7}"
          f" {'Cal-PDK':>8} {'Oracle':>7} {'gap':>6}")
    print("-"*62)
    for r in all_results:
        gap = r["psnr_cal"] - r["psnr_ora"]
        print(f"{r['name']:<25} {r['psnr_deg']:>6.1f} {r['psnr_g']:>7.1f}"
              f" {r['psnr_cal']:>8.1f} {r['psnr_ora']:>7.1f} {gap:>+6.1f}")


if __name__ == "__main__":
    main()