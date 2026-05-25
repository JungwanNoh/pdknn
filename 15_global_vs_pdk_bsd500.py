"""
15_global_vs_pdk_bsd500.py
==========================================
Global kernel vs Position-Dependent Kernel 비교 [BSD500]

4가지 degradation case:
  1. Vignetting
  2. PSF blur
  3. Coma aberration
  4. Non-uniform illumination

각 case마다:
  Global: 하나의 3x3 kernel -> 전체 이미지에 동일 적용
  PDK:    위치마다 다른 3x3 kernel

동일 조건: L1 + 0.1*grad_L1, n_iter=500, lr=0.02
출력: 15_comparison.png (4 case x 5 column)
      [raw | degraded | global | pdk | diff(pdk-global)]

Run:
  python 15_global_vs_pdk_bsd500.py
  python 15_global_vs_pdk_bsd500.py --n-iter 500 --n-images 4
"""

import argparse, os, glob
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

def to_np(t):
    return t.squeeze().detach().cpu().float().numpy()

def load_single_image(path, size):
    img = Image.open(path).convert("L")
    w, h = img.size; s = min(w, h)
    img = img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
    img = img.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(
        np.array(img, dtype=np.float32)/255.).unsqueeze(0).unsqueeze(0)

def load_dataset(args):
    if args.data_dir and os.path.isdir(args.data_dir):
        exts = ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG")
        paths = []
        for ext in exts:
            paths += glob.glob(os.path.join(args.data_dir,"**",ext), recursive=True)
            paths += glob.glob(os.path.join(args.data_dir, ext))
        paths = sorted(set(paths)); imgs = []
        for p in paths:
            try: imgs.append(load_single_image(p, args.img_size))
            except Exception: continue
            if len(imgs) == args.n_images: break
        if imgs:
            print(f"[Load] {len(imgs)} images from {args.data_dir}"); return imgs
    try:
        from skimage import data as skd; import skimage.color as skc
        bl=[skd.camera(),skd.astronaut(),skd.chelsea(),
            skd.coffee(),skd.horse(),skd.hubble_deep_field()]
        imgs=[]
        for arr in bl[:args.n_images]:
            if arr.ndim==3: arr=skc.rgb2gray(arr)
            arr=(arr-arr.min())/(arr.max()-arr.min()+1e-8)
            pil=Image.fromarray((arr*255).astype(np.uint8))
            s=min(pil.size); w,h=pil.size
            pil=pil.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
            pil=pil.resize((args.img_size,args.img_size),Image.BILINEAR)
            imgs.append(torch.from_numpy(
                np.array(pil,dtype=np.float32)/255.).unsqueeze(0).unsqueeze(0))
        print(f"[Load] BSD500 fallback (skimage): {len(imgs)} images")
        return imgs
    except Exception as e:
        print(f"[Warning] {e}"); return _syn(args.n_images, args.img_size)

def _syn(n, size):
    H=W=size; ys=torch.linspace(-1,1,H); xs=torch.linspace(-1,1,W)
    gy,gx=torch.meshgrid(ys,xs,indexing="ij")
    pts=[(0.5+0.25*torch.sin(gy*12)+0.25*torch.sin(gx*12)).clamp(0,1),
         (0.5+0.5*torch.cos((gy**2+gx**2).sqrt()*15)).clamp(0,1),
         ((gy+gx+1)/2).clamp(0,1),
         (0.5+0.5*torch.sign(torch.sin(gy*12)*torch.sin(gx*12))).clamp(0,1)]
    return [p.unsqueeze(0).unsqueeze(0) for p in pts[:n]]


# ============================================================================
# Convolution helpers
# ============================================================================

def spatially_varying_conv(x, kernels):
    """kernels: (H,W,9) position-dependent"""
    B, C, H, W = x.shape
    patches = F.unfold(x, kernel_size=3, padding=1).view(B, C, 9, H*W)
    k = kernels.view(H*W, 9).T.unsqueeze(0).unsqueeze(0)
    return (patches * k).sum(dim=2).view(B, C, H, W).clamp(0, 1)

def global_conv(x, kernel_9):
    """kernel_9: (9,) single kernel applied everywhere"""
    k = kernel_9.view(1, 1, 3, 3)
    return F.conv2d(x, k.expand(x.shape[1],1,3,3),
                    padding=1, groups=x.shape[1]).clamp(0, 1)


# ============================================================================
# Degradation functions
# ============================================================================

def degrade_vignetting(x, alpha=2.5):
    r,_,_ = radial_map(*x.shape[2:], x.device)
    V = 1.0/(1.0+alpha*r**2)
    return (x*V.unsqueeze(0).unsqueeze(0)).clamp(0,1)

def degrade_psf(x, sigma0=0.30, alpha_psf=1.20):
    r,_,_ = radial_map(*x.shape[2:], x.device)
    sigma = sigma0 + alpha_psf*r**2
    coords = torch.tensor([
        [-1.,-1.],[-1.,0.],[-1.,1.],
        [ 0.,-1.],[ 0.,0.],[ 0.,1.],
        [ 1.,-1.],[ 1.,0.],[ 1.,1.]], device=x.device)
    d2 = (coords**2).sum(-1)
    s  = sigma.unsqueeze(-1).clamp(0.1,1.0)
    k  = torch.exp(-d2/(2*s**2)); k=k/k.sum(-1,keepdim=True)
    return spatially_varying_conv(x, k)

def degrade_coma(x, sigma0=0.30, alpha_psf=1.20, coma_k=0.30, alpha_vig=2.0):
    r,gy,gx = radial_map(*x.shape[2:], x.device)
    r2=r**2; rs=r.clamp(1e-6)
    sigma=sigma0+alpha_psf*r2
    shift_y=coma_k*r2*(gy/rs); shift_x=coma_k*r2*(gx/rs)
    coords = torch.tensor([
        [-1.,-1.],[-1.,0.],[-1.,1.],
        [ 0.,-1.],[ 0.,0.],[ 0.,1.],
        [ 1.,-1.],[ 1.,0.],[ 1.,1.]], device=x.device)
    d2=(coords**2).sum(-1); s=sigma.unsqueeze(-1).clamp(0.1,1.0)
    g1=torch.exp(-d2/(2*s**2)); g1=g1/g1.sum(-1,keepdim=True)
    sy=shift_y.unsqueeze(-1); sx=shift_x.unsqueeze(-1)
    dy=coords[:,0].unsqueeze(0).unsqueeze(0)-sy
    dx=coords[:,1].unsqueeze(0).unsqueeze(0)-sx
    g2=torch.exp(-(dy**2+dx**2)/(2*s**2)); g2=g2/g2.sum(-1,keepdim=True)
    w=(r*0.8).clamp(0,0.8).unsqueeze(-1)
    k=(1-w)*g1+w*g2; k=k/k.sum(-1,keepdim=True)
    x=spatially_varying_conv(x,k)
    V=1.0/(1.0+alpha_vig*r2)
    return (x*V.unsqueeze(0).unsqueeze(0)).clamp(0,1)

def degrade_nonunif(x, cx_off=0.20, cy_off=0.15,
                    sigma_illum=0.60, ellipse_ratio=1.40, alpha_vig=1.50):
    r,gy,gx = radial_map(*x.shape[2:], x.device)
    V_rad=1.0/(1.0+alpha_vig*r**2)
    dy=gy-cy_off; dx=gx-cx_off; d2=dx**2+(dy*ellipse_ratio)**2
    ill=torch.exp(-d2/(2*sigma_illum**2)); ill=ill/ill.max().clamp(1e-5)
    V=(V_rad*(0.3+0.7*ill)).clamp(0,1)
    return (x*V.unsqueeze(0).unsqueeze(0)).clamp(0,1)


# ============================================================================
# Global kernel model (하나의 3x3 kernel, 위치 무관)
# ============================================================================

class GlobalKernel(nn.Module):
    """
    단일 3x3 kernel -> 전체 이미지에 동일 적용.
    kernel은 softmax로 정규화 (sum=1 유지).
    """
    def __init__(self):
        super().__init__()
        # delta 초기화: center=1, 나머지=0
        init = torch.zeros(9); init[4] = 3.0
        self.kernel_logits = nn.Parameter(init)

    def forward(self, x):
        k = torch.softmax(self.kernel_logits, dim=0)  # (9,) sum=1
        return global_conv(x, k)

    def get_kernel(self):
        return torch.softmax(self.kernel_logits, dim=0).detach().cpu().numpy()


# ============================================================================
# PDK models per case
# ============================================================================

class VignettingPDK(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self._alpha = nn.Parameter(torch.tensor(float(np.log(np.exp(1.0)-1.0))))
        r,_,_ = radial_map(H,W); self.register_buffer("r2",r**2)

    @property
    def alpha(self): return F.softplus(self._alpha)

    def forward(self, x):
        gain=1.0+self.alpha*self.r2
        zeros=torch.zeros(*self.r2.shape,9,device=x.device)
        idx=torch.full((*self.r2.shape,1),4,dtype=torch.long,device=x.device)
        k=zeros.scatter(2,idx,gain.unsqueeze(-1))
        return spatially_varying_conv(x,k)


class PSFPDK(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._s0  = nn.Parameter(torch.tensor(_sp(0.5)))
        self._aps = nn.Parameter(torch.tensor(_sp(0.5)))
        self._l0  = nn.Parameter(torch.tensor(_sp(0.5)))
        self._al  = nn.Parameter(torch.tensor(_sp(0.3)))
        r,_,_ = radial_map(H,W)
        self.register_buffer("r2",r**2)
        self.register_buffer("coords",torch.tensor([
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
        sigma=(self.sigma0+self.alpha_psf*self.r2).unsqueeze(-1).clamp(0.1,1.0)
        d2=(self.coords**2).sum(-1)
        g=torch.exp(-d2/(2*sigma**2)); g=g/g.sum(-1,keepdim=True)
        delta=torch.zeros_like(g); delta[:,:,4]=1.0
        lam=(self.lambda0+self.alpha_lam*self.r2).unsqueeze(-1).clamp(0.05,3.0)
        return spatially_varying_conv(x, delta+lam*(delta-g))


class ComaPDK(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._s0  = nn.Parameter(torch.tensor(_sp(0.5)))
        self._aps = nn.Parameter(torch.tensor(_sp(0.5)))
        self._ck  = nn.Parameter(torch.tensor(_sp(0.1)))
        self._l0  = nn.Parameter(torch.tensor(_sp(0.5)))
        self._al  = nn.Parameter(torch.tensor(_sp(0.3)))
        self._av  = nn.Parameter(torch.tensor(_sp(1.0)))
        r,gy,gx=radial_map(H,W)
        self.register_buffer("r",r); self.register_buffer("r2",r**2)
        self.register_buffer("gy",gy); self.register_buffer("gx",gx)
        self.register_buffer("coords",torch.tensor([
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
        vig_gain=1.0+self.alpha_vig*self.r2
        sigma=(self.sigma0+self.alpha_psf*self.r2).unsqueeze(-1).clamp(0.1,1.0)
        rs=self.r.clamp(1e-6)
        sy=(-self.coma_k*self.r2*self.gy/rs).unsqueeze(-1)
        sx=(-self.coma_k*self.r2*self.gx/rs).unsqueeze(-1)
        d2=(self.coords**2).sum(-1)
        g1=torch.exp(-d2/(2*sigma**2)); g1=g1/g1.sum(-1,keepdim=True)
        dy=self.coords[:,0].unsqueeze(0).unsqueeze(0)-sy
        dx=self.coords[:,1].unsqueeze(0).unsqueeze(0)-sx
        g2=torch.exp(-(dy**2+dx**2)/(2*sigma**2)); g2=g2/g2.sum(-1,keepdim=True)
        w=(self.r*0.8).clamp(0,0.8).unsqueeze(-1)
        g=(1-w)*g1+w*g2; g=g/g.sum(-1,keepdim=True)
        delta=torch.zeros_like(g); delta[:,:,4]=1.0
        lam=(self.lambda0+self.alpha_lam*self.r2).unsqueeze(-1).clamp(0.05,3.0)
        k=(delta+lam*(delta-g))*vig_gain.unsqueeze(-1)
        return spatially_varying_conv(x,k)


class NonUnifPDK(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._cx  = nn.Parameter(torch.tensor(0.0))
        self._cy  = nn.Parameter(torch.tensor(0.0))
        self._sig = nn.Parameter(torch.tensor(_sp(0.6)))
        self._ell = nn.Parameter(torch.tensor(_sp(1.0)))
        self._av  = nn.Parameter(torch.tensor(_sp(1.0)))
        r,gy,gx=radial_map(H,W)
        self.register_buffer("r2",r**2)
        self.register_buffer("gy",gy); self.register_buffer("gx",gx)

    @property
    def cx(self):  return torch.tanh(self._cx)*0.5
    @property
    def cy(self):  return torch.tanh(self._cy)*0.5
    @property
    def sig(self): return F.softplus(self._sig).clamp(0.2,2.0)
    @property
    def ell(self): return F.softplus(self._ell).clamp(0.3,3.0)
    @property
    def av(self):  return F.softplus(self._av)

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


# ============================================================================
# Training (공통 loss)
# ============================================================================

def train(model, deg, clean, n_iter, lr):
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n_iter)
    history=[]
    for i in range(n_iter):
        opt.zero_grad()
        out=model(deg)
        l1=F.l1_loss(out,clean)
        def gmap(t): return t[:,:,:,1:]-t[:,:,:,:-1], t[:,:,1:,:]-t[:,:,:-1,:]
        ox,oy=gmap(out); cx,cy=gmap(clean)
        loss=l1+0.1*(F.l1_loss(ox,cx)+F.l1_loss(oy,cy))
        loss.backward(); opt.step(); sched.step()
        history.append(loss.item())
    with torch.no_grad(): cor=model(deg).clamp(0,1)
    return cor, history


def run_case(deg, clean, GlobalModel, PDKModel, n_iter, lr):
    H,W = deg.shape[2], deg.shape[3]

    print("    [Global] training...")
    g_model = GlobalModel()
    g_cor, g_hist = train(g_model, deg, clean, n_iter, lr)

    print("    [PDK]    training...")
    p_model = PDKModel(H,W)
    p_cor, p_hist = train(p_model, deg, clean, n_iter, lr)

    return g_cor, p_cor, g_hist, p_hist


# ============================================================================
# Visualization
# ============================================================================

def save_comparison(results, res_dir):
    """
    results: list of dict per case:
      {name, raw, deg, g_cor, p_cor, psnr_deg, psnr_g, psnr_p,
       g_hist, p_hist}
    Layout: n_cases rows x 6 cols
      [raw | degraded | global | pdk | diff(global-raw) | diff(pdk-raw)]
    """
    n_cases = len(results)
    fig, axes = plt.subplots(n_cases, 6, figsize=(21, 3.8*n_cases + 0.5))
    if n_cases == 1: axes = axes[np.newaxis,:]

    col_titles = ["1. Raw (clean)", "2. Degraded",
                  "3. Global kernel", "4. PD kernel",
                  "5. |Global - Raw|", "6. |PDK - Raw|"]
    for c, t in enumerate(col_titles):
        axes[0,c].set_title(t, fontsize=10, fontweight="bold", pad=6)

    for r, res in enumerate(results):
        raw_np  = to_np(res["raw"])
        deg_np  = to_np(res["deg"])
        g_np    = to_np(res["g_cor"])
        p_np    = to_np(res["p_cor"])
        diff_g  = np.abs(g_np - raw_np)
        diff_p  = np.abs(p_np - raw_np)
        vmax_d  = max(diff_g.max(), diff_p.max(), 0.01)

        imgs = [raw_np, deg_np, g_np, p_np, diff_g, diff_p]
        cmaps = ["gray","gray","gray","gray","hot","hot"]
        vmins = [0,0,0,0,0,0]
        vmaxs = [1,1,1,1,vmax_d,vmax_d]

        for c,(img,cm,vn,vx) in enumerate(zip(imgs,cmaps,vmins,vmaxs)):
            axes[r,c].imshow(img, cmap=cm, vmin=vn, vmax=vx)
            axes[r,c].axis("off")

        # row label
        axes[r,0].set_ylabel(res["name"], fontsize=9, fontweight="bold",
                             rotation=0, labelpad=80, va="center")

        # PSNR annotation
        axes[r,1].text(0.5,-0.05,
                       f'PSNR={res["psnr_deg"]:.1f}dB',
                       transform=axes[r,1].transAxes,
                       ha="center",va="top",fontsize=8,color="dimgray")
        axes[r,2].text(0.5,-0.05,
                       f'PSNR={res["psnr_g"]:.1f}dB',
                       transform=axes[r,2].transAxes,
                       ha="center",va="top",fontsize=8,color="steelblue")
        axes[r,3].text(0.5,-0.05,
                       f'PSNR={res["psnr_p"]:.1f}dB  (+{res["psnr_p"]-res["psnr_g"]:.1f})',
                       transform=axes[r,3].transAxes,
                       ha="center",va="top",fontsize=8,color="tomato")

    fig.suptitle("Global kernel vs PD kernel [BSD500]", fontsize=13, y=1.01)
    plt.tight_layout(rect=[0,0.03,1,1])
    path = os.path.join(res_dir, "15_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_loss_curves(results, res_dir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]
    for ax, res in zip(axes, results):
        ax.plot(res["g_hist"], color="steelblue", linewidth=1.5, label="global")
        ax.plot(res["p_hist"], color="tomato",    linewidth=1.5, label="PDK")
        ax.set_title(res["name"], fontsize=10)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle("Training loss: Global vs PDK", fontsize=12)
    plt.tight_layout()
    path = os.path.join(res_dir, "15_loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_psnr_summary(results, res_dir):
    names = [r["name"] for r in results]
    psnr_deg = [r["psnr_deg"] for r in results]
    psnr_g   = [r["psnr_g"]   for r in results]
    psnr_p   = [r["psnr_p"]   for r in results]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x-w,   psnr_deg, w, label="Degraded",      color="#aaaaaa")
    ax.bar(x,     psnr_g,   w, label="Global kernel",  color="steelblue")
    ax.bar(x+w,   psnr_p,   w, label="PD kernel",      color="tomato")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("PSNR (dB)", fontsize=10)
    ax.set_title("PSNR summary: Global vs PD kernel [BSD500]", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # PSNR 값 표시
    for xi, (d,g,p) in enumerate(zip(psnr_deg,psnr_g,psnr_p)):
        ax.text(xi-w, d+0.3, f"{d:.1f}", ha="center", fontsize=7, color="#555")
        ax.text(xi,   g+0.3, f"{g:.1f}", ha="center", fontsize=7, color="steelblue")
        ax.text(xi+w, p+0.3, f"{p:.1f}", ha="center", fontsize=7, color="tomato")

    plt.tight_layout()
    path = os.path.join(res_dir, "15_psnr_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


# ============================================================================
# Args & Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Global vs PDK comparison [BSD500]")
    p.add_argument("--data-dir",  type=str,   default="")
    p.add_argument("--n-images",  type=int,   default=4)
    p.add_argument("--img-size",  type=int,   default=256)
    p.add_argument("--n-iter",    type=int,   default=500)
    p.add_argument("--lr",        type=float, default=0.02)
    p.add_argument("--res-dir",   type=str,   default="./res/global_vs_pdk")
    # degradation params
    p.add_argument("--vig-alpha",      type=float, default=2.50)
    p.add_argument("--psf-sigma0",     type=float, default=0.30)
    p.add_argument("--psf-alpha",      type=float, default=1.20)
    p.add_argument("--coma-k",         type=float, default=0.30)
    p.add_argument("--coma-alpha-vig", type=float, default=2.00)
    p.add_argument("--nu-cx",          type=float, default=0.20)
    p.add_argument("--nu-cy",          type=float, default=0.15)
    p.add_argument("--nu-sigma",       type=float, default=0.60)
    p.add_argument("--nu-ell",         type=float, default=1.40)
    p.add_argument("--nu-vig",         type=float, default=1.50)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.res_dir, exist_ok=True)
    imgs = load_dataset(args)
    if not imgs: return

    # 대표 이미지 1장으로 비교 (첫 번째)
    # 여러 장이면 평균 PSNR
    H = W = imgs[0].shape[-1]

    # 4가지 case 정의
    cases = [
        {
            "name": "Vignetting",
            "degrade_fn": lambda x: degrade_vignetting(x, args.vig_alpha),
            "GlobalModel": GlobalKernel,
            "PDKModel":    VignettingPDK,
        },
        {
            "name": "PSF blur",
            "degrade_fn": lambda x: degrade_psf(x, args.psf_sigma0, args.psf_alpha),
            "GlobalModel": GlobalKernel,
            "PDKModel":    PSFPDK,
        },
        {
            "name": "Coma aberration",
            "degrade_fn": lambda x: degrade_coma(
                x, args.psf_sigma0, args.psf_alpha, args.coma_k, args.coma_alpha_vig),
            "GlobalModel": GlobalKernel,
            "PDKModel":    ComaPDK,
        },
        {
            "name": "Non-uniform illum",
            "degrade_fn": lambda x: degrade_nonunif(
                x, args.nu_cx, args.nu_cy, args.nu_sigma, args.nu_ell, args.nu_vig),
            "GlobalModel": GlobalKernel,
            "PDKModel":    NonUnifPDK,
        },
    ]

    all_results = []

    for case in cases:
        print(f"\n[Case] {case['name']}")

        # 이미지별 학습 후 평균 PSNR
        psnr_degs, psnr_gs, psnr_ps = [], [], []
        g_cors, p_cors = [], []
        g_hist_final = p_hist_final = None

        for i, raw in enumerate(imgs):
            deg = case["degrade_fn"](raw)
            print(f"  image {i+1}/{len(imgs)}  PSNR_deg={psnr(raw,deg):.2f}dB")
            g_cor, p_cor, g_hist, p_hist = run_case(
                deg, raw,
                case["GlobalModel"], case["PDKModel"],
                args.n_iter, args.lr)
            psnr_degs.append(psnr(raw,deg))
            psnr_gs.append(psnr(raw,g_cor))
            psnr_ps.append(psnr(raw,p_cor))
            g_cors.append(g_cor); p_cors.append(p_cor)
            if i == 0:
                g_hist_final=g_hist; p_hist_final=p_hist

        print(f"  PSNR_deg={np.mean(psnr_degs):.2f}dB"
              f"  Global={np.mean(psnr_gs):.2f}dB"
              f"  PDK={np.mean(psnr_ps):.2f}dB"
              f"  Delta={np.mean(psnr_ps)-np.mean(psnr_gs):+.2f}dB")

        all_results.append({
            "name":     case["name"],
            "raw":      imgs[0],
            "deg":      case["degrade_fn"](imgs[0]),
            "g_cor":    g_cors[0],
            "p_cor":    p_cors[0],
            "psnr_deg": np.mean(psnr_degs),
            "psnr_g":   np.mean(psnr_gs),
            "psnr_p":   np.mean(psnr_ps),
            "g_hist":   g_hist_final,
            "p_hist":   p_hist_final,
        })

    # 시각화
    save_comparison(all_results, args.res_dir)
    save_loss_curves(all_results, args.res_dir)
    save_psnr_summary(all_results, args.res_dir)

    print(f"\n[Done] {os.path.abspath(args.res_dir)}")
    print("  15_comparison.png   -- 4 cases x 6 cols")
    print("  15_loss_curves.png  -- training loss global vs PDK")
    print("  15_psnr_summary.png -- PSNR bar chart")


if __name__ == "__main__":
    main()