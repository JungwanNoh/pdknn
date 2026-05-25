"""
16_calibration_based_pdk_bsd500.py
==========================================
Calibration-based PDK: offline training -> online inference

핵심 주장 검증:
  "PDK는 소자 특성(물리 파라미터)을 학습하므로
   calibration pattern으로 학습해도
   unseen 이미지에 적용 가능하다"

실험 구성:
  Phase 1 - Calibration (offline):
    알려진 test pattern만으로 kernel 파라미터 학습
    (체커보드, 사인파, 그래디언트, 동심원)
    clean pattern이 있으므로 학습 가능

  Phase 2 - Inference (online):
    학습된 kernel map 고정
    BSD500 unseen 이미지에 적용
    추가 학습 없음

  비교:
    A. Calibration PDK (ours)  <- 이게 실제 시나리오
    B. Oracle PDK              <- 각 이미지로 직접 학습 (upper bound)
    C. Global kernel           <- baseline
    D. No correction           <- degraded

4가지 degradation:
  1. Vignetting
  2. PSF blur
  3. Coma aberration
  4. Non-uniform illumination

출력:
  16_main_comparison.png   -- 4 case x 5 col
  16_psnr_summary.png      -- PSNR bar chart
  16_calibration_detail.png -- calibration pattern 학습 과정

Run:
  python 16_calibration_based_pdk_bsd500.py
  python 16_calibration_based_pdk_bsd500.py --n-cal-iter 500 --n-inf-images 6
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

def spatially_varying_conv(x, kernels):
    B, C, H, W = x.shape
    patches = F.unfold(x, kernel_size=3, padding=1).view(B, C, 9, H*W)
    k = kernels.view(H*W, 9).T.unsqueeze(0).unsqueeze(0)
    return (patches * k).sum(dim=2).view(B, C, H, W).clamp(0, 1)

def global_conv(x, kernel_9):
    k = kernel_9.view(1, 1, 3, 3)
    return F.conv2d(x, k.expand(x.shape[1],1,3,3),
                    padding=1, groups=x.shape[1]).clamp(0, 1)


# ============================================================================
# Calibration patterns (test pattern, 소자 특성 측정용)
# ============================================================================

def make_calibration_patterns(size, device="cpu"):
    """
    소자 calibration에 사용할 알려진 패턴들.
    실제 소자 실험에서는 이런 패턴을 디스플레이에 띄우고 촬영.

    패턴 선택 기준:
      - 다양한 주파수 성분 포함
      - 공간적으로 균일하게 분포
      - 엣지/텍스처/균일 영역 모두 포함
    """
    H = W = size
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    patterns = {
        "checker_fine":   (0.5+0.5*torch.sign(
                           torch.sin(gy*12)*torch.sin(gx*12))).clamp(0,1),
        "checker_coarse": (0.5+0.5*torch.sign(
                           torch.sin(gy*6)*torch.sin(gx*6))).clamp(0,1),
        "sine_horizontal":(0.5+0.5*torch.sin(gy*15)).clamp(0,1),
        "sine_vertical":  (0.5+0.5*torch.sin(gx*15)).clamp(0,1),
        "sine_diagonal":  (0.5+0.5*torch.sin(
                           (gy+gx)*10)).clamp(0,1),
        "concentric":     (0.5+0.5*torch.cos(
                           (gy**2+gx**2).sqrt()*20)).clamp(0,1),
        "gradient_x":     ((gx+1)/2).clamp(0,1),
        "gradient_y":     ((gy+1)/2).clamp(0,1),
        "gradient_radial":(1.0-(gy**2+gx**2).sqrt()/(2**0.5)).clamp(0,1),
        "random_smooth":  torch.tensor(
            __import__("scipy.ndimage",fromlist=["gaussian_filter"])
            .gaussian_filter(
                np.random.RandomState(42).rand(H,W).astype(np.float32), 5
            ), device=device) if True else (
            torch.rand(H,W,generator=torch.Generator().manual_seed(42))),
    }
    return {k: v.unsqueeze(0).unsqueeze(0) for k,v in patterns.items()}


def make_calibration_patterns_simple(size, device="cpu"):
    """scipy 없이 동작하는 버전"""
    H = W = size
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    # 랜덤 패턴을 Gaussian blur로 smooth하게 만들기
    gen = torch.Generator(); gen.manual_seed(42)
    raw_rand = torch.rand(1,1,H,W, generator=gen, device=device)
    k_size = 15; sigma = 5.0
    coords = torch.arange(k_size, device=device).float() - k_size//2
    g1d = torch.exp(-coords**2/(2*sigma**2))
    g1d = g1d/g1d.sum()
    gk = (g1d.unsqueeze(0)*g1d.unsqueeze(1)).view(1,1,k_size,k_size)
    smooth_rand = F.conv2d(raw_rand, gk, padding=k_size//2).squeeze()
    smooth_rand = (smooth_rand-smooth_rand.min())/(smooth_rand.max()-smooth_rand.min()+1e-5)

    patterns = {
        "checker_fine":    (0.5+0.5*torch.sign(
                            torch.sin(gy*12)*torch.sin(gx*12))).clamp(0,1),
        "checker_coarse":  (0.5+0.5*torch.sign(
                            torch.sin(gy*6)*torch.sin(gx*6))).clamp(0,1),
        "sine_h":          (0.5+0.5*torch.sin(gy*15)).clamp(0,1),
        "sine_v":          (0.5+0.5*torch.sin(gx*15)).clamp(0,1),
        "sine_diag":       (0.5+0.5*torch.sin((gy+gx)*10)).clamp(0,1),
        "concentric":      (0.5+0.5*torch.cos(
                            (gy**2+gx**2).sqrt()*20)).clamp(0,1),
        "gradient_x":      ((gx+1)/2).clamp(0,1),
        "gradient_y":      ((gy+1)/2).clamp(0,1),
        "gradient_radial": (1.0-(gy**2+gx**2).sqrt()/(2**0.5)).clamp(0,1),
        "smooth_random":   smooth_rand,
    }
    return {k: v.unsqueeze(0).unsqueeze(0) for k,v in patterns.items()}


# ============================================================================
# Degradation functions (동일)
# ============================================================================

def degrade_vignetting(x, alpha=2.5):
    r,_,_ = radial_map(*x.shape[2:], x.device)
    V = 1.0/(1.0+alpha*r**2)
    return (x*V.unsqueeze(0).unsqueeze(0)).clamp(0,1)

def degrade_psf(x, sigma0=0.30, alpha_psf=1.20):
    r,_,_ = radial_map(*x.shape[2:], x.device)
    sigma = sigma0+alpha_psf*r**2
    coords = torch.tensor([
        [-1.,-1.],[-1.,0.],[-1.,1.],
        [ 0.,-1.],[ 0.,0.],[ 0.,1.],
        [ 1.,-1.],[ 1.,0.],[ 1.,1.]], device=x.device)
    d2=(coords**2).sum(-1)
    s=sigma.unsqueeze(-1).clamp(0.1,1.0)
    k=torch.exp(-d2/(2*s**2)); k=k/k.sum(-1,keepdim=True)
    return spatially_varying_conv(x,k)

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
# PDK Models (15번과 동일)
# ============================================================================

class VignettingPDK(nn.Module):
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

    def get_params(self):
        return {"alpha": self.alpha.item()}


class PSFPDK(nn.Module):
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
        return spatially_varying_conv(x,delta+lam*(delta-g))

    def get_params(self):
        return {"sigma0":self.sigma0.item(),
                "alpha_psf":self.alpha_psf.item(),
                "lambda0":self.lambda0.item(),
                "alpha_lam":self.alpha_lam.item()}


class ComaPDK(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._s0=nn.Parameter(torch.tensor(_sp(0.5)))
        self._aps=nn.Parameter(torch.tensor(_sp(0.5)))
        self._ck=nn.Parameter(torch.tensor(_sp(0.1)))
        self._l0=nn.Parameter(torch.tensor(_sp(0.5)))
        self._al=nn.Parameter(torch.tensor(_sp(0.3)))
        self._av=nn.Parameter(torch.tensor(_sp(1.0)))
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

    def get_params(self):
        return {"sigma0":self.sigma0.item(),"alpha_psf":self.alpha_psf.item(),
                "coma_k":self.coma_k.item(),"alpha_vig":self.alpha_vig.item()}


class NonUnifPDK(nn.Module):
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

    def get_params(self):
        return {"cx":self.cx.item(),"cy":self.cy.item(),
                "sig":self.sig.item(),"ell":self.ell.item(),
                "av":self.av.item()}


class GlobalKernel(nn.Module):
    def __init__(self):
        super().__init__()
        init=torch.zeros(9); init[4]=3.0
        self.kernel_logits=nn.Parameter(init)

    def forward(self, x):
        k=torch.softmax(self.kernel_logits,dim=0)
        return global_conv(x,k)


# ============================================================================
# Training (공통 loss)
# ============================================================================

def loss_fn(out, clean):
    l1=F.l1_loss(out,clean)
    def gmap(t): return t[:,:,:,1:]-t[:,:,:,:-1], t[:,:,1:,:]-t[:,:,:-1,:]
    ox,oy=gmap(out); cx,cy=gmap(clean)
    return l1+0.1*(F.l1_loss(ox,cx)+F.l1_loss(oy,cy))


def train_on_patterns(model, patterns_deg, patterns_clean,
                      n_iter, lr, verbose=True):
    """
    Calibration: 여러 패턴을 배치로 학습.

    핵심: 이미지 content가 다양하므로
    content bias 없이 소자 특성(kernel 파라미터)만 수렴.
    """
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n_iter)
    history=[]
    pat_names=list(patterns_deg.keys())

    for i in range(n_iter):
        opt.zero_grad()
        total_loss=torch.tensor(0.0)
        # 모든 패턴에 대해 loss 합산 -> content bias 상쇄
        for name in pat_names:
            deg=patterns_deg[name]; clean=patterns_clean[name]
            out=model(deg)
            total_loss=total_loss+loss_fn(out,clean)
        total_loss=total_loss/len(pat_names)
        total_loss.backward(); opt.step(); sched.step()
        history.append(total_loss.item())
        if verbose and (i+1)%100==0:
            print(f"    iter {i+1:4d}/{n_iter}  loss={total_loss.item():.6f}")
    return history


def train_oracle(model, deg, clean, n_iter, lr):
    """Oracle: 해당 이미지로 직접 학습 (upper bound)"""
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n_iter)
    history=[]
    for i in range(n_iter):
        opt.zero_grad()
        out=model(deg); loss=loss_fn(out,clean)
        loss.backward(); opt.step(); sched.step()
        history.append(loss.item())
    with torch.no_grad(): cor=model(deg).clamp(0,1)
    return cor, history


# ============================================================================
# Inference with fixed kernel
# ============================================================================

def inference_fixed(model, deg):
    """
    학습된 파라미터로 kernel map 고정 후 적용.
    추가 학습 없음.
    """
    model.eval()
    with torch.no_grad():
        return model(deg).clamp(0,1)


# ============================================================================
# Load BSD500 unseen images
# ============================================================================

def load_unseen_images(args):
    if args.data_dir and os.path.isdir(args.data_dir):
        exts=("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG")
        paths=[]
        for ext in exts:
            paths+=glob.glob(os.path.join(args.data_dir,"**",ext),recursive=True)
            paths+=glob.glob(os.path.join(args.data_dir,ext))
        paths=sorted(set(paths)); imgs=[]
        for p in paths:
            try: imgs.append(load_single_image(p,args.img_size))
            except Exception: continue
            if len(imgs)==args.n_inf_images: break
        if imgs:
            print(f"[Load] {len(imgs)} unseen images from {args.data_dir}")
            return imgs
    try:
        from skimage import data as skd; import skimage.color as skc
        bl=[skd.camera(),skd.astronaut(),skd.chelsea(),
            skd.coffee(),skd.horse(),skd.hubble_deep_field()]
        imgs=[]
        for arr in bl[:args.n_inf_images]:
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
        print(f"[Warning] {e}"); return []


# ============================================================================
# Visualization
# ============================================================================

def save_main_comparison(case_results, res_dir):
    """
    4 case x 5 col:
    [raw | degraded | global | cal-PDK | oracle-PDK]
    """
    n_cases=len(case_results)
    fig,axes=plt.subplots(n_cases,5,figsize=(18,4*n_cases+0.5))
    if n_cases==1: axes=axes[np.newaxis,:]

    col_titles=["1. Raw (clean)","2. Degraded",
                "3. Global kernel","4. Cal-PDK (ours)","5. Oracle PDK"]
    for c,t in enumerate(col_titles):
        axes[0,c].set_title(t,fontsize=10,fontweight="bold",pad=6)

    for r,res in enumerate(case_results):
        raw_np=to_np(res["raw"]); deg_np=to_np(res["deg"])
        g_np=to_np(res["g_cor"]); cal_np=to_np(res["cal_cor"])
        ora_np=to_np(res["ora_cor"])

        for c,(img,cmap) in enumerate(zip(
            [raw_np,deg_np,g_np,cal_np,ora_np],
            ["gray"]*5)):
            axes[r,c].imshow(img,cmap=cmap,vmin=0,vmax=1)
            axes[r,c].axis("off")

        # row label
        axes[r,0].set_ylabel(res["name"],fontsize=9,fontweight="bold",
                             rotation=0,labelpad=80,va="center")

        # PSNR annotations
        for c,(pv,color) in enumerate(zip(
            [res["psnr_deg"],res["psnr_g"],
             res["psnr_cal"],res["psnr_ora"]],
            ["dimgray","steelblue","tomato","green"]), start=1):
            axes[r,c].text(0.5,-0.06,f"PSNR={pv:.1f}dB",
                           transform=axes[r,c].transAxes,
                           ha="center",va="top",fontsize=8,color=color)

        # Cal vs Oracle gap
        gap=res["psnr_cal"]-res["psnr_ora"]
        axes[r,4].text(0.5,-0.12,
                       f"gap vs oracle: {gap:+.1f}dB",
                       transform=axes[r,4].transAxes,
                       ha="center",va="top",fontsize=7.5,color="gray")

    fig.suptitle("Calibration-based PDK vs Oracle vs Global [BSD500]",
                 fontsize=13,y=1.01)
    plt.tight_layout(rect=[0,0.04,1,1])
    path=os.path.join(res_dir,"16_main_comparison.png")
    fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_psnr_summary(case_results, res_dir):
    names=[r["name"] for r in case_results]
    pd_=[r["psnr_deg"] for r in case_results]
    pg_ =[r["psnr_g"]   for r in case_results]
    pc_ =[r["psnr_cal"] for r in case_results]
    po_ =[r["psnr_ora"] for r in case_results]

    x=np.arange(len(names)); w=0.18
    fig,ax=plt.subplots(figsize=(12,5))
    ax.bar(x-1.5*w, pd_, w, label="Degraded",      color="#aaaaaa")
    ax.bar(x-0.5*w, pg_, w, label="Global kernel",  color="steelblue")
    ax.bar(x+0.5*w, pc_, w, label="Cal-PDK (ours)", color="tomato")
    ax.bar(x+1.5*w, po_, w, label="Oracle PDK",     color="green",alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(names,fontsize=10)
    ax.set_ylabel("PSNR (dB)",fontsize=10)
    ax.set_title("PSNR: Calibration PDK vs Oracle vs Global [BSD500]",fontsize=12)
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3,axis="y")

    for xi,(d,g,c,o) in enumerate(zip(pd_,pg_,pc_,po_)):
        ax.text(xi-1.5*w,d+0.3,f"{d:.1f}",ha="center",fontsize=6.5,color="#666")
        ax.text(xi-0.5*w,g+0.3,f"{g:.1f}",ha="center",fontsize=6.5,color="steelblue")
        ax.text(xi+0.5*w,c+0.3,f"{c:.1f}",ha="center",fontsize=6.5,color="tomato")
        ax.text(xi+1.5*w,o+0.3,f"{o:.1f}",ha="center",fontsize=6.5,color="green")

    plt.tight_layout()
    path=os.path.join(res_dir,"16_psnr_summary.png")
    fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_calibration_detail(cal_patterns, cal_histories, res_dir):
    """
    calibration 패턴 시각화 + 학습 곡선.
    행0: 전체 calibration 패턴 (n_pats개)
    행1: case별 loss curve (n_cases개, 빈 subplot 없음)
    """
    n_pats  = len(cal_patterns)
    n_cases = len(cal_histories)

    # 행0: n_pats열, 행1: n_cases열 → 각 행의 열 수가 달라서 gridspec 사용
    fig = plt.figure(figsize=(max(n_pats, n_cases) * 2.8, 7))
    import matplotlib.gridspec as gridspec

    # 행0: 패턴 (n_pats개)
    gs0 = gridspec.GridSpec(1, n_pats, figure=fig,
                            left=0.03, right=0.97, top=0.92, bottom=0.52,
                            wspace=0.1)
    for c, (name, img) in enumerate(cal_patterns.items()):
        ax = fig.add_subplot(gs0[0, c])
        ax.imshow(to_np(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=8, pad=3)
        ax.axis("off")

    # 행1: loss curve (n_cases개)
    gs1 = gridspec.GridSpec(1, n_cases, figure=fig,
                            left=0.05, right=0.97, top=0.45, bottom=0.08,
                            wspace=0.35)
    colors = ["steelblue", "tomato", "green", "orange"]
    for c, (case_name, hist) in enumerate(cal_histories.items()):
        ax = fig.add_subplot(gs1[0, c])
        ax.plot(hist, color=colors[c % len(colors)], linewidth=1.5)
        ax.set_title(f"Cal loss: {case_name}", fontsize=9, pad=4)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    fig.suptitle("Calibration patterns (top) & training loss per case (bottom)",
                 fontsize=12)
    path = os.path.join(res_dir, "16_calibration_detail.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_params_comparison(case_true_params, case_cal_params,
                           case_ora_params, res_dir):
    """
    true params vs calibration-learned params vs oracle params 비교
    """
    n_cases=len(case_true_params)
    fig,axes=plt.subplots(1,n_cases,figsize=(5*n_cases,4))
    if n_cases==1: axes=[axes]

    for ax,(case_name,true_p) in zip(axes,case_true_params.items()):
        cal_p=case_cal_params.get(case_name,{})
        ora_p=case_ora_params.get(case_name,{})
        params=list(true_p.keys())
        x=np.arange(len(params))
        w=0.25
        true_vals=[true_p[p] for p in params]
        cal_vals =[cal_p.get(p,0) for p in params]
        ora_vals =[ora_p.get(p,0) for p in params]
        ax.bar(x-w,  true_vals, w, label="True",      color="#aaaaaa")
        ax.bar(x,    cal_vals,  w, label="Cal-PDK",   color="tomato")
        ax.bar(x+w,  ora_vals,  w, label="Oracle",    color="green",alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(params,fontsize=8,rotation=15)
        ax.set_title(case_name,fontsize=10); ax.legend(fontsize=8)
        ax.grid(True,alpha=0.3,axis="y")

    fig.suptitle("True vs Calibration-learned vs Oracle params",fontsize=12)
    plt.tight_layout()
    path=os.path.join(res_dir,"16_params_comparison.png")
    fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


# ============================================================================
# Args & Main
# ============================================================================

def parse_args():
    p=argparse.ArgumentParser(
        description="Calibration-based PDK [BSD500]")
    p.add_argument("--data-dir",     type=str,   default="")
    p.add_argument("--n-inf-images", type=int,   default=4,
                   help="inference에 사용할 unseen 이미지 수")
    p.add_argument("--img-size",     type=int,   default=256)
    p.add_argument("--n-cal-iter",   type=int,   default=500,
                   help="calibration 학습 반복 횟수")
    p.add_argument("--n-ora-iter",   type=int,   default=500,
                   help="oracle 학습 반복 횟수")
    p.add_argument("--lr",           type=float, default=0.02)
    p.add_argument("--res-dir",      type=str,   default="./res/calibration_pdk")
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
    args=parse_args()
    os.makedirs(args.res_dir,exist_ok=True)
    H=W=args.img_size

    # ── Phase 1: Calibration patterns 준비 ──────────────────────────
    print("[Phase 1] Preparing calibration patterns...")
    cal_patterns_clean=make_calibration_patterns_simple(H)
    n_pats=len(cal_patterns_clean)
    print(f"  {n_pats} calibration patterns: {list(cal_patterns_clean.keys())}")

    # ── unseen inference 이미지 로드 ─────────────────────────────────
    print("[Load] Unseen inference images...")
    unseen_imgs=load_unseen_images(args)
    if not unseen_imgs:
        print("[Error] No images loaded"); return
    print(f"  {len(unseen_imgs)} unseen images loaded")

    # ── case 정의 ────────────────────────────────────────────────────
    cases=[
        {
            "name":       "Vignetting",
            "degrade_fn": lambda x: degrade_vignetting(x,args.vig_alpha),
            "true_params":{"alpha":args.vig_alpha},
            "PDKModel":   VignettingPDK,
        },
        {
            "name":       "PSF blur",
            "degrade_fn": lambda x: degrade_psf(x,args.psf_sigma0,args.psf_alpha),
            "true_params":{"sigma0":args.psf_sigma0,"alpha_psf":args.psf_alpha},
            "PDKModel":   PSFPDK,
        },
        {
            "name":       "Coma aberration",
            "degrade_fn": lambda x: degrade_coma(
                x,args.psf_sigma0,args.psf_alpha,args.coma_k,args.coma_alpha_vig),
            "true_params":{"sigma0":args.psf_sigma0,"coma_k":args.coma_k,
                           "alpha_vig":args.coma_alpha_vig},
            "PDKModel":   ComaPDK,
        },
        {
            "name":       "Non-uniform illum",
            "degrade_fn": lambda x: degrade_nonunif(
                x,args.nu_cx,args.nu_cy,args.nu_sigma,args.nu_ell,args.nu_vig),
            "true_params":{"cx_off":args.nu_cx,"cy_off":args.nu_cy,
                           "sigma_illum":args.nu_sigma},
            "PDKModel":   NonUnifPDK,
        },
    ]

    all_case_results=[]
    cal_histories={}
    case_cal_params={}; case_ora_params={}
    case_true_params={c["name"]:c["true_params"] for c in cases}

    for case in cases:
        print(f"\n{'='*50}")
        print(f"[Case] {case['name']}")
        degrade=case["degrade_fn"]

        # calibration 패턴 degradation
        print("  [Cal] Applying degradation to cal patterns...")
        cal_patterns_deg={
            name: degrade(img)
            for name,img in cal_patterns_clean.items()
        }

        # ── Phase 1: Calibration training ────────────────────────────
        print(f"  [Cal] Training on {n_pats} patterns "
              f"(n_iter={args.n_cal_iter})...")
        cal_model=case["PDKModel"](H,W)
        cal_hist=train_on_patterns(
            cal_model,
            cal_patterns_deg, cal_patterns_clean,
            n_iter=args.n_cal_iter, lr=args.lr)
        cal_histories[case["name"]]=cal_hist
        case_cal_params[case["name"]]=cal_model.get_params()
        print(f"  [Cal] learned params: {case_cal_params[case['name']]}")

        # Global kernel calibration
        print("  [Global] Training global kernel on cal patterns...")
        g_model=GlobalKernel()
        train_on_patterns(g_model,
                          cal_patterns_deg,cal_patterns_clean,
                          n_iter=args.n_cal_iter,lr=args.lr,
                          verbose=False)

        # ── Phase 2: Inference on unseen images ──────────────────────
        print(f"  [Inference] Applying fixed kernel to {len(unseen_imgs)} unseen images...")
        psnr_degs,psnr_gs,psnr_cals,psnr_oras=[],[],[],[]
        ora_cors=[]

        for i,raw in enumerate(unseen_imgs):
            deg=degrade(raw)

            # Calibration PDK: 고정 kernel 적용 (no training)
            cal_cor=inference_fixed(cal_model,deg)

            # Global: 고정 kernel 적용
            g_cor=inference_fixed(g_model,deg)

            # Oracle PDK: 해당 이미지로 직접 학습 (upper bound)
            print(f"    Oracle training img {i+1}/{len(unseen_imgs)}...")
            ora_model=case["PDKModel"](H,W)
            ora_cor,_=train_oracle(ora_model,deg,raw,
                                   n_iter=args.n_ora_iter,lr=args.lr)

            psnr_degs.append(psnr(raw,deg))
            psnr_gs.append(psnr(raw,g_cor))
            psnr_cals.append(psnr(raw,cal_cor))
            psnr_oras.append(psnr(raw,ora_cor))
            ora_cors.append(ora_cor)

            if i==0:
                case_ora_params[case["name"]]=ora_model.get_params()

        pd_=np.mean(psnr_degs); pg_=np.mean(psnr_gs)
        pc_=np.mean(psnr_cals); po_=np.mean(psnr_oras)

        print(f"  PSNR  Degraded={pd_:.2f}  Global={pg_:.2f}"
              f"  Cal-PDK={pc_:.2f}  Oracle={po_:.2f}"
              f"  gap={pc_-po_:+.2f}dB")

        # 첫 번째 이미지로 시각화
        raw0=unseen_imgs[0]; deg0=degrade(raw0)
        cal_cor0=inference_fixed(cal_model,deg0)
        g_cor0=inference_fixed(g_model,deg0)

        all_case_results.append({
            "name":     case["name"],
            "raw":      raw0,
            "deg":      deg0,
            "g_cor":    g_cor0,
            "cal_cor":  cal_cor0,
            "ora_cor":  ora_cors[0],
            "psnr_deg": pd_,
            "psnr_g":   pg_,
            "psnr_cal": pc_,
            "psnr_ora": po_,
        })

    # ── 시각화 ────────────────────────────────────────────────────────
    print("\n[Saving results...]")
    save_main_comparison(all_case_results,args.res_dir)
    save_psnr_summary(all_case_results,args.res_dir)
    save_calibration_detail(cal_patterns_clean,cal_histories,args.res_dir)
    save_params_comparison(case_true_params,case_cal_params,
                           case_ora_params,args.res_dir)

    print(f"\n[Done] {os.path.abspath(args.res_dir)}")
    print("  16_main_comparison.png    -- Cal-PDK vs Oracle vs Global")
    print("  16_psnr_summary.png       -- PSNR bar chart (4 methods)")
    print("  16_calibration_detail.png -- cal patterns + loss curves")
    print("  16_params_comparison.png  -- true vs cal vs oracle params")

    # PSNR 요약 출력
    print("\n[Summary]")
    print(f"{'Case':<22} {'Deg':>6} {'Global':>7} {'Cal-PDK':>8} {'Oracle':>7} {'gap':>6}")
    print("-"*60)
    for res in all_case_results:
        gap=res["psnr_cal"]-res["psnr_ora"]
        print(f"{res['name']:<22} {res['psnr_deg']:>6.1f} "
              f"{res['psnr_g']:>7.1f} {res['psnr_cal']:>8.1f} "
              f"{res['psnr_ora']:>7.1f} {gap:>+6.1f}")


if __name__ == "__main__":
    main()