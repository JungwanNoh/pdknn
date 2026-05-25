"""
18_dataset_based_pdk_bsd500.py
==========================================
Dataset-based PDK: 실제 이미지로 학습 -> unseen 이미지에 적용

핵심 질문:
  calibration pattern 대신 실제 이미지로 학습해도
  unseen 이미지에 kernel이 일반화되는가?

세 가지 방식 직접 비교:
  A. Cal-PDK    : test pattern 10개로 학습 (16번)
  B. Dataset-PDK: 실제 이미지 N장으로 학습 (새로운 방식)
  C. Oracle PDK : 각 이미지 직접 학습 (upper bound)
  D. Global     : baseline

추가 실험:
  N = 5, 10, 20 장으로 학습량 변화에 따른 성능 변화

4가지 degradation:
  Vignetting / PSF blur / Coma / Non-uniform illum

핵심 차이:
  Cal-PDK:     수학적 패턴 (주파수/공간 설계)
  Dataset-PDK: 자연 이미지 (content diversity로 bias 상쇄)

Run:
  python 18_dataset_based_pdk_bsd500.py
  python 18_dataset_based_pdk_bsd500.py --n-train 20 --n-inf 6
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

def _make_synthetic(n, size):
    """
    BSD500 / skimage 없을 때 사용하는 synthetic 이미지.
    다양한 주파수/패턴으로 N장 생성 → content diversity 확보.
    """
    H = W = size
    ys = torch.linspace(-1, 1, H); xs = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r = (gy**2+gx**2).sqrt()/(2**0.5)
    imgs = []
    seeds = range(n)
    freqs_y = [4,6,8,10,12,15,18,20,25,30,5,7,9,11,13,16,22,28,35,40]
    freqs_x = [4,8,6,12,10,20,15,25,18,30,7,5,11,9,16,13,28,22,40,35]
    for i in seeds:
        fy = freqs_y[i % len(freqs_y)]
        fx = freqs_x[i % len(freqs_x)]
        t = i % 6
        if t == 0:
            p = (0.5+0.5*torch.sign(torch.sin(gy*fy)*torch.sin(gx*fx))).clamp(0,1)
        elif t == 1:
            p = (0.5+0.3*torch.sin(gy*fy)+0.2*torch.cos(gx*fx)).clamp(0,1)
        elif t == 2:
            p = (0.5+0.5*torch.cos(r*(fy+fx)/2)).clamp(0,1)
        elif t == 3:
            p = ((gy*torch.sin(torch.tensor(i*0.5))+gx*torch.cos(torch.tensor(i*0.5))+1)/2).clamp(0,1)
        elif t == 4:
            p = (0.5+0.4*torch.sin((gy+gx)*fy*0.7)).clamp(0,1)
        else:
            gen = torch.Generator(); gen.manual_seed(i*17+42)
            raw = torch.rand(1,1,H,W, generator=gen)
            k_size=15; sig=max(3.0, H/40)
            c1d = torch.arange(k_size).float()-k_size//2
            g1d = torch.exp(-c1d**2/(2*sig**2)); g1d=g1d/g1d.sum()
            gk  = (g1d.unsqueeze(0)*g1d.unsqueeze(1)).view(1,1,k_size,k_size)
            import torch.nn.functional as _F
            sr = _F.conv2d(raw,gk,padding=k_size//2).squeeze()
            p  = (sr-sr.min())/(sr.max()-sr.min()+1e-5)
        imgs.append(p.unsqueeze(0).unsqueeze(0))
    return imgs


def load_images(data_dir, n, size, offset=0, allow_synthetic=True):
    """
    data_dir에서 실제 이미지 n장 로드.
    allow_synthetic=False이면 실제 이미지가 부족해도 synthetic으로 보충하지 않음.
    """
    imgs = []

    # 1. 실제 데이터셋
    if data_dir and os.path.isdir(data_dir):
        exts = ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG")
        paths = []
        for ext in exts:
            paths += glob.glob(os.path.join(data_dir,"**",ext), recursive=True)
            paths += glob.glob(os.path.join(data_dir, ext))
        paths = sorted(set(paths))
        print(f"[Load] Found {len(paths)} images in {data_dir}")
        for p in paths[offset:offset+n]:
            try: imgs.append(load_single_image(p, size))
            except Exception: continue
        if imgs:
            actual = min(len(imgs), n)
            print(f"[Load] Using {actual} real images")
            return imgs[:actual]
    else:
        print(f"[Load] Path not found: '{data_dir}'")

    # 실제 이미지 없을 때
    if not allow_synthetic:
        print(f"[Error] No real images found in '{data_dir}'")
        return []

    # 2. skimage fallback
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from skimage import data as skd
            import skimage.color as skc
            bl = [skd.camera(), skd.astronaut(), skd.chelsea(),
                  skd.coffee(), skd.horse(), skd.hubble_deep_field()]
        for arr in bl:
            if arr.ndim == 3: arr = skc.rgb2gray(arr)
            arr = arr.astype(np.float32)
            mn, mx = arr.min(), arr.max()
            arr = (arr - mn) / (mx - mn + 1e-8)
            pil = Image.fromarray((arr*255).astype(np.uint8))
            s = min(pil.size); w,h = pil.size
            pil = pil.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
            pil = pil.resize((size,size), Image.BILINEAR)
            imgs.append(torch.from_numpy(
                np.array(pil,dtype=np.float32)/255.).unsqueeze(0).unsqueeze(0))
        print(f"[Load] skimage fallback: {len(imgs)} images")
    except Exception as e:
        print(f"[Warning skimage] {e}")

    # 3. synthetic 보충
    total_needed = offset + n
    if len(imgs) < total_needed:
        needed = total_needed - len(imgs)
        print(f"[Fallback] Generating {needed} synthetic images...")
        imgs += _make_synthetic(needed, size)

    result = imgs[offset:offset+n]
    print(f"[Load] Final: {len(result)} images")
    return result

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
# Calibration patterns (16번과 동일)
# ============================================================================

def make_cal_patterns(size, device="cpu"):
    H = W = size
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gen = torch.Generator(); gen.manual_seed(42)
    raw = torch.rand(1,1,H,W, generator=gen, device=device)
    k_size=15; sig=5.0
    c1d = torch.arange(k_size, device=device).float()-k_size//2
    g1d = torch.exp(-c1d**2/(2*sig**2)); g1d=g1d/g1d.sum()
    gk  = (g1d.unsqueeze(0)*g1d.unsqueeze(1)).view(1,1,k_size,k_size)
    sr  = F.conv2d(raw,gk,padding=k_size//2).squeeze()
    sr  = (sr-sr.min())/(sr.max()-sr.min()+1e-5)
    pats = {
        "checker_fine":    (0.5+0.5*torch.sign(torch.sin(gy*12)*torch.sin(gx*12))).clamp(0,1),
        "checker_coarse":  (0.5+0.5*torch.sign(torch.sin(gy*6)*torch.sin(gx*6))).clamp(0,1),
        "sine_h":          (0.5+0.5*torch.sin(gy*15)).clamp(0,1),
        "sine_v":          (0.5+0.5*torch.sin(gx*15)).clamp(0,1),
        "sine_diag":       (0.5+0.5*torch.sin((gy+gx)*10)).clamp(0,1),
        "concentric":      (0.5+0.5*torch.cos((gy**2+gx**2).sqrt()*20)).clamp(0,1),
        "gradient_x":      ((gx+1)/2).clamp(0,1),
        "gradient_y":      ((gy+1)/2).clamp(0,1),
        "gradient_radial": (1.0-(gy**2+gx**2).sqrt()/(2**0.5)).clamp(0,1),
        "smooth_random":   sr,
    }
    return {k: v.unsqueeze(0).unsqueeze(0) for k,v in pats.items()}


# ============================================================================
# Degradation
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
    r,gy,gx=radial_map(*x.shape[2:], x.device)
    r2=r**2; rs=r.clamp(1e-6)
    sigma=sigma0+alpha_psf*r2
    coords=torch.tensor([
        [-1.,-1.],[-1.,0.],[-1.,1.],
        [ 0.,-1.],[ 0.,0.],[ 0.,1.],
        [ 1.,-1.],[ 1.,0.],[ 1.,1.]], device=x.device)
    d2=(coords**2).sum(-1); s=sigma.unsqueeze(-1).clamp(0.1,1.0)
    shift_y=coma_k*r2*(gy/rs); shift_x=coma_k*r2*(gx/rs)
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
    r,gy,gx=radial_map(*x.shape[2:], x.device)
    V_rad=1.0/(1.0+alpha_vig*r**2)
    dy=gy-cy_off; dx=gx-cx_off; d2=dx**2+(dy*ellipse_ratio)**2
    ill=torch.exp(-d2/(2*sigma_illum**2)); ill=ill/ill.max().clamp(1e-5)
    V=(V_rad*(0.3+0.7*ill)).clamp(0,1)
    return (x*V.unsqueeze(0).unsqueeze(0)).clamp(0,1)


# ============================================================================
# PDK Models (16번과 동일)
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
    def get_params(self): return {"alpha":self.alpha.item()}

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
        return {"sigma0":self.sigma0.item(),"alpha_psf":self.alpha_psf.item(),
                "lambda0":self.lambda0.item(),"alpha_lam":self.alpha_lam.item()}

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
# Training
# ============================================================================

def train_on_dataset(model, train_imgs, degrade_fn, n_iter, lr, verbose=True):
    """
    실제 이미지 N장으로 학습.
    매 iteration마다 모든 이미지에 대해 loss 합산.
    → content가 다양 → content bias 상쇄 → 물리 파라미터만 수렴.

    Cal-PDK와의 차이:
      Cal-PDK:     수학적 패턴 (주파수 균일)
      Dataset-PDK: 자연 이미지 (content 다양성)
    """
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    history = []
    # 미리 degraded 이미지 생성 (반복 계산 방지)
    pairs = [(img, degrade_fn(img)) for img in train_imgs]

    for i in range(n_iter):
        opt.zero_grad()
        total = torch.tensor(0.0)
        for clean, deg in pairs:
            out   = model(deg)
            total = total + loss_fn(out, clean)
        total = total / len(pairs)
        total.backward(); opt.step(); sched.step()
        history.append(total.item())
        if verbose and (i+1) % 100 == 0:
            print(f"    iter {i+1:4d}/{n_iter}  loss={total.item():.6f}")
    return history


def train_on_patterns(model, patterns_clean, degrade_fn, n_iter, lr, verbose=False):
    """Cal-PDK: calibration pattern으로 학습 (16번 방식)"""
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    history = []
    pairs = [(img, degrade_fn(img)) for img in patterns_clean.values()]
    for i in range(n_iter):
        opt.zero_grad()
        total = torch.tensor(0.0)
        for clean, deg in pairs:
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
        return model(deg).clamp(0,1)


def inference_fixed(model, deg):
    model.eval()
    with torch.no_grad():
        return model(deg).clamp(0,1)


# ============================================================================
# N장 실험: 학습량에 따른 성능 변화
# ============================================================================

def run_n_sweep(train_imgs_all, inf_imgs, degrade_fn,
                PDKModel, n_iter, lr, n_list, H, W):
    """
    n_list = [5, 10, 20, ...] 장으로 각각 학습 후 inf 이미지에 적용
    returns: {n: mean_psnr}
    """
    results = {}
    for n in n_list:
        train_imgs = train_imgs_all[:n]
        model = PDKModel(H, W)
        train_on_dataset(model, train_imgs, degrade_fn,
                         n_iter=n_iter, lr=lr, verbose=False)
        psnrs = []
        for raw in inf_imgs:
            deg = degrade_fn(raw)
            cor = inference_fixed(model, deg)
            psnrs.append(psnr(raw, cor))
        results[n] = np.mean(psnrs)
        print(f"    N={n:3d}  PSNR={results[n]:.2f}dB")
    return results


# ============================================================================
# Visualization
# ============================================================================

def save_main_comparison(case_results, res_dir):
    """
    4 case x 5 col:
    [raw | deg | cal-pdk | dataset-pdk | oracle]
    + 5열 diff map: |method - raw|

    unseen 이미지에 대한 비교 결과 저장.
    """
    n = len(case_results)
    fig, axes = plt.subplots(n, 6, figsize=(22, 4*n+0.5))
    if n == 1: axes = axes[np.newaxis,:]

    col_titles = ["1. Raw (clean)", "2. Degraded",
                  "3. Cal-PDK", "4. Dataset-PDK",
                  "5. Oracle PDK", "6. |Cal - Dataset| diff"]
    for c, t in enumerate(col_titles):
        axes[0,c].set_title(t, fontsize=9, fontweight="bold", pad=6)

    colors = ["dimgray", "tomato", "darkorange", "green"]

    for r, res in enumerate(case_results):
        cal_np  = to_np(res["cal_cor"])
        ds_np   = to_np(res["ds_cor"])
        ora_np  = to_np(res["ora_cor"])
        raw_np  = to_np(res["raw"])
        deg_np  = to_np(res["deg"])
        diff_np = np.abs(cal_np - ds_np)

        for c, img in enumerate([raw_np, deg_np, cal_np, ds_np, ora_np]):
            axes[r,c].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[r,c].axis("off")

        # diff map (col 5)
        vmax_d = max(diff_np.max(), 0.01)
        axes[r,5].imshow(diff_np, cmap="hot", vmin=0, vmax=vmax_d)
        axes[r,5].axis("off")

        axes[r,0].set_ylabel(res["name"], fontsize=9, fontweight="bold",
                             rotation=0, labelpad=90, va="center")

        # PSNR annotations
        for c, (pv, col) in enumerate(zip(
            [res["psnr_deg"], res["psnr_cal"],
             res["psnr_ds"], res["psnr_ora"]], colors), start=1):
            axes[r,c].text(0.5,-0.06, f"PSNR={pv:.1f}dB",
                           transform=axes[r,c].transAxes,
                           ha="center", va="top", fontsize=8.5, color=col)

        # Dataset vs Cal gap, Dataset vs Oracle gap
        gap_cal = res["psnr_ds"] - res["psnr_cal"]
        gap_ora = res["psnr_ds"] - res["psnr_ora"]
        axes[r,3].text(0.5,-0.12,
                       f"vs Cal: {gap_cal:+.1f}  vs Oracle: {gap_ora:+.1f}",
                       transform=axes[r,3].transAxes,
                       ha="center", va="top", fontsize=7.5, color="gray")

    fig.suptitle("Unseen image test: Cal-PDK vs Dataset-PDK vs Oracle [BSD500]",
                 fontsize=13, y=1.01)
    plt.tight_layout(rect=[0,0.04,1,1])
    path = os.path.join(res_dir, "18_main_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_psnr_summary(case_results, res_dir):
    names = [r["name"]     for r in case_results]
    pd_   = [r["psnr_deg"] for r in case_results]
    pc_   = [r["psnr_cal"] for r in case_results]
    pds_  = [r["psnr_ds"]  for r in case_results]
    po_   = [r["psnr_ora"] for r in case_results]

    x = np.arange(len(names)); w = 0.18
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x-1.5*w, pd_,  w, label="Degraded",         color="#aaaaaa")
    ax.bar(x-0.5*w, pc_,  w, label="Cal-PDK (pattern)", color="tomato")
    ax.bar(x+0.5*w, pds_, w, label="Dataset-PDK",       color="darkorange")
    ax.bar(x+1.5*w, po_,  w, label="Oracle PDK",        color="green", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("PSNR (dB)", fontsize=10)
    ax.set_title("Unseen image PSNR: Cal-PDK vs Dataset-PDK vs Oracle [BSD500]",
                 fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    for xi, vals in enumerate(zip(pd_, pc_, pds_, po_)):
        cols = ["#666", "tomato", "darkorange", "green"]
        offs = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
        for xoff, v, col in zip(offs, vals, cols):
            ax.text(xi+xoff, v+0.3, f"{v:.1f}",
                    ha="center", fontsize=7, color=col)

    plt.tight_layout()
    path = os.path.join(res_dir, "18_psnr_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_n_sweep(sweep_results, cal_psnrs, oracle_psnrs, res_dir):
    """
    학습 이미지 수 N에 따른 Dataset-PDK PSNR 변화
    Cal-PDK (패턴 기반) / Oracle 참조선 포함
    """
    n_cases = len(sweep_results)
    fig, axes = plt.subplots(1, n_cases, figsize=(5*n_cases, 4.5))
    if n_cases == 1: axes = [axes]
    colors = ["steelblue","tomato","green","orange"]

    for ax, (case_name, sweep), cal_p, ora_p, col in zip(
            axes, sweep_results.items(),
            cal_psnrs, oracle_psnrs, colors):
        ns   = sorted(sweep.keys())
        vals = [sweep[n] for n in ns]
        ax.plot(ns, vals, "o-", color=col, linewidth=2,
                markersize=6, label="Dataset-PDK")
        ax.axhline(cal_p, color="tomato",  linewidth=1.5,
                   linestyle="--", label=f"Cal-PDK ({cal_p:.1f}dB)")
        ax.axhline(ora_p, color="green",   linewidth=1.5,
                   linestyle=":", label=f"Oracle ({ora_p:.1f}dB)")
        ax.set_xlabel("# training images (N)", fontsize=9)
        ax.set_ylabel("PSNR (dB)", fontsize=9)
        ax.set_title(case_name, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xticks(ns)

    fig.suptitle("Dataset-PDK: PSNR vs # training images "
                 "(Cal-PDK=pattern baseline, Oracle=upper bound)",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(res_dir, "18_n_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_loss_comparison(case_results, res_dir):
    """Cal-PDK vs Dataset-PDK 학습 곡선 비교"""
    n = len(case_results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]
    for ax, res in zip(axes, case_results):
        ax.plot(res["cal_hist"], color="tomato",    linewidth=1.5,
                label="Cal-PDK (pattern)")
        ax.plot(res["ds_hist"],  color="darkorange", linewidth=1.5,
                label="Dataset-PDK")
        ax.set_title(res["name"], fontsize=10)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("Training loss: Cal-PDK vs Dataset-PDK", fontsize=12)
    plt.tight_layout()
    path = os.path.join(res_dir, "18_loss_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_params_comparison(case_results, res_dir):
    """true / cal / dataset / oracle 파라미터 비교"""
    n = len(case_results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]
    for ax, res in zip(axes, case_results):
        tp = res["true_params"]
        cp = res["cal_params"]
        dp = res["ds_params"]
        op = res["ora_params"]
        params = list(tp.keys())
        x = np.arange(len(params)); w = 0.18
        ax.bar(x-1.5*w, [tp[p] for p in params], w,
               label="True", color="#aaaaaa")
        ax.bar(x-0.5*w, [cp.get(p,0) for p in params], w,
               label="Cal-PDK", color="tomato")
        ax.bar(x+0.5*w, [dp.get(p,0) for p in params], w,
               label="Dataset-PDK", color="darkorange")
        ax.bar(x+1.5*w, [op.get(p,0) for p in params], w,
               label="Oracle", color="green", alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(params, fontsize=8, rotation=15)
        ax.set_title(res["name"], fontsize=10)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("True vs Cal-PDK vs Dataset-PDK vs Oracle params",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(res_dir, "18_params_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Saved] {path}")


def save_figure_images(all_results, res_dir):
    """
    논문 figure용 개별 이미지 저장.
    Vignetting / PSF blur 케이스에서
    raw / degraded / corrected(Dataset-PDK) 각각 저장.
    타이틀 없음, axis 없음, 여백 없음.
    """
    target_cases = ["Vignetting", "PSF blur"]
    fig_dir = os.path.join(res_dir, "figure_images")
    os.makedirs(fig_dir, exist_ok=True)

    for res in all_results:
        name = res["name"]
        if name not in target_cases:
            continue

        tag = name.lower().replace(" ", "_")  # "vignetting" / "psf_blur"

        imgs = {
            "raw":        res["raw"],
            "degraded":   res["deg"],
            "corrected":  res["ds_cor"],
        }

        for label, tensor in imgs.items():
            arr = tensor.squeeze().detach().cpu().float().numpy()
            # clamp
            arr = arr.clip(0, 1)

            fig, ax = plt.subplots(1, 1,
                                   figsize=(arr.shape[1]/100,
                                            arr.shape[0]/100),
                                   dpi=300)
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            path = os.path.join(fig_dir, f"{tag}_{label}.png")
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print(f"  [Figure] {path}")

    print(f"  [Figure] Saved to {fig_dir}/")


# ============================================================================
# Args & Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Dataset-based PDK vs Cal-PDK [BSD500]")
    p.add_argument("--train-dir",  type=str,   default="./data/BSD300/images/train",
                   help="학습용 이미지 폴더 (default: BSD300/images/train)")
    p.add_argument("--inf-dir",    type=str,   default="./data/BSD300/images/test",
                   help="inference용 이미지 폴더 (default: BSD300/images/test)")
    p.add_argument("--data-dir",   type=str,   default="",
                   help="train/inf-dir 대신 단일 폴더 사용 시 (train/inf 자동 분리)")
    p.add_argument("--n-train",    type=int,   default=10,
                   help="Dataset-PDK 학습 이미지 수 (default:10)")
    p.add_argument("--n-inf",      type=int,   default=4,
                   help="inference 이미지 수 (default:4)")
    p.add_argument("--img-size",   type=int,   default=256)
    p.add_argument("--n-iter",     type=int,   default=500)
    p.add_argument("--n-ora-iter", type=int,   default=500)
    p.add_argument("--lr",         type=float, default=0.02)
    p.add_argument("--n-sweep",    type=str,   default="5,10,20",
                   help="N sweep 값들 (쉼표 구분, default:5,10,20)")
    p.add_argument("--res-dir",    type=str,   default="./res/dataset_pdk")
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
    H = W = args.img_size
    n_sweep_list = [int(x) for x in args.n_sweep.split(",")]
    n_total_needed = max(n_sweep_list) + args.n_inf

    # 이미지 로드: train / inference 완전 분리
    max_n_train = max(n_sweep_list)

    if args.data_dir:
        # 단일 폴더: 앞 N장 train, 뒤 n_inf장 inference
        print(f"[Load] Single dir mode: '{args.data_dir}'")
        all_imgs = load_images(args.data_dir, max_n_train + args.n_inf, H,
                                allow_synthetic=False)
        train_imgs = all_imgs[:max_n_train]
        inf_imgs   = all_imgs[max_n_train:max_n_train+args.n_inf]
    else:
        # 분리 폴더: train-dir / inf-dir
        print(f"[Load] Train dir:     '{args.train_dir}'")
        print(f"[Load] Inference dir: '{args.inf_dir}'")
        # train: 실제 이미지만 (synthetic 보충 금지)
        train_imgs = load_images(args.train_dir, max_n_train, H,
                                 allow_synthetic=False)
        # inference: 실제 이미지만 (synthetic 보충 금지)
        inf_imgs   = load_images(args.inf_dir,   args.n_inf,  H,
                                 allow_synthetic=False)
        if not train_imgs or not inf_imgs:
            print("[Error] BSD300 이미지를 읽지 못했습니다.")
            print("  경로를 확인하거나 --data-dir로 단일 폴더를 지정하세요.")
            return

    if not train_imgs:
        print("[Error] No training images"); return
    if not inf_imgs:
        print("[Error] No inference images"); return

    # n_sweep_list를 실제 로드된 train 이미지 수로 클램핑
    n_sweep_list = [min(n, len(train_imgs)) for n in n_sweep_list]
    n_sweep_list = sorted(set(n_sweep_list))
    if max(n_sweep_list) < args.n_train:
        print(f"  [Warning] n-train clamped to {max(n_sweep_list)} "
              f"(only {len(train_imgs)} real images available)")

    print(f"  Train: {len(train_imgs)} imgs  |  Inference: {len(inf_imgs)} imgs (unseen)")

    # calibration patterns (Cal-PDK용)
    cal_pats = make_cal_patterns(H)
    print(f"  Cal patterns: {len(cal_pats)}")

    # case 정의
    cases = [
        {
            "name":        "Vignetting",
            "degrade_fn":  lambda x: degrade_vignetting(x, args.vig_alpha),
            "true_params": {"alpha": args.vig_alpha},
            "PDKModel":    VignettingPDK,
        },
        {
            "name":        "PSF blur",
            "degrade_fn":  lambda x: degrade_psf(x, args.psf_sigma0, args.psf_alpha),
            "true_params": {"sigma0": args.psf_sigma0, "alpha_psf": args.psf_alpha},
            "PDKModel":    PSFPDK,
        },
        {
            "name":        "Coma aberration",
            "degrade_fn":  lambda x: degrade_coma(
                x, args.psf_sigma0, args.psf_alpha, args.coma_k, args.coma_alpha_vig),
            "true_params": {"sigma0": args.psf_sigma0, "coma_k": args.coma_k,
                            "alpha_vig": args.coma_alpha_vig},
            "PDKModel":    ComaPDK,
        },
        {
            "name":        "Non-uniform illum",
            "degrade_fn":  lambda x: degrade_nonunif(
                x, args.nu_cx, args.nu_cy, args.nu_sigma, args.nu_ell, args.nu_vig),
            "true_params": {"cx_off": args.nu_cx, "cy_off": args.nu_cy,
                            "sigma_illum": args.nu_sigma},
            "PDKModel":    NonUnifPDK,
        },
    ]

    all_results = []
    all_sweep   = {}
    cal_psnrs   = []
    oracle_psnrs = []

    for case in cases:
        print(f"\n{'='*55}")
        print(f"[Case] {case['name']}")
        degrade   = case["degrade_fn"]
        PDKModel  = case["PDKModel"]

        # ── Cal-PDK (패턴 기반) ──────────────────────────────────────
        print("  [Cal-PDK] training on calibration patterns...")
        cal_model = PDKModel(H, W)
        cal_hist  = train_on_patterns(
            cal_model, cal_pats, degrade, args.n_iter, args.lr)

        # ── Dataset-PDK (이미지 기반, n_train장) ────────────────────
        print(f"  [Dataset-PDK] training on {args.n_train} images...")
        ds_model = PDKModel(H, W)
        ds_hist  = train_on_dataset(
            ds_model, train_imgs[:args.n_train],
            degrade, args.n_iter, args.lr)

        # ── Global kernel ─────────────────────────────────────────────
        print("  [Global] training on cal patterns...")
        g_model = GlobalKernel()
        train_on_patterns(g_model, cal_pats, degrade,
                          args.n_iter, args.lr, verbose=False)

        # ── Inference on unseen images ────────────────────────────────
        print(f"  [Inference] {len(inf_imgs)} unseen images...")
        psnr_degs,psnr_gs,psnr_cals,psnr_dss,psnr_oras = [],[],[],[],[]
        ora_cors = []; ora_params0 = None

        for i, raw in enumerate(inf_imgs):
            deg     = degrade(raw)
            g_cor   = inference_fixed(g_model,   deg)
            cal_cor = inference_fixed(cal_model,  deg)
            ds_cor  = inference_fixed(ds_model,   deg)

            print(f"    Oracle img {i+1}/{len(inf_imgs)}...")
            ora_model = PDKModel(H, W)
            ora_cor   = train_oracle(ora_model, deg, raw,
                                     args.n_ora_iter, args.lr)

            psnr_degs.append(psnr(raw,deg))
            psnr_gs.append(psnr(raw,g_cor))
            psnr_cals.append(psnr(raw,cal_cor))
            psnr_dss.append(psnr(raw,ds_cor))
            psnr_oras.append(psnr(raw,ora_cor))
            ora_cors.append(ora_cor)
            if i == 0:
                ora_params0 = ora_model.get_params()
                deg0=deg; g_cor0=g_cor; cal_cor0=cal_cor
                ds_cor0=ds_cor; ora_cor0=ora_cor

        pd_=np.mean(psnr_degs); pg_=np.mean(psnr_gs)
        pc_=np.mean(psnr_cals); pds_=np.mean(psnr_dss); po_=np.mean(psnr_oras)

        print(f"  PSNR  Deg={pd_:.1f}  Global={pg_:.1f}"
              f"  Cal={pc_:.1f}  Dataset={pds_:.1f}  Oracle={po_:.1f}")
        print(f"  Dataset vs Cal: {pds_-pc_:+.1f}dB  "
              f"Dataset vs Oracle: {pds_-po_:+.1f}dB")

        cal_psnrs.append(pc_)
        oracle_psnrs.append(po_)

        # ── N sweep ─────────────────────────────────────────────────
        print(f"  [N-sweep] {n_sweep_list}...")
        sweep = run_n_sweep(train_imgs, inf_imgs, degrade,
                            PDKModel, args.n_iter, args.lr,
                            n_sweep_list, H, W)
        all_sweep[case["name"]] = sweep

        all_results.append({
            "name":        case["name"],
            "raw":         inf_imgs[0],
            "deg":         deg0,
            "g_cor":       g_cor0,
            "cal_cor":     cal_cor0,
            "ds_cor":      ds_cor0,
            "ora_cor":     ora_cor0,
            "psnr_deg":    pd_,
            "psnr_g":      pg_,
            "psnr_cal":    pc_,
            "psnr_ds":     pds_,
            "psnr_ora":    po_,
            "true_params": case["true_params"],
            "cal_params":  cal_model.get_params(),
            "ds_params":   ds_model.get_params(),
            "ora_params":  ora_params0,
            "cal_hist":    cal_hist,
            "ds_hist":     ds_hist,
        })

    # 시각화
    print("\n[Saving results...]")
    save_main_comparison(all_results, args.res_dir)
    save_psnr_summary(all_results, args.res_dir)
    save_loss_comparison(all_results, args.res_dir)
    save_params_comparison(all_results, args.res_dir)
    save_n_sweep(all_sweep, cal_psnrs, oracle_psnrs, args.res_dir)
    save_figure_images(all_results, args.res_dir)

    print(f"\n[Done] {os.path.abspath(args.res_dir)}")
    print("  18_main_comparison.png  -- 4 case x 6 col")
    print("  18_psnr_summary.png     -- 5-method PSNR bar chart")
    print("  18_loss_comparison.png  -- Cal vs Dataset loss curves")
    print("  18_params_comparison.png-- params 비교")
    print("  18_n_sweep.png          -- N에 따른 PSNR 변화")

    print("\n[Summary]")
    print(f"{'Case':<22} {'Deg':>5} {'Cal':>6} {'Dataset':>8}"
          f" {'Oracle':>7} {'DS-Cal':>7} {'DS-Ora':>7}")
    print("-"*65)
    for r in all_results:
        print(f"{r['name']:<22} {r['psnr_deg']:>5.1f}"
              f" {r['psnr_cal']:>6.1f} {r['psnr_ds']:>8.1f}"
              f" {r['psnr_ora']:>7.1f}"
              f" {r['psnr_ds']-r['psnr_cal']:>+7.1f}"
              f" {r['psnr_ds']-r['psnr_ora']:>+7.1f}")


if __name__ == "__main__":
    main()