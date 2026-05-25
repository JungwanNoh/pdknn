"""
12_combined_bsd500.py
============================================
PSF blur + Vignetting -- 3x3 PD Kernel [BSD500]

Degradation:
  Step 1: Position-dependent PSF blur
            sigma(r) = sigma0 + alpha_psf * r^2
  Step 2: Vignetting
            V(r) = 1 / (1 + alpha_vig * r^2)

Restoration (learned 3x3 PD kernel):
  k(r) = vig_gain(r) * (delta + lambda(r) * (delta - Gaussian(sigma(r))))
  vig_gain(r) = 1 + alpha_vig_r * r^2

  Parameters (5 total):
    sigma0, alpha_psf  -- PSF blur model
    lambda0, alpha_lam -- deblur strength (independent)
    alpha_vig_r        -- vignetting gain

  Loss: L1 + 0.1 * gradient_L1

Run:
  python 12_combined_bsd500.py
  python 12_combined_bsd500.py --sigma0 0.3 --alpha-psf 1.2 --alpha-vig 2.5
  python 12_combined_bsd500.py --data-dir /path/to/BSD500
"""

import argparse, os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
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
    w, h = img.size; s = min(w, h)
    img = img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

def load_images_from_dir(img_dir, n, size):
    exts = ('*.png','*.jpg','*.jpeg','*.PNG','*.JPG','*.JPEG')
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(img_dir,'**',ext), recursive=True)
        paths += glob.glob(os.path.join(img_dir, ext))
    paths = sorted(set(paths)); imgs = []
    for p in paths:
        try: imgs.append(load_single_image(p, size))
        except Exception: continue
        if len(imgs) == n: break
    return imgs


# ============================================================================
# Degradation: PSF blur -> Vignetting
# ============================================================================

def make_gaussian_3x3(sigma_map):
    coords = torch.tensor([
        [-1.,-1.],[-1.,0.],[-1.,1.],
        [ 0.,-1.],[ 0.,0.],[ 0.,1.],
        [ 1.,-1.],[ 1.,0.],[ 1.,1.],
    ], device=sigma_map.device)
    d2 = (coords**2).sum(-1)
    s  = sigma_map.unsqueeze(-1).clamp(0.1, 1.0)
    k  = torch.exp(-d2 / (2 * s**2))
    return k / k.sum(-1, keepdim=True)

def apply_psf_blur(x, sigma0, alpha_psf):
    B, C, H, W = x.shape
    sigma = sigma0 + alpha_psf * radial_map(H, W, x.device)**2
    return spatially_varying_conv(x, make_gaussian_3x3(sigma))

def apply_vignetting(x, alpha_vig):
    B, C, H, W = x.shape
    r2 = radial_map(H, W, x.device)**2
    V  = 1.0 / (1.0 + alpha_vig * r2)
    return (x * V.unsqueeze(0).unsqueeze(0)).clamp(0, 1)

def apply_degradation(x, sigma0, alpha_psf, alpha_vig):
    """PSF blur -> Vignetting (물리적 순서)"""
    x = apply_psf_blur(x, sigma0, alpha_psf)
    x = apply_vignetting(x, alpha_vig)
    return x


# ============================================================================
# Spatially Varying Convolution
# ============================================================================

def spatially_varying_conv(x, kernels):
    B, C, H, W = x.shape
    patches = F.unfold(x, kernel_size=3, padding=1).view(B, C, 9, H*W)
    k = kernels.view(H*W, 9).T.unsqueeze(0).unsqueeze(0)
    return (patches * k).sum(dim=2).view(B, C, H, W).clamp(0, 1)


# ============================================================================
# Restoration: 3x3 PD Kernel (PSF deblur + Vignetting gain)
# ============================================================================

class CombinedRestorationPDK(nn.Module):
    """
    PSF blur + Vignetting 동시 보정.

    Restoration kernel:
      k(r) = vig_gain(r) * (delta + lambda(r) * (delta - Gaussian(sigma(r))))

      vig_gain(r)  = 1 + alpha_vig_r * r^2   <- vignetting 역보정
      sigma(r)     = sigma0 + alpha_psf * r^2 <- PSF 모델
      lambda(r)    = lambda0 + alpha_lam * r^2 <- deblur 강도 (독립)

    파라미터 5개:
      sigma0, alpha_psf  -- PSF blur 모델링
      lambda0, alpha_lam -- deblur 강도
      alpha_vig_r        -- vignetting 역보정 강도

    kernel 해석:
      vignetting: center weight를 위치별로 스케일 (1x1 효과)
      PSF deblur: 9개 weight 모두 활용 (진짜 3x3 효과)
      -> 두 효과가 하나의 3x3 kernel로 통합
    """
    def __init__(self, H, W,
                 sigma0_init=0.5,  alpha_psf_init=0.5,
                 lambda0_init=0.3, alpha_lam_init=0.5,
                 alpha_vig_init=1.0):
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._sigma0    = nn.Parameter(torch.tensor(_sp(sigma0_init)))
        self._alpha_psf = nn.Parameter(torch.tensor(_sp(alpha_psf_init)))
        self._lambda0   = nn.Parameter(torch.tensor(_sp(lambda0_init)))
        self._alpha_lam = nn.Parameter(torch.tensor(_sp(alpha_lam_init)))
        self._alpha_vig = nn.Parameter(torch.tensor(_sp(alpha_vig_init)))

        r = radial_map(H, W)
        self.register_buffer('r2', r**2)
        self.register_buffer('coords', torch.tensor([
            [-1.,-1.],[-1.,0.],[-1.,1.],
            [ 0.,-1.],[ 0.,0.],[ 0.,1.],
            [ 1.,-1.],[ 1.,0.],[ 1.,1.]]))

    @property
    def sigma0(self):    return F.softplus(self._sigma0)
    @property
    def alpha_psf(self): return F.softplus(self._alpha_psf)
    @property
    def lambda0(self):   return F.softplus(self._lambda0)
    @property
    def alpha_lam(self): return F.softplus(self._alpha_lam)
    @property
    def alpha_vig(self): return F.softplus(self._alpha_vig)

    def forward(self, x):
        # vignetting 역보정 gain
        vig_gain = 1.0 + self.alpha_vig * self.r2         # (H,W)

        # PSF Gaussian
        sigma = self.sigma0 + self.alpha_psf * self.r2    # (H,W)
        sc    = sigma.unsqueeze(-1).clamp(0.1, 1.0)       # (H,W,1)
        d2    = (self.coords**2).sum(-1)
        g     = torch.exp(-d2 / (2 * sc**2))
        g     = g / g.sum(-1, keepdim=True)

        # deblur 강도 (독립)
        lam   = (self.lambda0 + self.alpha_lam * self.r2
                 ).unsqueeze(-1).clamp(0.05, 3.0)         # (H,W,1)

        delta = torch.zeros_like(g); delta[:,:,4] = 1.0

        # PSF deblur kernel
        k_psf = delta + lam * (delta - g)                 # (H,W,9)

        # vignetting gain 곱: center weight에 gain 적용
        # -> k = vig_gain * k_psf (전체 스케일)
        k = k_psf * vig_gain.unsqueeze(-1)                # (H,W,9)

        return spatially_varying_conv(x, k)


def train_pdk(deg, clean, n_iter=500, lr=0.02,
              lambda_smooth=0.1):
    """
    lambda_smooth: alpha_lam L2 regularization 강도
      L_reg = lambda_smooth * alpha_lam^2
      -> alpha_lam이 커지면 penalty
      -> lambda(r)의 급격한 증가 억제
      -> edge 과보정 완화

    lambda_smooth 값 가이드:
      0.01 : 약한 억제 (거의 영향 없음)
      0.10 : 중간 억제 (기본값, edge 과보정 완화)
      0.50 : 강한 억제 (alpha_lam → 0에 가까워짐)
      1.00 : 매우 강한 억제 (lambda 거의 균일해짐)
    """
    B, C, H, W = deg.shape
    model = CombinedRestorationPDK(H, W)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    history, reg_history = [], []
    for i in range(n_iter):
        opt.zero_grad()
        out = model(deg)

        # reconstruction loss
        l1  = F.l1_loss(out, clean)
        def gmap(t): return t[:,:,:,1:]-t[:,:,:,:-1], t[:,:,1:,:]-t[:,:,:-1,:]
        ox,oy = gmap(out); cx,cy = gmap(clean)
        l_rec = l1 + 0.1*(F.l1_loss(ox,cx)+F.l1_loss(oy,cy))

        # alpha_lam L2 regularization
        # alpha_lam^2 로 lambda(r)의 급격한 증가 억제
        l_reg = lambda_smooth * model.alpha_lam ** 2

        loss = l_rec + l_reg
        loss.backward(); opt.step(); sched.step()
        history.append(l_rec.item())
        reg_history.append(l_reg.item())
        if (i+1) % 100 == 0:
            print(f'  iter {i+1:4d}/{n_iter}'
                  f'  rec={l_rec.item():.5f}'
                  f'  reg={l_reg.item():.5f}'
                  f'  s0={model.sigma0.item():.3f}'
                  f'  a_psf={model.alpha_psf.item():.3f}'
                  f'  lam0={model.lambda0.item():.3f}'
                  f'  a_lam={model.alpha_lam.item():.3f}'
                  f'  a_vig={model.alpha_vig.item():.3f}')
    with torch.no_grad(): cor = model(deg).clamp(0,1)
    return cor, {
        'sigma0':    model.sigma0.item(),
        'alpha_psf': model.alpha_psf.item(),
        'lambda0':   model.lambda0.item(),
        'alpha_lam': model.alpha_lam.item(),
        'alpha_vig': model.alpha_vig.item(),
    }, history, reg_history


# ============================================================================
# Visualization
# ============================================================================

def _draw_3x3(ax, k9, title):
    k = k9.reshape(3,3); vmax = max(abs(k.max()),abs(k.min()),1e-6)
    ax.imshow(k, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    for i in range(3):
        for j in range(3):
            v = k[i,j]; c = 'white' if abs(v)>vmax*0.5 else 'black'
            ax.text(j,i,f'{v:.3f}',ha='center',va='center',
                    fontsize=10,color=c,fontweight='bold')
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['-1','0','+1'],fontsize=8)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(['-1','0','+1'],fontsize=8)
    ax.set_title(title, fontsize=9, pad=5)


def save_comparison(raw_list, deg_list, cor_list,
                    true_params, learned_params, res_dir, tag):
    n = len(raw_list)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n+1))
    if n == 1: axes = axes[np.newaxis,:]
    pd_ = np.mean([psnr(r,d) for r,d in zip(raw_list,deg_list)])
    pc_ = np.mean([psnr(r,c) for r,c in zip(raw_list,cor_list)])
    cols = [
        (raw_list, '1. Raw (clean)', ''),
        (deg_list, '2. Degraded (PSF + Vignetting)',
         f'PSNR={pd_:.1f}dB | '
         f's0={true_params["sigma0"]:.2f} '
         f'a_psf={true_params["alpha_psf"]:.2f} '
         f'a_vig={true_params["alpha_vig"]:.2f}'),
        (cor_list, '3. Corrected (learned PDK)',
         f'PSNR={pc_:.1f}dB | '
         f's0={learned_params["sigma0"]:.2f} '
         f'a_psf={learned_params["alpha_psf"]:.2f} '
         f'lam0={learned_params["lambda0"]:.2f} '
         f'a_vig={learned_params["alpha_vig"]:.2f}'),
    ]
    for c,(lst,title,sub) in enumerate(cols):
        axes[0,c].set_title(title, fontsize=10, fontweight='bold', pad=8)
        for r in range(n):
            axes[r,c].imshow(to_np(lst[r]), cmap='gray', vmin=0, vmax=1)
            axes[r,c].axis('off')
        if sub:
            axes[0,c].text(0.5,-0.05,sub,transform=axes[0,c].transAxes,
                           ha='center',va='top',fontsize=8,color='dimgray')
    fig.suptitle(f'PSF + Vignetting -- 3x3 PD Kernel [{tag}]',
                 fontsize=13, y=1.01)
    plt.tight_layout(rect=[0,0.05,1,1])
    path = os.path.join(res_dir,'comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'  [Saved] {path}')


def save_kernel_map(true_params, learned_params, loss_history, reg_history, size, res_dir):
    H = W = size
    r  = radial_map(H, W).numpy(); r2 = r**2
    zones = [('center',0.00),('mid',0.50),('edge',0.85)]

    s0t=true_params['sigma0']; apt=true_params['alpha_psf']
    avt=true_params['alpha_vig']
    s0l=learned_params['sigma0']; apl=learned_params['alpha_psf']
    l0l=learned_params['lambda0']; all_=learned_params['alpha_lam']
    avl=learned_params['alpha_vig']

    def _kdeg(s0, ap, av, rv):
        """degradation kernel: PSF Gaussian * vignetting (pointwise)"""
        co=np.array([[-1,-1],[-1,0],[-1,1],
                     [ 0,-1],[ 0,0],[ 0,1],
                     [ 1,-1],[ 1,0],[ 1,1]], dtype=float)
        d2=(co**2).sum(-1); s=max(s0+ap*rv**2, 0.1)
        g=np.exp(-d2/(2*s**2)); g=g/g.sum()
        V=1.0/(1.0+av*rv**2)   # vignetting factor (scalar)
        return g * V           # PSF scaled by vignetting

    def _kres(s0, ap, l0, al, av, rv):
        """restoration kernel: vig_gain * (delta + lam*(delta-G))"""
        co=np.array([[-1,-1],[-1,0],[-1,1],
                     [ 0,-1],[ 0,0],[ 0,1],
                     [ 1,-1],[ 1,0],[ 1,1]], dtype=float)
        d2=(co**2).sum(-1); s=max(s0+ap*rv**2, 0.1)
        g=np.exp(-d2/(2*s**2)); g=g/g.sum()
        delta=np.zeros(9); delta[4]=1.0
        lam=min(max(l0+al*rv**2, 0.05), 3.0)
        k_psf = delta + lam*(delta - g)
        vig_gain = 1.0 + av*rv**2
        return k_psf * vig_gain

    # sigma map, vignetting map, combined
    sigma_t = s0t + apt*r2
    vig_t   = 1.0/(1.0+avt*r2)
    sigma_l = s0l + apl*r2
    vig_l   = 1.0/(1.0+avl*r2)

    fig = plt.figure(figsize=(20, 24))
    gs  = fig.add_gridspec(5, 4, hspace=0.55, wspace=0.4,
                           height_ratios=[1.2, 1.2, 1, 1, 1])

    # 행0: true maps
    for col,(sm,lbl) in enumerate([
        (sigma_t, f'PSF sigma (true)\ns0={s0t:.2f} a={apt:.2f}'),
        (vig_t,   f'Vignetting V(r) (true)\na_vig={avt:.2f}'),
        (sigma_t*vig_t, 'Combined (sigma*V)\ntrue'),
    ]):
        ax=fig.add_subplot(gs[0,col])
        im=ax.imshow(sm, cmap='hot'); ax.set_title(lbl,fontsize=9,pad=4); ax.axis('off')
        for _,zr in zones:
            ax.plot(W/2+zr*(W/2)*0.72, H/2-zr*(H/2)*0.72,'o',
                    markersize=7,markerfacecolor='none',
                    markeredgecolor='cyan',markeredgewidth=2)
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)

    ax_l=fig.add_subplot(gs[0,3])
    ax_l.plot(loss_history,color='steelblue',linewidth=1.5,label='rec loss')
    ax_l.plot(reg_history, color='tomato',  linewidth=1.5,label='reg loss')
    ax_l.legend(fontsize=7)
    ax_l.set_title('Training loss\n(rec + alpha_lam reg)',fontsize=9)
    ax_l.set_xlabel('Iteration',fontsize=8); ax_l.set_ylabel('Loss',fontsize=8)
    ax_l.grid(True,alpha=0.3)

    # 행1: learned maps
    for col,(sm,lbl) in enumerate([
        (sigma_l, f'PSF sigma (learned)\ns0={s0l:.2f} a={apl:.2f}'),
        (vig_l,   f'Vignetting V(r) (learned)\na_vig={avl:.2f}'),
        (np.abs(sigma_t-sigma_l)+np.abs(vig_t-vig_l), '|true-learned|\n(sigma + vig)'),
    ]):
        ax=fig.add_subplot(gs[1,col])
        im=ax.imshow(sm, cmap='hot'); ax.set_title(lbl,fontsize=9,pad=4); ax.axis('off')
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)

    # lambda profile (행1 col3)
    ax_lam=fig.add_subplot(gs[1,3])
    rv=np.linspace(0,1,100); rv2=rv**2
    ax_lam.plot(rv, np.minimum(l0l+all_*rv2, 3.0),
                color='tomato',linewidth=2,label=f'learned l0={l0l:.2f} a={all_:.2f}')
    ax_lam.set_xlabel('r',fontsize=8); ax_lam.set_ylabel('lambda(r)',fontsize=8)
    ax_lam.set_title('Deblur strength lambda(r)',fontsize=9)
    ax_lam.legend(fontsize=8); ax_lam.grid(True,alpha=0.3)

    # 행2~4: zone별 3x3 kernel
    for row,(zn,zr) in enumerate(zones):
        ax=fig.add_subplot(gs[row+2,0])
        _draw_3x3(ax,_kdeg(s0t,apt,avt,zr),
                  f'Degrade kernel\n(PSF+Vig) r={zr:.2f}')
        ax.set_ylabel(f'zone: {zn}',fontsize=10,labelpad=6,
                      color='dimgray',fontweight='bold')

        ax=fig.add_subplot(gs[row+2,1])
        _draw_3x3(ax,_kres(s0t,apt,s0t*1.5,0.0,avt,zr),
                  f'Restore (true params)\nr={zr:.2f}')

        ax=fig.add_subplot(gs[row+2,2])
        _draw_3x3(ax,_kres(s0l,apl,l0l,all_,avl,zr),
                  f'Restore (learned)\nr={zr:.2f}')

        ax=fig.add_subplot(gs[row+2,3])
        kd=_kdeg(s0t,apt,avt,zr)
        kr=_kres(s0l,apl,l0l,all_,avl,zr)
        conv=np.convolve(kd,kr[::-1]); mid=len(conv)//2
        ax.bar(range(len(conv)),conv,color='steelblue',alpha=0.7)
        ax.axvline(mid,color='red',linewidth=1.5,linestyle='--',label='ideal')
        ax.set_title(f'Degrade*Restore r={zr:.2f}\n(ideal: spike)',fontsize=9)
        ax.set_xlabel('coeff index',fontsize=8); ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

    fig.suptitle('PSF+Vignetting -- 3x3 PD Kernel map',fontsize=14,y=1.01)
    path=os.path.join(res_dir,'kernel_map.png')
    fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f'  [Saved] {path}')


def save_profile(true_params, learned_params, res_dir):
    r=np.linspace(0,1,200); r2=r**2
    s0t=true_params['sigma0']; apt=true_params['alpha_psf']; avt=true_params['alpha_vig']
    s0l=learned_params['sigma0']; apl=learned_params['alpha_psf']
    l0l=learned_params['lambda0']; all_=learned_params['alpha_lam']
    avl=learned_params['alpha_vig']

    fig,axes=plt.subplots(1,3,figsize=(14,4))

    axes[0].plot(r,s0t+apt*r2,label=f'true  s0={s0t:.3f} a={apt:.3f}',
                 color='black',linewidth=2)
    axes[0].plot(r,s0l+apl*r2,label=f'learned s0={s0l:.3f} a={apl:.3f}',
                 color='tomato',linewidth=1.5,linestyle='--')
    axes[0].set_xlabel('r'); axes[0].set_ylabel('sigma(r)')
    axes[0].set_title('PSF sigma profile'); axes[0].legend(); axes[0].grid(True,alpha=0.3)

    axes[1].plot(r,1/(1+avt*r2),label=f'true  a_vig={avt:.3f}',
                 color='black',linewidth=2)
    axes[1].plot(r,1/(1+avl*r2),label=f'learned a_vig={avl:.3f}',
                 color='steelblue',linewidth=1.5,linestyle='--')
    axes[1].set_xlabel('r'); axes[1].set_ylabel('V(r)')
    axes[1].set_title('Vignetting profile'); axes[1].legend(); axes[1].grid(True,alpha=0.3)

    axes[2].plot(r,np.minimum(l0l+all_*r2,3.0),
                 label=f'learned l0={l0l:.3f} a={all_:.3f}',
                 color='tomato',linewidth=2)
    axes[2].set_xlabel('r'); axes[2].set_ylabel('lambda(r)')
    axes[2].set_title('Deblur strength lambda(r)')
    axes[2].legend(); axes[2].grid(True,alpha=0.3)

    plt.tight_layout()
    path=os.path.join(res_dir,'profile.png')
    fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f'  [Saved] {path}')


# ============================================================================
# Dataset loader & args
# ============================================================================

def load_dataset(args):
    if args.data_dir and os.path.isdir(args.data_dir):
        imgs=load_images_from_dir(args.data_dir,args.n_images,args.img_size)
        if imgs: print(f'[Load] BSD500: {len(imgs)} images'); return imgs
    try:
        from skimage import data as skd; import skimage.color as skc
        builtins=[skd.camera(),skd.astronaut(),skd.chelsea(),
                  skd.coffee(),skd.horse(),skd.hubble_deep_field()]
        imgs=[]
        for arr in builtins[:args.n_images]:
            if arr.ndim==3: arr=skc.rgb2gray(arr)
            arr=(arr-arr.min())/(arr.max()-arr.min()+1e-8)
            pil=Image.fromarray((arr*255).astype(np.uint8))
            s=min(pil.size); w,h=pil.size
            pil=pil.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
            pil=pil.resize((args.img_size,args.img_size),Image.BILINEAR)
            imgs.append(torch.from_numpy(
                np.array(pil,dtype=np.float32)/255.).unsqueeze(0).unsqueeze(0))
        print(f'[Load] BSD500 fallback (skimage): {len(imgs)} images')
        return imgs
    except Exception as e:
        print(f'[Warning] {e} -- fallback'); return _synthetic(args.n_images,args.img_size)

def _synthetic(n,size):
    H=W=size; ys=torch.linspace(-1,1,H); xs=torch.linspace(-1,1,W)
    gy,gx=torch.meshgrid(ys,xs,indexing='ij')
    pts=[(0.5+0.25*torch.sin(gy*20)+0.25*torch.sin(gx*20)).clamp(0,1),
         (0.5+0.5*torch.cos((gy**2+gx**2).sqrt()*15)).clamp(0,1),
         ((gy+gx+1)/2).clamp(0,1),
         (0.5+0.5*torch.sign(torch.sin(gy*12)*torch.sin(gx*12))).clamp(0,1)]
    return [p.unsqueeze(0).unsqueeze(0) for p in pts[:n]]

def parse_args():
    p=argparse.ArgumentParser(description='PSF+Vignetting PDK -- BSD500')
    p.add_argument('--data-dir',  type=str,   default='')
    p.add_argument('--n-images',  type=int,   default=4)
    p.add_argument('--img-size',  type=int,   default=256)
    p.add_argument('--sigma0',    type=float, default=0.30)
    p.add_argument('--alpha-psf', type=float, default=1.20)
    p.add_argument('--alpha-vig', type=float, default=2.50)
    p.add_argument('--lambda-smooth', type=float, default=0.10,
                   help='alpha_lam L2 regularization (default: 0.10)')
    p.add_argument('--n-iter',    type=int,   default=500)
    p.add_argument('--lr',        type=float, default=0.02)
    p.add_argument('--res-dir',   type=str,   default='./res/combined_bsd500')
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    os.makedirs(args.res_dir, exist_ok=True)
    imgs = load_dataset(args)
    if not imgs: return
    true_params = {
        'sigma0': args.sigma0, 'alpha_psf': args.alpha_psf,
        'alpha_vig': args.alpha_vig,
    }
    H = W = imgs[0].shape[-1]
    print(f'[Config] size={H}  sigma0={args.sigma0}'
          f'  alpha_psf={args.alpha_psf}  alpha_vig={args.alpha_vig}')

    raw_list,deg_list,cor_list,all_lp,all_lh,all_lh_reg = [],[],[],[],[],[]
    for i, raw in enumerate(imgs):
        deg = apply_degradation(raw, args.sigma0, args.alpha_psf, args.alpha_vig)
        print(f'[Image {i+1}/{len(imgs)}]  PSNR_deg={psnr(raw,deg):.2f}dB  Learning...')
        cor,lp,lh,lh_reg = train_pdk(deg, raw,
                                      n_iter=args.n_iter, lr=args.lr,
                                      lambda_smooth=args.lambda_smooth)
        print(f'  PSNR_cor={psnr(raw,cor):.2f}dB')
        raw_list.append(raw); deg_list.append(deg); cor_list.append(cor)
        all_lp.append(lp); all_lh.append(lh); all_lh_reg.append(lh_reg)

    learned_params = all_lp[0]   # 시각화는 첫 번째 이미지 기준
    loss_history   = all_lh[0]

    print(f'\n[True]    sigma0={true_params["sigma0"]:.3f}'
          f'  alpha_psf={true_params["alpha_psf"]:.3f}'
          f'  alpha_vig={true_params["alpha_vig"]:.3f}')
    print(f'[Learned] sigma0={learned_params["sigma0"]:.3f}'
          f'  alpha_psf={learned_params["alpha_psf"]:.3f}'
          f'  lambda0={learned_params["lambda0"]:.3f}'
          f'  alpha_lam={learned_params["alpha_lam"]:.3f}'
          f'  alpha_vig={learned_params["alpha_vig"]:.3f}')
    print('[All imgs] ' + '  '.join(
        [f'img{i+1}: s0={lp["sigma0"]:.3f} a_psf={lp["alpha_psf"]:.3f}'
         f' a_vig={lp["alpha_vig"]:.3f}'
         for i,lp in enumerate(all_lp)]))

    tag = args.res_dir.rstrip('/').split('_')[-1].upper()
    save_comparison(raw_list,deg_list,cor_list,
                    true_params,learned_params,args.res_dir,tag)
    save_kernel_map(true_params,learned_params,loss_history,all_lh_reg[0],H,args.res_dir)
    save_profile(true_params,learned_params,args.res_dir)
    print(f'\n[Done] {os.path.abspath(args.res_dir)}')
    print('  comparison.png  -- raw / degraded / corrected')
    print('  kernel_map.png  -- sigma/vig maps + zone kernels')
    print('  profile.png     -- PSF sigma, vignetting, lambda profiles')


if __name__ == '__main__':
    main()
