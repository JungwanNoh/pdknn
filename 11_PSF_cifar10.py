"""
11_PSF_cifar10.py
============================================
Position-dependent PSF blur only -- 3x3 PD Kernel [CIFAR-10]

Degradation:
  PSF blur only (NO vignetting)
  sigma(r) = sigma0 + alpha_psf * r^2

Restoration (learned):
  k(r) = delta + lambda(r) * (delta - Gaussian(sigma(r)))
  lambda(r) = min(sigma(r)*1.5, 2.0)
  Loss: L1 + 0.1 * gradient_L1

Run:
  python 11_PSF_cifar10.py
  python 11_PSF_cifar10.py --sigma0 0.3 --alpha-psf 1.2 --n-iter 500
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
# Degradation: Position-dependent PSF blur
# ============================================================================

def make_gaussian_3x3(sigma_map):
    """sigma_map: (H,W) -> (H,W,9) normalized Gaussian kernels."""
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
    """sigma(r) = sigma0 + alpha_psf * r^2"""
    B, C, H, W = x.shape
    sigma = sigma0 + alpha_psf * radial_map(H, W, x.device)**2
    return spatially_varying_conv(x, make_gaussian_3x3(sigma))


# ============================================================================
# Spatially Varying Convolution
# ============================================================================

def spatially_varying_conv(x, kernels):
    B, C, H, W = x.shape
    patches = F.unfold(x, kernel_size=3, padding=1).view(B, C, 9, H*W)
    k = kernels.view(H*W, 9).T.unsqueeze(0).unsqueeze(0)
    return (patches * k).sum(dim=2).view(B, C, H, W).clamp(0, 1)


# ============================================================================
# Restoration: 3x3 PD Kernel
# ============================================================================

class PSFRestorationPDK(nn.Module):
    """
    학습 파라미터: sigma0, alpha_psf (2개)
    k(r) = delta + lambda(r)*(delta - Gaussian(sigma(r)))
    lambda(r) = min(sigma(r)*1.5, 2.0)
    """
    def __init__(self, H, W, sigma0_init=0.5, alpha_psf_init=0.5):
        super().__init__()
        self._sigma0    = nn.Parameter(torch.tensor(
            float(np.log(np.exp(sigma0_init)-1.0))))
        self._alpha_psf = nn.Parameter(torch.tensor(
            float(np.log(np.exp(alpha_psf_init)-1.0))))
        r = radial_map(H, W)
        self.register_buffer('r2', r**2)
        self.register_buffer('coords', torch.tensor([
            [-1.,-1.],[-1.,0.],[-1.,1.],
            [ 0.,-1.],[ 0.,0.],[ 0.,1.],
            [ 1.,-1.],[ 1.,0.],[ 1.,1.],]))

    @property
    def sigma0(self):    return F.softplus(self._sigma0)
    @property
    def alpha_psf(self): return F.softplus(self._alpha_psf)

    def forward(self, x):
        sigma   = self.sigma0 + self.alpha_psf * self.r2
        sc      = sigma.unsqueeze(-1).clamp(0.1, 1.0)
        d2      = (self.coords**2).sum(-1)
        g       = torch.exp(-d2/(2*sc**2))
        g       = g / g.sum(-1, keepdim=True)
        delta   = torch.zeros_like(g); delta[:,:,4] = 1.0
        lam     = (sc * 1.5).clamp(0.1, 2.0)
        return spatially_varying_conv(x, delta + lam*(delta - g))


def train_pdk(deg, clean, n_iter=500, lr=0.02):
    B, C, H, W = deg.shape
    model = PSFRestorationPDK(H, W)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    history = []
    for i in range(n_iter):
        opt.zero_grad()
        out = model(deg)
        l1  = F.l1_loss(out, clean)
        def gmap(t): return t[:,:,:,1:]-t[:,:,:,:-1], t[:,:,1:,:]-t[:,:,:-1,:]
        ox,oy = gmap(out); cx,cy = gmap(clean)
        loss = l1 + 0.1*(F.l1_loss(ox,cx)+F.l1_loss(oy,cy))
        loss.backward(); opt.step(); sched.step()
        history.append(loss.item())
        if (i+1) % 100 == 0:
            print(f'  iter {i+1:4d}/{n_iter}  loss={loss.item():.5f}'
                  f'  s0={model.sigma0.item():.3f}'
                  f'  a={model.alpha_psf.item():.3f}')
    with torch.no_grad(): cor = model(deg).clamp(0,1)
    return cor, {'sigma0': model.sigma0.item(),
                  'alpha_psf': model.alpha_psf.item()}, history


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
    pd = np.mean([psnr(r,d) for r,d in zip(raw_list,deg_list)])
    pc = np.mean([psnr(r,c) for r,c in zip(raw_list,cor_list)])
    cols = [
        (raw_list, '1. Raw (clean)', ''),
        (deg_list, '2. Degraded (PSF blur)',
         f'PSNR={pd:.1f}dB  s0={true_params["sigma0"]:.2f}'
         f'  a={true_params["alpha_psf"]:.2f}'),
        (cor_list, '3. Corrected (learned PDK)',
         f'PSNR={pc:.1f}dB  s0={learned_params["sigma0"]:.2f}'
         f'  a={learned_params["alpha_psf"]:.2f}'),
    ]
    for c,(lst,title,sub) in enumerate(cols):
        axes[0,c].set_title(title, fontsize=11, fontweight='bold', pad=8)
        for r in range(n):
            axes[r,c].imshow(to_np(lst[r]), cmap='gray', vmin=0, vmax=1)
            axes[r,c].axis('off')
        if sub:
            axes[0,c].text(0.5,-0.04,sub,transform=axes[0,c].transAxes,
                           ha='center',va='top',fontsize=8.5,color='dimgray')
    fig.suptitle(f'PSF blur only -- 3x3 PD Kernel [{tag}]', fontsize=13, y=1.01)
    plt.tight_layout(rect=[0,0.04,1,1])
    path = os.path.join(res_dir,'comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'  [Saved] {path}')


def save_kernel_map(true_params, learned_params, loss_history, size, res_dir):
    H = W = size; r = radial_map(H,W).numpy(); r2 = r**2
    zones = [('center',0.00),('mid',0.50),('edge',0.85)]
    s0t=true_params['sigma0']; apt=true_params['alpha_psf']
    s0l=learned_params['sigma0']; apl=learned_params['alpha_psf']

    def _kdeg(s0,ap,rv):
        co=np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]],dtype=float)
        d2=(co**2).sum(-1); s=max(s0+ap*rv**2,0.1)
        k=np.exp(-d2/(2*s**2)); return k/k.sum()
    def _kres(s0,ap,rv):
        co=np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]],dtype=float)
        d2=(co**2).sum(-1); s=max(s0+ap*rv**2,0.1)
        g=np.exp(-d2/(2*s**2)); g=g/g.sum()
        delta=np.zeros(9); delta[4]=1.0; lam=min(s*1.5,2.0)
        return delta+lam*(delta-g)

    fig = plt.figure(figsize=(18,22))
    gs  = fig.add_gridspec(4,4,hspace=0.6,wspace=0.4,height_ratios=[1.4,1,1,1])

    sm_t=s0t+apt*r2; sm_l=s0l+apl*r2; vmax_s=sm_t.max()
    for col,(sm,lbl) in enumerate([
        (sm_t,  f'Sigma map (true)\ns0={s0t:.2f}  a={apt:.2f}'),
        (sm_l,  f'Sigma map (learned)\ns0={s0l:.2f}  a={apl:.2f}'),
        (np.abs(sm_t-sm_l), '|true - learned| sigma'),
    ]):
        ax=fig.add_subplot(gs[0,col])
        im=ax.imshow(sm,cmap='hot',vmin=0,vmax=vmax_s)
        ax.set_title(lbl,fontsize=10,pad=5); ax.axis('off')
        for _,zr in zones:
            ax.plot(W/2+zr*(W/2)*0.72, H/2-zr*(H/2)*0.72,'o',
                    markersize=8,markerfacecolor='none',
                    markeredgecolor='cyan',markeredgewidth=2.0)
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)

    al=fig.add_subplot(gs[0,3])
    al.plot(loss_history,color='steelblue',linewidth=1.5)
    al.set_title('Training loss\n(L1+0.1*grad)',fontsize=10)
    al.set_xlabel('Iteration',fontsize=9); al.set_ylabel('Loss',fontsize=9)
    al.grid(True,alpha=0.3)

    for row,(zn,zr) in enumerate(zones):
        ax=fig.add_subplot(gs[row+1,0])
        _draw_3x3(ax,_kdeg(s0t,apt,zr),
                  f'PSF (degrade)\ntrue r={zr:.2f}  s={s0t+apt*zr**2:.2f}')
        ax.set_ylabel(f'zone: {zn}',fontsize=11,labelpad=6,
                      color='dimgray',fontweight='bold')
        ax=fig.add_subplot(gs[row+1,1])
        _draw_3x3(ax,_kres(s0t,apt,zr),f'Restore (true params)\nr={zr:.2f}')
        ax=fig.add_subplot(gs[row+1,2])
        _draw_3x3(ax,_kres(s0l,apl,zr),f'Restore (learned)\nr={zr:.2f}')
        ax=fig.add_subplot(gs[row+1,3])
        kd=_kdeg(s0t,apt,zr); kr=_kres(s0l,apl,zr)
        conv=np.convolve(kd,kr[::-1]); mid=len(conv)//2
        ax.bar(range(len(conv)),conv,color='steelblue',alpha=0.7)
        ax.axvline(mid,color='red',linewidth=1.5,linestyle='--',label='ideal')
        ax.set_title(f'PSF*Restore r={zr:.2f}\n(ideal: spike at center)',fontsize=9)
        ax.set_xlabel('coeff index',fontsize=8); ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

    fig.suptitle('PSF blur only -- 3x3 PD Kernel map',fontsize=14,y=1.01)
    path=os.path.join(res_dir,'kernel_map.png')
    fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f'  [Saved] {path}')


def save_sigma_profile(true_params, learned_params, res_dir):
    r=np.linspace(0,1,200); r2=r**2
    s0t=true_params['sigma0']; apt=true_params['alpha_psf']
    s0l=learned_params['sigma0']; apl=learned_params['alpha_psf']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(r,s0t+apt*r2,label=f'true s0={s0t:.3f} a={apt:.3f}',
                 color='black',linewidth=2)
    axes[0].plot(r,s0l+apl*r2,label=f'learned s0={s0l:.3f} a={apl:.3f}',
                 color='tomato',linewidth=1.5,linestyle='--')
    axes[0].set_xlabel('r'); axes[0].set_ylabel('sigma(r)')
    axes[0].set_title('PSF sigma profile'); axes[0].legend(); axes[0].grid(True,alpha=0.3)
    axes[1].plot(r,np.minimum((s0t+apt*r2)*1.5,2.0),
                 label='lambda (true)',color='black',linewidth=2)
    axes[1].plot(r,np.minimum((s0l+apl*r2)*1.5,2.0),
                 label='lambda (learned)',color='tomato',linewidth=1.5,linestyle='--')
    axes[1].set_xlabel('r'); axes[1].set_ylabel('lambda')
    axes[1].set_title('Deblur strength'); axes[1].legend(); axes[1].grid(True,alpha=0.3)
    plt.tight_layout()
    path=os.path.join(res_dir,'sigma_profile.png')
    fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f'  [Saved] {path}')


# ============================================================================
# Dataset loader & args
# ============================================================================

def load_dataset(args):
    try:
        import torchvision, torchvision.transforms as T
        tf = T.Compose([T.Grayscale(), T.Resize(args.img_size), T.ToTensor()])
        ds = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=tf)
        imgs,idx=[],list(range(0,len(ds),max(1,len(ds)//args.n_images)))
        for i in idx[:args.n_images]:
            img,_=ds[i]; imgs.append(img.unsqueeze(0))
        print(f'[Load] CIFAR-10: {len(imgs)} images ({args.img_size}x{args.img_size})')
        return imgs
    except Exception as e:
        print(f'[Warning] {e} -- fallback'); return _synthetic(args.n_images, args.img_size)

def _synthetic(n, size):
    H=W=size; ys=torch.linspace(-1,1,H); xs=torch.linspace(-1,1,W)
    gy,gx=torch.meshgrid(ys,xs,indexing='ij')
    pts=[(0.5+0.5*torch.sign(torch.sin(gy*8)*torch.sin(gx*8))).clamp(0,1),
         (0.5+0.25*torch.sin(gy*12)+0.25*torch.sin(gx*12)).clamp(0,1),
         ((gx+1)/2).clamp(0,1),
         (0.5+0.5*torch.cos((gy**2+gx**2).sqrt()*10)).clamp(0,1)]
    return [p.unsqueeze(0).unsqueeze(0) for p in pts[:n]]

def parse_args():
    p=argparse.ArgumentParser(description='PSF only PDK -- CIFAR-10')
    p.add_argument('--data-dir',  type=str,   default='./data')
    p.add_argument('--n-images',  type=int,   default=4)
    p.add_argument('--img-size',  type=int,   default=128)
    p.add_argument('--sigma0',    type=float, default=0.30)
    p.add_argument('--alpha-psf', type=float, default=1.20)
    p.add_argument('--n-iter',    type=int,   default=500)
    p.add_argument('--lr',        type=float, default=0.02)
    p.add_argument('--res-dir',   type=str,   default='./res/psf_only_cifar10')
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    os.makedirs(args.res_dir, exist_ok=True)
    imgs = load_dataset(args)
    if not imgs: return
    true_params = {'sigma0': args.sigma0, 'alpha_psf': args.alpha_psf}
    H = W = imgs[0].shape[-1]
    print(f'[Config] size={H}  sigma0={args.sigma0}  alpha_psf={args.alpha_psf}')
    raw_list,deg_list,cor_list,all_lp,all_lh = [],[],[],[],[]
    for i, raw in enumerate(imgs):
        deg = apply_psf_blur(raw, args.sigma0, args.alpha_psf)
        print(f'[Image {i+1}/{len(imgs)}]  PSNR_deg={psnr(raw,deg):.2f}dB  Learning...')
        cor,lp,lh = train_pdk(deg, raw, n_iter=args.n_iter, lr=args.lr)
        print(f'  PSNR_cor={psnr(raw,cor):.2f}dB')
        raw_list.append(raw); deg_list.append(deg); cor_list.append(cor)
        all_lp.append(lp); all_lh.append(lh)
    # visualization uses first image params (not mean)
    learned_params = all_lp[0]
    loss_history   = all_lh[0]
    print(f'\n[True]    sigma0={true_params["sigma0"]:.3f}  alpha_psf={true_params["alpha_psf"]:.3f}')
    print(f'[Learned img1] sigma0={learned_params["sigma0"]:.3f}  alpha_psf={learned_params["alpha_psf"]:.3f}')
    print('[All imgs] ' + '  '.join(
        [f'img{i+1}: s0={lp["sigma0"]:.3f} a={lp["alpha_psf"]:.3f}'
         for i, lp in enumerate(all_lp)]))
    tag = args.res_dir.rstrip('/').split('_')[-1].upper()
    save_comparison(raw_list,deg_list,cor_list,
                    true_params,learned_params,args.res_dir,tag)
    save_kernel_map(true_params,learned_params,loss_history,H,args.res_dir)
    save_sigma_profile(true_params,learned_params,args.res_dir)
    print(f'\n[Done] {os.path.abspath(args.res_dir)}')
    print('  comparison.png    -- raw / degraded / corrected')
    print('  kernel_map.png    -- sigma map + zone kernels')
    print('  sigma_profile.png -- PSF sigma & deblur strength')


if __name__ == '__main__':
    main()
