"""
13_coma_aberration_div2k.py
============================================
Coma Aberration -- 3x3 PD Kernel [DIV2K]
Run: python 13_coma_aberration_div2k.py
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

def radial_map(H, W, device='cpu'):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    r = (gy**2 + gx**2).sqrt() / (2**0.5)
    return r, gy, gx

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
    return torch.from_numpy(
        np.array(img, dtype=np.float32)/255.).unsqueeze(0).unsqueeze(0)

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

def spatially_varying_conv(x, kernels):
    B, C, H, W = x.shape
    patches = F.unfold(x, kernel_size=3, padding=1).view(B, C, 9, H*W)
    k = kernels.view(H*W, 9).T.unsqueeze(0).unsqueeze(0)
    return (patches * k).sum(dim=2).view(B, C, H, W).clamp(0, 1)

def _draw_3x3(ax, k9, title):
    k = k9.reshape(3,3); vmax = max(abs(k.max()),abs(k.min()),1e-6)
    ax.imshow(k, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    for i in range(3):
        for j in range(3):
            v=k[i,j]; c='white' if abs(v)>vmax*0.5 else 'black'
            ax.text(j,i,f'{v:.3f}',ha='center',va='center',
                    fontsize=10,color=c,fontweight='bold')
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['-1','0','+1'],fontsize=8)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(['-1','0','+1'],fontsize=8)
    ax.set_title(title, fontsize=9, pad=5)

# ============================================================================
# Coma Aberration
# PSF(r,theta) = (1-w)*G(sigma) + w*G(sigma, shift)
#   w = r*0.8,  shift = coma_k*r^2*(sin(theta), cos(theta))
# ============================================================================

def _coma_kernel_map(sigma_map, shift_y, shift_x):
    coords = torch.tensor([
        [-1.,-1.],[-1.,0.],[-1.,1.],
        [ 0.,-1.],[ 0.,0.],[ 0.,1.],
        [ 1.,-1.],[ 1.,0.],[ 1.,1.],
    ], device=sigma_map.device)
    d2 = (coords**2).sum(-1)
    s  = sigma_map.unsqueeze(-1).clamp(0.1, 1.0)
    g1 = torch.exp(-d2/(2*s**2)); g1=g1/g1.sum(-1,keepdim=True)
    sy = shift_y.unsqueeze(-1); sx = shift_x.unsqueeze(-1)
    dy = coords[:,0].unsqueeze(0).unsqueeze(0) - sy
    dx = coords[:,1].unsqueeze(0).unsqueeze(0) - sx
    g2 = torch.exp(-(dy**2+dx**2)/(2*s**2)); g2=g2/g2.sum(-1,keepdim=True)
    r,_,_ = radial_map(sigma_map.shape[0], sigma_map.shape[1], sigma_map.device)
    w = (r*0.8).clamp(0,0.8).unsqueeze(-1)
    k = (1-w)*g1 + w*g2
    return k/k.sum(-1,keepdim=True)

def apply_coma_blur(x, sigma0, alpha_psf, coma_k):
    B,C,H,W=x.shape
    r,gy,gx=radial_map(H,W,x.device)
    r2=r**2; rs=r.clamp(1e-6)
    sigma=sigma0+alpha_psf*r2
    shift_y=coma_k*r2*(gy/rs); shift_x=coma_k*r2*(gx/rs)
    return spatially_varying_conv(x, _coma_kernel_map(sigma,shift_y,shift_x))

def apply_degradation(x, sigma0, alpha_psf, coma_k, alpha_vig):
    B,C,H,W=x.shape
    r,_,_=radial_map(H,W,x.device)
    x=apply_coma_blur(x,sigma0,alpha_psf,coma_k)
    V=1.0/(1.0+alpha_vig*r**2)
    return (x*V.unsqueeze(0).unsqueeze(0)).clamp(0,1)

class ComaRestorationPDK(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._sigma0    = nn.Parameter(torch.tensor(_sp(0.5)))
        self._alpha_psf = nn.Parameter(torch.tensor(_sp(0.5)))
        self._coma_k    = nn.Parameter(torch.tensor(_sp(0.1)))
        self._lambda0   = nn.Parameter(torch.tensor(_sp(0.5)))
        self._alpha_lam = nn.Parameter(torch.tensor(_sp(0.3)))
        self._alpha_vig = nn.Parameter(torch.tensor(_sp(1.0)))
        r,gy,gx=radial_map(H,W)
        self.register_buffer('r',r); self.register_buffer('r2',r**2)
        self.register_buffer('gy',gy); self.register_buffer('gx',gx)
        self.register_buffer('coords',torch.tensor([
            [-1.,-1.],[-1.,0.],[-1.,1.],
            [ 0.,-1.],[ 0.,0.],[ 0.,1.],
            [ 1.,-1.],[ 1.,0.],[ 1.,1.]]))

    @property
    def sigma0(self):    return F.softplus(self._sigma0)
    @property
    def alpha_psf(self): return F.softplus(self._alpha_psf)
    @property
    def coma_k(self):    return F.softplus(self._coma_k)
    @property
    def lambda0(self):   return F.softplus(self._lambda0)
    @property
    def alpha_lam(self): return F.softplus(self._alpha_lam)
    @property
    def alpha_vig(self): return F.softplus(self._alpha_vig)

    def forward(self, x):
        vig_gain=1.0+self.alpha_vig*self.r2
        sigma=self.sigma0+self.alpha_psf*self.r2
        sc=sigma.unsqueeze(-1).clamp(0.1,1.0)
        rs=self.r.clamp(1e-6)
        sy=(-self.coma_k*self.r2*self.gy/rs).unsqueeze(-1)
        sx=(-self.coma_k*self.r2*self.gx/rs).unsqueeze(-1)
        d2=(self.coords**2).sum(-1)
        g1=torch.exp(-d2/(2*sc**2)); g1=g1/g1.sum(-1,keepdim=True)
        dy=self.coords[:,0].unsqueeze(0).unsqueeze(0)-sy
        dx=self.coords[:,1].unsqueeze(0).unsqueeze(0)-sx
        g2=torch.exp(-(dy**2+dx**2)/(2*sc**2)); g2=g2/g2.sum(-1,keepdim=True)
        w=(self.r*0.8).clamp(0,0.8).unsqueeze(-1)
        g=(1-w)*g1+w*g2; g=g/g.sum(-1,keepdim=True)
        delta=torch.zeros_like(g); delta[:,:,4]=1.0
        lam=(self.lambda0+self.alpha_lam*self.r2).unsqueeze(-1).clamp(0.05,3.0)
        k=(delta+lam*(delta-g))*vig_gain.unsqueeze(-1)
        return spatially_varying_conv(x,k)

def train_pdk(deg, clean, n_iter=500, lr=0.02, lambda_smooth=0.05):
    model=ComaRestorationPDK(*deg.shape[2:])
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n_iter)
    history=[]
    for i in range(n_iter):
        opt.zero_grad()
        out=model(deg)
        l1=F.l1_loss(out,clean)
        def gmap(t): return t[:,:,:,1:]-t[:,:,:,:-1], t[:,:,1:,:]-t[:,:,:-1,:]
        ox,oy=gmap(out); cx,cy=gmap(clean)
        l_rec=l1+0.1*(F.l1_loss(ox,cx)+F.l1_loss(oy,cy))
        l_reg=lambda_smooth*model.alpha_lam**2
        (l_rec+l_reg).backward(); opt.step(); sched.step()
        history.append(l_rec.item())
        if (i+1)%100==0:
            print(f'  iter {i+1:4d}/{n_iter}  rec={l_rec.item():.5f}'
                  f'  coma_k={model.coma_k.item():.3f}'
                  f'  a_vig={model.alpha_vig.item():.3f}')
    with torch.no_grad(): cor=model(deg).clamp(0,1)
    return cor, {
        'sigma0':model.sigma0.item(),'alpha_psf':model.alpha_psf.item(),
        'coma_k':model.coma_k.item(),'lambda0':model.lambda0.item(),
        'alpha_lam':model.alpha_lam.item(),'alpha_vig':model.alpha_vig.item()
    }, history

def save_comparison(raw_list,deg_list,cor_list,true_params,learned_params,res_dir,dsname):
    n=len(raw_list)
    fig,axes=plt.subplots(n,3,figsize=(12,4*n+1))
    if n==1: axes=axes[np.newaxis,:]
    pd_=np.mean([psnr(r,d) for r,d in zip(raw_list,deg_list)])
    pc_=np.mean([psnr(r,c) for r,c in zip(raw_list,cor_list)])
    cols=[
        (raw_list,'1. Raw (clean)',''),
        (deg_list,'2. Degraded (Coma+Vignetting)',
         f'PSNR={pd_:.1f}dB | coma_k={true_params["coma_k"]:.2f}'
         f' s0={true_params["sigma0"]:.2f} a_vig={true_params["alpha_vig"]:.2f}'),
        (cor_list,'3. Corrected (learned PDK)',
         f'PSNR={pc_:.1f}dB | coma_k={learned_params["coma_k"]:.2f}'
         f' s0={learned_params["sigma0"]:.2f} a_vig={learned_params["alpha_vig"]:.2f}'),
    ]
    for c,(lst,title,sub) in enumerate(cols):
        axes[0,c].set_title(title,fontsize=10,fontweight='bold',pad=8)
        for r in range(n):
            axes[r,c].imshow(to_np(lst[r]),cmap='gray',vmin=0,vmax=1); axes[r,c].axis('off')
        if sub:
            axes[0,c].text(0.5,-0.06,sub,transform=axes[0,c].transAxes,
                           ha='center',va='top',fontsize=8,color='dimgray')
    fig.suptitle(f'Coma Aberration -- 3x3 PD Kernel [{dsname}]',fontsize=13,y=1.01)
    plt.tight_layout(rect=[0,0.05,1,1])
    path=os.path.join(res_dir,'comparison.png')
    fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig); print(f'  [Saved] {path}')

def save_degradation_map(true_params,learned_params,loss_history,size,res_dir):
    H=W=size; r,gy,gx=radial_map(H,W)
    r=r.numpy(); r2=r**2; gy=gy.numpy(); gx=gx.numpy()
    s0t=true_params['sigma0']; apt=true_params['alpha_psf']
    ckt=true_params['coma_k']; avt=true_params['alpha_vig']
    s0l=learned_params['sigma0']; apl=learned_params['alpha_psf']
    ckl=learned_params['coma_k']
    fig,axes=plt.subplots(2,4,figsize=(18,9))
    for ax,(data,lbl) in zip(axes[0],[
        (s0t+apt*r2,f'PSF sigma (true)\ns0={s0t:.2f} a={apt:.2f}'),
        (ckt*r2,f'Coma shift mag (true)\ncoma_k={ckt:.3f}'),
        (1/(1+avt*r2),f'Vignetting (true)\na_vig={avt:.2f}'),
        (np.abs((s0t+apt*r2)-(s0l+apl*r2)),'|sigma: true-learned|'),
    ]):
        im=ax.imshow(data,cmap='hot'); ax.set_title(lbl,fontsize=9,pad=4); ax.axis('off')
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    zones=[('center',0.0,0.0),('right',0.0,0.6),('top-right',0.42,0.42)]
    for ax,(zn,ry,rx) in zip(axes[1,:3],zones):
        rv=np.sqrt(ry**2+rx**2)/(2**0.5); rs=max(rv,1e-6)
        sy_v=ckt*rv**2*(ry/rs); sx_v=ckt*rv**2*(rx/rs)
        co=np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]],dtype=float)
        d2=(co**2).sum(-1); s=max(s0t+apt*rv**2,0.1)
        g1=np.exp(-d2/(2*s**2)); g1=g1/g1.sum()
        dy_=co[:,0]-sy_v; dx_=co[:,1]-sx_v
        g2=np.exp(-(dy_**2+dx_**2)/(2*s**2)); g2=g2/g2.sum()
        w=min(rv*0.8,0.8); k=(1-w)*g1+w*g2; k=k/k.sum()
        _draw_3x3(ax,k,f'PSF @ {zn}\nr={rv:.2f} shift=({sy_v:.2f},{sx_v:.2f})')
    ax_l=axes[1,3]; ax_l.plot(loss_history,color='steelblue',linewidth=1.5)
    ax_l.set_title('Training loss',fontsize=9); ax_l.set_xlabel('Iteration',fontsize=8)
    ax_l.set_ylabel('Loss',fontsize=8); ax_l.grid(True,alpha=0.3)
    fig.suptitle('Coma Aberration -- PSF kernel analysis',fontsize=13,y=1.01)
    plt.tight_layout()
    path=os.path.join(res_dir,'degradation_map.png')
    fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig); print(f'  [Saved] {path}')

def main():
    args=parse_args(); os.makedirs(args.res_dir,exist_ok=True)
    imgs=load_dataset(args)
    if not imgs: return
    true_params={'sigma0':args.sigma0,'alpha_psf':args.alpha_psf,
                 'coma_k':args.coma_k,'alpha_vig':args.alpha_vig}
    H=W=imgs[0].shape[-1]
    print(f'[Config] coma_k={args.coma_k} s0={args.sigma0} a_psf={args.alpha_psf} a_vig={args.alpha_vig}')
    raw_list,deg_list,cor_list,all_lp,all_lh=[],[],[],[],[]
    for i,raw in enumerate(imgs):
        deg=apply_degradation(raw,args.sigma0,args.alpha_psf,args.coma_k,args.alpha_vig)
        print(f'[Image {i+1}/{len(imgs)}]  PSNR_deg={psnr(raw,deg):.2f}dB  Learning...')
        cor,lp,lh=train_pdk(deg,raw,n_iter=args.n_iter,lr=args.lr,lambda_smooth=args.lambda_smooth)
        print(f'  PSNR_cor={psnr(raw,cor):.2f}dB')
        raw_list.append(raw); deg_list.append(deg); cor_list.append(cor)
        all_lp.append(lp); all_lh.append(lh)
    learned_params=all_lp[0]; loss_history=all_lh[0]
    print(f'\n[True]    {true_params}\n[Learned] {learned_params}')
    dsname=args.res_dir.rstrip('/').split('_')[-1].upper()
    save_comparison(raw_list,deg_list,cor_list,true_params,learned_params,args.res_dir,dsname)
    save_degradation_map(true_params,learned_params,loss_history,H,args.res_dir)
    print(f'\n[Done] {os.path.abspath(args.res_dir)}')

if __name__ == '__main__':
    main()

def load_dataset(args):
    if args.data_dir and os.path.isdir(args.data_dir):
        imgs=load_images_from_dir(args.data_dir,args.n_images,args.img_size)
        if imgs: return imgs
    print('[Fallback] synthetic'); return _syn(args.n_images,args.img_size)

def _syn(n,size):
    H=W=size; ys=torch.linspace(-1,1,H); xs=torch.linspace(-1,1,W)
    gy,gx=torch.meshgrid(ys,xs,indexing='ij')
    pts=[(0.5+0.5*torch.sign(torch.sin(gy*20)*torch.sin(gx*20))).clamp(0,1),
         (0.5+0.25*torch.sin(gy*20)+0.25*torch.sin(gx*20)).clamp(0,1),
         ((gx+1)/2).clamp(0,1),
         (0.5+0.5*torch.cos((gy**2+gx**2).sqrt()*10)).clamp(0,1)]
    return [p.unsqueeze(0).unsqueeze(0) for p in pts[:n]]

def parse_args():
    p=argparse.ArgumentParser(description='Coma PDK -- DIV2K')
    p.add_argument('--data-dir',     type=str,   default='')
    p.add_argument('--n-images',     type=int,   default=4)
    p.add_argument('--img-size',     type=int,   default=512)
    p.add_argument('--sigma0',       type=float, default=0.30)
    p.add_argument('--alpha-psf',    type=float, default=1.20)
    p.add_argument('--coma-k',       type=float, default=0.30)
    p.add_argument('--alpha-vig',    type=float, default=2.00)
    p.add_argument('--lambda-smooth',type=float, default=0.05)
    p.add_argument('--n-iter',       type=int,   default=500)
    p.add_argument('--lr',           type=float, default=0.02)
    p.add_argument('--res-dir',      type=str,   default='./res/coma_div2k')
    return p.parse_args()
