"""
14_non_uniform_illum_bsd500.py
============================================
Non-uniform Illumination -- 3x3 PD Kernel [BSD500]
Run: python 14_non_uniform_illum_bsd500.py
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


def parse_args():
    p=argparse.ArgumentParser(description='Non-uniform Illum PDK -- BSD500')
    p.add_argument('--data-dir',      type=str,   default='')
    p.add_argument('--n-images',      type=int,   default=4)
    p.add_argument('--img-size',      type=int,   default=256)
    p.add_argument('--cx-off',        type=float, default=0.20)
    p.add_argument('--cy-off',        type=float, default=0.15)
    p.add_argument('--sigma-illum',   type=float, default=0.60)
    p.add_argument('--ellipse-ratio', type=float, default=1.40)
    p.add_argument('--alpha-vig',     type=float, default=1.50)
    p.add_argument('--sigma-psf',     type=float, default=0.0)
    p.add_argument('--n-iter',        type=int,   default=500)
    p.add_argument('--lr',            type=float, default=0.02)
    p.add_argument('--res-dir',       type=str,   default='./res/nonunif_bsd500')
    return p.parse_args()


def _syn(n,size):
    H=W=size; ys=torch.linspace(-1,1,H); xs=torch.linspace(-1,1,W)
    gy,gx=torch.meshgrid(ys,xs,indexing='ij')
    pts=[(0.5+0.5*torch.sign(torch.sin(gy*12)*torch.sin(gx*12))).clamp(0,1),
         (0.5+0.25*torch.sin(gy*12)+0.25*torch.sin(gx*12)).clamp(0,1),
         ((gx+1)/2).clamp(0,1),
         (0.5+0.5*torch.cos((gy**2+gx**2).sqrt()*10)).clamp(0,1)]
    return [p.unsqueeze(0).unsqueeze(0) for p in pts[:n]]


def load_dataset(args):
    if args.data_dir and os.path.isdir(args.data_dir):
        imgs=load_images_from_dir(args.data_dir,args.n_images,args.img_size)
        if imgs: return imgs
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
        print(f'[Load] BSD500 fallback: {len(imgs)} images'); return imgs
    except Exception as e:
        print(f'[Warning] {e}'); return _syn(args.n_images,args.img_size)


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
# Non-uniform Illumination
# V(x,y) = V_rad(r) * illum(x-cx, y-cy, sigma, ellipse)
# ============================================================================

def _illum(gy, gx, r2, cx, cy, sig, ell, av):
    V_rad=1.0/(1.0+av*r2)
    dy=gy-cy; dx=gx-cx; d2=dx**2+(dy*ell)**2
    ill=torch.exp(-d2/(2*sig**2)); ill=ill/ill.max().clamp(1e-5)
    return (V_rad*(0.3+0.7*ill)).clamp(0,1)

def apply_degradation(x, cx_off, cy_off, sigma_illum,
                      ellipse_ratio, alpha_vig, sigma_psf=0.0):
    B,C,H,W=x.shape
    r,gy,gx=radial_map(H,W,x.device)
    if sigma_psf>0:
        co=torch.tensor([[-1.,-1.],[-1.,0.],[-1.,1.],
                          [ 0.,-1.],[ 0.,0.],[ 0.,1.],
                          [ 1.,-1.],[ 1.,0.],[ 1.,1.]],device=x.device)
        d2=(co**2).sum(-1); g=torch.exp(-d2/(2*sigma_psf**2)); g=g/g.sum()
        x=F.conv2d(x,g.view(1,1,3,3),padding=1)
    V=_illum(gy,gx,r**2,cx_off,cy_off,sigma_illum,ellipse_ratio,alpha_vig)
    return (x*V.unsqueeze(0).unsqueeze(0)).clamp(0,1)

class NonUniformRestorationPDK(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._cx_off       =nn.Parameter(torch.tensor(0.0))
        self._cy_off       =nn.Parameter(torch.tensor(0.0))
        self._sigma_illum  =nn.Parameter(torch.tensor(_sp(0.6)))
        self._ellipse_ratio=nn.Parameter(torch.tensor(_sp(1.0)))
        self._alpha_vig    =nn.Parameter(torch.tensor(_sp(1.0)))
        r,gy,gx=radial_map(H,W)
        self.register_buffer('r2',r**2)
        self.register_buffer('gy',gy); self.register_buffer('gx',gx)

    @property
    def cx_off(self):        return torch.tanh(self._cx_off)*0.5
    @property
    def cy_off(self):        return torch.tanh(self._cy_off)*0.5
    @property
    def sigma_illum(self):   return F.softplus(self._sigma_illum).clamp(0.2,2.0)
    @property
    def ellipse_ratio(self): return F.softplus(self._ellipse_ratio).clamp(0.3,3.0)
    @property
    def alpha_vig(self):     return F.softplus(self._alpha_vig)

    def forward(self, x):
        B,C,H,W=x.shape
        V=_illum(self.gy,self.gx,self.r2,self.cx_off,self.cy_off,
                 self.sigma_illum,self.ellipse_ratio,self.alpha_vig).clamp(1e-3,1.0)
        gain=(1.0/V).clamp(1.0,10.0)
        zeros=torch.zeros(H,W,9,device=x.device)
        idx=torch.full((H,W,1),4,dtype=torch.long,device=x.device)
        kernels=zeros.scatter(2,idx,gain.unsqueeze(-1))
        return spatially_varying_conv(x,kernels)

def train_pdk(deg, clean, n_iter=500, lr=0.02, **kwargs):
    model=NonUniformRestorationPDK(*deg.shape[2:])
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
        if (i+1)%100==0:
            print(f'  iter {i+1:4d}/{n_iter}  loss={loss.item():.5f}'
                  f'  cx={model.cx_off.item():.3f} cy={model.cy_off.item():.3f}'
                  f'  sig={model.sigma_illum.item():.3f}'
                  f'  ell={model.ellipse_ratio.item():.3f}')
    with torch.no_grad(): cor=model(deg).clamp(0,1)
    return cor, {
        'cx_off':model.cx_off.item(),'cy_off':model.cy_off.item(),
        'sigma_illum':model.sigma_illum.item(),
        'ellipse_ratio':model.ellipse_ratio.item(),
        'alpha_vig':model.alpha_vig.item()
    }, history

def save_comparison(raw_list,deg_list,cor_list,true_params,learned_params,res_dir,dsname):
    n=len(raw_list)
    fig,axes=plt.subplots(n,3,figsize=(12,4*n+1))
    if n==1: axes=axes[np.newaxis,:]
    pd_=np.mean([psnr(r,d) for r,d in zip(raw_list,deg_list)])
    pc_=np.mean([psnr(r,c) for r,c in zip(raw_list,cor_list)])
    cols=[
        (raw_list,'1. Raw (clean)',''),
        (deg_list,'2. Degraded (Non-uniform Illum)',
         f'PSNR={pd_:.1f}dB | cx={true_params["cx_off"]:.2f}'
         f' cy={true_params["cy_off"]:.2f}'
         f' sig={true_params["sigma_illum"]:.2f} ell={true_params["ellipse_ratio"]:.2f}'),
        (cor_list,'3. Corrected (learned PDK)',
         f'PSNR={pc_:.1f}dB | cx={learned_params["cx_off"]:.2f}'
         f' cy={learned_params["cy_off"]:.2f}'
         f' sig={learned_params["sigma_illum"]:.2f} ell={learned_params["ellipse_ratio"]:.2f}'),
    ]
    for c,(lst,title,sub) in enumerate(cols):
        axes[0,c].set_title(title,fontsize=10,fontweight='bold',pad=8)
        for r in range(n):
            axes[r,c].imshow(to_np(lst[r]),cmap='gray',vmin=0,vmax=1); axes[r,c].axis('off')
        if sub:
            axes[0,c].text(0.5,-0.06,sub,transform=axes[0,c].transAxes,
                           ha='center',va='top',fontsize=8,color='dimgray')
    fig.suptitle(f'Non-uniform Illumination -- 3x3 PD Kernel [{dsname}]',fontsize=13,y=1.01)
    plt.tight_layout(rect=[0,0.05,1,1])
    path=os.path.join(res_dir,'comparison.png')
    fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig); print(f'  [Saved] {path}')

def save_degradation_map(true_params,learned_params,loss_history,size,res_dir):
    H=W=size; r,gy,gx=radial_map(H,W)
    r2=r.numpy()**2; gy=gy.numpy(); gx=gx.numpy()
    def illum_np(cx,cy,sig,ell,av):
        V_rad=1/(1+av*r2); dy=gy-cy; dx=gx-cx; d2=dx**2+(dy*ell)**2
        ill=np.exp(-d2/(2*sig**2)); ill=ill/max(ill.max(),1e-5)
        return np.clip(V_rad*(0.3+0.7*ill),0,1), ill
    cxt=true_params['cx_off']; cyt=true_params['cy_off']
    sit=true_params['sigma_illum']; elt=true_params['ellipse_ratio']; avt=true_params['alpha_vig']
    cxl=learned_params['cx_off']; cyl=learned_params['cy_off']
    sil=learned_params['sigma_illum']; ell_=learned_params['ellipse_ratio']; avl=learned_params['alpha_vig']
    V_t,ill_t=illum_np(cxt,cyt,sit,elt,avt); V_l,ill_l=illum_np(cxl,cyl,sil,ell_,avl)
    fig,axes=plt.subplots(2,4,figsize=(18,9))
    for ax,(data,lbl) in zip(axes[0],[
        (V_t,f'Illum map (true)\ncx={cxt:.2f} cy={cyt:.2f}'),
        (ill_t,f'Pattern (true)\nsig={sit:.2f} ell={elt:.2f}'),
        (1/(1+avt*r2),f'Vignetting (true)\na_vig={avt:.2f}'),
        (np.abs(V_t-V_l),'|V: true-learned|'),
    ]):
        im=ax.imshow(data,cmap='hot'); ax.set_title(lbl,fontsize=9,pad=4); ax.axis('off')
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    for ax,(data,lbl) in zip(axes[1,:3],[
        (V_l,f'Illum map (learned)\ncx={cxl:.2f} cy={cyl:.2f}'),
        (ill_l,f'Pattern (learned)\nsig={sil:.2f} ell={ell_:.2f}'),
        (1/(1+avl*r2),f'Vignetting (learned)\na_vig={avl:.2f}'),
    ]):
        im=ax.imshow(data,cmap='hot'); ax.set_title(lbl,fontsize=9,pad=4); ax.axis('off')
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    ax_l=axes[1,3]; ax_l.plot(loss_history,color='steelblue',linewidth=1.5)
    ax_l.set_title('Training loss',fontsize=9); ax_l.set_xlabel('Iteration',fontsize=8)
    ax_l.set_ylabel('Loss',fontsize=8); ax_l.grid(True,alpha=0.3)
    fig.suptitle('Non-uniform Illumination -- map analysis',fontsize=13,y=1.01)
    plt.tight_layout()
    path=os.path.join(res_dir,'degradation_map.png')
    fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig); print(f'  [Saved] {path}')


def main():
    args=parse_args(); os.makedirs(args.res_dir,exist_ok=True)
    imgs=load_dataset(args)
    if not imgs: return
    true_params={'cx_off':args.cx_off,'cy_off':args.cy_off,
                 'sigma_illum':args.sigma_illum,'ellipse_ratio':args.ellipse_ratio,
                 'alpha_vig':args.alpha_vig}
    H=W=imgs[0].shape[-1]
    print(f'[Config] cx={args.cx_off} cy={args.cy_off} sig={args.sigma_illum}'
          f' ell={args.ellipse_ratio} a_vig={args.alpha_vig}')
    raw_list,deg_list,cor_list,all_lp,all_lh=[],[],[],[],[]
    for i,raw in enumerate(imgs):
        deg=apply_degradation(raw,args.cx_off,args.cy_off,args.sigma_illum,
                              args.ellipse_ratio,args.alpha_vig,args.sigma_psf)
        print(f'[Image {i+1}/{len(imgs)}]  PSNR_deg={psnr(raw,deg):.2f}dB  Learning...')
        cor,lp,lh=train_pdk(deg,raw,n_iter=args.n_iter,lr=args.lr)
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