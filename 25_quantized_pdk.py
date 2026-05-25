"""
25_quantized_pdk.py
==========================================
Quantization-Aware PDK 실험

동기:
  하드웨어(OKA/QDCTF)에서 conductance level은 discrete
  → kernel weight를 N-bit로 quantize했을 때 성능 저하 분석
  → "몇 bit면 충분한가" 를 PSNR/SSIM로 정량화

실험 구성:
  A. Full-precision PDK (reference)
  B. Post-training quantization (PTQ)
     학습 후 kernel 값을 N-bit로 round
     → N = 2, 3, 4, 5, 6, 8 bit
  C. Quantization-aware training (QAT)
     학습 중 STE(Straight-Through Estimator)로 quantize
     → N = 2, 3, 4, 5, 6, 8 bit

출력:
  25_ptq_vs_qat.png     -- bit수별 PSNR 비교 (PTQ vs QAT)
  25_kernel_error.png   -- bit수별 kernel weight 오차 분포
  25_summary_table.png  -- 수치 요약 테이블

Run:
  python 25_quantized_pdk.py
  python 25_quantized_pdk.py \\
      --train-dir ./data/BSD300/images/train \\
      --inf-dir   ./data/BSD300/images/test  \\
      --n-train 20 --n-inf 4 --img-size 256  \\
      --n-iter 800
"""

import argparse, os, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

BIT_LEVELS = [2, 3, 4, 5, 6, 8]


# ============================================================================
# Utilities (23번과 동일)
# ============================================================================

def radial_map(H, W, device="cpu"):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r = (gy**2 + gx**2).sqrt() / (2**0.5)
    return r, gy, gx

def psnr(a, b):
    mse = F.mse_loss(a.float(), b.float()).item()
    return 99.9 if mse < 1e-10 else 10*np.log10(1.0/mse)

def ssim_simple(a, b):
    a = a.float(); b = b.float()
    mu_a = F.avg_pool2d(a,11,stride=1,padding=5)
    mu_b = F.avg_pool2d(b,11,stride=1,padding=5)
    mu_a2=mu_a**2; mu_b2=mu_b**2; mu_ab=mu_a*mu_b
    sig_a2=F.avg_pool2d(a**2,11,stride=1,padding=5)-mu_a2
    sig_b2=F.avg_pool2d(b**2,11,stride=1,padding=5)-mu_b2
    sig_ab=F.avg_pool2d(a*b,11,stride=1,padding=5)-mu_ab
    c1,c2=0.01**2,0.03**2
    return ((2*mu_ab+c1)*(2*sig_ab+c2)/
            ((mu_a2+mu_b2+c1)*(sig_a2+sig_b2+c2))).mean().item()

def to_np(t): return t.squeeze().detach().cpu().float().numpy()

def load_images(data_dir, n, size):
    exts = ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG")
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(data_dir,"**",ext), recursive=True)
        paths += glob.glob(os.path.join(data_dir, ext))
    paths = sorted(set(paths))
    random.shuffle(paths)
    imgs = []
    for p in paths:
        try:
            img = Image.open(p).convert("L")
            w,h = img.size; s=min(w,h)
            img = img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
            img = img.resize((size,size),Image.BILINEAR)
            t = torch.from_numpy(
                np.array(img,dtype=np.float32)/255.
            ).unsqueeze(0).unsqueeze(0)
            imgs.append(t)
        except: continue
        if len(imgs)==n: break
    print(f"  [Load] {len(imgs)} images from {data_dir}")
    return imgs

def load_images_by_names(data_dir, names, size):
    exts = (".png",".jpg",".jpeg",".PNG",".JPG",".JPEG")
    imgs = []
    for name in names:
        found = None
        for ext in exts:
            c = os.path.join(data_dir, name+ext)
            if os.path.exists(c): found=c; break
        if found is None:
            for ext in exts:
                hits = glob.glob(os.path.join(data_dir,"**",name+ext),recursive=True)
                if hits: found=hits[0]; break
        if found is None: print(f"  [Warning] {name} not found"); continue
        try:
            img = Image.open(found).convert("L")
            w,h=img.size; s=min(w,h)
            img=img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
            img=img.resize((size,size),Image.BILINEAR)
            t=torch.from_numpy(np.array(img,dtype=np.float32)/255.).unsqueeze(0).unsqueeze(0)
            imgs.append(t)
            print(f"  [Load] {found}")
        except Exception as e: print(f"  [Error] {e}")
    return imgs

def spatially_varying_conv(x, kernels):
    B,C,H,W = x.shape
    patches = F.unfold(x,kernel_size=3,padding=1).view(B,C,9,H*W)
    k = kernels.view(H*W,9).T.unsqueeze(0).unsqueeze(0)
    return (patches*k).sum(dim=2).view(B,C,H,W).clamp(0,1)

def loss_fn(out, clean):
    l1 = F.l1_loss(out,clean)
    def gmap(t): return t[:,:,:,1:]-t[:,:,:,:-1], t[:,:,1:,:]-t[:,:,:-1,:]
    ox,oy=gmap(out); cx,cy=gmap(clean)
    return l1+0.1*(F.l1_loss(ox,cx)+F.l1_loss(oy,cy))

def degrade_coma_vig(x, sigma0=0.40, alpha_psf=2.0,
                     coma_k=0.60, alpha_vig=4.0):
    r,gy,gx = radial_map(*x.shape[2:],x.device)
    r2=r**2; rs=r.clamp(1e-6)
    sigma=(sigma0+alpha_psf*r2).clamp(0.1,2.0)
    shift_y=coma_k*r2*(gy/rs); shift_x=coma_k*r2*(gx/rs)
    coords=torch.tensor([
        [-1.,-1.],[-1.,0.],[-1.,1.],
        [ 0.,-1.],[ 0.,0.],[ 0.,1.],
        [ 1.,-1.],[ 1.,0.],[ 1.,1.]], device=x.device)
    d2=(coords**2).sum(-1)
    s=sigma.unsqueeze(-1)
    g1=torch.exp(-d2/(2*s**2)); g1=g1/g1.sum(-1,keepdim=True)
    sy=shift_y.unsqueeze(-1); sx=shift_x.unsqueeze(-1)
    dy=coords[:,0].unsqueeze(0).unsqueeze(0)-sy
    dx=coords[:,1].unsqueeze(0).unsqueeze(0)-sx
    g2=torch.exp(-(dy**2+dx**2)/(2*s**2)); g2=g2/g2.sum(-1,keepdim=True)
    w=(r*0.9).clamp(0,0.9).unsqueeze(-1)
    k=(1-w)*g1+w*g2; k=k/k.sum(-1,keepdim=True)
    x=spatially_varying_conv(x,k)
    V=1.0/(1.0+alpha_vig*r2)
    return (x*V.unsqueeze(0).unsqueeze(0)).clamp(0,1)


# ============================================================================
# Quantization helpers
# ============================================================================

def quantize_kernels(kernels_hwk, n_bits):
    """
    (H,W,9) kernel map을 n_bits로 uniform quantize.
    범위: [k_min, k_max] → 2^n_bits levels
    """
    k_min = kernels_hwk.min()
    k_max = kernels_hwk.max()
    levels = 2**n_bits - 1
    scale = (k_max - k_min) / levels
    if scale < 1e-8: return kernels_hwk.copy()
    q = np.round((kernels_hwk - k_min) / scale) * scale + k_min
    return q.astype(np.float32)

def ste_quantize(k_tensor, n_bits, k_min, k_max):
    """
    Straight-Through Estimator quantization.
    forward: quantize, backward: pass-through gradient
    """
    levels = 2**n_bits - 1
    scale = (k_max - k_min) / levels
    if scale < 1e-8: return k_tensor
    k_norm = (k_tensor - k_min) / scale
    k_q = torch.round(k_norm).detach() - k_norm.detach() + k_norm
    return k_q * scale + k_min


# ============================================================================
# PDK Models
# ============================================================================

class ComaPDK(nn.Module):
    def __init__(self, H, W, n_bits=None, qat=False):
        """
        n_bits: None = full precision, int = quantization bit width
        qat:    True = quantization-aware training (STE)
                False = post-training quantization (applied at inference)
        """
        super().__init__()
        def _sp(v): return float(np.log(np.exp(v)-1.0))
        self._s0  = nn.Parameter(torch.tensor(_sp(0.40)))
        self._aps = nn.Parameter(torch.tensor(_sp(1.0)))
        self._ck  = nn.Parameter(torch.tensor(_sp(0.20)))
        self._l0  = nn.Parameter(torch.tensor(_sp(0.50)))
        self._al  = nn.Parameter(torch.tensor(_sp(0.30)))
        self._av  = nn.Parameter(torch.tensor(_sp(2.0)))
        r,gy,gx = radial_map(H,W)
        self.register_buffer("r",  r.contiguous())
        self.register_buffer("r2", r.pow(2).contiguous())
        self.register_buffer("gy", gy.contiguous())
        self.register_buffer("gx", gx.contiguous())
        self.register_buffer("coords", torch.tensor([
            [-1.,-1.],[-1.,0.],[-1.,1.],
            [ 0.,-1.],[ 0.,0.],[ 0.,1.],
            [ 1.,-1.],[ 1.,0.],[ 1.,1.]]))
        self.n_bits = n_bits
        self.qat    = qat
        self._k_min = None
        self._k_max = None

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

    def _build_kernels(self):
        vig_gain = 1.0+self.alpha_vig*self.r2
        sigma=(self.sigma0+self.alpha_psf*self.r2).unsqueeze(-1).clamp(0.1,2.0)
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
        return k

    def forward(self, x):
        k = self._build_kernels()
        if self.n_bits is not None and self.qat:
            # QAT: STE quantize during training
            k_min = k.detach().min()
            k_max = k.detach().max()
            k = ste_quantize(k, self.n_bits, k_min, k_max)
        return spatially_varying_conv(x, k)

    def get_kernels(self):
        with torch.no_grad():
            k = self._build_kernels().detach().cpu().numpy()
        return k

    def get_kernels_quantized(self, n_bits):
        k = self.get_kernels()
        return quantize_kernels(k, n_bits)


# ============================================================================
# Training
# ============================================================================

def train_dataset(model, train_imgs, degrade_fn, n_iter, lr, verbose=True):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    pairs = [(img, degrade_fn(img)) for img in train_imgs]
    for i in range(n_iter):
        opt.zero_grad()
        total = torch.tensor(0.0)
        for clean, deg in pairs:
            total = total + loss_fn(model(deg), clean)
        total = total / len(pairs)
        total.backward(); opt.step(); sched.step()
        if verbose and (i+1)%200==0:
            print(f"    iter {i+1}/{n_iter}  loss={total.item():.5f}")

def infer_with_kernels(x, kernels_hwk):
    """numpy kernel map으로 직접 inference"""
    k = torch.from_numpy(kernels_hwk).float()
    return spatially_varying_conv(x, k)


# ============================================================================
# Evaluation
# ============================================================================

def eval_model(model, inf_imgs, degrade_fn):
    """full-precision 모델 평가"""
    results = []
    model.eval()
    with torch.no_grad():
        for clean in inf_imgs:
            deg = degrade_fn(clean)
            out = model(deg).clamp(0,1)
            results.append(dict(
                psnr=psnr(clean,out),
                ssim=ssim_simple(clean,out),
                psnr_deg=psnr(clean,deg),
            ))
    return results

def eval_quantized(kernels_fp, n_bits, inf_imgs, degrade_fn):
    """PTQ: full-precision kernel → quantize → inference"""
    k_q = quantize_kernels(kernels_fp, n_bits)
    k_q_t = torch.from_numpy(k_q).float()
    results = []
    with torch.no_grad():
        for clean in inf_imgs:
            deg = degrade_fn(clean)
            out = spatially_varying_conv(deg, k_q_t).clamp(0,1)
            results.append(dict(
                psnr=psnr(clean,out),
                ssim=ssim_simple(clean,out),
            ))
    return results


# ============================================================================
# Visualization
# ============================================================================

def save_bit_comparison(fp_results, ptq_dict, qat_dict, res_dir):
    """bit수별 PSNR/SSIM 비교 (PTQ vs QAT)"""
    fp_psnr = np.mean([r['psnr'] for r in fp_results])
    fp_ssim = np.mean([r['ssim'] for r in fp_results])
    deg_psnr = np.mean([r['psnr_deg'] for r in fp_results])

    bits = BIT_LEVELS
    ptq_psnr = [np.mean([r['psnr'] for r in ptq_dict[b]]) for b in bits]
    ptq_ssim = [np.mean([r['ssim'] for r in ptq_dict[b]]) for b in bits]
    qat_psnr = [np.mean([r['psnr'] for r in qat_dict[b]]) for b in bits]
    qat_ssim = [np.mean([r['ssim'] for r in qat_dict[b]]) for b in bits]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, ptq_vals, qat_vals, fp_val, deg_val, ylabel, title in [
        (axes[0], ptq_psnr, qat_psnr, fp_psnr, deg_psnr,
         'PSNR (dB)', 'PSNR vs bit width'),
        (axes[1], ptq_ssim, qat_ssim, fp_ssim, None,
         'SSIM', 'SSIM vs bit width'),
    ]:
        ax.plot(bits, ptq_vals, 'o-', color='#D85A30',
                linewidth=2, markersize=7, label='PTQ')
        ax.plot(bits, qat_vals, 's-', color='#185FA5',
                linewidth=2, markersize=7, label='QAT')
        ax.axhline(fp_val, linestyle='--', color='#1D9E75',
                   linewidth=1.5, label=f'Full precision ({fp_val:.2f})')
        if deg_val is not None:
            ax.axhline(deg_val, linestyle=':', color='#888780',
                       linewidth=1.2, label=f'Degraded ({deg_val:.2f})')
        ax.set_xlabel('Bit width', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(bits)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Quantization effect on PDK correction performance',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(res_dir, '25_ptq_vs_qat.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {path}')


def save_kernel_error(fp_kernels, res_dir):
    """bit수별 kernel quantization error 분포"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for idx, n_bits in enumerate(BIT_LEVELS):
        ax = axes[idx]
        k_q = quantize_kernels(fp_kernels, n_bits)
        err = (k_q - fp_kernels).ravel()
        ax.hist(err, bins=80, color='#D85A30', alpha=0.8,
                edgecolor='none')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        mae = np.abs(err).mean()
        rms = np.sqrt((err**2).mean())
        ax.set_title(f'{n_bits}-bit  MAE={mae:.4f}  RMSE={rms:.4f}',
                     fontsize=10)
        ax.set_xlabel('Quantization error (k_q - k_fp)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('PTQ kernel quantization error distribution',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(res_dir, '25_kernel_error.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {path}')


def save_summary_table(fp_results, ptq_dict, qat_dict, res_dir):
    bits = BIT_LEVELS
    fp_psnr = np.mean([r['psnr'] for r in fp_results])
    fp_ssim = np.mean([r['ssim'] for r in fp_results])
    deg_psnr = np.mean([r['psnr_deg'] for r in fp_results])

    rows = [['Degraded', f'{deg_psnr:.2f}', '-', '-', '-']]
    rows.append(['Full precision', f'{fp_psnr:.2f}',
                 f'{fp_ssim:.4f}', '-', '-'])
    for b in bits:
        ptq_p = np.mean([r['psnr'] for r in ptq_dict[b]])
        ptq_s = np.mean([r['ssim'] for r in ptq_dict[b]])
        qat_p = np.mean([r['psnr'] for r in qat_dict[b]])
        qat_s = np.mean([r['ssim'] for r in qat_dict[b]])
        rows.append([
            f'{b}-bit',
            f'{ptq_p:.2f}',
            f'{ptq_s:.4f}',
            f'{qat_p:.2f}',
            f'{qat_s:.4f}',
        ])

    headers = ['Method', 'PTQ PSNR', 'PTQ SSIM', 'QAT PSNR', 'QAT SSIM']
    fig, ax = plt.subplots(figsize=(10, 0.5+0.5*len(rows)))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=headers,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for j in range(len(headers)):
        tbl[0,j].set_facecolor('#2C2C2A')
        tbl[0,j].set_text_props(color='white', fontweight='bold')
    # full precision row 강조
    for j in range(len(headers)):
        tbl[2,j].set_facecolor('#E1F5EE')
        tbl[2,j].set_text_props(color='#085041', fontweight='bold')
    plt.suptitle('Quantization sensitivity summary',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(res_dir, '25_summary_table.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {path}')

    # 터미널 출력
    print('\n' + '='*65)
    print(f'{"Method":<16} {"PTQ PSNR":>10} {"PTQ SSIM":>10}'
          f' {"QAT PSNR":>10} {"QAT SSIM":>10}')
    print('-'*65)
    for row in rows:
        print(f'{row[0]:<16} {row[1]:>10} {row[2]:>10}'
              f' {row[3]:>10} {row[4]:>10}')
    print('='*65)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-dir', type=str,
                   default='./data/BSD300/images/train')
    p.add_argument('--inf-dir',   type=str,
                   default='./data/BSD300/images/test')
    p.add_argument('--n-train',   type=int, default=20)
    p.add_argument('--img-size',  type=int, default=256)
    p.add_argument('--n-iter',    type=int, default=800)
    p.add_argument('--lr',        type=float, default=0.02)
    p.add_argument('--res-dir',   type=str,
                   default='./res/quant_pdk')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.res_dir, exist_ok=True)
    H = W = args.img_size
    INF_NAMES = ['102061', '143090', '103070', '145086']

    degrade = lambda x: degrade_coma_vig(x, 0.40, 2.0, 0.60, 4.0)

    # ── 데이터 로드 ────────────────────────────────────────────
    print('[Load] Train images...')
    train_imgs = load_images(args.train_dir, args.n_train, H)
    print('[Load] Inference images (fixed)...')
    inf_imgs   = load_images_by_names(args.inf_dir, INF_NAMES, H)

    # ── Step 1: Full-precision PDK 학습 ────────────────────────
    print('\n[Train] Full-precision PDK...')
    fp_model = ComaPDK(H, W, n_bits=None)
    train_dataset(fp_model, train_imgs, degrade, args.n_iter, args.lr)
    fp_model.eval()
    fp_results = eval_model(fp_model, inf_imgs, degrade)
    fp_kernels = fp_model.get_kernels()  # (H,W,9) numpy
    torch.save(fp_model.state_dict(),
               os.path.join(args.res_dir, 'fp_model.pt'))

    fp_psnr = np.mean([r['psnr'] for r in fp_results])
    print(f'  Full-precision PSNR: {fp_psnr:.2f} dB')

    # ── Step 2: PTQ (post-training quantization) ───────────────
    print('\n[PTQ] Applying quantization to trained kernels...')
    ptq_dict = {}
    for n_bits in BIT_LEVELS:
        ptq_dict[n_bits] = eval_quantized(fp_kernels, n_bits,
                                          inf_imgs, degrade)
        p = np.mean([r['psnr'] for r in ptq_dict[n_bits]])
        print(f'  {n_bits}-bit PTQ PSNR: {p:.2f} dB  '
              f'(gap: {fp_psnr-p:.2f} dB)')

    # ── Step 3: QAT (quantization-aware training) ──────────────
    print('\n[QAT] Training with quantization-aware STE...')
    qat_dict = {}
    for n_bits in BIT_LEVELS:
        print(f'  Training {n_bits}-bit QAT...')
        qat_model = ComaPDK(H, W, n_bits=n_bits, qat=True)
        train_dataset(qat_model, train_imgs, degrade,
                      args.n_iter, args.lr, verbose=False)
        qat_model.eval()
        qat_dict[n_bits] = eval_model(qat_model, inf_imgs, degrade)
        p = np.mean([r['psnr'] for r in qat_dict[n_bits]])
        print(f'  {n_bits}-bit QAT PSNR: {p:.2f} dB  '
              f'(gap: {fp_psnr-p:.2f} dB)')

    # ── 시각화 ────────────────────────────────────────────────
    print('\n[Saving results...]')
    save_bit_comparison(fp_results, ptq_dict, qat_dict, args.res_dir)
    save_kernel_error(fp_kernels, args.res_dir)
    save_summary_table(fp_results, ptq_dict, qat_dict, args.res_dir)

    print(f'\n[Done] {os.path.abspath(args.res_dir)}')
    print('  25_ptq_vs_qat.png     -- bit수별 PSNR/SSIM 비교')
    print('  25_kernel_error.png   -- quantization error 분포')
    print('  25_summary_table.png  -- 수치 요약')


if __name__ == '__main__':
    main()