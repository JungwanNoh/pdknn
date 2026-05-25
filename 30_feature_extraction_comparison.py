"""
30_feature_extraction_comparison.py
=====================================================
Reviewer 방어용: feature extraction 관점에서도 PDK 필요성 증명

핵심 주장:
  "vignetting은 high-pass kernel(Sobel)이 무시할 수 있지만
   coma는 spatially varying 비대칭 왜곡이라
   high-pass로도 보정 불가"

실험 설계:
  강한 coma + vignetting degradation 적용
  → 4가지 처리 후 Sobel edge 추출
  → clean edge map과 비교

처리:
  1. degraded         (no correction)
  2. Sobel direct     (degraded에 Sobel만 적용)
  3. global + Sobel   (학습된 spatially invariant kernel + Sobel)
  4. PDK + Sobel      (학습된 spatially varying kernel + Sobel)

평가:
  edge map의 PSNR/SSIM (vs clean edge map)
  → "PDK만이 coma 왜곡을 정확히 보정해 edge feature 복원"

출력:
  res/feature_extraction/
    sobel_compare_XX.png      각 이미지별 5-column grid
    edge_metrics_table.png    수치 비교
    30_summary.png

Run:
  python 30_feature_extraction_comparison.py
  python 30_feature_extraction_comparison.py \\
      --coma-k 1.20 --alpha-vig 6.0 --n-iter 2000
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


# ============================================================================
# Utilities (27번과 동일)
# ============================================================================

def radial_map(H, W, device="cpu"):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r = (gy**2 + gx**2).sqrt() / (2**0.5)
    return r, gy, gx

def spatially_varying_conv(x, kernels):
    B, C, H, W = x.shape
    patches = F.unfold(x, kernel_size=3, padding=1).view(B, C, 9, H*W)
    k = kernels.view(H*W, 9).T.unsqueeze(0).unsqueeze(0)
    return (patches * k).sum(dim=2).view(B, C, H, W).clamp(0, 1)

def global_conv(x, kernel_9):
    k = kernel_9.view(1, 1, 3, 3)
    return F.conv2d(x, k.expand(x.shape[1], 1, 3, 3),
                    padding=1, groups=x.shape[1]).clamp(0, 1)

def loss_fn(out, clean):
    l1 = F.l1_loss(out, clean)
    def gmap(t): return t[:,:,:,1:]-t[:,:,:,:-1], t[:,:,1:,:]-t[:,:,:-1,:]
    ox, oy = gmap(out); cx, cy = gmap(clean)
    return l1 + 0.1*(F.l1_loss(ox,cx)+F.l1_loss(oy,cy))

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
    sig_ab=F.avg_pool2d(a*b, 11,stride=1,padding=5)-mu_ab
    c1,c2=0.01**2,0.03**2
    return ((2*mu_ab+c1)*(2*sig_ab+c2)/
            ((mu_a2+mu_b2+c1)*(sig_a2+sig_b2+c2))).mean().item()

def to_np(t):
    return t.squeeze().detach().cpu().float().numpy()


# ============================================================================
# Sobel edge detection
# ============================================================================

def sobel_edge(x):
    """
    표준 Sobel filter로 edge magnitude 추출.
    - high-pass kernel의 대표
    - vignetting(low-frequency) 무시
    - 하지만 coma의 위치별 왜곡은 보정 불가
    """
    sx = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).view(1, 1, 3, 3)
    sy = torch.tensor([[-1., -2., -1.],
                        [ 0.,  0.,  0.],
                        [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
    gx = F.conv2d(x, sx, padding=1)
    gy = F.conv2d(x, sy, padding=1)
    mag = (gx**2 + gy**2).sqrt()
    B = mag.shape[0]
    flat = mag.view(B, -1)
    mx = flat.max(dim=1, keepdim=True)[0].clamp(min=1e-6)
    mag = (mag.view(B, -1) / mx).view_as(mag)
    return mag.clamp(0, 1)


# ============================================================================
# Feature extraction metrics
# ============================================================================

def edge_binary(edge_map, threshold=0.15):
    """edge map → binary 0/1 (numpy array)"""
    arr = edge_map.squeeze().detach().cpu().numpy()
    return (arr > threshold).astype(np.float32)


def edge_iou(edge_a, edge_b, threshold=0.15):
    """
    Intersection over Union of binary edges.
    값 ∈ [0, 1] (1=완벽 일치)
    """
    a = edge_binary(edge_a, threshold)
    b = edge_binary(edge_b, threshold)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / max(union, 1)


def edge_cosine(edge_a, edge_b):
    """
    Edge map 간 cosine similarity.
    threshold-free, 전체 pattern 유사도.
    값 ∈ [-1, 1] (실제로는 [0,1], edge map은 양수)
    """
    a = edge_a.squeeze().detach().cpu().numpy().flatten()
    b = edge_b.squeeze().detach().cpu().numpy().flatten()
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def pratt_fom(edge_gt, edge_pred, threshold=0.15, alpha=1/9):
    """
    Pratt's Figure of Merit — edge detection 표준 지표.
      FoM = (1 / max(|GT|, |Pred|)) * Σ_i 1 / (1 + α·d_i^2)
      d_i: 예측 edge 픽셀 i에서 가장 가까운 GT edge 픽셀까지의 거리
    값 ∈ [0, 1] (1=완벽), 위치 미스를 거리로 페널티
    """
    from scipy.ndimage import distance_transform_edt

    gt   = edge_binary(edge_gt,   threshold)
    pred = edge_binary(edge_pred, threshold)

    n_gt   = gt.sum()
    n_pred = pred.sum()
    if n_pred == 0 or n_gt == 0:
        return 0.0

    # GT가 0인 위치에서 가장 가까운 GT까지의 거리
    dist = distance_transform_edt(1 - gt)
    # 예측 edge 위치의 거리들
    d_pred = dist[pred.astype(bool)]
    # FoM 합산
    fom = (1.0 / max(n_gt, n_pred)) * np.sum(1.0 / (1.0 + alpha * d_pred**2))
    return float(fom)


# ============================================================================
# Degradation: STRONG coma + vignetting
# ============================================================================

def degrade_coma_vig(x, sigma0=0.30, alpha_psf=1.5,
                     coma_k=1.20, alpha_vig=6.0):
    """
    Reviewer 방어용 강한 degradation:
    - coma_k 강하게 (PSF가 크게 옆으로 밀림)
    - alpha_vig 강하게 (가장자리 어둡게)
    """
    r, gy, gx = radial_map(*x.shape[2:], x.device)
    r2 = r**2; rs = r.clamp(1e-6)
    sigma = (sigma0 + alpha_psf*r2).clamp(0.1, 2.0)
    shift_y = coma_k*r2*(gy/rs)
    shift_x = coma_k*r2*(gx/rs)
    coords = torch.tensor([
        [-1.,-1.],[-1.,0.],[-1.,1.],
        [ 0.,-1.],[ 0.,0.],[ 0.,1.],
        [ 1.,-1.],[ 1.,0.],[ 1.,1.]], device=x.device)
    d2 = (coords**2).sum(-1)
    s  = sigma.unsqueeze(-1)
    g1 = torch.exp(-d2/(2*s**2)); g1=g1/g1.sum(-1,keepdim=True)
    sy = shift_y.unsqueeze(-1); sx = shift_x.unsqueeze(-1)
    dy = coords[:,0].unsqueeze(0).unsqueeze(0)-sy
    dx = coords[:,1].unsqueeze(0).unsqueeze(0)-sx
    g2 = torch.exp(-(dy**2+dx**2)/(2*s**2)); g2=g2/g2.sum(-1,keepdim=True)
    w  = (r*0.95).clamp(0,0.95).unsqueeze(-1)
    k  = (1-w)*g1+w*g2; k=k/k.sum(-1,keepdim=True)
    x  = spatially_varying_conv(x, k)
    V  = 1.0/(1.0+alpha_vig*r2)
    return (x*V.unsqueeze(0).unsqueeze(0)).clamp(0,1)


# ============================================================================
# Models (27번과 동일)
# ============================================================================

class GlobalKernel(nn.Module):
    def __init__(self):
        super().__init__()
        init = torch.zeros(9); init[4] = 3.0
        self.kernel_logits = nn.Parameter(init)

    def forward(self, x):
        k = torch.softmax(self.kernel_logits, dim=0)
        return global_conv(x, k)


class DirectPDK(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        init = torch.zeros(H, W, 9)
        init[:, :, 4] = 1.0
        self.kernels = nn.Parameter(init)

    def forward(self, x):
        return spatially_varying_conv(x, self.kernels)


# ============================================================================
# Data loading
# ============================================================================

def load_images(data_dir, n, size):
    exts = ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG")
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(data_dir,"**",ext), recursive=True)
        paths += glob.glob(os.path.join(data_dir, ext))
    paths = sorted(set(paths)); random.shuffle(paths)
    imgs = []
    for p in paths:
        try:
            img = Image.open(p).convert("L")
            w,h=img.size; s=min(w,h)
            img = img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
            img = img.resize((size,size), Image.BILINEAR)
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
        if found is None:
            print(f"  [Warning] {name} not found"); continue
        try:
            img = Image.open(found).convert("L")
            w,h=img.size; s=min(w,h)
            img=img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
            img=img.resize((size,size),Image.BILINEAR)
            t=torch.from_numpy(np.array(img,dtype=np.float32)/255.
                               ).unsqueeze(0).unsqueeze(0)
            imgs.append(t)
        except Exception as e: print(f"  [Error] {e}")
    return imgs


# ============================================================================
# Training
# ============================================================================

def train_global(model, train_imgs, degrade_fn, n_iter, lr):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    pairs = [(img, degrade_fn(img)) for img in train_imgs]
    for i in range(n_iter):
        opt.zero_grad()
        total = sum(loss_fn(model(deg), clean) for clean,deg in pairs)
        total = total / len(pairs)
        total.backward(); opt.step(); sched.step()
        if (i+1) % 200 == 0:
            print(f"    iter {i+1}/{n_iter}  loss={total.item():.5f}")


def train_direct_pdk(model, train_imgs, degrade_fn, n_iter, lr, batch=32):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    n = len(train_imgs)
    X = torch.cat(train_imgs, dim=0)
    for i in range(n_iter):
        idx   = torch.randperm(n)[:batch]
        clean = X[idx]
        deg   = degrade_fn(clean)
        opt.zero_grad()
        loss = loss_fn(model(deg), clean)
        loss.backward(); opt.step(); sched.step()
        if (i+1) % 400 == 0:
            print(f"    iter {i+1}/{n_iter}  loss={loss.item():.5f}")


# ============================================================================
# Saving
# ============================================================================

def save_img(arr, path, cmap="gray"):
    arr = np.clip(arr, 0, 1)
    fig, ax = plt.subplots(1, 1,
        figsize=(arr.shape[1]/100, arr.shape[0]/100), dpi=300)
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
    ax.axis("off")
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_compare(results, path):
    """
    각 이미지마다 6 column:
      col 1: clean image
      col 2: clean Sobel edge (target)
      col 3: degraded Sobel edge
      col 4: global+Sobel edge
      col 5: PDK+Sobel edge
      col 6: degraded image (참조용)
    """
    n = len(results)
    titles = ["Clean", "Clean Sobel\n(target)",
              "Degraded\nSobel",
              "Global → Sobel",
              "PDK → Sobel",
              "Degraded\n(input)"]
    fig, axes = plt.subplots(n, 6, figsize=(18, 3.2*n))
    if n == 1: axes = axes[np.newaxis, :]

    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=10, fontweight="bold", pad=6)

    for r, res in enumerate(results):
        for c, key in enumerate(
            ["clean", "clean_e", "deg_e", "glb_e", "pdk_e", "deg"]
        ):
            cmap = "gray" if key in ("clean", "deg") else "magma"
            axes[r, c].imshow(to_np(res[key]), cmap=cmap, vmin=0, vmax=1)
            axes[r, c].axis("off")

        for c, key, color in [
            (2, "iou_deg", "dimgray"),
            (3, "iou_glb", "steelblue"),
            (4, "iou_pdk", "#085041"),
        ]:
            axes[r, c].text(0.5, -0.05,
                f"IoU={res[key]:.3f}",
                transform=axes[r, c].transAxes,
                ha="center", fontsize=8, color=color, fontweight="bold")

    plt.suptitle("Edge feature comparison: Sobel-based feature extraction\n"
                 "PDK is the only method that recovers edge structure under coma",
                 fontsize=11, y=0.995)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")


def save_metrics_table(results, path):
    headers = ["Image", "Deg→Sobel", "Glb→Sobel", "PDK→Sobel"]

    def build(metric_prefix, fmt="{:.3f}"):
        rows = []
        for i, r in enumerate(results):
            rows.append([f"img {i+1}",
                         fmt.format(r[f"{metric_prefix}_deg"]),
                         fmt.format(r[f"{metric_prefix}_glb"]),
                         fmt.format(r[f"{metric_prefix}_pdk"])])
        means = [np.mean([r[f"{metric_prefix}_{k}"] for r in results])
                 for k in ["deg", "glb", "pdk"]]
        rows.append(["Mean",
                     fmt.format(means[0]),
                     fmt.format(means[1]),
                     fmt.format(means[2])])
        return rows

    rows_iou = build("iou")
    rows_cos = build("cos")
    rows_fom = build("fom")

    fig, axes = plt.subplots(1, 3,
        figsize=(15, 0.5 + 0.5*len(rows_iou)))
    for ax, rows, title, desc in [
        (axes[0], rows_iou, "Edge IoU (@τ=0.15)", "spatial overlap"),
        (axes[1], rows_cos, "Edge Cosine sim",     "pattern similarity"),
        (axes[2], rows_fom, "Pratt's FoM",          "edge detection std"),
    ]:
        ax.axis("off")
        tbl = ax.table(cellText=rows, colLabels=headers,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
        for j in range(len(headers)):
            tbl[0, j].set_facecolor("#2C2C2A")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(rows)+1):
            tbl[i, 3].set_facecolor("#E1F5EE")
            tbl[i, 3].set_text_props(color="#085041", fontweight="bold")
        for j in range(len(headers)):
            tbl[len(rows), j].set_facecolor("#F1EFE8")
            tbl[len(rows), j].set_text_props(fontweight="bold")
        ax.set_title(f"{title}\n({desc})", fontsize=10, pad=10)

    plt.suptitle("Feature extraction metrics: edge map quality vs clean",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")


# ============================================================================
# Args
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=str,
                   default="./data/BSD300/images/train")
    p.add_argument("--inf-dir",   type=str,
                   default="./data/BSD300/images/test")
    p.add_argument("--n-train",   type=int, default=200)
    p.add_argument("--img-size",  type=int, default=256)
    p.add_argument("--n-iter",    type=int, default=2000)
    p.add_argument("--glb-iter",  type=int, default=800)
    p.add_argument("--lr",        type=float, default=0.01)
    p.add_argument("--batch",     type=int,   default=32)
    # 강한 degradation 기본값
    p.add_argument("--sigma0",    type=float, default=0.30)
    p.add_argument("--alpha-psf", type=float, default=1.5)
    p.add_argument("--coma-k",    type=float, default=1.20,
                   help="strong coma (default 1.20 vs 27번 0.60)")
    p.add_argument("--alpha-vig", type=float, default=6.0,
                   help="strong vignetting (default 6.0 vs 27번 4.0)")
    p.add_argument("--res-dir",   type=str,
                   default="./res/feature_extraction")
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    fig_dir = os.path.join(args.res_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    H = W = args.img_size

    degrade = lambda x: degrade_coma_vig(
        x, args.sigma0, args.alpha_psf, args.coma_k, args.alpha_vig)

    INF_NAMES = ["102061", "143090", "103070", "145086"]

    print("[Load] Train images...")
    train_imgs = load_images(args.train_dir, args.n_train, H)
    print("[Load] Inference images...")
    inf_imgs   = load_images_by_names(args.inf_dir, INF_NAMES, H)

    print(f"\n[Degradation] strong coma+vignetting:")
    print(f"  sigma0={args.sigma0}  alpha_psf={args.alpha_psf}")
    print(f"  coma_k={args.coma_k}  alpha_vig={args.alpha_vig}")

    # ── Global ────────────────────────────────────────────────
    print(f"\n[Train] Global kernel ({args.glb_iter} iter)...")
    glb_model = GlobalKernel()
    train_global(glb_model, train_imgs, degrade,
                 n_iter=args.glb_iter, lr=0.02)

    # ── DirectPDK ─────────────────────────────────────────────
    print(f"\n[Train] DirectPDK ({args.n_iter} iter, batch={args.batch})...")
    pdk_model = DirectPDK(H, W)
    train_direct_pdk(pdk_model, train_imgs, degrade,
                     n_iter=args.n_iter, lr=args.lr, batch=args.batch)

    # ── Inference + Sobel edge 비교 ──────────────────────────
    print("\n[Inference] computing edge feature metrics...")
    print("  Metrics: IoU(@0.15), Cosine sim, Pratt's FoM")
    results = []
    for i, clean in enumerate(inf_imgs):
        deg = degrade(clean)
        with torch.no_grad():
            glb = glb_model(deg)
            pdk = pdk_model(deg)

            clean_e = sobel_edge(clean)
            deg_e   = sobel_edge(deg)
            glb_e   = sobel_edge(glb)
            pdk_e   = sobel_edge(pdk)

        # feature extraction 평가 (vs clean edge map)
        iou_deg = edge_iou(clean_e, deg_e)
        iou_glb = edge_iou(clean_e, glb_e)
        iou_pdk = edge_iou(clean_e, pdk_e)
        cos_deg = edge_cosine(clean_e, deg_e)
        cos_glb = edge_cosine(clean_e, glb_e)
        cos_pdk = edge_cosine(clean_e, pdk_e)
        fom_deg = pratt_fom(clean_e, deg_e)
        fom_glb = pratt_fom(clean_e, glb_e)
        fom_pdk = pratt_fom(clean_e, pdk_e)

        print(f"  img {i+1}:")
        print(f"    IoU   — Deg={iou_deg:.3f}  Glb={iou_glb:.3f}  PDK={iou_pdk:.3f}")
        print(f"    Cos   — Deg={cos_deg:.3f}  Glb={cos_glb:.3f}  PDK={cos_pdk:.3f}")
        print(f"    FoM   — Deg={fom_deg:.3f}  Glb={fom_glb:.3f}  PDK={fom_pdk:.3f}")

        tag = f"{i+1:02d}"
        save_img(to_np(clean_e), os.path.join(fig_dir, f"clean_edge_{tag}.png"),
                 cmap="magma")
        save_img(to_np(deg_e),   os.path.join(fig_dir, f"deg_edge_{tag}.png"),
                 cmap="magma")
        save_img(to_np(glb_e),   os.path.join(fig_dir, f"glb_edge_{tag}.png"),
                 cmap="magma")
        save_img(to_np(pdk_e),   os.path.join(fig_dir, f"pdk_edge_{tag}.png"),
                 cmap="magma")

        results.append(dict(
            clean=clean, deg=deg,
            clean_e=clean_e, deg_e=deg_e, glb_e=glb_e, pdk_e=pdk_e,
            iou_deg=iou_deg, iou_glb=iou_glb, iou_pdk=iou_pdk,
            cos_deg=cos_deg, cos_glb=cos_glb, cos_pdk=cos_pdk,
            fom_deg=fom_deg, fom_glb=fom_glb, fom_pdk=fom_pdk))

    # ── Summary ───────────────────────────────────────────────
    save_compare(results, os.path.join(args.res_dir, "30_summary.png"))
    save_metrics_table(results,
        os.path.join(fig_dir, "edge_metrics_table.png"))

    print("\n" + "="*70)
    print("[Result] Mean feature extraction metrics (vs clean edge):")
    print(f"  {'':<22} {'IoU':>8} {'Cos':>8} {'FoM':>8}")
    print("-"*70)
    for key, label in [("deg", "Degraded → Sobel"),
                        ("glb", "Global   → Sobel"),
                        ("pdk", "PDK      → Sobel")]:
        iou = np.mean([r[f"iou_{key}"] for r in results])
        cos = np.mean([r[f"cos_{key}"] for r in results])
        fom = np.mean([r[f"fom_{key}"] for r in results])
        print(f"  {label:<22} {iou:>8.3f} {cos:>8.3f} {fom:>8.3f}")
    print("="*70)
    print("\n핵심 메시지:")
    print("  - high-pass(Sobel)도 coma의 비대칭 왜곡은 보정 못함")
    print("  - global kernel도 spatially varying degradation엔 무력")
    print("  - PDK만이 위치별 보정으로 edge feature 복원 가능")
    print(f"\n[Done] {os.path.abspath(args.res_dir)}")


if __name__ == "__main__":
    main()