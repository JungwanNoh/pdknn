"""
31_localized_degradation_demo.py
=====================================================
설명용 위치별 degradation + 보정 비교.

degradation 배치 (의도적으로 직관적):
  - 왼쪽 위 모서리: 강한 coma (asymmetric blur)
  - 중앙 영역:      약한 coma
  - 오른쪽 아래:    vignetting (어두워짐)
  - 그 외:          변형 없음

비교: clean / degraded / Global / PDK 4가지 저장

Run:
  python 31_localized_degradation_demo.py
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


def make_localized_kernels(H, W):
    """
    위치별 광학 degradation 커널 생성.
      - 좌상~중앙(파란 영역)  : Coma (asymmetric blur)
      - 우상(빨간 영역)         : Spherical aberration (등방 큰 blur)
      - 하중앙(하늘색 영역)     : Vignetting only (kernel은 identity)

    영역 경계는 sigmoid로 부드럽게 처리하여 픽셀 jump 방지
    """
    coords = torch.tensor([
        [-1.,-1.],[-1.,0.],[-1.,1.],
        [ 0.,-1.],[ 0.,0.],[ 0.,1.],
        [ 1.,-1.],[ 1.,0.],[ 1.,1.]])

    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    def smooth_step(x, edge=0.0, sharpness=15.0):
        """sigmoid based smooth mask: x < edge → 0, x > edge → 1"""
        return torch.sigmoid(-(x - edge) * sharpness)  # x < edge에서 1

    def smooth_step_pos(x, edge=0.0, sharpness=15.0):
        return torch.sigmoid((x - edge) * sharpness)   # x > edge에서 1

    # 부드러운 마스크 (경계에서 0→1 점진 변화)
    coma_mask  = smooth_step(gy, 0.2) * smooth_step(gx, 0.2)
    spher_mask = smooth_step(gy, 0.0) * smooth_step_pos(gx, 0.05)

    # 우상에서 spherical 우선 (coma와 겹치는 영역에서 spherical만 적용)
    coma_mask = coma_mask * (1 - spher_mask)

    # ── Coma: 좌상 방향 PSF shift ──────────────────────
    coma_shift_max = 0.95
    shift_y = -coma_mask * coma_shift_max
    shift_x = -coma_mask * coma_shift_max

    # ── Sigma map ──────────────────────────────────────
    sigma_map = torch.full((H, W), 0.4)
    sigma_map = sigma_map + coma_mask * 0.6
    sigma_map = sigma_map + spher_mask * 1.0

    # ── Kernel 계산 ──────────────────────────────────────
    sigma = sigma_map.unsqueeze(-1).clamp(0.1, 1.5)
    sy = shift_y.unsqueeze(-1)
    sx = shift_x.unsqueeze(-1)
    dy = coords[:, 0].unsqueeze(0).unsqueeze(0) - sy
    dx = coords[:, 1].unsqueeze(0).unsqueeze(0) - sx
    g  = torch.exp(-(dy**2 + dx**2) / (2 * sigma**2))
    g  = g / g.sum(-1, keepdim=True)

    return g


def make_vignetting_map(H, W):
    """
    전역 radial vignetting.
    이미지 중심은 그대로, 가장자리(corners)로 갈수록 어두워짐.
    """
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    r = (gy**2 + gx**2).sqrt() / (2**0.5)   # 0(center) ~ 1(corner)
    # alpha 강도 조정 (값 클수록 어두움)
    alpha = 4.0
    V = 1.0 / (1.0 + alpha * r**2)
    return V


def degrade_localized(x):
    H, W = x.shape[-2:]
    kernels = make_localized_kernels(H, W)
    out = spatially_varying_conv(x, kernels)
    V = make_vignetting_map(H, W).unsqueeze(0).unsqueeze(0)
    return (out * V).clamp(0, 1)


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


def load_one_image(path, size):
    img = Image.open(path).convert("L")
    w, h = img.size; s = min(w, h)
    img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def load_train_images(data_dir, n, size):
    exts = ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG")
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(data_dir,"**",ext), recursive=True)
        paths += glob.glob(os.path.join(data_dir, ext))
    paths = sorted(set(paths)); random.shuffle(paths)
    imgs = []
    for p in paths:
        try:
            imgs.append(load_one_image(p, size))
        except: continue
        if len(imgs) == n: break
    print(f"  [Load] {len(imgs)} train images")
    return imgs


def find_image(data_dir, name):
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    for ext in exts:
        c = os.path.join(data_dir, name + ext)
        if os.path.exists(c): return c
        hits = glob.glob(os.path.join(data_dir, "**", name + ext),
                         recursive=True)
        if hits: return hits[0]
    return None


def train_global(model, train_imgs, degrade_fn, n_iter, lr):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    pairs = [(img, degrade_fn(img)) for img in train_imgs]
    for i in range(n_iter):
        opt.zero_grad()
        total = sum(loss_fn(model(deg), clean) for clean, deg in pairs)
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


def save_region_overlay(img_np, path, dpi=600):
    """
    이미지 위에 degradation 영역을 색상 오버레이로 표시.
    (보통 degraded 이미지에 그려서 어떤 영역에 어떤 효과가 들어갔는지 시각화)
      - 좌상~중앙 (파란):  Coma
      - 우상 (빨간):       Spherical aberration
      - 가장자리 (하늘):   Vignetting (radial, 전역)
    """
    H, W = img_np.shape
    fig, ax = plt.subplots(1, 1,
        figsize=(W/100, H/100), dpi=dpi)
    ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)

    ys = np.linspace(-1, 1, H)
    xs = np.linspace(-1, 1, W)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    r = np.sqrt(gy**2 + gx**2) / np.sqrt(2)

    # 영역 마스크 (degrade와 동일 binary 근사)
    coma_mask  = (gy < 0.2) & (gx < 0.2)
    spher_mask = (gy < 0.0) & (gx > 0.05)
    coma_only  = coma_mask & ~spher_mask
    # vignetting은 가장자리 ring으로 표시 (r > 0.6 영역)
    vig_ring   = r > 0.65

    overlay = np.zeros((H, W, 4))
    # Vignetting 먼저 (배경처럼) — alpha를 거리에 따라 점진
    vig_alpha = np.clip((r - 0.6) / 0.4, 0, 1) * 0.5  # 0~0.5 점진
    overlay[..., 0] = np.where(vig_ring, 0.40, 0)
    overlay[..., 1] = np.where(vig_ring, 0.85, 0)
    overlay[..., 2] = np.where(vig_ring, 1.0,  0)
    overlay[..., 3] = np.where(vig_ring, vig_alpha, 0)

    # Coma (덮어쓰기)
    overlay[coma_only]  = [0.25, 0.55, 1.0, 0.32]
    # Spherical (덮어쓰기)
    overlay[spher_mask] = [1.0, 0.30, 0.30, 0.32]

    ax.imshow(overlay)

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_img(arr, path, dpi=300):
    arr = np.clip(arr, 0, 1)
    fig, ax = plt.subplots(1, 1,
        figsize=(arr.shape[1]/100, arr.shape[0]/100), dpi=dpi)
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_summary(arrs_named, name, path, psnrs):
    titles = list(arrs_named.keys())
    n = len(titles)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    for i, t in enumerate(titles):
        axes[i].imshow(np.clip(arrs_named[t], 0, 1),
                       cmap="gray", vmin=0, vmax=1)
        axes[i].axis("off")
        axes[i].set_title(t, fontsize=12, fontweight="bold", pad=6)
        if t in psnrs:
            axes[i].text(0.5, -0.04,
                f"PSNR={psnrs[t]:.2f} dB",
                transform=axes[i].transAxes,
                ha="center", fontsize=10,
                color="#085041" if t == "PDK" else "dimgray",
                fontweight="bold")
    plt.suptitle(f"Localized degradation correction  ({name})",
                 fontsize=13, y=1.0)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=str,
                   default="./data/BSD300/images/train")
    p.add_argument("--inf-dir",   type=str,
                   default="./data/BSD300/images/test")
    p.add_argument("--name",      type=str, default="102061")
    p.add_argument("--img-size",  type=int, default=256)
    p.add_argument("--n-train",   type=int, default=200)
    p.add_argument("--n-iter",    type=int, default=2000)
    p.add_argument("--glb-iter",  type=int, default=800)
    p.add_argument("--lr",        type=float, default=0.01)
    p.add_argument("--batch",     type=int, default=32)
    p.add_argument("--res-dir",   type=str,
                   default="./res/localized_demo")
    p.add_argument("--dpi",       type=int, default=600)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.res_dir, exist_ok=True)
    H = W = args.img_size

    src = find_image(args.inf_dir, args.name)
    if src is None:
        print(f"[Error] {args.name} not found in {args.inf_dir}")
        return
    print(f"[Inference target] {src}")

    print("\n[Load] Train images...")
    train_imgs = load_train_images(args.train_dir, args.n_train, H)

    print(f"\n[Train] Global kernel ({args.glb_iter} iter)...")
    glb_model = GlobalKernel()
    train_global(glb_model, train_imgs, degrade_localized,
                 n_iter=args.glb_iter, lr=0.02)

    print(f"\n[Train] DirectPDK ({args.n_iter} iter, batch={args.batch})...")
    pdk_model = DirectPDK(H, W)
    train_direct_pdk(pdk_model, train_imgs, degrade_localized,
                     n_iter=args.n_iter, lr=args.lr, batch=args.batch)

    print("\n[Inference]")
    clean = load_one_image(src, H)
    deg   = degrade_localized(clean)
    with torch.no_grad():
        glb = glb_model(deg)
        pdk = pdk_model(deg)

    p_deg = psnr(clean, deg)
    p_glb = psnr(clean, glb)
    p_pdk = psnr(clean, pdk)
    print(f"  Degraded: {p_deg:.2f} dB")
    print(f"  Global:   {p_glb:.2f} dB")
    print(f"  PDK:      {p_pdk:.2f} dB")

    clean_np = clean.squeeze().numpy()
    deg_np   = deg.squeeze().numpy()
    glb_np   = glb.squeeze().detach().numpy()
    pdk_np   = pdk.squeeze().detach().numpy()

    base = args.name
    out_clean = os.path.join(args.res_dir, f"{base}_clean.png")
    out_deg   = os.path.join(args.res_dir, f"{base}_degraded.png")
    out_reg   = os.path.join(args.res_dir, f"{base}_degraded_region.png")
    out_glb   = os.path.join(args.res_dir, f"{base}_global.png")
    out_pdk   = os.path.join(args.res_dir, f"{base}_pdk.png")
    save_img(clean_np, out_clean, dpi=args.dpi)
    save_img(deg_np,   out_deg,   dpi=args.dpi)
    save_region_overlay(deg_np, out_reg, dpi=args.dpi)
    save_img(glb_np,   out_glb,   dpi=args.dpi)
    save_img(pdk_np,   out_pdk,   dpi=args.dpi)

    print(f"\n[Saved] individual:")
    for p in [out_clean, out_deg, out_reg, out_glb, out_pdk]:
        print(f"  {p}")

    arrs = {
        "Clean":    clean_np,
        "Degraded": deg_np,
        "Global":   glb_np,
        "PDK":      pdk_np,
    }
    psnrs = {"Degraded": p_deg, "Global": p_glb, "PDK": p_pdk}
    save_summary(arrs, base,
                 os.path.join(args.res_dir, f"{base}_compare.png"),
                 psnrs)

    print("\nDegradation 배치:")
    print("  - 왼쪽 위 모서리: 강한 coma (asymmetric blur)")
    print("  - 중앙 영역:      약한 coma")
    print("  - 오른쪽 아래:    vignetting (어두워짐)")


if __name__ == "__main__":
    main()