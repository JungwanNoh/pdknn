"""
09_image_processing_compare.py
================================
이미지 처리 비교 시각화 (학습 없음 — PNG 저장만)

3가지 상황별 raw / degraded / global / pdk 비교:
  Case 1. 조도 불균일 (Vignetting)
  Case 2. 왜곡 (Barrel Distortion)
  Case 3. 조도 불균일 + 왜곡 (Combined)

출력 구조:
  ./res/image_processing/
    01_vignetting/
      comparison.png          ← raw | degraded | global | pdk (N행 비교)
      pdk_zones.png           ← PDK 구역 가중치 맵
      pdk_correction_map.png  ← 보정량 히트맵
    02_distortion/  (동일 구조)
    03_combined/    (동일 구조)

이미지 소스 (우선순위):
  1. --img-dir 지정 경로
  2. EuroSAT (clean RGB 위성 이미지)
  3. APTOS (fundus, 이미 vignetting 있음 — vignetting 실험엔 부적합)
  4. 합성 이미지 (체커보드, 그래디언트 패턴)

Run:
  python 09_image_processing_compare.py
  python 09_image_processing_compare.py --n-images 6 --img-size 128
  python 09_image_processing_compare.py --img-dir /path/to/images
"""

import argparse
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================================
# 1. 이미지 로드
# ============================================================================

def _load_pil_images(img_dir, n, img_size):
    """디렉토리(+하위 디렉토리)에서 PIL 이미지 로드 → (1,H,W) float tensor [0,1]."""
    exts = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG')
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(img_dir, '**', ext), recursive=True)
        paths += glob.glob(os.path.join(img_dir, ext))
    # 중복 제거 + 정렬
    paths = sorted(set(paths))[:n * 10]

    imgs = []
    for p in paths:
        try:
            img = Image.open(p).convert('L')        # grayscale

            # 중앙 정사각형 크롭 → 종횡비 유지
            w, h  = img.size
            s     = min(w, h)
            left  = (w - s) // 2
            top   = (h - s) // 2
            img   = img.crop((left, top, left + s, top + s))

            img = img.resize((img_size, img_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            imgs.append(torch.from_numpy(arr).unsqueeze(0))
        except Exception:
            continue
        if len(imgs) == n:
            break
    return imgs


def _make_synthetic_images(n, img_size):
    """
    합성 테스트 이미지 생성.
    다양한 패턴 포함: 체커보드, 방사형 그래디언트, 사인파, 엣지 패턴
    """
    imgs = []
    H = W = img_size
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')

    patterns = [
        # 체커보드 — 왜곡/샤프닝 효과 잘 보임
        (0.5 + 0.5 * torch.sign(torch.sin(gy * 8) * torch.sin(gx * 8))).clamp(0, 1),
        # 방사형 그래디언트 — vignetting 전후 대비 명확
        (1.0 - (gy**2 + gx**2).sqrt() / (2**0.5)).clamp(0, 1),
        # 사인파 격자
        (0.5 + 0.25 * torch.sin(gy * 12) + 0.25 * torch.sin(gx * 12)).clamp(0, 1),
        # 대각선 엣지
        ((gy + gx + 1) / 2).clamp(0, 1),
        # 동심원
        (0.5 + 0.5 * torch.cos((gy**2 + gx**2).sqrt() * 10)).clamp(0, 1),
        # 랜덤 노이즈 (시드 고정)
        torch.rand(H, W),
    ]

    for i, p in enumerate(patterns):
        if len(imgs) >= n:
            break
        imgs.append(p.unsqueeze(0))
    return imgs


def load_images(args):
    """
    이미지 로드. 우선순위: 지정 경로 → EuroSAT → APTOS → 합성.
    반환: List[(1,H,W)] float32 [0,1]
    """
    n  = args.n_images
    sz = args.img_size

    # 1) 사용자 지정 경로
    if args.img_dir:
        imgs = _load_pil_images(args.img_dir, n, sz)
        if imgs:
            print(f'[Load] {len(imgs)} images from {args.img_dir}')
            return imgs

    # 2) EuroSAT — clean RGB 위성 이미지 (vignetting/distortion 없음)
    eurosat_candidates = [
        './data/EuroSAT/2750',
        './data/eurosat/2750',
        './data/EuroSAT',
        './data/eurosat',
    ]
    # torchvision 캐시 경로도 탐색
    eurosat_candidates += glob.glob('./data/**/EuroSAT*', recursive=True)
    eurosat_candidates += glob.glob('./data/**/eurosat*', recursive=True)
    for d in sorted(set(eurosat_candidates)):
        if os.path.isdir(d):
            imgs = _load_pil_images(d, n, sz)
            if imgs:
                print(f'[Load] {len(imgs)} EuroSAT images from {d}')
                return imgs

    # 3) APTOS (fundus) — 이미 vignetting 있어 combined/distortion 실험에만 참고용
    aptos_dirs = [
        './data/aptos/train_images',
        './data/aptos/train_images/train_images',
    ]
    for d in aptos_dirs:
        if os.path.isdir(d):
            imgs = _load_pil_images(d, n, sz)
            if imgs:
                print(f'[Load] {len(imgs)} APTOS fundus images (주의: 이미 vignetting 있음)')
                return imgs

    # 4) 합성 이미지
    print('[Load] No real images found — using synthetic patterns.')
    imgs = _make_synthetic_images(n, sz)
    return imgs[:n]


# ============================================================================
# 2. 열화(Degradation) 함수
# ============================================================================

def apply_vignetting(x: torch.Tensor, alpha: float = 2.5) -> torch.Tensor:
    """
    Vignetting: I_obs = I_true × V(r),  V(r) = 1/(1 + α·r²)
    alpha가 클수록 가장자리가 더 어두워짐.
    x: (B,1,H,W) or (1,H,W)
    """
    batched = x.dim() == 4
    if not batched:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    device = x.device
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    r2  = gy**2 + gx**2
    V   = 1.0 / (1.0 + alpha * r2)            # vignetting factor (1=center, <1=edge)
    out = (x * V.unsqueeze(0).unsqueeze(0)).clamp(0, 1)
    return out if batched else out.squeeze(0)


def apply_barrel_distortion(x: torch.Tensor, k1: float = 0.4,
                             edge_blur_sigma: float = 2.5) -> torch.Tensor:
    """
    렌즈 왜곡 시뮬레이션 (Option C):
      1) 기하학적 배럴 왜곡:
         출력 픽셀 (gx,gy) 는 입력 (gx/(1+k1·r²), gy/(1+k1·r²)) 에서 샘플링
         → 중심 확대 + 가장자리 압축 효과
      2) 반경별 가변 블러 (실제 렌즈 수차 시뮬레이션):
         - 중심부: 블러 없음 (paraxial region)
         - 가장자리: 강한 블러 (field curvature, coma aberration)

    x: (B,1,H,W) or (1,H,W)
    """
    batched = x.dim() == 4
    if not batched:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    device = x.device

    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    r  = (gx**2 + gy**2).sqrt()               # (H,W)
    r2 = r ** 2

    # --- Step 1: 기하학적 배럴 왜곡 ---
    # 출력 (gx,gy) → 입력 (gx/(1+k1*r²), gy/(1+k1*r²)) 샘플링
    distortion = 1.0 + k1 * r2
    grid = torch.stack([gx / distortion,
                        gy / distortion], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    warped = F.grid_sample(x.float(), grid,
                           align_corners=True, padding_mode='border')  # (B,1,H,W)

    # --- Step 2: 반경별 가변 블러 ---
    r_norm = (r / (2**0.5)).clamp(0, 1).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    max_blur_size = 11
    blur_k = _gauss_kernel(max_blur_size, edge_blur_sigma, device)
    blurred = F.conv2d(warped, blur_k.expand(C, 1, max_blur_size, max_blur_size),
                       padding=max_blur_size // 2, groups=C)

    # center(r=0): warped 그대로  /  edge(r=1): blurred에 가까움
    blend_weight = r_norm ** 2
    out = (1 - blend_weight) * warped + blend_weight * blurred

    return out.clamp(0, 1) if batched else out.squeeze(0).clamp(0, 1)


def apply_combined(x: torch.Tensor,
                   alpha: float = 2.5, k1: float = 0.4) -> torch.Tensor:
    """Barrel distortion → Vignetting."""
    return apply_vignetting(apply_barrel_distortion(x, k1), alpha)


# ============================================================================
# 3. 전처리 함수
# ============================================================================

# --- Kernels ---
def _gauss_kernel(size, sigma, device):
    coords = torch.arange(size, device=device).float() - size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    k2d = g.unsqueeze(0) * g.unsqueeze(1)
    return k2d.view(1, 1, size, size)


def _apply_conv(x, k):
    """x: (B,1,H,W), k: (1,1,kH,kW)"""
    p = k.shape[-1] // 2
    return F.conv2d(x, k.to(x.device), padding=p)


def _inverse_barrel_grid(H, W, k1, device):
    """
    배럴 왜곡 역변환 샘플링 그리드 생성.

    Forward (apply_barrel_distortion):
        D(gx, gy) = original(gx/(1+k1·r²), gy/(1+k1·r²))

    Inverse (undistort):
        original(u, v) ≈ D(u·(1+k1·r²_u), v·(1+k1·r²_u))
        where r²_u = u²+v²  (출력 좌표 기준)

    즉, 출력 픽셀 (u,v) 는 왜곡 이미지에서 (u*(1+k1*r²), v*(1+k1*r²)) 위치를 샘플링.
    → 가장자리일수록 더 바깥쪽을 샘플링 → 안으로 당겨진 픽셀 복원.

    Returns: grid (H, W, 2) in [-1, 1] range
    """
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    r2 = gx**2 + gy**2
    inv_factor = 1.0 + k1 * r2   # ≥ 1: 가장자리일수록 바깥쪽 샘플링
    grid = torch.stack([gx * inv_factor,
                        gy * inv_factor], dim=-1)  # (H, W, 2)
    return grid.clamp(-1, 1)


def _apply_inverse_barrel(x: torch.Tensor, k1: float) -> torch.Tensor:
    """
    배럴 왜곡 역변환 적용.
    x: (B,1,H,W)
    Returns: (B,1,H,W) — 기하학적 왜곡 보정됨
    """
    B, C, H, W = x.shape
    grid = _inverse_barrel_grid(H, W, k1, x.device)  # (H,W,2)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)    # (B,H,W,2)
    return F.grid_sample(x.float(), grid,
                         align_corners=True, padding_mode='border').clamp(0, 1)


# --- Global ---
def global_vignetting_correct(x: torch.Tensor) -> torch.Tensor:
    """
    Global gamma (γ=0.55) — uniform brightening.
    Cannot distinguish center (over-exposed) from edge (under-exposed).
    """
    return x.clamp(1e-6, 1).pow(0.55)


def global_distortion_correct(x: torch.Tensor, k1: float = 0.4) -> torch.Tensor:
    """
    Global 왜곡 보정:
      Step 1: 역기하학적 변환 (배럴 왜곡 역변환) — 중심/가장자리 동일하게 적용
      Step 2: 균일한 unsharp masking (위치 무관, 중간 강도)

    문제:
      - 역기하학 변환 후에도 가장자리는 여전히 resampling blur 남아 있음
      - Global sharpening은 모든 영역에 동일한 강도를 적용
        → 중심(원래 선명): 과보정 → 노이즈 증폭, ringing artifact
        → 가장자리(여전히 흐릿): 과소보정 → blur 잔존
    PDK가 왜 필요한지 보여주는 baseline.
    """
    # Step 1: 역기하학 변환
    undistorted = _apply_inverse_barrel(x, k1)

    # Step 2: 균일 sharpening (중간 강도 — 가장자리에는 부족, 중심에는 과함)
    B, C, H, W = undistorted.shape
    blur = _apply_conv(undistorted, _gauss_kernel(11, 2.5, undistorted.device))
    sharpened = (undistorted + 0.8 * (undistorted - blur)).clamp(0, 1)
    return sharpened


def global_combined_correct(x: torch.Tensor, k1: float = 0.4) -> torch.Tensor:
    return global_distortion_correct(global_vignetting_correct(x), k1=k1)


# --- PDK ---
def _radial_map(H, W, device):
    """Returns r (H,W) normalised [0,1], gy, gx."""
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    r = (gy**2 + gx**2).sqrt() / (2**0.5)
    return r, gy, gx


def pdk_vignetting_correct(x: torch.Tensor):
    """
    위치별 vignetting 보정:
      I_corr(r) = I_obs(r) × (1 + α·r²)
    α는 이미지별로 중심 대비 평균 밝기 비율로 추정.

    Returns: (corrected, zone_mask_dict)
    """
    B, C, H, W = x.shape
    device = x.device
    r, _, _ = _radial_map(H, W, device)      # (H,W)
    r4d = r.unsqueeze(0).unsqueeze(0)         # (1,1,H,W)

    # Per-image α estimation
    center_mask = (r4d < 0.2).float()
    I_c   = (x * center_mask).sum([2,3]) / center_mask.sum().clamp(1)
    I_m   = x.mean([2,3])
    alpha = ((I_c / I_m.clamp(1e-5)) - 1).clamp(0.1, 4.0).view(B, C, 1, 1)

    # Zone masks
    inner_m  = torch.sigmoid(-(r4d - 0.30) / 0.04)
    outer_m  = torch.sigmoid( (r4d - 0.65) / 0.04)
    middle_m = (1 - inner_m - outer_m).clamp(0, 1)

    corr_i = 1.0 + 0.25 * alpha * r4d**2
    corr_m = 1.0 + 1.0  * alpha * r4d**2
    corr_o = 1.0 + 2.0  * alpha * r4d**2

    out = (inner_m * x * corr_i
         + middle_m * x * corr_m
         + outer_m  * x * corr_o).clamp(0, 1)

    zones = {
        'inner (low correction)'  : inner_m.squeeze().cpu().numpy(),
        'middle (std correction)' : middle_m.squeeze().cpu().numpy(),
        'outer (high correction)' : outer_m.squeeze().cpu().numpy(),
        'correction_strength'     : (inner_m*0.25 + middle_m*1.0 + outer_m*2.0
                                     ).squeeze().cpu().numpy(),
    }
    return out, zones


def pdk_distortion_correct(x: torch.Tensor, k1: float = 0.4):
    """
    PDK 왜곡 보정 — 2단계 위치 적응형 복원:

    Step 1: 역기하학 변환 (Global과 동일)
      - 배럴 왜곡 역변환: (u,v) → D(u*(1+k1*r²), v*(1+k1*r²)) 샘플링
      - 중심/가장자리 모두 기하학 복원

    Step 2: 위치별(PDK) sharpening — 여기서 Global과 차별화
      - Inner  r<0.30: 거의 no-op (원본이 선명했던 구역, 역변환으로 이미 충분)
      - Middle 0.30-0.65: 중간 강도 sharpening
      - Outer  r>0.65: 강력 sharpening
        (가장자리 edge blur = resampling + 렌즈 수차 → 집중 복원 필요)

    Global 대비 PDK 우위:
      Global: 역변환 후 균일 sharpening → 중심 과보정, 가장자리 과소보정
      PDK:    역변환 후 위치별 sharpening → 가장자리 집중 복원, 중심 보존

    Returns: (corrected, zone_mask_dict)
    """
    B, C, H, W = x.shape
    device = x.device

    # Step 1: 역기하학 변환 (모든 픽셀에 동일하게)
    undistorted = _apply_inverse_barrel(x, k1)

    # Step 2: 위치별 sharpening
    r, _, _ = _radial_map(H, W, device)
    r4d = r.unsqueeze(0).unsqueeze(0)

    inner_m  = torch.sigmoid(-(r4d - 0.30) / 0.04)
    outer_m  = torch.sigmoid( (r4d - 0.65) / 0.04)
    middle_m = (1 - inner_m - outer_m).clamp(0, 1)

    def _unsharp(img, blur_size, blur_sigma, strength):
        blur = _apply_conv(img, _gauss_kernel(blur_size, blur_sigma, device))
        return (img + strength * (img - blur)).clamp(0, 1)

    sharp_i = _unsharp(undistorted, 3,  0.8, 0.05)   # 중심: 거의 손대지 않음
    sharp_m = _unsharp(undistorted, 7,  2.0, 0.80)   # 중간: 적당한 복원
    sharp_o = _unsharp(undistorted, 11, 2.5, 2.50)   # 가장자리: 강력 복원

    out = (inner_m * sharp_i + middle_m * sharp_m + outer_m * sharp_o).clamp(0, 1)

    zones = {
        'inner (minimal sharpen)' : inner_m.squeeze().cpu().numpy(),
        'middle (mid sharpen)'    : middle_m.squeeze().cpu().numpy(),
        'outer (strong sharpen)'  : outer_m.squeeze().cpu().numpy(),
        'sharpen_strength'        : (inner_m*0.05 + middle_m*0.80 + outer_m*2.50
                                     ).squeeze().cpu().numpy(),
    }
    return out, zones


def pdk_combined_correct(x: torch.Tensor, k1: float = 0.4):
    """Barrel undistortion (PDK sharpening) → Vignetting correction."""
    x_sharp, zones_d = pdk_distortion_correct(x, k1=k1)
    x_corr,  zones_v = pdk_vignetting_correct(x_sharp)

    # Merge zone info
    zones = {}
    for k, v in zones_d.items():
        zones[f'dist_{k}'] = v
    for k, v in zones_v.items():
        zones[f'vig_{k}'] = v
    zones['combined_strength'] = zones_d.get('sharpen_strength',
                                              np.ones((x.shape[2], x.shape[3])))
    return x_corr, zones


# ============================================================================
# 4. 시각화 유틸
# ============================================================================

def _to_np(t: torch.Tensor) -> np.ndarray:
    """(1,H,W) or (H,W) → (H,W) numpy [0,1]."""
    return t.squeeze().detach().cpu().float().numpy()


def _save_comparison(res_dir: str, case_name: str,
                     images: list, img_size: int):
    """
    comparison.png 저장.

    images: List of dict with keys:
      'title': str
      'data': List[np.ndarray (H,W)]   — each element = one sample image
    """
    n_cols  = len(images)
    n_rows  = len(images[0]['data'])
    fig_w   = n_cols * 2.5
    fig_h   = n_rows * 2.5 + 0.8

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w, fig_h),
                              squeeze=False)

    for c, col in enumerate(images):
        axes[0, c].set_title(col['title'], fontsize=8, pad=4)
        for r in range(n_rows):
            ax  = axes[r, c]
            arr = col['data'][r]
            ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

    fig.suptitle(case_name, fontsize=10, y=1.01)
    plt.tight_layout(pad=0.5)
    path = os.path.join(res_dir, 'comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Saved] {path}')


def _save_zone_map(res_dir: str, zones: dict, img_size: int,
                   case_suffix: str = ''):
    """
    pdk_zones.png: 각 zone 마스크를 컬러맵으로 시각화.
    """
    zone_items = list(zones.items())
    n = len(zone_items)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3), squeeze=False)

    for i, (name, mask) in enumerate(zone_items):
        ax = axes[0, i]
        if mask.ndim == 3:
            mask = mask.squeeze()
        im = ax.imshow(mask, cmap='hot', vmin=0, vmax=mask.max() or 1)
        ax.set_title(name, fontsize=7)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'PDK Zone Weights{case_suffix}', fontsize=9)
    plt.tight_layout()
    path = os.path.join(res_dir, 'pdk_zones.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Saved] {path}')


def _save_correction_map(res_dir: str,
                          originals: list,
                          pdk_outputs: list):
    """
    pdk_correction_map.png: |PDK - degraded| 차이 히트맵.
    PDK가 얼마나, 어디를 보정했는지 시각화.
    """
    n = len(originals)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5), squeeze=False)

    for i in range(n):
        diff = np.abs(pdk_outputs[i] - originals[i])
        axes[0, i].imshow(pdk_outputs[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'PDK output #{i+1}', fontsize=7)
        axes[0, i].axis('off')
        im = axes[1, i].imshow(diff, cmap='hot', vmin=0, vmax=diff.max() or 0.1)
        axes[1, i].set_title(f'|PDK - degraded|', fontsize=7)
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

    fig.suptitle('PDK Correction Magnitude\n'
                 '(bright = large correction applied)', fontsize=9)
    plt.tight_layout()
    path = os.path.join(res_dir, 'pdk_correction_map.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Saved] {path}')


# ============================================================================
# 5. 케이스별 실행
# ============================================================================

def run_case(case_id: int, case_name: str,
             raw_imgs: list,
             degrade_fn,
             global_fn,
             pdk_fn,
             res_root: str):
    """
    공통 실행 로직.
    raw_imgs: List[(1,H,W) tensor]
    global_fn: x -> corrected (no zone return)
    pdk_fn:    x -> (corrected, zones)
    """
    case_dir = os.path.join(res_root, f'{case_id:02d}_{case_name}')
    os.makedirs(case_dir, exist_ok=True)
    print(f'\n[Case {case_id}] {case_name}  →  {case_dir}')

    raw_np, deg_np, glo_np, pdk_np = [], [], [], []
    last_zones = {}

    for img in raw_imgs:
        img_b = img.unsqueeze(0)             # (1,1,H,W)

        deg = degrade_fn(img_b)              # (1,1,H,W)
        glo = global_fn(deg).clamp(0, 1)    # (1,1,H,W)
        pdk_out, zones = pdk_fn(deg)         # (1,1,H,W), dict

        raw_np.append(_to_np(img))
        deg_np.append(_to_np(deg))
        glo_np.append(_to_np(glo))
        pdk_np.append(_to_np(pdk_out))
        last_zones = zones

    # comparison.png
    _save_comparison(
        case_dir, case_name,
        images=[
            {'title': 'Original (raw)',      'data': raw_np},
            {'title': 'Degraded',            'data': deg_np},
            {'title': 'Global correction',   'data': glo_np},
            {'title': 'PDK correction',      'data': pdk_np},
        ],
        img_size=raw_imgs[0].shape[-1],
    )

    # pdk_zones.png
    _save_zone_map(case_dir, last_zones, raw_imgs[0].shape[-1],
                   case_suffix=f' ({case_name})')

    # pdk_correction_map.png  (PDK output vs degraded — 보정량)
    _save_correction_map(case_dir, deg_np, pdk_np)

    print(f'  [Done] Case {case_id}: {len(raw_imgs)} images processed.')


# ============================================================================
# 6. Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='이미지 처리 비교 시각화: vignetting / distortion / combined')
    p.add_argument('--img-dir',    type=str,  default='',
                   help='이미지 폴더 경로 (비워두면 자동 탐색)')
    p.add_argument('--n-images',   type=int,  default=4,
                   help='비교에 사용할 이미지 수 (default: 4)')
    p.add_argument('--img-size',   type=int,  default=128,
                   help='리사이즈 크기 (default: 128)')
    p.add_argument('--res-dir',    type=str,  default='./res/image_processing',
                   help='결과 저장 폴더')
    p.add_argument('--vignetting-alpha', type=float, default=2.5,
                   help='Vignetting 강도 (default: 2.5)')
    p.add_argument('--barrel-k1',        type=float, default=0.4,
                   help='Barrel distortion 계수 (default: 0.4)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.res_dir, exist_ok=True)

    # 이미지 로드
    imgs = load_images(args)
    if not imgs:
        print('ERROR: No images loaded.')
        return
    print(f'[Load] {len(imgs)} images, size={args.img_size}×{args.img_size}')

    alpha = args.vignetting_alpha
    k1    = args.barrel_k1

    # ── Case 1: 조도 불균일 ──
    run_case(
        case_id=1,
        case_name='vignetting',
        raw_imgs=imgs,
        degrade_fn=lambda x: apply_vignetting(x, alpha=alpha),
        global_fn=lambda x: global_vignetting_correct(x),
        pdk_fn=pdk_vignetting_correct,
        res_root=args.res_dir,
    )

    # ── Case 2: 왜곡 ──
    run_case(
        case_id=2,
        case_name='distortion',
        raw_imgs=imgs,
        degrade_fn=lambda x: apply_barrel_distortion(x, k1=k1),
        global_fn=lambda x: global_distortion_correct(x, k1=k1),
        pdk_fn=lambda x: pdk_distortion_correct(x, k1=k1),
        res_root=args.res_dir,
    )

    # ── Case 3: 조도 불균일 + 왜곡 ──
    run_case(
        case_id=3,
        case_name='combined',
        raw_imgs=imgs,
        degrade_fn=lambda x: apply_combined(x, alpha=alpha, k1=k1),
        global_fn=lambda x: global_combined_correct(x, k1=k1),
        pdk_fn=lambda x: pdk_combined_correct(x, k1=k1),
        res_root=args.res_dir,
    )

    print(f'\n모든 결과 저장 완료: {os.path.abspath(args.res_dir)}')
    print('\n출력 파일:')
    for case_dir in ['01_vignetting', '02_distortion', '03_combined']:
        print(f'  {args.res_dir}/{case_dir}/')
        print(f'    comparison.png        — raw | degraded | global | pdk')
        print(f'    pdk_zones.png         — PDK 구역 가중치 맵')
        print(f'    pdk_correction_map.png — 보정량 히트맵')


if __name__ == '__main__':
    main()
