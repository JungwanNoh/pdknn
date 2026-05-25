"""
08_aptos.py
===========
APTOS 2019 Diabetic Retinopathy — Fundus Vignetting PDK Correction

WHY fundus images need PDK preprocessing
-----------------------------------------
Fundus cameras illuminate the retina through a narrow pupil aperture.
This produces RADIAL VIGNETTING: the center of the image receives more light
than the periphery, creating a systematic brightness gradient.

  Center (r < 0.3) : over-bright, possibly saturated optic disc
  Middle (0.3-0.65): reasonably exposed retinal vessels
  Outer  (r > 0.65): under-exposed — microaneurysms, hemorrhages,
                      neovascularization are systematically dimmed here

Clinical consequence: peripheral DR lesions (which appear at the outer
retina) are systematically MISSED or under-weighted because the darkened
periphery suppresses their contrast.

WHY global preprocessing fails
--------------------------------
  Global gamma/CLAHE: brightens ALL pixels by the same function.
    → Over-brightens the center (already saturated optic disc)
    → Under-corrects the periphery (needs MORE correction)
    → Cannot know WHICH pixels are peripheral without position info

  PDK: knows r for every pixel → applies STRONGER correction at larger r
    → Center: gentle normalization (avoid over-saturation)
    → Periphery: aggressive brightness recovery + contrast boost

This correction is MATHEMATICALLY IMPOSSIBLE without position information.
PDK (position-dependent kernel) is the minimum requirement.

PDK vignetting model
---------------------
  Observed: I_obs(r) = I_true(r) × V(r)
  Vignetting: V(r) = 1 / (1 + α·r²)       (cos⁴-law approximation)
  Correction: I_corr(r) = I_obs(r) / V(r) = I_obs(r) × (1 + α·r²)

  α is estimated per-image from the ratio of center vs. mean brightness.

Dataset: APTOS 2019 Blindness Detection (Kaggle)
  - 3,662 retinal fundus images
  - Labels: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR
  - Natural vignetting from fundus camera optics

Download (auto-handled by this script):
  kaggle competitions download -c aptos2019-blindness-detection -p ./data/aptos
  OR manual: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

Run:
  python 08_aptos.py --all --epochs 20
  python 08_aptos.py --preprocess fundus_pdk --model KANHyperPDKNN --epochs 20
"""

import os
import csv
import sys
import argparse
import zipfile
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    raise ImportError("pip install Pillow")

try:
    import pandas as pd
except ImportError:
    raise ImportError("pip install pandas")


# ===========================================================================
# 0. Dataset download
# ===========================================================================

DATA_ROOT  = './data/aptos'
CSV_PATH   = os.path.join(DATA_ROOT, 'train.csv')
IMAGES_DIR = os.path.join(DATA_ROOT, 'train_images')   # updated in main()


def _find_images_dir():
    """Auto-detect train_images folder (handles nested extraction)."""
    candidates = [
        os.path.join(DATA_ROOT, 'train_images'),
        os.path.join(DATA_ROOT, 'train_images', 'train_images'),
        os.path.join(DATA_ROOT, 'train_images', 'images'),
        DATA_ROOT,
    ]
    for p in candidates:
        if os.path.isdir(p):
            files = [f for f in os.listdir(p)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                print(f'[Data] Images found at: {p}  ({len(files)} files)')
                return p
    raise FileNotFoundError(
        f'No image files found in any of: {candidates}\n'
        f'Please check your extraction path.'
    )


def _try_kaggle_download():
    """Try to download via Kaggle API. Returns True on success."""
    try:
        result = subprocess.run(
            ['kaggle', 'competitions', 'download',
             '-c', 'aptos2019-blindness-detection',
             '-p', DATA_ROOT],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f'  kaggle CLI error: {result.stderr.strip()}')
            return False

        # Unzip
        zip_path = os.path.join(DATA_ROOT,
                                'aptos2019-blindness-detection.zip')
        if os.path.exists(zip_path):
            print('  Extracting...')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATA_ROOT)
            os.remove(zip_path)
        return True

    except FileNotFoundError:
        print('  kaggle CLI not found.')
        return False
    except subprocess.TimeoutExpired:
        print('  kaggle download timed out.')
        return False


def ensure_data():
    """
    Download APTOS 2019 if not already present.
    Priority: kaggle CLI → manual instructions.
    """
    if os.path.isdir(IMAGES_DIR) and os.path.isfile(CSV_PATH):
        n = len(os.listdir(IMAGES_DIR))
        print(f'[Data] APTOS found: {n} images in {IMAGES_DIR}')
        return

    os.makedirs(DATA_ROOT, exist_ok=True)
    print('[Data] APTOS 2019 not found. Attempting auto-download...')

    # --- Try kaggle API ---
    if _try_kaggle_download():
        print('[Data] Download complete.')
        return

    # --- Kaggle API not available: try pip install ---
    print('[Data] Trying to install kaggle package...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle', '-q'])

    print('\n' + '=' * 65)
    print('MANUAL DOWNLOAD REQUIRED')
    print('=' * 65)
    print('1. Go to: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data')
    print('2. Accept competition rules and download:')
    print('     train_images.zip')
    print('     train.csv')
    print(f'3. Extract to: {os.path.abspath(DATA_ROOT)}')
    print('   Expected structure:')
    print(f'     {DATA_ROOT}/train.csv')
    print(f'     {DATA_ROOT}/train_images/0a09aa7356c0.png  ...')
    print()
    print('OR set up Kaggle API:')
    print('  https://www.kaggle.com/docs/api')
    print('  Then re-run this script.')
    print('=' * 65)
    sys.exit(1)


# ===========================================================================
# 1. Dataset
# ===========================================================================

class APTOSDataset(Dataset):
    """
    APTOS 2019 retinal fundus images.
    Labels: 0-4 (DR severity: No DR → Proliferative DR).
    Input: grayscale 64×64, 1 channel (hardware constraint).
    """
    def __init__(self, df, img_dir, transform=None):
        self.img_dir   = img_dir
        self.transform = transform

        # Build id_code → actual filepath map at init time
        # (handles .png / .jpg / .jpeg regardless of extraction method)
        available = {}
        for fname in os.listdir(img_dir):
            stem, ext = os.path.splitext(fname)
            if ext.lower() in ('.png', '.jpg', '.jpeg'):
                available[stem] = os.path.join(img_dir, fname)

        # Filter df to only rows with a matching image file
        mask = df['id_code'].isin(available)
        missing = (~mask).sum()
        if missing > 0:
            print(f'  [Dataset] WARNING: {missing} id_codes have no matching '
                  f'image file and will be skipped.')
        self.df       = df[mask].reset_index(drop=True)
        self.img_map  = available
        print(f'  [Dataset] {len(self.df)} images loaded from {img_dir}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.img_map[row['id_code']]
        img      = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row['diagnosis'])
        return img, label


def get_dataloaders(batch_size=64, num_workers=2, img_size=64):
    df        = pd.read_csv(CSV_PATH)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # Normalize to [0,1] range — vignetting correction works on [0,1]
        # Final normalization happens AFTER preprocessing in the train loop
    ])

    full_ds  = APTOSDataset(df, IMAGES_DIR, transform)
    n_total  = len(full_ds)
    n_train  = int(n_total * 0.8)
    n_val    = n_total - n_train
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    print(f'APTOS  Train: {n_train} | Val: {n_val}')
    return train_loader, val_loader


# ===========================================================================
# 2. Preprocessing
# ===========================================================================

def _normalize_batch(x):
    """Normalize to zero-mean unit-std after preprocessing."""
    mean = x.mean(dim=[2, 3], keepdim=True)
    std  = x.std(dim=[2, 3], keepdim=True).clamp(min=1e-5)
    return (x - mean) / std


def raw_preprocess(x):
    """Identity — just normalize."""
    return _normalize_batch(x)


def global_preprocess(x):
    """
    Global gamma correction (γ=0.6) to brighten the image uniformly.

    This is what a POSITION-AGNOSTIC approach does:
      - Brightens center (already well-exposed) — causes over-saturation
      - Brightens edge (under-exposed) — helps but same amount as center

    A global approach CANNOT know which pixels need more correction.
    """
    x_gamma = x.clamp(min=1e-6).pow(0.6)   # γ < 1 → brighten
    return _normalize_batch(x_gamma)


def fundus_pdk_preprocess(x):
    """
    Position-Dependent Kernel vignetting correction for fundus images.

    Vignetting model (cos⁴-law approximation):
        I_obs(r) = I_true(r) × 1/(1 + α·r²)

    Correction:
        I_corr(r) = I_obs(r) × (1 + α·r²)

    α is estimated PER IMAGE from the brightness ratio:
        α = (I_center / I_mean - 1) / r_mean²

    Zone-specific application:
        Inner  r < 0.30 : mild correction (α_eff = 0.3α) — avoid over-saturation
        Middle 0.30-0.65: standard correction (α_eff = α)
        Outer  r > 0.65 : full correction + contrast boost (α_eff = 1.5α)

    This correction is IMPOSSIBLE without knowing r for each pixel.
    PDK provides exactly this: different operation per spatial location.
    """
    B, C, H, W = x.shape
    device     = x.device

    # --- Radial distance map ---
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    r  = (gy**2 + gx**2).sqrt()           # [0, sqrt(2)]
    r  = (r / r.max()).unsqueeze(0).unsqueeze(0)   # (1,1,H,W) ∈ [0,1]

    # --- Per-image α estimation ---
    center_mask = (r < 0.2).float()
    I_center = (x * center_mask).sum(dim=[2,3]) / center_mask.sum().clamp(1)
    I_mean   = x.mean(dim=[2,3])
    # α: how much darker the mean is compared to center
    alpha = ((I_center / I_mean.clamp(min=1e-5)) - 1.0).clamp(min=0.1, max=3.0)
    alpha = alpha.view(B, C, 1, 1)         # (B,C,1,1)

    # --- Vignetting correction factor per position ---
    correction = 1.0 + alpha * r**2        # (B,C,H,W) — stronger at edge

    # --- Zone masks (soft sigmoid) ---
    inner_mask  = torch.sigmoid(-(r - 0.30) / 0.04)           # r < 0.30
    outer_mask  = torch.sigmoid( (r - 0.65) / 0.04)           # r > 0.65
    middle_mask = 1.0 - inner_mask - outer_mask

    # Zone-specific correction strength
    corr_inner  = 1.0 + 0.3 * alpha * r**2   # gentle
    corr_middle = correction                   # standard
    corr_outer  = 1.0 + 1.5 * alpha * r**2   # aggressive

    x_corr = (inner_mask  * x * corr_inner
            + middle_mask * x * corr_middle
            + outer_mask  * x * corr_outer).clamp(0.0, 1.0)

    return _normalize_batch(x_corr)


PREPROCESS_FNS = {
    'raw'       : raw_preprocess,
    'global'    : global_preprocess,
    'fundus_pdk': fundus_pdk_preprocess,
}


# ===========================================================================
# 3. KAN / KANHyperPDKNN / BaselineCNN
# ===========================================================================

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 grid_size=5, spline_order=3, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.spline_order = spline_order
        self.n_basis      = grid_size + spline_order
        h    = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)
        self.spline_weight  = nn.Parameter(
            torch.randn(in_features, out_features, self.n_basis) * 0.1)
        self.residual_scale = nn.Parameter(
            torch.ones(in_features, out_features) * 0.1)

    def b_splines(self, x):
        x     = x.unsqueeze(-1)
        grid  = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            dl = grid[k:-1]   - grid[:-(k+1)]
            dr = grid[k+1:]   - grid[1:-k]
            safe = lambda n, d: n / d.clamp(min=1e-8)
            left  = safe(x - grid[:-(k+1)], dl) * bases[..., :-1]
            right = safe(grid[k+1:] - x,    dr) * bases[..., 1:]
            bases = left + right
        return bases

    def forward(self, x):
        bs  = self.b_splines(x)
        return (torch.einsum('bin,ion->bo', bs, self.spline_weight)
              + torch.einsum('bi,io->bo', F.silu(x), self.residual_scale))


class KAN(nn.Module):
    def __init__(self, dims, grid_size=5, spline_order=3):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLayer(dims[i], dims[i+1], grid_size, spline_order)
            for i in range(len(dims)-1)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class KANHyperPDKConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 input_height, input_width,
                 stride=1, padding=0, kan_hidden=32, kan_latent=32,
                 grid_size=5, spline_order=3):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        kH = kW = kernel_size
        self.H_out = (input_height + 2*padding - kH) // stride + 1
        self.W_out = (input_width  + 2*padding - kW) // stride + 1
        self.kernel_flat  = out_channels * in_channels * kH * kW
        self.kan            = KAN([2, kan_hidden, kan_latent], grid_size, spline_order)
        self.kernel_decoder = nn.Linear(kan_latent, self.kernel_flat)
        self.bias_decoder   = nn.Linear(kan_latent, out_channels)
        ys = torch.linspace(-1, 1, self.H_out)
        xs = torch.linspace(-1, 1, self.W_out)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('coords',
            torch.stack([gy.flatten(), gx.flatten()], dim=-1))

    def _generate_kernels(self):
        latent  = self.kan(self.coords)
        kernels = self.kernel_decoder(latent).view(
            self.H_out * self.W_out, self.out_channels, -1)
        biases  = self.bias_decoder(latent)
        return kernels, biases

    def forward(self, x, pos_chunk=256):
        B               = x.size(0)
        kH = kW         = self.kernel_size
        P               = self.H_out * self.W_out
        kernels, biases = self._generate_kernels()

        patches   = F.unfold(x, kernel_size=kH,
                             padding=self.padding, stride=self.stride)
        patches   = patches.permute(0, 2, 1)     # (B, P, kflat)
        kernels_t = kernels.permute(0, 2, 1)     # (P, kflat, out_ch)

        chunks = []
        for s in range(0, P, pos_chunk):
            e = min(s + pos_chunk, P)
            chunks.append(torch.einsum('bpk,pko->bpo',
                                       patches[:, s:e],
                                       kernels_t[s:e]))
        out = torch.cat(chunks, dim=1)            # (B, P, out_ch)
        return (out + biases.unsqueeze(0)).permute(0, 2, 1).view(
            B, self.out_channels, self.H_out, self.W_out)


class KANHyperPDKNN(nn.Module):
    def __init__(self, num_classes=5, img_h=64, img_w=64,
                 in_channels=1, kan_hidden=32, kan_latent=32):
        super().__init__()
        self.block1 = KANHyperPDKConv2d(
            in_channels, 32, 3, img_h, img_w, padding=1,
            kan_hidden=kan_hidden, kan_latent=kan_latent)
        self.bn1 = nn.BatchNorm2d(32)
        self.block2 = KANHyperPDKConv2d(
            32, 64, 3, img_h//2, img_w//2, padding=1,
            kan_hidden=kan_hidden, kan_latent=kan_latent)
        self.bn2 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.block1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.block2(x))), 2)
        return self.fc(self.gap(x).flatten(1))


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=5, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        return self.fc(self.gap(x).flatten(1))


def build_model(model_name, device, num_classes=5, img_size=64):
    if model_name == 'KANHyperPDKNN':
        m = KANHyperPDKNN(num_classes=num_classes,
                           img_h=img_size, img_w=img_size, in_channels=1)
    else:
        m = BaselineCNN(num_classes=num_classes, in_channels=1)
    return m.to(device)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# ===========================================================================
# 4. Train / Eval
# ===========================================================================

def train_one_epoch(model, loader, optimizer, criterion,
                    device, preprocess_fn):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs = preprocess_fn(imgs)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, preprocess_fn):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs = preprocess_fn(imgs)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


# ===========================================================================
# 5. Single run
# ===========================================================================

def run_one(preprocess, model_name, args, device, res_dir):
    print(f'\n{"=" * 62}')
    print(f'  Model      : {model_name}')
    print(f'  Preprocess : {preprocess}')
    print(f'{"=" * 62}')

    model = build_model(model_name, device, num_classes=5,
                        img_size=args.img_size)
    print(f'  Params: {count_params(model):,}')

    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    preprocess_fn = PREPROCESS_FNS[preprocess]

    csv_path = os.path.join(res_dir, f'{model_name}_{preprocess}.csv')
    best_acc  = 0.0

    with open(csv_path, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['epoch', 'train_loss', 'train_acc',
                         'val_loss', 'val_acc'])

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, preprocess_fn)
            vl_loss, vl_acc = evaluate(
                model, val_loader, criterion, device, preprocess_fn)
            scheduler.step()
            best_acc = max(best_acc, vl_acc)

            writer.writerow([epoch, f'{tr_loss:.6f}', f'{tr_acc:.6f}',
                             f'{vl_loss:.6f}', f'{vl_acc:.6f}'])
            csvf.flush()

            print(f'  [{epoch:03d}/{args.epochs}] '
                  f'tr={tr_loss:.4f}/{tr_acc:.3f} | '
                  f'val={vl_loss:.4f}/{vl_acc:.3f} | best={best_acc:.3f}')

    torch.save(model.state_dict(),
               os.path.join(res_dir, f'{model_name}_{preprocess}.pth'))
    print(f'  Best val acc: {best_acc:.4f}')
    return best_acc


# ===========================================================================
# 6. Vignetting visualisation (before training)
# ===========================================================================

def save_vignetting_visualisation(res_dir, img_size=64):
    """
    Show raw / global / fundus_pdk side-by-side on a few APTOS images.
    Visualises the vignetting correction effect before training.
    """
    df = pd.read_csv(CSV_PATH).head(8)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])

    images = []
    for _, row in df.iterrows():
        p = os.path.join(IMAGES_DIR, row['id_code'] + '.png')
        if not os.path.exists(p):
            p = os.path.join(IMAGES_DIR, row['id_code'] + '.jpg')
        if not os.path.exists(p):
            continue
        images.append(transform(Image.open(p).convert('RGB')))
        if len(images) == 4:
            break

    if not images:
        print('[VIS] No images found for visualisation.')
        return

    batch = torch.stack(images)   # (N,1,H,W)

    raw_out = raw_preprocess(batch)
    glo_out = global_preprocess(batch)
    pdk_out = fundus_pdk_preprocess(batch)

    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[None, :]

    titles = ['Original (raw tensor)',
              'Raw normalized',
              'Global γ=0.6',
              'Fundus PDK (radial vignetting correction)']
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=8)

    def _show(ax, img_tensor):
        arr = img_tensor.squeeze().numpy()
        ax.imshow(arr, cmap='gray')
        ax.axis('off')

    for row in range(n):
        _show(axes[row, 0], batch[row])
        _show(axes[row, 1], raw_out[row])
        _show(axes[row, 2], glo_out[row])
        _show(axes[row, 3], pdk_out[row])

    fig.suptitle(
        'APTOS fundus vignetting correction:\n'
        'PDK applies STRONGER brightening at larger radii '
        '(darker periphery receives more correction)',
        fontsize=9
    )
    plt.tight_layout()
    path = os.path.join(res_dir, 'vignetting_correction_vis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[VIS] Vignetting correction visualisation → {path}')


# ===========================================================================
# 7. Comparison plot + summary
# ===========================================================================

def plot_comparison(res_dir, model_name, preprocesses):
    colors = {'raw': 'tab:blue', 'global': 'tab:orange',
              'fundus_pdk': 'tab:green'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for pp in preprocesses:
        csv_path = os.path.join(res_dir, f'{model_name}_{pp}.csv')
        if not os.path.exists(csv_path):
            continue
        epochs, tr_acc, vl_acc = [], [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                epochs.append(int(row['epoch']))
                tr_acc.append(float(row['train_acc']))
                vl_acc.append(float(row['val_acc']))
        c = colors.get(pp, 'black')
        axes[0].plot(epochs, [float(x) for x in
                               [row['train_loss'] for row in
                                csv.DictReader(open(csv_path))]],
                     '--', color=c, alpha=0.4, label=f'{pp} (train)')
        # re-read for val_loss
        axes[0].plot(epochs,
                     [float(r['val_loss']) for r in
                      csv.DictReader(open(csv_path))],
                     '-', color=c, label=pp)
        axes[1].plot(epochs, tr_acc, '--', color=c, alpha=0.4)
        axes[1].plot(epochs, vl_acc, '-', color=c, label=pp)

    for ax, title, ylabel in zip(
        axes,
        [f'Loss ({model_name})', f'Accuracy ({model_name})'],
        ['Cross-entropy', 'Accuracy']
    ):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        'APTOS 2019 DR: raw vs global vs fundus_pdk\n'
        'fundus_pdk applies position-dependent vignetting correction',
        fontsize=9
    )
    plt.tight_layout()
    path = os.path.join(res_dir, f'comparison_{model_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] → {path}')


def print_summary(results: dict):
    print('\n' + '=' * 62)
    print(f'  {"Model+Preprocess":<35} {"Best Val Acc":>12}')
    print('  ' + '-' * 50)
    for key, acc in results.items():
        print(f'  {key:<35} {acc:>12.4f}')
    print('=' * 62)

    # PDK gain analysis
    for model_name in ['KANHyperPDKNN', 'BaselineCNN']:
        raw_key = f'{model_name}_raw'
        glo_key = f'{model_name}_global'
        pdk_key = f'{model_name}_fundus_pdk'
        if raw_key in results and pdk_key in results:
            gain_vs_raw    = results[pdk_key] - results[raw_key]
            gain_vs_global = results[pdk_key] - results.get(glo_key, float('nan'))
            print(f'\n  [{model_name}]')
            print(f'    PDK vs raw    : {gain_vs_raw:+.4f}')
            print(f'    PDK vs global : {gain_vs_global:+.4f}')


# ===========================================================================
# 8. Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='APTOS 2019 DR — fundus PDK vignetting correction')
    p.add_argument('--preprocess', default='fundus_pdk',
                   choices=['raw', 'global', 'fundus_pdk'])
    p.add_argument('--model',      default='KANHyperPDKNN',
                   choices=['KANHyperPDKNN', 'BaselineCNN'])
    p.add_argument('--all',        action='store_true',
                   help='Run raw/global/fundus_pdk × KANHyperPDKNN+BaselineCNN')
    p.add_argument('--epochs',     type=int,   default=20)
    p.add_argument('--batch-size', type=int,   default=64,  dest='batch_size')
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--img-size',   type=int,   default=64,  dest='img_size')
    p.add_argument('--num-workers',type=int,   default=2,   dest='num_workers')
    p.add_argument('--res-dir',    type=str,   default='./res/aptos',
                   dest='res_dir')
    p.add_argument('--no-vis',     action='store_true',
                   help='Skip vignetting visualisation')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device  : {device}')

    # 1. Download data if needed
    ensure_data()

    # 1b. Auto-detect actual images directory (handles nested zip extraction)
    global IMAGES_DIR
    IMAGES_DIR = _find_images_dir()

    # 2. Create result dir
    os.makedirs(args.res_dir, exist_ok=True)

    # 3. Visualise vignetting correction (once)
    if not args.no_vis:
        save_vignetting_visualisation(args.res_dir, img_size=args.img_size)

    # 4. Run experiments
    if args.all:
        combinations = [
            ('KANHyperPDKNN', pp)
            for pp in ['raw', 'global', 'fundus_pdk']
        ] + [
            ('BaselineCNN', pp)
            for pp in ['raw', 'global', 'fundus_pdk']
        ]
    else:
        combinations = [(args.model, args.preprocess)]

    results = {}
    for model_name, preprocess in combinations:
        key = f'{model_name}_{preprocess}'
        results[key] = run_one(preprocess, model_name, args, device, args.res_dir)

    # 5. Plot per model
    for model_name in (['KANHyperPDKNN', 'BaselineCNN'] if args.all
                       else [args.model]):
        plot_comparison(args.res_dir, model_name,
                        ['raw', 'global', 'fundus_pdk'])

    # 6. Summary
    print_summary(results)


if __name__ == '__main__':
    main()
