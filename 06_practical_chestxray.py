"""
06_practical_chestxray.py
Position-Dependent Kernel (PDK) Preprocessing for Chest X-ray Classification

WHY chest X-ray benefits from position-dependent preprocessing
===============================================================
Unlike natural images (CIFAR, ImageNet), chest radiographs have *spatially
consistent anatomy*:

  - Lung fields always occupy the lateral thirds of the image (left and right).
  - The heart silhouette is invariably center-left.
  - Rib edges form predictable diagonal/curved patterns in mid-lateral zones.
  - The mediastinum is a nearly-uniform bright strip along the vertical center.
  - Background (outside the body) is a dark, featureless region at corners.

This anatomical consistency means that:
  (a) A lesion or infiltrate in the left-lower lung field will ALWAYS appear in
      roughly the same spatial region across patients (given standard PA views).
  (b) The diaphragm boundary and costophrenic angles are geometrically stable.
  (c) Uniform blur (global preprocessing) destroys fine pleural/vascular detail
      that carries diagnostic information, but is wasteful to preserve in the
      uniform background corners.

PDK preprocessing exploits this structure by assigning *spatially-aware* roles:
  - recover  : use Laplacian of Gaussian (LoG) to highlight blob/lesion regions
               that lose high-frequency detail after blur — especially important
               in lung parenchyma where nodules and infiltrates reside.
  - preserve : retain strong anatomical boundaries (lung borders, rib edges, and
               diaphragm lines) that carry structural diagnostic cues.
  - suppress : apply Gaussian smoothing to low-information background regions
               (homogeneous tissue, air outside body) to reduce noise.

An adaptive intensity threshold further adapts to image brightness: darker
images tend to represent aerated (healthy) lungs and require more sensitive
recovery, while brighter images (consolidated/fluid-filled) may tolerate
stronger suppression in flat regions.

Usage
-----
  python 06_practical_chestxray.py --all --epochs 20
  python 06_practical_chestxray.py --preprocess medical_pdk --epochs 20
  python 06_practical_chestxray.py --preprocess raw --epochs 20
"""

import os
import csv
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from medmnist import ChestMNIST
except ImportError:
    raise ImportError("medmnist not installed. Run: pip install medmnist")


# ---------------------------------------------------------------------------
# 1. KAN Layers
# ---------------------------------------------------------------------------

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network layer.

    Uses B-spline basis functions built with the recurrence relation
      `for k in range(1, self.spline_order+1)`
    combined with a SiLU residual branch.
    """

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
            grid_size + 2 * spline_order + 1
        )
        self.register_buffer('grid', grid)

        self.spline_weight  = nn.Parameter(
            torch.randn(in_features, out_features, self.n_basis) * 0.1)
        self.residual_scale = nn.Parameter(
            torch.ones(in_features, out_features) * 0.1)

    def b_splines(self, x):
        """Evaluate B-spline bases via Cox-de Boor recurrence."""
        x     = x.unsqueeze(-1)          # (..., in_features, 1)
        grid  = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            denom_left  = grid[k:-1]   - grid[:-(k + 1)]
            denom_right = grid[k + 1:] - grid[1:-k]
            safe_div    = lambda n, d: n / d.clamp(min=1e-8)
            left  = safe_div(x - grid[:-(k + 1)], denom_left)  * bases[..., :-1]
            right = safe_div(grid[k + 1:] - x,    denom_right) * bases[..., 1:]
            bases = left + right
        return bases                     # (..., in_features, n_basis)

    def forward(self, x):
        bs         = self.b_splines(x)
        spline_out = torch.einsum('bin,ion->bo', bs, self.spline_weight)
        silu_out   = torch.einsum('bi,io->bo', F.silu(x), self.residual_scale)
        return spline_out + silu_out


class KAN(nn.Module):
    def __init__(self, dims, grid_size=5, spline_order=3):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLayer(dims[i], dims[i + 1], grid_size, spline_order)
            for i in range(len(dims) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# 2. KANHyperPDKConv2d — KAN-generated position-dependent kernels
# ---------------------------------------------------------------------------

class KANHyperPDKConv2d(nn.Module):
    """
    Hypernetwork-style PDK conv where a KAN maps spatial coordinates
    (y, x) ∈ [-1, 1]² to per-position convolutional kernels.

    Coordinates are registered as a buffer via torch.linspace(-1, 1).
    _generate_kernels calls self.kan(self.coords).
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 input_height, input_width,
                 stride=1, padding=0,
                 kan_hidden=32, kan_latent=32,
                 grid_size=5, spline_order=3):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding

        kH = kW = kernel_size
        self.H_out = (input_height + 2 * padding - kH) // stride + 1
        self.W_out = (input_width  + 2 * padding - kW) // stride + 1
        self.kernel_flat = out_channels * in_channels * kH * kW

        self.kan            = KAN([2, kan_hidden, kan_latent], grid_size, spline_order)
        self.kernel_decoder = nn.Linear(kan_latent, self.kernel_flat)
        self.bias_decoder   = nn.Linear(kan_latent, out_channels)

        # Coordinate grid: shape (H_out * W_out, 2), values in [-1, 1]
        ys = torch.linspace(-1, 1, self.H_out)
        xs = torch.linspace(-1, 1, self.W_out)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
        self.register_buffer('coords', coords)

    def _generate_kernels(self):
        P       = self.H_out * self.W_out
        latent  = self.kan(self.coords)                         # (P, kan_latent)
        kernels = self.kernel_decoder(latent).view(
            P, self.out_channels, -1)                           # (P, C_out, C_in*kH*kW)
        biases  = self.bias_decoder(latent)                     # (P, C_out)
        return kernels, biases

    def forward(self, x, pos_chunk: int = 256):
        B   = x.size(0)
        kH  = kW = self.kernel_size
        kernels, biases = self._generate_kernels()
        P = self.H_out * self.W_out

        patches = F.unfold(x, kernel_size=kH,
                           padding=self.padding, stride=self.stride)
        patches    = patches.permute(0, 2, 1)       # (B, P, kflat)
        kernels_t  = kernels.permute(0, 2, 1)       # (P, kflat, out_ch)

        out_chunks = []
        for start in range(0, P, pos_chunk):
            end     = min(start + pos_chunk, P)
            o = torch.einsum('bpk,pko->bpo',
                             patches[:, start:end, :],
                             kernels_t[start:end, :, :])
            out_chunks.append(o)

        out = torch.cat(out_chunks, dim=1)           # (B, P, out_ch)
        out = (out + biases.unsqueeze(0)).permute(0, 2, 1).view(
            B, self.out_channels, self.H_out, self.W_out)
        return out


# ---------------------------------------------------------------------------
# 3. KANHyperPDKNN
# ---------------------------------------------------------------------------

class KANHyperPDKNN(nn.Module):
    """
    Two-block KANHyperPDK network for grayscale (1-channel) classification.

      block1 : KANHyperPDKConv2d(in_channels, 32, k=3, padding=1)
      bn1    : BatchNorm2d(32)
      MaxPool2d(2)
      block2 : KANHyperPDKConv2d(32, 64, k=3, padding=1)  [on H/2 × W/2]
      bn2    : BatchNorm2d(64)
      MaxPool2d(2)
      gap    : AdaptiveAvgPool2d(1)
      fc     : Linear(64, num_classes)
    """

    def __init__(self, in_channels=1, img_h=64, img_w=64,
                 num_classes=2, kan_hidden=32, kan_latent=32):
        super().__init__()

        self.block1 = KANHyperPDKConv2d(
            in_channels, 32, 3, img_h, img_w,
            padding=1, kan_hidden=kan_hidden, kan_latent=kan_latent)
        self.bn1    = nn.BatchNorm2d(32)

        h1, w1 = img_h // 2, img_w // 2

        self.block2 = KANHyperPDKConv2d(
            32, 64, 3, h1, w1,
            padding=1, kan_hidden=kan_hidden, kan_latent=kan_latent)
        self.bn2    = nn.BatchNorm2d(64)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.block1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.block2(x))), 2)
        return self.fc(self.gap(x).flatten(1))


class BaselineCNN(nn.Module):
    """Standard spatially-shared CNN for comparison."""

    def __init__(self, in_channels=1, num_classes=2):
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


# ---------------------------------------------------------------------------
# 4. ChestMNIST DataLoader (binary label)
# ---------------------------------------------------------------------------

def get_chestxray_loaders(batch_size=128, data_root='./data'):
    """
    Load ChestMNIST (64×64 grayscale) with medmnist.

    Multi-label → binary: label=1 if any pathology present, 0 if normal.
    Normalisation: mean=(0.5,) std=(0.5,) per medmnist convention.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    train_set = ChestMNIST(split='train', transform=transform,
                           download=True, size=64, root=data_root)
    val_set   = ChestMNIST(split='val',   transform=transform,
                           download=True, size=64, root=data_root)
    test_set  = ChestMNIST(split='test',  transform=transform,
                           download=True, size=64, root=data_root)

    # Wrap to binarise multi-label targets
    class BinaryWrapper(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, label = self.base[idx]
            # label shape: (14,) numpy array — 1 if any pathology
            binary = int(np.any(label.astype(np.int32) > 0))
            return img, binary

    train_loader = DataLoader(BinaryWrapper(train_set), batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(BinaryWrapper(val_set),   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(BinaryWrapper(test_set),  batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 5. Preprocessing kernels
# ---------------------------------------------------------------------------

def _binomial_kernel():
    k = torch.tensor([[1., 2., 1.],
                       [2., 4., 2.],
                       [1., 2., 1.]]) / 16.0
    return k.float()


def _highboost_kernel():
    k = torch.tensor([[ 0.0, -0.25,  0.0],
                       [-0.25,  2.0, -0.25],
                       [ 0.0, -0.25,  0.0]])
    return k.float()


def _log_kernel():
    """Laplacian of Gaussian (LoG) — 3×3 discrete Laplacian."""
    k = torch.tensor([[0.,  1., 0.],
                       [1., -4., 1.],
                       [0.,  1., 0.]])
    return k.float()


def _apply_kernel(x, k):
    ch = x.shape[1]
    w  = k.to(x.device, x.dtype).view(1, 1, 3, 3).repeat(ch, 1, 1, 1)
    return torch.clamp(F.conv2d(x, w, padding=1, groups=ch), -3.0, 3.0)


def _sobel_mag(x):
    sx = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    sy = torch.tensor([[-1., -2., -1.],
                        [ 0.,  0.,  0.],
                        [ 1.,  2.,  1.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    return torch.sqrt(F.conv2d(x, sx, padding=1) ** 2
                      + F.conv2d(x, sy, padding=1) ** 2 + 1e-8)


# ---------------------------------------------------------------------------
# 5a. Preprocessing functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def raw_preprocess_batch(x):
    """Identity preprocessing — no transformation."""
    return x, None


@torch.no_grad()
def global_preprocess_batch(x):
    """Global binomial blur — same kernel applied everywhere."""
    return _apply_kernel(x, _binomial_kernel()), None


@torch.no_grad()
def medical_pdk_preprocess_batch(x):
    """
    Chest X-ray specific PDK preprocessing.

    X-ray anatomy constraints exploited:
      - Lung fields are in lateral (left/right) spatial regions.
      - Heart is center-left → dense tissue → less need for edge recovery.
      - Background corners (dark, uniform) can be aggressively suppressed.

    Pipeline:
      1. Compute LoG response to detect blob-like lesions/nodules.
         Laplacian kernel: [[0,1,0],[1,-4,1],[0,1,0]]
      2. Compute Sobel edge magnitude on raw vs. globally blurred image.
      3. missed = edges present in raw but lost after blur (pathological detail).
      4. recover_mask  : where LoG response is high AND detail was lost after blur.
                         Adaptive threshold based on image mean intensity —
                         darker images (aerated lungs) → lower threshold →
                         more sensitive recovery.
      5. preserve_mask : strong anatomical boundaries (Sobel > per-image median),
                         excluding recover zones.
      6. suppress_mask : remaining uniform / background regions.

    Returns (preprocessed_batch, zone_stats_dict).
    """
    # --- Step 1: LoG blob/lesion detection ---
    log_resp  = _apply_kernel(x, _log_kernel())          # (B, 1, H, W)
    log_abs   = log_resp.abs()
    log_maxv  = log_abs.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8
    log_n     = log_abs / log_maxv                        # normalised [0, 1]

    # --- Step 2: Missed edges after global blur ---
    global_out  = _apply_kernel(x, _binomial_kernel())
    edge_raw    = _sobel_mag(x)
    edge_global = _sobel_mag(global_out)
    missed      = torch.relu(edge_raw - edge_global)
    maxv        = missed.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8
    missed_n    = missed / maxv                           # normalised [0, 1]

    # --- Step 3: Adaptive threshold (darker image → more sensitive) ---
    # image mean in normalised space: brighter chest (pleural effusion, pneumonia)
    # has higher mean; darker = aerated lung = fine vessels need recovery.
    img_mean  = x.flatten(1).mean(dim=1)                  # (B,)  in ~[-1, 1]
    # map from [-1, 1] to threshold [0.20, 0.45]:
    #   dark image (mean ~ -1) → threshold ~ 0.20 (very sensitive)
    #   bright image (mean ~ +1) → threshold ~ 0.45 (less sensitive)
    threshold = (0.325 + 0.125 * img_mean).view(-1, 1, 1, 1).clamp(0.15, 0.50)

    # recover: high LoG response AND missed detail
    recover_score = 0.5 * log_n + 0.5 * missed_n
    recover_mask  = (recover_score > threshold).float()

    # --- Step 4: Preserve strong anatomical boundaries ---
    med           = edge_global.flatten(1).median(dim=1)[0].view(-1, 1, 1, 1)
    preserve_mask = ((edge_global > med) * (1.0 - recover_mask)).float()

    # --- Step 5: Suppress everything else ---
    suppress_mask = 1.0 - torch.clamp(recover_mask + preserve_mask, 0.0, 1.0)

    # --- Compose output ---
    out = (preserve_mask * x
           + recover_mask  * _apply_kernel(x, _highboost_kernel())
           + suppress_mask * _apply_kernel(x, _binomial_kernel()))
    out = torch.clamp(out, -3.0, 3.0)

    zone_stats = {
        'recover' : recover_mask.mean().item(),
        'preserve': preserve_mask.mean().item(),
        'suppress': suppress_mask.mean().item(),
    }
    return out, zone_stats


def apply_preprocess(x, preprocess):
    if preprocess == 'raw':
        return raw_preprocess_batch(x)
    elif preprocess == 'global':
        return global_preprocess_batch(x)
    elif preprocess == 'medical_pdk':
        return medical_pdk_preprocess_batch(x)
    else:
        raise ValueError(f"Unknown preprocess: {preprocess}")


# ---------------------------------------------------------------------------
# 6. Train / Eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, preprocess):
    model.train()
    total_loss = correct = total = 0
    zone_accum  = {'recover': 0., 'preserve': 0., 'suppress': 0.}
    zone_batches = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        imgs, zs = apply_preprocess(imgs, preprocess)
        if zs is not None:
            for k in zone_accum:
                zone_accum[k] += zs[k]
            zone_batches += 1

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    zs_avg = ({k: v / zone_batches for k, v in zone_accum.items()}
               if zone_batches else None)
    return total_loss / total, correct / total, zs_avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, preprocess):
    model.eval()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        imgs, _ = apply_preprocess(imgs, preprocess)
        logits  = model(imgs)
        loss    = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# 7. run_one: single preprocess experiment
# ---------------------------------------------------------------------------

def count_params(m):
    return sum(p.numel() for p in m.parameters())


def run_one(preprocess, args, device, res_dir):
    print(f'\n{"=" * 60}')
    print(f'Preprocessing : {preprocess}')
    print(f'{"=" * 60}')

    model = KANHyperPDKNN(
        in_channels=1, img_h=64, img_w=64, num_classes=2
    ).to(device)
    print(f'Model  : {model.__class__.__name__}  ({count_params(model):,} params)')

    csv_path = os.path.join(res_dir, f'KANHyperPDKNN_{preprocess}.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(['epoch', 'train_loss', 'train_acc',
                     'test_loss',  'test_acc',
                     'zone_recover', 'zone_preserve', 'zone_suppress'])

    train_loader, val_loader, test_loader = get_chestxray_loaders(
        batch_size=args.batch_size, data_root=args.data_root)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    best_acc  = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_zone = train_one_epoch(
            model, train_loader, optimizer, criterion, device, preprocess)
        te_loss, te_acc = evaluate(
            model, test_loader, criterion, device, preprocess)
        scheduler.step()
        best_acc = max(best_acc, te_acc)

        # zone stats for CSV (None for raw/global)
        zr = tr_zone['recover']  if tr_zone else float('nan')
        zp = tr_zone['preserve'] if tr_zone else float('nan')
        zs = tr_zone['suppress'] if tr_zone else float('nan')

        writer.writerow([epoch,
                         f'{tr_loss:.6f}', f'{tr_acc:.6f}',
                         f'{te_loss:.6f}', f'{te_acc:.6f}',
                         f'{zr:.4f}', f'{zp:.4f}', f'{zs:.4f}'])
        csv_file.flush()

        msg = (f'[{epoch:03d}/{args.epochs}] '
               f'train loss={tr_loss:.4f} acc={tr_acc:.3f} | '
               f'test  loss={te_loss:.4f} acc={te_acc:.3f} | '
               f'best={best_acc:.3f}')
        if tr_zone:
            msg += (f"  zones: R={tr_zone['recover']:.3f}"
                    f" P={tr_zone['preserve']:.3f}"
                    f" S={tr_zone['suppress']:.3f}")
        print(msg)

    csv_file.close()
    pth_path = os.path.join(res_dir, f'KANHyperPDKNN_{preprocess}.pth')
    torch.save(model.state_dict(), pth_path)
    print(f'Saved model → {pth_path}')
    print(f'Saved CSV   → {csv_path}')
    return best_acc


# ---------------------------------------------------------------------------
# 8. Comparison plot
# ---------------------------------------------------------------------------

def plot_comparison(res_dir, preprocesses):
    """
    Plot test accuracy curves for all preprocessing modes on one figure
    and save to res_dir/comparison_KANHyperPDKNN.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = {'raw': '#1f77b4', 'global': '#ff7f0e', 'medical_pdk': '#2ca02c'}

    for pp in preprocesses:
        csv_path = os.path.join(res_dir, f'KANHyperPDKNN_{pp}.csv')
        if not os.path.exists(csv_path):
            continue
        epochs, tr_accs, te_accs = [], [], []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                tr_accs.append(float(row['train_acc']))
                te_accs.append(float(row['test_acc']))

        c = colors.get(pp, None)
        axes[0].plot(epochs, tr_accs, label=pp, color=c, marker='o', markersize=3)
        axes[1].plot(epochs, te_accs, label=pp, color=c, marker='o', markersize=3)

    for ax, title in zip(axes, ['Train Accuracy', 'Test Accuracy']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle('ChestMNIST — KANHyperPDKNN Preprocessing Comparison\n'
                 '(binary: any-pathology vs. normal)',
                 fontsize=12)
    plt.tight_layout()
    fname = os.path.join(res_dir, 'comparison_KANHyperPDKNN.png')
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'Comparison plot → {fname}')


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='ChestXray PDK Preprocessing Comparison (ChestMNIST)')
    parser.add_argument('--preprocess', type=str, default='medical_pdk',
                        choices=['raw', 'global', 'medical_pdk'],
                        help='Preprocessing mode to run (ignored if --all)')
    parser.add_argument('--all',        action='store_true',
                        help='Run raw / global / medical_pdk sequentially')
    parser.add_argument('--epochs',     type=int,   default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int,   default=128,
                        help='Mini-batch size (default: 128)')
    parser.add_argument('--lr',         type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--data-root',  type=str,   default='./data',
                        help='Directory to download/cache medmnist data')
    parser.add_argument('--res-dir',    type=str,   default='./res/chestxray',
                        help='Directory to save CSVs and plots')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res_dir = args.res_dir
    os.makedirs(res_dir, exist_ok=True)

    print(f'Device        : {device}')
    print(f'Results dir   : {res_dir}')
    print(f'Epochs        : {args.epochs}')
    print(f'Batch size    : {args.batch_size}')
    print(f'Learning rate : {args.lr}')

    targets = ['raw', 'global', 'medical_pdk'] if args.all else [args.preprocess]

    results = {}
    for pp in targets:
        results[pp] = run_one(pp, args, device, res_dir)

    print(f'\n{"=" * 60}')
    print('Final Best Accuracy Comparison')
    print(f'{"=" * 60}')
    for pp, acc in results.items():
        print(f'  {pp:<15s} : {acc:.4f}')

    if len(targets) > 1:
        plot_comparison(res_dir, targets)


if __name__ == '__main__':
    main()
