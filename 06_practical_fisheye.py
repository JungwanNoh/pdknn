"""
06_practical_fisheye.py
Synthetic Fisheye Distortion on EuroSAT — Positional (Radial) PDK

Demonstrates RADIAL PDK rule: preprocessing zones are determined by
POSITION (radial distance from image center), NOT by image content.
This is the key distinction from content-adaptive PDK rules.

  - Center  (r < 0.4): minimal barrel distortion -> preserve (identity)
  - Middle  (0.4-0.7): moderate distortion        -> global (gaussian smooth)
  - Edge    (r > 0.7): severe distortion           -> highboost correction

Dataset: EuroSAT via torchvision (auto-downloaded), grayscale 64x64, 10 classes.
Barrel distortion (k1=0.3) is applied to ALL inputs before preprocessing.

Preprocessing choices:
  raw_distorted         — barrel distortion only, no further processing
  global_distorted      — barrel distortion + global gaussian
  radial_pdk_distorted  — barrel distortion + zone-based radial PDK

Run:
  python 06_practical_fisheye.py --all --epochs 20
  python 06_practical_fisheye.py --preprocess radial_pdk_distorted --epochs 20
"""

import os
import csv
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Barrel distortion
# ---------------------------------------------------------------------------

def apply_barrel_distortion(x, k1=0.3):
    """
    Apply synthetic barrel distortion to a batch of images.

    x: (B, C, H, W) tensor
    k1: distortion coefficient (positive = barrel, center compressed / edges stretched)

    Uses inverse mapping via grid_sample:
      distorted pixel at (x_d, y_d) comes from undistorted (x_d/distortion, y_d/distortion)
    """
    B, C, H, W = x.shape
    # Normalized coordinates in [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing='ij'
    )
    r2          = grid_x ** 2 + grid_y ** 2
    distortion  = 1 + k1 * r2           # barrel: center compressed, edges stretched
    grid_x_d    = grid_x / distortion
    grid_y_d    = grid_y / distortion
    grid = torch.stack([grid_x_d, grid_y_d], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    return F.grid_sample(x, grid, align_corners=True, padding_mode='border')


# ---------------------------------------------------------------------------
# 2. KAN / Model (identical to 06_practical_lgg.py)
# ---------------------------------------------------------------------------

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
            grid_size + 2 * spline_order + 1
        )
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
            denom_left  = grid[k:-1]  - grid[:-(k+1)]
            denom_right = grid[k+1:]  - grid[1:-k]
            safe_div    = lambda n, d: n / d.clamp(min=1e-8)
            left  = safe_div(x - grid[:-(k+1)], denom_left)  * bases[..., :-1]
            right = safe_div(grid[k+1:] - x,    denom_right) * bases[..., 1:]
            bases = left + right
        return bases

    def forward(self, x):
        bs         = self.b_splines(x)
        spline_out = torch.einsum('bin,ion->bo', bs, self.spline_weight)
        silu_out   = torch.einsum('bi,io->bo', F.silu(x), self.residual_scale)
        return spline_out + silu_out


class KAN(nn.Module):
    def __init__(self, dims, grid_size=5, spline_order=3):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLayer(dims[i], dims[i+1], grid_size, spline_order)
            for i in range(len(dims) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
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
        self.H_out = (input_height + 2 * padding - kH) // stride + 1
        self.W_out = (input_width  + 2 * padding - kW) // stride + 1
        self.kernel_flat = out_channels * in_channels * kH * kW
        self.kan            = KAN([2, kan_hidden, kan_latent], grid_size, spline_order)
        self.kernel_decoder = nn.Linear(kan_latent, self.kernel_flat)
        self.bias_decoder   = nn.Linear(kan_latent, out_channels)
        ys = torch.linspace(-1, 1, self.H_out)
        xs = torch.linspace(-1, 1, self.W_out)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
        self.register_buffer('coords', coords)

    def _generate_kernels(self):
        P       = self.H_out * self.W_out
        latent  = self.kan(self.coords)
        kernels = self.kernel_decoder(latent).view(P, self.out_channels, -1)
        biases  = self.bias_decoder(latent)
        return kernels, biases

    def forward(self, x, pos_chunk: int = 256):
        """
        pos_chunk: number of spatial positions processed per bmm call.
                   Lower = less peak VRAM.  256 works on 8-10 GB GPUs.
        """
        B    = x.size(0)
        kH = kW = self.kernel_size
        kernels, biases = self._generate_kernels()   # (P, out_ch, kflat), (P, out_ch)
        P = self.H_out * self.W_out

        patches = F.unfold(x, kernel_size=kH, padding=self.padding, stride=self.stride)
        # patches: (B, C*kH*kW, P)  →  (B, P, kflat)
        patches = patches.permute(0, 2, 1)

        # kernels: (P, out_ch, kflat)  →  (P, kflat, out_ch)
        kernels_t = kernels.permute(0, 2, 1)

        # Process positions in chunks to bound peak VRAM
        out_chunks = []
        for start in range(0, P, pos_chunk):
            end      = min(start + pos_chunk, P)
            p_chunk  = patches[:, start:end, :]           # (B, chunk, kflat)
            k_chunk  = kernels_t[start:end, :, :]         # (chunk, kflat, out_ch)
            # einsum: batch(B) × chunk × kflat , chunk × kflat × out_ch
            o_chunk  = torch.einsum('bpk,pko->bpo', p_chunk, k_chunk)  # (B, chunk, out_ch)
            out_chunks.append(o_chunk)

        out = torch.cat(out_chunks, dim=1)               # (B, P, out_ch)
        out = (out + biases.unsqueeze(0)).permute(0, 2, 1).view(
            B, self.out_channels, self.H_out, self.W_out)
        return out


class KANHyperPDKNN(nn.Module):
    def __init__(self, in_channels=1, img_h=64, img_w=64,
                 num_classes=10, kan_hidden=32, kan_latent=32):
        super().__init__()
        self.block1 = KANHyperPDKConv2d(
            in_channels, 32, 3, img_h, img_w, padding=1,
            kan_hidden=kan_hidden, kan_latent=kan_latent)
        self.bn1    = nn.BatchNorm2d(32)
        h1, w1      = img_h // 2, img_w // 2
        self.block2 = KANHyperPDKConv2d(
            32, 64, 3, h1, w1, padding=1,
            kan_hidden=kan_hidden, kan_latent=kan_latent)
        self.bn2    = nn.BatchNorm2d(64)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.block1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.block2(x))), 2)
        return self.fc(self.gap(x).flatten(1))


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
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
# 3. Preprocessing helpers
# ---------------------------------------------------------------------------

def _make_kernel(t):
    if t in ('binomial', 'gaussian'):
        k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0
    elif t == 'highboost':
        k = torch.tensor([[0., -0.25, 0.], [-0.25, 2.0, -0.25], [0., -0.25, 0.]])
    else:
        raise ValueError(t)
    return k.float()


def _apply_kernel(x, k):
    ch = x.shape[1]
    w  = k.to(x.device, x.dtype).view(1, 1, 3, 3).repeat(ch, 1, 1, 1)
    return torch.clamp(F.conv2d(x, w, padding=1, groups=ch), -3.0, 3.0)


def global_preprocess_batch(x):
    return _apply_kernel(x, _make_kernel('gaussian'))


# ---------------------------------------------------------------------------
# 4. Radial PDK Rule (POSITIONAL zones, not content-based)
# ---------------------------------------------------------------------------

def radial_pdk_preprocess_batch(x):
    """
    Radial PDK Rule — KEY DIFFERENCE: zones are based on POSITION (radial distance
    from image center), NOT on image content.

    This is intentional: barrel distortion severity is a function of radius,
    so the correction should also be spatially fixed.

      Center  (r_norm < 0.4): minimal distortion  -> preserve (identity)
      Middle  (0.4 - 0.7)  : moderate distortion  -> global   (gaussian)
      Edge    (r_norm > 0.7): severe distortion    -> highboost (correction)

    NOTE: Because zones are FIXED by position and do NOT depend on image
    content, zone_stats will be constant across all batches and images.
    This is intentional behavior — the radial masks are computed once from
    a fixed coordinate grid.
    """
    H, W = x.shape[2], x.shape[3]

    ys = torch.linspace(-1, 1, H, device=x.device)
    xs = torch.linspace(-1, 1, W, device=x.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    r       = torch.sqrt(grid_y ** 2 + grid_x ** 2)  # [0, sqrt(2)]
    r_norm  = r / r.max()                             # [0, 1]

    # Fixed positional masks — independent of image content
    preserve_mask = torch.sigmoid((0.4 - r_norm) * 15).unsqueeze(0).unsqueeze(0)
    edge_mask     = torch.sigmoid((r_norm - 0.7) * 15).unsqueeze(0).unsqueeze(0)
    middle_mask   = 1.0 - preserve_mask - edge_mask

    out = (preserve_mask * x
           + middle_mask * _apply_kernel(x, _make_kernel('gaussian'))
           + edge_mask   * _apply_kernel(x, _make_kernel('highboost')))

    # zone_stats are constant because masks are position-based (not content-based)
    zone_stats = {
        'center': preserve_mask.mean().item(),
        'middle': middle_mask.mean().item(),
        'edge'  : edge_mask.mean().item(),
    }
    return torch.clamp(out, -3.0, 3.0), zone_stats


def apply_preprocess(x, preprocess, k1=0.3):
    """
    Apply barrel distortion first, then the selected preprocessing.
    All three variants receive distorted input.
    """
    # Step 1: always apply barrel distortion
    x = apply_barrel_distortion(x, k1=k1)

    # Step 2: preprocessing on top of distorted image
    zone_stats = None
    if preprocess == 'global_distorted':
        x = global_preprocess_batch(x)
    elif preprocess == 'radial_pdk_distorted':
        x, zone_stats = radial_pdk_preprocess_batch(x)
    # 'raw_distorted' => barrel distortion only, nothing more

    return x, zone_stats


# ---------------------------------------------------------------------------
# 5. Data — EuroSAT (torchvision, auto-download), grayscale 64x64
# ---------------------------------------------------------------------------

def get_eurosat_loaders(batch_size=128, num_workers=2, data_root='./data'):
    T = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    train_set = torchvision.datasets.EuroSAT(
        root=data_root, download=True, transform=T)

    # EuroSAT has no official train/test split; use 80/20
    n_total = len(train_set)
    n_train = int(n_total * 0.8)
    n_test  = n_total - n_train
    generator = torch.Generator().manual_seed(42)
    train_set, test_set = torch.utils.data.random_split(
        train_set, [n_train, n_test], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"EuroSAT  Train: {n_train} | Test: {n_test}")
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 6. Train / Eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, preprocess):
    model.train()
    total_loss = correct = total = 0
    zone_accum  = {'center': 0., 'middle': 0., 'edge': 0.}
    zone_batches = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, zs     = apply_preprocess(imgs, preprocess)
        if zs:
            for k in zone_accum:
                if k in zs:
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

    zs_avg = {k: v / zone_batches for k, v in zone_accum.items()} \
             if zone_batches else None
    return total_loss / total, correct / total, zs_avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, preprocess):
    model.eval()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, _      = apply_preprocess(imgs, preprocess)
        logits       = model(imgs)
        total_loss  += criterion(logits, labels).item() * imgs.size(0)
        correct     += (logits.argmax(1) == labels).sum().item()
        total       += imgs.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# 7. Experiment runner
# ---------------------------------------------------------------------------

def count_params(m):
    return sum(p.numel() for p in m.parameters())


def run_one(preprocess, args, device, res_dir):
    print(f'\n{"=" * 60}')
    print(f'preprocess = {preprocess}')
    print(f'{"=" * 60}')

    model = KANHyperPDKNN(
        in_channels=1, img_h=64, img_w=64, num_classes=10
    ).to(device)
    print(f'Model: {model.__class__.__name__}  ({count_params(model):,} params)')

    csv_path = os.path.join(res_dir, f'KANHyperPDKNN_{preprocess}.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    train_loader, test_loader = get_eurosat_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc  = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_zone = train_one_epoch(
            model, train_loader, optimizer, criterion, device, preprocess)
        te_loss, te_acc = evaluate(
            model, test_loader, criterion, device, preprocess)
        scheduler.step()
        best_acc = max(best_acc, te_acc)

        writer.writerow([epoch, f'{tr_loss:.6f}', f'{tr_acc:.6f}',
                         f'{te_loss:.6f}', f'{te_acc:.6f}'])
        csv_file.flush()

        msg = (f'[{epoch:03d}/{args.epochs}] '
               f'train loss={tr_loss:.4f} acc={tr_acc:.3f} | '
               f'test  loss={te_loss:.4f} acc={te_acc:.3f} | best={best_acc:.3f}')
        if tr_zone:
            # NOTE: for radial_pdk, zone stats are constant (position-based, not content)
            msg += (f" | zones: C={tr_zone.get('center', 0):.3f}"
                    f" M={tr_zone.get('middle', 0):.3f}"
                    f" E={tr_zone.get('edge', 0):.3f}")
        print(msg)

    csv_file.close()
    pth_path = os.path.join(res_dir, f'KANHyperPDKNN_{preprocess}.pth')
    torch.save(model.state_dict(), pth_path)
    print(f'Saved -> {pth_path}')
    print(f'CSV   -> {csv_path}')
    return best_acc


def plot_comparison(res_dir, preprocesses):
    plt.figure(figsize=(8, 5))
    for pp in preprocesses:
        csv_path = os.path.join(res_dir, f'KANHyperPDKNN_{pp}.csv')
        if not os.path.exists(csv_path):
            continue
        epochs, accs = [], []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                accs.append(float(row['test_acc']))
        plt.plot(epochs, accs, label=pp, marker='o', markersize=3)

    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('EuroSAT Fisheye — Radial PDK Comparison (KANHyperPDKNN)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(res_dir, 'comparison_KANHyperPDKNN.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f'Plot -> {fname}')


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='EuroSAT Fisheye — Radial PDK Rule (Positional Zones)')
    parser.add_argument('--preprocess', type=str, default='radial_pdk_distorted',
                        choices=['raw_distorted', 'global_distorted',
                                 'radial_pdk_distorted'])
    parser.add_argument('--all',        action='store_true',
                        help='Run raw_distorted / global_distorted / radial_pdk_distorted')
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch-size', type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--num-workers',type=int,   default=2)
    parser.add_argument('--data-root',  type=str,   default='./data')
    parser.add_argument('--res-dir',    type=str,   default='./res/fisheye')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res_dir = args.res_dir
    os.makedirs(res_dir, exist_ok=True)
    print(f'Device  : {device}')
    print(f'Results -> {res_dir}')
    print(
        '\nNOTE: In radial_pdk_distorted, zone statistics are constant across batches.\n'
        '      This is intentional — zones are position-based (radial), not content-based.\n'
    )

    targets = (['raw_distorted', 'global_distorted', 'radial_pdk_distorted']
               if args.all else [args.preprocess])
    results = {}
    for pp in targets:
        results[pp] = run_one(pp, args, device, res_dir)

    print(f'\n{"=" * 60}')
    print('Final Best Accuracy Comparison')
    print(f'{"=" * 60}')
    for pp, acc in results.items():
        print(f'  {pp:30s} : {acc:.4f}')

    plot_comparison(res_dir, targets)


if __name__ == '__main__':
    main()
