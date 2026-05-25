"""
06_practical_lgg.py
Brain Tumor MRI (LGG) — Binary Classification (tumor / no-tumor)

Demonstrates MRI PDK rule: tumor regions have high LOCAL CONTRAST
relative to surrounding tissue, so adaptive local-contrast masking
separates tumor boundaries (recover), strong edges (preserve),
and uniform normal tissue (suppress).

Dataset download (run once):
  kaggle datasets download mateuszbuda/lgg-mri-segmentation -p ./data/lgg --unzip
  Or manually from: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

Expected structure:
  ./data/lgg/kaggle_3m/{patient_id}/{patient_id}_N.tif
  ./data/lgg/kaggle_3m/{patient_id}/{patient_id}_N_mask.tif

Run:
  python 06_practical_lgg.py --all --epochs 20
  python 06_practical_lgg.py --preprocess mri_pdk --epochs 20
"""

import os
import csv
import argparse
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required: pip install Pillow")


# ---------------------------------------------------------------------------
# 0. Data download instructions
# ---------------------------------------------------------------------------

_DOWNLOAD_MSG = """
============================================================
LGG MRI Dataset NOT FOUND.

To download:
  kaggle datasets download mateuszbuda/lgg-mri-segmentation -p ./data/lgg --unzip

Or manually from:
  https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

Expected structure:
  ./data/lgg/kaggle_3m/{patient_id}/{patient_id}_N.tif
  ./data/lgg/kaggle_3m/{patient_id}/{patient_id}_N_mask.tif
============================================================
"""


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

class LGGDataset(Dataset):
    """
    Brain MRI LGG Segmentation dataset.
    Label: 1 (tumor) if the corresponding mask has any nonzero pixel, else 0.
    Images are loaded as grayscale, resized to 64x64, and normalized.
    """

    def __init__(self, root='./data/lgg', img_size=64, transform=None):
        self.transform = transform
        self.samples = []  # list of (image_path, label)

        kaggle_root = os.path.join(root, 'kaggle_3m')
        if not os.path.isdir(kaggle_root):
            print(_DOWNLOAD_MSG)
            raise FileNotFoundError(
                f"Dataset not found at {kaggle_root}. "
                "See instructions above."
            )

        # Gather all .tif files that are NOT masks
        tif_files = glob.glob(os.path.join(kaggle_root, '**', '*.tif'), recursive=True)
        img_files = [f for f in tif_files if not f.endswith('_mask.tif')]

        if len(img_files) == 0:
            print(_DOWNLOAD_MSG)
            raise FileNotFoundError(
                "No .tif image files found. "
                "Check dataset structure."
            )

        for img_path in sorted(img_files):
            mask_path = img_path.replace('.tif', '_mask.tif')
            if not os.path.isfile(mask_path):
                continue
            # Determine label: any nonzero pixel in mask => tumor (1)
            mask = np.array(Image.open(mask_path))
            label = 1 if mask.max() > 0 else 0
            self.samples.append((img_path, label))

        print(f"LGGDataset: {len(self.samples)} samples found.")
        n_tumor  = sum(1 for _, l in self.samples if l == 1)
        n_notumor = len(self.samples) - n_tumor
        print(f"  tumor={n_tumor}, no-tumor={n_notumor}")

        self.img_size = img_size
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')  # grayscale
        img = self.base_transform(img)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_lgg_loaders(batch_size=64, num_workers=2,
                    data_root='./data/lgg', img_size=64, seed=42):
    dataset = LGGDataset(root=data_root, img_size=img_size)

    n_total = len(dataset)
    n_test  = int(n_total * 0.2)
    n_train = n_total - n_test

    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train: {n_train} | Test: {n_test}")
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 2. KAN / Model
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
        B = x.size(0)
        kH = kW = self.kernel_size
        kernels, biases = self._generate_kernels()
        P = self.H_out * self.W_out

        patches   = F.unfold(x, kernel_size=kH, padding=self.padding, stride=self.stride)
        patches   = patches.permute(0, 2, 1)    # (B, P, kflat)
        kernels_t = kernels.permute(0, 2, 1)    # (P, kflat, out_ch)

        out_chunks = []
        for start in range(0, P, pos_chunk):
            end = min(start + pos_chunk, P)
            o = torch.einsum('bpk,pko->bpo',
                             patches[:, start:end, :],
                             kernels_t[start:end, :, :])
            out_chunks.append(o)

        out = torch.cat(out_chunks, dim=1)      # (B, P, out_ch)
        out = (out + biases.unsqueeze(0)).permute(0, 2, 1).view(
            B, self.out_channels, self.H_out, self.W_out)
        return out


class KANHyperPDKNN(nn.Module):
    def __init__(self, in_channels=1, img_h=64, img_w=64,
                 num_classes=2, kan_hidden=32, kan_latent=32):
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
    def __init__(self, num_classes=2, in_channels=1):
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


def _sobel_mag(x):
    sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                       device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                       device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    return torch.sqrt(F.conv2d(x, sx, padding=1) ** 2
                      + F.conv2d(x, sy, padding=1) ** 2 + 1e-8)


def global_preprocess_batch(x):
    return _apply_kernel(x, _make_kernel('gaussian'))


# ---------------------------------------------------------------------------
# 4. MRI PDK Rule
# ---------------------------------------------------------------------------

@torch.no_grad()
def mri_pdk_preprocess_batch(x):
    """
    MRI PDK Rule:
      Brain MRI tumor regions have HIGH LOCAL CONTRAST relative to surrounding tissue.

      1. local_contrast = |x - local_mean|  using avg_pool2d (7x7 window)
      2. Normalize local_contrast per image to [0, 1]
      3. recover_mask  = sigmoid((local_contrast_n - adaptive_threshold) * 10)
         adaptive_threshold = 0.3 - 0.15 * complexity_n
         (complex images: lower threshold => more recover)
      4. preserve_mask = strong Sobel edges * (1 - recover_mask)
      5. suppress_mask = remainder (uniform normal tissue)
      6. recover  -> highboost (enhance tumor boundary)
         preserve -> identity  (keep strong structural edges)
         suppress -> gaussian  (smooth normal tissue)
    """
    # --- Local contrast ---
    local_mean     = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
    local_contrast = torch.abs(x - local_mean)

    # Normalize per image
    maxv = local_contrast.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8
    local_contrast_n = local_contrast / maxv  # [0, 1]

    # --- Adaptive threshold based on image complexity ---
    complexity   = local_contrast.flatten(1).std(dim=1)           # (B,)
    c_min, c_max = complexity.min(), complexity.max() + 1e-8
    complexity_n = (complexity - c_min) / (c_max - c_min)         # [0, 1]

    adaptive_threshold = (0.3 - 0.15 * complexity_n).view(-1, 1, 1, 1)  # [0.15, 0.30]

    # --- Masks ---
    recover_mask = torch.sigmoid((local_contrast_n - adaptive_threshold) * 10)

    # preserve_mask: strong absolute Sobel edges, outside recover zone
    sobel = _sobel_mag(x)
    sobel_max  = sobel.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8
    sobel_n    = sobel / sobel_max
    edge_thresh = torch.quantile(sobel_n.flatten(1), 0.60, dim=1).view(-1, 1, 1, 1)
    preserve_mask = torch.sigmoid((sobel_n - edge_thresh) * 10) * (1 - recover_mask)

    suppress_mask = 1.0 - torch.clamp(recover_mask + preserve_mask, 0.0, 1.0)

    # --- Apply processing per zone ---
    out = (recover_mask  * _apply_kernel(x, _make_kernel('highboost'))
           + preserve_mask * x
           + suppress_mask * _apply_kernel(x, _make_kernel('gaussian')))

    zone_stats = {
        'recover' : recover_mask.mean().item(),
        'preserve': preserve_mask.mean().item(),
        'suppress': suppress_mask.mean().item(),
    }
    return torch.clamp(out, -3.0, 3.0), zone_stats


def apply_preprocess(x, preprocess):
    zone_stats = None
    if preprocess == 'global':
        x = global_preprocess_batch(x)
    elif preprocess == 'mri_pdk':
        x, zone_stats = mri_pdk_preprocess_batch(x)
    # 'raw' => no-op
    return x, zone_stats


# ---------------------------------------------------------------------------
# 5. Train / Eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, preprocess):
    model.train()
    total_loss = correct = total = 0
    zone_accum  = {'recover': 0., 'preserve': 0., 'suppress': 0.}
    zone_batches = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, zs     = apply_preprocess(imgs, preprocess)
        if zs:
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
# 6. Experiment runner
# ---------------------------------------------------------------------------

def count_params(m):
    return sum(p.numel() for p in m.parameters())


def run_one(preprocess, args, device, res_dir):
    print(f'\n{"=" * 60}')
    print(f'preprocess = {preprocess}')
    print(f'{"=" * 60}')

    model = KANHyperPDKNN(
        in_channels=1, img_h=64, img_w=64, num_classes=2
    ).to(device)
    print(f'Model: {model.__class__.__name__}  ({count_params(model):,} params)')

    csv_path = os.path.join(res_dir, f'KANHyperPDKNN_{preprocess}.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    train_loader, test_loader = get_lgg_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        img_size=64,
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
            msg += (f" | zones: R={tr_zone['recover']:.3f}"
                    f" P={tr_zone['preserve']:.3f}"
                    f" S={tr_zone['suppress']:.3f}")
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
    plt.title('LGG Brain Tumor MRI — Preprocessing Comparison (KANHyperPDKNN)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(res_dir, 'comparison_KANHyperPDKNN.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f'Plot -> {fname}')


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='LGG Brain Tumor MRI — Binary Classification with MRI PDK Rule')
    parser.add_argument('--preprocess', type=str, default='mri_pdk',
                        choices=['raw', 'global', 'mri_pdk'])
    parser.add_argument('--all',        action='store_true',
                        help='Run raw / global / mri_pdk sequentially')
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch-size', type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--num-workers',type=int,   default=2)
    parser.add_argument('--data-root',  type=str,   default='./data/lgg')
    parser.add_argument('--res-dir',    type=str,   default='./res/lgg')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res_dir = args.res_dir
    os.makedirs(res_dir, exist_ok=True)
    print(f'Device  : {device}')
    print(f'Results -> {res_dir}')

    targets = ['raw', 'global', 'mri_pdk'] if args.all else [args.preprocess]
    results = {}
    for pp in targets:
        results[pp] = run_one(pp, args, device, res_dir)

    print(f'\n{"=" * 60}')
    print('Final Best Accuracy Comparison')
    print(f'{"=" * 60}')
    for pp, acc in results.items():
        print(f'  {pp:15s} : {acc:.4f}')

    plot_comparison(res_dir, targets)


if __name__ == '__main__':
    main()
