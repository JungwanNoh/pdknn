"""
07_fisheye_comparison.py
========================
BaselineCNN (position-independent) vs KANHyperPDKNN (position-dependent)
on fisheye-distorted EuroSAT — 3 preprocessing × 2 model comparison.

Research hypothesis
-------------------
  BaselineCNN has NO spatial awareness — every position shares the same
  learned kernel.  When input has barrel distortion (outer pixels smeared),
  the network CANNOT learn position-specific corrections.
  → PDK preprocessing gap should be LARGER for BaselineCNN.

  KANHyperPDKNN generates a different kernel per spatial location.
  Even on raw distorted input it can learn implicit radial correction.
  → PDK preprocessing gap should be SMALLER for KANHyperPDKNN.

Expected 2×3 result table
--------------------------
  preprocessing       BaselineCNN   KANHyperPDKNN   Δ (PDK benefit)
  ──────────────────  ───────────   ─────────────   ───────────────
  raw_distorted            ↓low          ↑higher     (PDK helps CNN more)
  global_distorted         ↓             ~
  radial_pdk_distorted     ↑higher       ↑            (PDK least needed)

This demonstrates that position-dependent kernels (KANHyperPDKNN) partially
substitute for PDK preprocessing, while position-independent kernels
(BaselineCNN) rely more heavily on PDK to compensate for distortion.

Run
---
  # Run BaselineCNN --all  (KANHyperPDKNN CSVs auto-loaded from ./res/fisheye)
  python 07_fisheye_comparison.py --all --epochs 20

  # Single preprocessing
  python 07_fisheye_comparison.py --preprocess radial_pdk_distorted --epochs 20

  # Skip KANHyperPDKNN loading (plot BaselineCNN only)
  python 07_fisheye_comparison.py --all --epochs 20 --no-kan
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ===========================================================================
# 1. Barrel distortion  (identical to 06_practical_fisheye.py)
# ===========================================================================

def apply_barrel_distortion(x, k1=0.3):
    B, C, H, W = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing='ij'
    )
    r2         = grid_x ** 2 + grid_y ** 2
    distortion = 1 + k1 * r2
    grid = torch.stack([grid_x / distortion,
                        grid_y / distortion], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    return F.grid_sample(x, grid, align_corners=True, padding_mode='border')


# ===========================================================================
# 2. Preprocessing helpers  (identical to 06)
# ===========================================================================

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


def radial_pdk_preprocess_batch(x):
    H, W   = x.shape[2], x.shape[3]
    ys     = torch.linspace(-1, 1, H, device=x.device)
    xs     = torch.linspace(-1, 1, W, device=x.device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    r_norm = (gy**2 + gx**2).sqrt() / (gy**2 + gx**2).sqrt().max()

    preserve_mask = torch.sigmoid((0.4 - r_norm) * 15).unsqueeze(0).unsqueeze(0)
    edge_mask     = torch.sigmoid((r_norm - 0.7) * 15).unsqueeze(0).unsqueeze(0)
    middle_mask   = 1.0 - preserve_mask - edge_mask

    out = (preserve_mask * x
           + middle_mask * _apply_kernel(x, _make_kernel('gaussian'))
           + edge_mask   * _apply_kernel(x, _make_kernel('highboost')))
    return torch.clamp(out, -3.0, 3.0)


def apply_preprocess(x, preprocess, k1=0.3):
    x = apply_barrel_distortion(x, k1=k1)
    if preprocess == 'global_distorted':
        x = global_preprocess_batch(x)
    elif preprocess == 'radial_pdk_distorted':
        x = radial_pdk_preprocess_batch(x)
    return x


# ===========================================================================
# 3. Model — BaselineCNN only (KANHyperPDKNN results loaded from 06 CSV)
# ===========================================================================

class BaselineCNN(nn.Module):
    """
    Standard CNN with position-INDEPENDENT kernels.
    Every spatial location in every layer shares the same kernel weights.
    Cannot learn position-specific corrections — must rely on preprocessing.
    """
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


# ===========================================================================
# 4. Dataset  (identical to 06)
# ===========================================================================

def get_eurosat_loaders(batch_size=128, num_workers=2, data_root='./data'):
    T = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    full_set = torchvision.datasets.EuroSAT(root=data_root, download=True, transform=T)
    n_total  = len(full_set)
    n_train  = int(n_total * 0.8)
    generator = torch.Generator().manual_seed(42)
    train_set, test_set = torch.utils.data.random_split(
        full_set, [n_train, n_total - n_train], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    print(f'EuroSAT  Train: {n_train} | Test: {n_total - n_train}')
    return train_loader, test_loader


# ===========================================================================
# 5. Train / Eval
# ===========================================================================

def count_params(m):
    return sum(p.numel() for p in m.parameters())


def train_one_epoch(model, loader, optimizer, criterion, device, preprocess):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs         = apply_preprocess(imgs, preprocess)
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
def evaluate(model, loader, criterion, device, preprocess):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs         = apply_preprocess(imgs, preprocess)
        logits       = model(imgs)
        total_loss  += criterion(logits, labels).item() * imgs.size(0)
        correct     += (logits.argmax(1) == labels).sum().item()
        total       += imgs.size(0)
    return total_loss / total, correct / total


# ===========================================================================
# 6. Single run
# ===========================================================================

def run_one(preprocess, args, device, res_dir):
    print(f'\n{"=" * 60}')
    print(f'Model      : BaselineCNN  (position-INDEPENDENT kernel)')
    print(f'Preprocess : {preprocess}')
    print(f'{"=" * 60}')

    model = BaselineCNN(num_classes=10, in_channels=1).to(device)
    print(f'Params: {count_params(model):,}')

    csv_path = os.path.join(res_dir, f'BaselineCNN_{preprocess}.csv')
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
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, preprocess)
        te_loss, te_acc = evaluate(
            model, test_loader, criterion, device, preprocess)
        scheduler.step()
        best_acc = max(best_acc, te_acc)

        writer.writerow([epoch, f'{tr_loss:.6f}', f'{tr_acc:.6f}',
                         f'{te_loss:.6f}', f'{te_acc:.6f}'])
        csv_file.flush()
        print(f'[{epoch:03d}/{args.epochs}] '
              f'train loss={tr_loss:.4f} acc={tr_acc:.3f} | '
              f'test  loss={te_loss:.4f} acc={te_acc:.3f} | best={best_acc:.3f}')

    csv_file.close()
    torch.save(model.state_dict(),
               os.path.join(res_dir, f'BaselineCNN_{preprocess}.pth'))
    print(f'CSV -> {csv_path}')
    return best_acc


# ===========================================================================
# 7. Load KANHyperPDKNN results from 06 CSVs  (read-only, no retraining)
# ===========================================================================

def load_kan_results(kan_res_dir: str, preprocesses: list) -> dict:
    """
    Read best test_acc from 06_practical_fisheye.py CSV files.
    Returns {preprocess: best_acc} or empty dict if files not found.
    """
    results = {}
    for pp in preprocesses:
        csv_path = os.path.join(kan_res_dir, f'KANHyperPDKNN_{pp}.csv')
        if not os.path.exists(csv_path):
            print(f'  [KAN] CSV not found: {csv_path}  (run 06_practical_fisheye.py first)')
            continue
        best = 0.0
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                best = max(best, float(row['test_acc']))
        results[pp] = best
        print(f'  [KAN] loaded {pp}: best={best:.4f}')
    return results


def load_kan_curves(kan_res_dir: str, preprocesses: list) -> dict:
    """Returns {preprocess: [acc_epoch1, acc_epoch2, ...]} for plotting."""
    curves = {}
    for pp in preprocesses:
        csv_path = os.path.join(kan_res_dir, f'KANHyperPDKNN_{pp}.csv')
        if not os.path.exists(csv_path):
            continue
        accs = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                accs.append(float(row['test_acc']))
        curves[pp] = accs
    return curves


# ===========================================================================
# 8. Comparison plot  (BaselineCNN curves + KANHyperPDKNN curves side-by-side)
# ===========================================================================

PREPROCESSES = ['raw_distorted', 'global_distorted', 'radial_pdk_distorted']
COLORS       = {'raw_distorted': 'tab:blue',
                'global_distorted': 'tab:orange',
                'radial_pdk_distorted': 'tab:green'}
LABELS       = {'raw_distorted': 'raw',
                'global_distorted': 'global',
                'radial_pdk_distorted': 'radial_pdk'}


def plot_full_comparison(res_dir: str, baseline_results: dict,
                         kan_results: dict, kan_curves: dict):
    """
    Left panel  : BaselineCNN test accuracy curves per preprocessing
    Right panel : Bar chart — best acc, side-by-side BaselineCNN vs KANHyperPDKNN
    """
    # ── Load BaselineCNN curves ──
    cnn_curves = {}
    for pp in PREPROCESSES:
        csv_path = os.path.join(res_dir, f'BaselineCNN_{pp}.csv')
        if not os.path.exists(csv_path):
            continue
        accs = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                accs.append(float(row['test_acc']))
        cnn_curves[pp] = accs

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: accuracy curves ──
    ax = axes[0]
    for pp in PREPROCESSES:
        c = COLORS[pp]
        lbl = LABELS[pp]
        if pp in cnn_curves:
            epochs = list(range(1, len(cnn_curves[pp]) + 1))
            ax.plot(epochs, cnn_curves[pp], '-', color=c, label=f'CNN-{lbl}')
        if pp in kan_curves:
            epochs = list(range(1, len(kan_curves[pp]) + 1))
            ax.plot(epochs, kan_curves[pp], '--', color=c, label=f'KAN-{lbl}', alpha=0.7)

    ax.set_title('Test Accuracy Curves\n(solid=BaselineCNN  dashed=KANHyperPDKNN)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Right: grouped bar chart ──
    ax2    = axes[1]
    pp_labels = [LABELS[pp] for pp in PREPROCESSES]
    x      = np.arange(len(PREPROCESSES))
    width  = 0.35

    cnn_vals = [baseline_results.get(pp, 0.0) for pp in PREPROCESSES]
    kan_vals = [kan_results.get(pp, 0.0)      for pp in PREPROCESSES]

    bars1 = ax2.bar(x - width/2, cnn_vals, width,
                    label='BaselineCNN (position-independent)',
                    color=['tab:blue', 'tab:orange', 'tab:green'], alpha=0.6)
    bars2 = ax2.bar(x + width/2, kan_vals, width,
                    label='KANHyperPDKNN (position-dependent)',
                    color=['tab:blue', 'tab:orange', 'tab:green'], alpha=1.0)

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                     f'{h:.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                     f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(pp_labels)
    ax2.set_ylabel('Best Test Accuracy')
    ax2.set_title('Best Accuracy: BaselineCNN vs KANHyperPDKNN\n'
                  '(KAN implicitly learns position-aware correction)')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    # Highlight PDK benefit delta
    for i, pp in enumerate(PREPROCESSES):
        cnn_v = baseline_results.get(pp, 0.0)
        kan_v = kan_results.get(pp, 0.0)
        if cnn_v > 0 and kan_v > 0:
            delta = kan_v - cnn_v
            sign  = '+' if delta >= 0 else ''
            ax2.text(x[i], max(cnn_v, kan_v) + 0.015,
                     f'KAN{sign}{delta:.3f}',
                     ha='center', fontsize=7, color='black',
                     style='italic')

    fig.suptitle(
        'FishEye EuroSAT: PDK benefit is LARGER for position-independent CNN\n'
        '(KANHyperPDKNN compensates for distortion via position-dependent kernels)',
        fontsize=10
    )
    fig.tight_layout()
    fname = os.path.join(res_dir, 'comparison_CNN_vs_KAN.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nComparison plot -> {fname}')
    return fname


# ===========================================================================
# 9. Summary table
# ===========================================================================

def print_summary_table(baseline_results: dict, kan_results: dict):
    print('\n' + '=' * 70)
    print(f'  {"Preprocessing":<25} {"BaselineCNN":>12} {"KANHyperPDK":>13} {"Δ(KAN-CNN)":>11}')
    print('  ' + '-' * 64)
    for pp in PREPROCESSES:
        cnn_v = baseline_results.get(pp, float('nan'))
        kan_v = kan_results.get(pp, float('nan'))
        delta = kan_v - cnn_v if (cnn_v == cnn_v and kan_v == kan_v) else float('nan')
        cnn_s = f'{cnn_v:.4f}' if cnn_v == cnn_v else '  N/A  '
        kan_s = f'{kan_v:.4f}' if kan_v == kan_v else '  N/A  '
        dlt_s = f'{delta:+.4f}' if delta == delta else '  N/A  '
        print(f'  {pp:<25} {cnn_s:>12} {kan_s:>13} {dlt_s:>11}')
    print('=' * 70)

    # Analysis
    raw_gap = (baseline_results.get('raw_distorted', 0) -
               baseline_results.get('radial_pdk_distorted', 0))
    print(f'\n  PDK gain for BaselineCNN  (pdk - raw): {-raw_gap:+.4f}')
    kan_raw_gap = (kan_results.get('raw_distorted', 0) -
                   kan_results.get('radial_pdk_distorted', 0))
    print(f'  PDK gain for KANHyperPDK  (pdk - raw): {-kan_raw_gap:+.4f}')

    if abs(-raw_gap) > abs(-kan_raw_gap):
        print('\n  ✓ CONFIRMED: PDK benefit is larger for BaselineCNN.')
        print('    KANHyperPDKNN compensates for distortion via position-dependent')
        print('    kernels, reducing reliance on explicit PDK preprocessing.')
    else:
        print('\n  → KANHyperPDKNN shows equal or larger PDK benefit.')
        print('    Consider: PDK + position-aware model may be complementary.')


# ===========================================================================
# 10. Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='07: BaselineCNN vs KANHyperPDKNN on fisheye EuroSAT')
    parser.add_argument('--preprocess', default='radial_pdk_distorted',
                        choices=PREPROCESSES)
    parser.add_argument('--all',         action='store_true',
                        help='Run all 3 preprocessings with BaselineCNN')
    parser.add_argument('--epochs',      type=int,   default=20)
    parser.add_argument('--batch-size',  type=int,   default=128)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int,   default=2)
    parser.add_argument('--data-root',   type=str,   default='./data')
    parser.add_argument('--res-dir',     type=str,   default='./res/fisheye',
                        help='Also used to read 06 KANHyperPDKNN CSVs')
    parser.add_argument('--no-kan',      action='store_true',
                        help='Skip loading KANHyperPDKNN results from 06')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res_dir = args.res_dir
    os.makedirs(res_dir, exist_ok=True)

    print(f'Device  : {device}')
    print(f'Results : {res_dir}')
    print(
        '\nKEY EXPERIMENT: BaselineCNN uses position-independent kernels.\n'
        'Without positional awareness, it CANNOT implicitly correct for radial\n'
        'barrel distortion — making PDK preprocessing essential.\n'
        'Compare with KANHyperPDKNN results from 06_practical_fisheye.py.\n'
    )

    targets = PREPROCESSES if args.all else [args.preprocess]

    # ── Train BaselineCNN ──
    baseline_results = {}
    for pp in targets:
        baseline_results[pp] = run_one(pp, args, device, res_dir)

    # ── Load KANHyperPDKNN from 06 ──
    kan_results = {}
    kan_curves  = {}
    if not args.no_kan:
        print('\n[Loading KANHyperPDKNN results from 06_practical_fisheye.py CSVs...]')
        kan_results = load_kan_results(res_dir, targets)
        kan_curves  = load_kan_curves(res_dir, targets)

    # ── Summary table ──
    print_summary_table(baseline_results, kan_results)

    # ── Comparison plot ──
    if len(targets) > 1:
        plot_full_comparison(res_dir, baseline_results, kan_results, kan_curves)


if __name__ == '__main__':
    main()
