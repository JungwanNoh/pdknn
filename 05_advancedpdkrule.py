"""
05_advancedpdkrule.py
Adaptive PDK Preprocessing — 가치 검증 실험

목표:
  기존 PDK rule의 문제 (zone 비율 고정) 를 해결하고,
  adaptive PDK 가 raw / global 대비 성능 향상을 보임을 실험으로 증명.

입력: Grayscale 1ch (white LED 소자 데이터 기준)

PDK rule 개선:
  [기존] 고정 threshold 0.35  → zone 비율 항상 R=0.15, P=0.37, S=0.48
  [개선] 이미지별 complexity 기반 adaptive threshold
         → 복잡한 이미지: recover ↑  /  단순한 이미지: suppress ↑

실행:
  python 05_advancedpdkrule.py --all --epochs 20
  python 05_advancedpdkrule.py --preprocess adaptive_pdk --epochs 20
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
import numpy as np


# ---------------------------------------------------------------------------
# 1. KAN (동일)
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
        self.spline_weight  = nn.Parameter(torch.randn(in_features, out_features, self.n_basis) * 0.1)
        self.residual_scale = nn.Parameter(torch.ones(in_features, out_features) * 0.1)

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


# ---------------------------------------------------------------------------
# 2. Model (grayscale 1ch 입력)
# ---------------------------------------------------------------------------

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

    def forward(self, x):
        B = x.size(0)
        kH = kW = self.kernel_size
        kernels, biases = self._generate_kernels()
        patches = F.unfold(x, kernel_size=kH, padding=self.padding, stride=self.stride)
        patches = patches.permute(0, 2, 1)
        kernels = kernels.permute(0, 2, 1)
        out = torch.bmm(
            patches.reshape(B * self.H_out * self.W_out, 1, -1),
            kernels.unsqueeze(0).expand(B, -1, -1, -1).reshape(
                B * self.H_out * self.W_out, -1, self.out_channels)
        ).squeeze(1)
        out = out.view(B, self.H_out * self.W_out, self.out_channels)
        out = (out + biases.unsqueeze(0)).permute(0, 2, 1).view(
            B, self.out_channels, self.H_out, self.W_out)
        return out


class KANHyperPDKNN(nn.Module):
    def __init__(self, num_classes=10, img_h=32, img_w=32,
                 in_channels=1, kan_hidden=32, kan_latent=32):
        super().__init__()
        self.block1 = KANHyperPDKConv2d(in_channels, 32, 3, img_h, img_w, padding=1,
                                         kan_hidden=kan_hidden, kan_latent=kan_latent)
        self.bn1    = nn.BatchNorm2d(32)
        h1, w1      = img_h // 2, img_w // 2
        self.block2 = KANHyperPDKConv2d(32, 64, 3, h1, w1, padding=1,
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
        self.fc    = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        return self.fc(self.gap(x).flatten(1))


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------

def _make_kernel(t):
    if t in ('binomial', 'gaussian'):
        k = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]]) / 16.0
    elif t == 'highboost':
        k = torch.tensor([[0.,-0.25,0.],[-0.25,2.0,-0.25],[0.,-0.25,0.]])
    else:
        raise ValueError(t)
    return k.float()

def _apply_kernel(x, k):
    ch = x.shape[1]
    w  = k.to(x.device, x.dtype).view(1,1,3,3).repeat(ch,1,1,1)
    return torch.clamp(F.conv2d(x, w, padding=1, groups=ch), -3.0, 3.0)

def _sobel_mag(x):
    sx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    sy = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    return torch.sqrt(F.conv2d(x,sx,padding=1)**2 + F.conv2d(x,sy,padding=1)**2 + 1e-8)

def global_preprocess_batch(x):
    return _apply_kernel(x, _make_kernel('binomial'))


@torch.no_grad()
def pdk_preprocess_batch(x):
    """기존 fixed-threshold PDK (비교 기준선)."""
    edge_raw    = _sobel_mag(x)
    global_out  = _apply_kernel(x, _make_kernel('binomial'))
    edge_global = _sobel_mag(global_out)
    missed      = torch.relu(edge_raw - edge_global)
    maxv        = missed.flatten(1).max(dim=1)[0].view(-1,1,1,1) + 1e-8
    missed_n    = missed / maxv
    recover_mask  = (missed_n > 0.35).float()
    med           = edge_global.flatten(1).median(dim=1)[0].view(-1,1,1,1)
    preserve_mask = ((edge_global > med) * (1 - recover_mask)).float()
    suppress_mask = 1.0 - torch.clamp(recover_mask + preserve_mask, 0, 1)
    out = (preserve_mask * x
           + recover_mask  * _apply_kernel(x, _make_kernel('highboost'))
           + suppress_mask * _apply_kernel(x, _make_kernel('gaussian')))
    zone_stats = {k: v.mean().item() for k, v in
                  zip(['recover','preserve','suppress'], [recover_mask, preserve_mask, suppress_mask])}
    return torch.clamp(out, -3.0, 3.0), zone_stats


@torch.no_grad()
def adaptive_pdk_preprocess_batch(x):
    """
    Adaptive PDK: 이미지별 complexity에 따라 threshold 동적 결정.

    개선 사항:
      1. recover threshold: 고정 0.35 → 이미지 complexity 기반 adaptive
         복잡한 이미지(edge 많음) → threshold ↓ → recover zone ↑
         단순한 이미지(flat 많음) → threshold ↑ → suppress zone ↑
      2. preserve threshold: 고정 median → 상위 40% edge (이미지별 percentile)
      3. soft mask: 하드 binary → sigmoid로 부드럽게
         경계 artifact 감소
    """
    edge_raw    = _sobel_mag(x)
    global_out  = _apply_kernel(x, _make_kernel('binomial'))
    edge_global = _sobel_mag(global_out)

    missed  = torch.relu(edge_raw - edge_global)
    maxv    = missed.flatten(1).max(dim=1)[0].view(-1,1,1,1) + 1e-8
    missed_n = missed / maxv

    # --- 1. Adaptive recover threshold ---
    # 이미지 complexity = missed edge의 표준편차
    complexity = missed.flatten(1).std(dim=1)                       # (B,)
    c_min, c_max = complexity.min(), complexity.max() + 1e-8
    complexity_n = (complexity - c_min) / (c_max - c_min)           # [0, 1]
    # complexity 높을수록 threshold 낮춤 (더 많이 recover)
    threshold = (0.4 - 0.25 * complexity_n).view(-1, 1, 1, 1)      # [0.15, 0.40]

    # soft recover mask (sigmoid)
    recover_mask = torch.sigmoid((missed_n - threshold) * 12)       # soft boundary

    # --- 2. Adaptive preserve threshold (상위 40% edge 보존) ---
    edge_thresh = torch.quantile(
        edge_global.flatten(1), 0.60, dim=1
    ).view(-1, 1, 1, 1)
    preserve_mask = torch.sigmoid((edge_global - edge_thresh) * 10) \
                    * (1 - recover_mask)

    # --- 3. suppress = 나머지 ---
    suppress_mask = 1.0 - torch.clamp(recover_mask + preserve_mask, 0, 1)

    out = (preserve_mask * x
           + recover_mask  * _apply_kernel(x, _make_kernel('highboost'))
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
    elif preprocess == 'pdk':
        x, zone_stats = pdk_preprocess_batch(x)
    elif preprocess == 'adaptive_pdk':
        x, zone_stats = adaptive_pdk_preprocess_batch(x)
    return x, zone_stats


# ---------------------------------------------------------------------------
# 4. Data (grayscale)
# ---------------------------------------------------------------------------

def get_cifar10_loaders(batch_size=128, num_workers=2, data_root='./data'):
    T_train = transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4734,), (0.2516,)),
    ])
    T_test = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.4734,), (0.2516,)),
    ])
    train_set = torchvision.datasets.CIFAR10(data_root, train=True,  download=True, transform=T_train)
    test_set  = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=T_test)
    return (DataLoader(train_set, batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
            DataLoader(test_set,  batch_size, shuffle=False, num_workers=num_workers, pin_memory=True))


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
        imgs, zs = apply_preprocess(imgs, preprocess)
        if zs:
            for k in zone_accum: zone_accum[k] += zs[k]
            zone_batches += 1
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    zs_avg = {k: v/zone_batches for k,v in zone_accum.items()} if zone_batches else None
    return total_loss/total, correct/total, zs_avg

@torch.no_grad()
def evaluate(model, loader, criterion, device, preprocess):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, _ = apply_preprocess(imgs, preprocess)
        logits  = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss/total, correct/total


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def count_params(m): return sum(p.numel() for p in m.parameters())


def run_one(preprocess, args, device, res_dir):
    print(f'\n{"="*60}')
    print(f'preprocess = {preprocess}')
    print(f'{"="*60}')

    model = KANHyperPDKNN(num_classes=10, in_channels=1).to(device) \
            if args.model == 'KANHyperPDKNN' \
            else BaselineCNN(num_classes=10, in_channels=1).to(device)
    print(f'Model : {model.__class__.__name__}  ({count_params(model):,} params)')

    csv_path = os.path.join(res_dir, f'{args.model}_{preprocess}.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    train_loader, test_loader = get_cifar10_loaders(args.batch_size, args.num_workers, args.data_root)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc  = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_zone = train_one_epoch(
            model, train_loader, optimizer, criterion, device, preprocess)
        te_loss, te_acc          = evaluate(
            model, test_loader,  criterion, device, preprocess)
        scheduler.step()
        best_acc = max(best_acc, te_acc)

        writer.writerow([epoch, f'{tr_loss:.6f}', f'{tr_acc:.6f}',
                         f'{te_loss:.6f}', f'{te_acc:.6f}'])
        csv_file.flush()

        msg = (f'[{epoch:03d}/{args.epochs}] '
               f'train loss={tr_loss:.4f} acc={tr_acc:.3f} | '
               f'test loss={te_loss:.4f} acc={te_acc:.3f} | best={best_acc:.3f}')
        if tr_zone:
            msg += (f" | zones: R={tr_zone['recover']:.3f}"
                    f" P={tr_zone['preserve']:.3f}"
                    f" S={tr_zone['suppress']:.3f}")
        print(msg)

    csv_file.close()
    pth_path = os.path.join(res_dir, f'{args.model}_{preprocess}.pth')
    torch.save(model.state_dict(), pth_path)
    print(f'Saved → {pth_path}')
    print(f'CSV   → {csv_path}')
    return best_acc


def plot_comparison(res_dir, model_name, preprocesses):
    """모든 preprocess의 test accuracy curve를 한 그래프에 비교."""
    plt.figure(figsize=(8, 5))
    for pp in preprocesses:
        csv_path = os.path.join(res_dir, f'{model_name}_{pp}.csv')
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
    plt.title(f'Preprocessing Comparison — {model_name} (grayscale 1ch)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(res_dir, f'comparison_{model_name}.png')
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f'Plot → {fname}')


def main():
    parser = argparse.ArgumentParser(description='Adaptive PDK Rule Experiment')
    parser.add_argument('--model',       type=str, default='KANHyperPDKNN',
                        choices=['KANHyperPDKNN', 'BaselineCNN'])
    parser.add_argument('--preprocess',  type=str, default='adaptive_pdk',
                        choices=['raw', 'global', 'pdk', 'adaptive_pdk'])
    parser.add_argument('--all',         action='store_true',
                        help='raw / global / pdk / adaptive_pdk 전부 실행')
    parser.add_argument('--epochs',      type=int,   default=20)
    parser.add_argument('--batch-size',  type=int,   default=128)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int,   default=2)
    parser.add_argument('--data-root',   type=str,   default='./data')
    parser.add_argument('--res-dir',     type=str,   default='./res')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res_dir = args.res_dir
    os.makedirs(res_dir, exist_ok=True)
    print(f'Device  : {device}')
    print(f'Results → {res_dir}')

    targets = ['raw', 'global', 'pdk', 'adaptive_pdk'] if args.all else [args.preprocess]
    results = {}
    for pp in targets:
        results[pp] = run_one(pp, args, device, res_dir)

    # 최종 성능 비교 출력
    print(f'\n{"="*60}')
    print('최종 Best Accuracy 비교')
    print(f'{"="*60}')
    for pp, acc in results.items():
        print(f'  {pp:15s} : {acc:.4f}')

    # 비교 그래프 저장
    plot_comparison(res_dir, args.model, targets)


if __name__ == '__main__':
    main()
