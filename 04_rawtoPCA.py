"""
04_rawtoPCA.py
Classification (raw / global / pdk) + 학습 후 Kernel PCA 분석

03_global_pdk_HyperPDK.py 와 동일한 classification 구조를 유지하면서,
학습이 끝난 KANHyperPDKNN 의 block1 / block2 에서 위치별 kernel weight 를 추출하고
PCA 로 1D 로 압축하여 raw (x, y) 좌표와 연결한다.

분석 파이프라인:
  1. 학습 (또는 --load-path 로 기존 모델 로드)
  2. block1 / block2 의 _generate_kernels() 호출 → (P, kernel_flat)
  3. PCA → (P, 1)  : "위치별 kernel 의 주요 변화 방향"
  4. heatmap 시각화 : PCA 값을 spatial grid 에 매핑
  5. (선택) pykan KAN([2, 5, 1]) 으로 raw (x,y) → PCA 값 학습
           → model.plot() 으로 symbolic 해석 시도

실행 예시:
  python 04_rawtoPCA.py --preprocess raw --epochs 20
  python 04_rawtoPCA.py --preprocess pdk --load-path KANHyperPDKNN_pdk.pth --pca-only
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# 1. KAN (03_global_pdk_HyperPDK.py 와 동일)
# ---------------------------------------------------------------------------

class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3,
                 grid_range: tuple = (-1.0, 1.0)):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
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
        x    = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            denom_left  = grid[k:-1]  - grid[:-(k+1)]
            denom_right = grid[k+1:]  - grid[1:-k]
            safe_div = lambda num, den: num / den.clamp(min=1e-8)
            left  = safe_div(x - grid[:-(k+1)], denom_left)  * bases[..., :-1]
            right = safe_div(grid[k+1:] - x,    denom_right) * bases[..., 1:]
            bases = left + right
        return bases

    def forward(self, x):
        bs        = self.b_splines(x)
        spline_out = torch.einsum('bin,ion->bo', bs, self.spline_weight)
        silu_out   = torch.einsum('bi,io->bo', F.silu(x), self.residual_scale)
        return spline_out + silu_out


class KAN(nn.Module):
    def __init__(self, dims: list[int], grid_size: int = 5, spline_order: int = 3):
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
# 2. KANHyperPDKConv2d — raw (x, y) 입력 버전
#    sinusoidal encoding 제거: KAN 입력 = raw (y, x) 2D 좌표 ∈ [-1, 1]
#    → KAN이 학습하는 함수가 직접 "위치 → kernel" 로 해석 가능
# ---------------------------------------------------------------------------

class KANHyperPDKConv2d(nn.Module):
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

        # KAN 입력: raw (y, x) 2D → kan_hidden → kan_latent
        self.kan            = KAN([2, kan_hidden, kan_latent], grid_size, spline_order)
        self.kernel_decoder = nn.Linear(kan_latent, self.kernel_flat)
        self.bias_decoder   = nn.Linear(kan_latent, out_channels)

        # raw 좌표 [-1, 1] 로 정규화 (KAN grid_range 에 맞춤)
        ys = torch.linspace(-1, 1, self.H_out)
        xs = torch.linspace(-1, 1, self.W_out)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
        self.register_buffer('coords', coords)     # (P, 2)  raw (y, x) ∈ [-1, 1]

    def _generate_kernels(self):
        P       = self.H_out * self.W_out
        latent  = self.kan(self.coords)             # raw (y,x) → latent
        kernels = self.kernel_decoder(latent).view(P, self.out_channels, -1)
        biases  = self.bias_decoder(latent)
        return kernels, biases

    def forward(self, x):
        B    = x.size(0)
        kH   = kW = self.kernel_size
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


# ---------------------------------------------------------------------------
# 4. Models (03 와 동일)
# ---------------------------------------------------------------------------

class KANHyperPDKNN(nn.Module):
    def __init__(self, num_classes=10, img_h=32, img_w=32,
                 kan_hidden=32, kan_latent=32):
        super().__init__()
        self.block1 = KANHyperPDKConv2d(1,  32, 3, img_h, img_w, padding=1,
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
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  32, 3, padding=1)
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
# 5. Preprocessing (03 와 동일)
# ---------------------------------------------------------------------------

def _make_kernel(t):
    if t == 'identity':
        k = torch.tensor([[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]])
    elif t in ('binomial', 'gaussian'):
        k = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]]) / 16.0
    elif t == 'highboost':
        k = torch.tensor([[0.,-0.25,0.],[-0.25,2.0,-0.25],[0.,-0.25,0.]])
    else:
        raise ValueError(t)
    return k.float()

def _apply_kernel_rgb(x, k):
    ch = x.shape[1]
    w  = k.to(x.device, x.dtype).view(1,1,3,3).repeat(ch,1,1,1)
    return torch.clamp(F.conv2d(x, w, padding=1, groups=ch), -3.0, 3.0)

def global_preprocess_batch(x):
    return _apply_kernel_rgb(x, _make_kernel('binomial'))

def _sobel_mag(g):
    sx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device=g.device, dtype=g.dtype).view(1,1,3,3)
    sy = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], device=g.device, dtype=g.dtype).view(1,1,3,3)
    return torch.sqrt(F.conv2d(g,sx,padding=1)**2 + F.conv2d(g,sy,padding=1)**2 + 1e-8)

def _to_gray(x):
    return 0.2989*x[:,0:1] + 0.5870*x[:,1:2] + 0.1140*x[:,2:3]

@torch.no_grad()
def pdk_preprocess_batch(x):
    # x: (B, 1, H, W) — grayscale 입력
    edge_raw    = _sobel_mag(x)
    global_out  = _apply_kernel_rgb(x, _make_kernel('binomial'))
    edge_global = _sobel_mag(global_out)
    missed      = torch.relu(edge_raw - edge_global)
    maxv        = missed.flatten(1).max(dim=1)[0].view(-1,1,1,1) + 1e-8
    missed_n    = missed / maxv
    recover_mask  = (missed_n > 0.35).float()
    med           = edge_global.flatten(1).median(dim=1)[0].view(-1,1,1,1)
    preserve_mask = ((edge_global > med) * (1 - recover_mask)).float()
    suppress_mask = 1.0 - torch.clamp(recover_mask + preserve_mask, 0, 1)
    out = (preserve_mask * x
           + recover_mask  * _apply_kernel_rgb(x, _make_kernel('highboost'))
           + suppress_mask * _apply_kernel_rgb(x, _make_kernel('gaussian')))
    zone_stats = {k: v.mean().item() for k, v in
                  zip(['recover','preserve','suppress'], [recover_mask, preserve_mask, suppress_mask])}
    return torch.clamp(out, -3.0, 3.0), zone_stats


# ---------------------------------------------------------------------------
# 6. Data (03 와 동일)
# ---------------------------------------------------------------------------

def get_cifar10_loaders(batch_size=128, num_workers=2, data_root='./data'):
    T_train = transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4734,), (0.2516,)),   # CIFAR-10 grayscale
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
# 7. Train / Eval (03 와 동일)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, preprocess):
    model.train()
    total_loss = correct = total = 0
    zone_accum = {'recover': 0., 'preserve': 0., 'suppress': 0.}
    zone_batches = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if preprocess == 'global':
            imgs = global_preprocess_batch(imgs)
        elif preprocess == 'pdk':
            imgs, zs = pdk_preprocess_batch(imgs)
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
    zs = {k: v/zone_batches for k,v in zone_accum.items()} if zone_batches else None
    return total_loss/total, correct/total, zs

@torch.no_grad()
def evaluate(model, loader, criterion, device, preprocess):
    model.eval()
    total_loss = correct = total = 0
    zone_accum = {'recover': 0., 'preserve': 0., 'suppress': 0.}
    zone_batches = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if preprocess == 'global':
            imgs = global_preprocess_batch(imgs)
        elif preprocess == 'pdk':
            imgs, zs = pdk_preprocess_batch(imgs)
            for k in zone_accum: zone_accum[k] += zs[k]
            zone_batches += 1
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    zs = {k: v/zone_batches for k,v in zone_accum.items()} if zone_batches else None
    return total_loss/total, correct/total, zs


# ---------------------------------------------------------------------------
# 8. Kernel PCA 분석
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_kernels(conv: KANHyperPDKConv2d) -> np.ndarray:
    """
    KANHyperPDKConv2d 에서 위치별 kernel weight 를 추출.
    return: (P, kernel_flat)  numpy array
    """
    kernels, _ = conv._generate_kernels()          # (P, C_out, C_in*kH*kW)
    P = kernels.shape[0]
    return kernels.reshape(P, -1).cpu().numpy()    # (P, kernel_flat)


def run_pca(kernels: np.ndarray, n_components: int = 3):
    """
    PCA 적용.
    return:
      pca_vals : (P, n_components)
      explained : explained variance ratio array
    """
    pca = PCA(n_components=n_components)
    pca_vals = pca.fit_transform(kernels)          # (P, n_components)
    return pca_vals, pca.explained_variance_ratio_


def visualize_kernel_pca(pca_vals: np.ndarray,
                         H: int, W: int,
                         block_name: str,
                         explained: np.ndarray,
                         res_dir: str,
                         preprocess: str):
    n_comp = pca_vals.shape[1]
    fig, axes = plt.subplots(1, n_comp, figsize=(5 * n_comp, 4))
    if n_comp == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        heatmap = pca_vals[:, i].reshape(H, W)
        im = ax.imshow(heatmap, cmap='RdBu_r', origin='upper')
        ax.set_title(f'{block_name} PC{i+1}\n(var {explained[i]*100:.1f}%)')
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f'Kernel PCA [{preprocess}] — {block_name}', y=1.02)
    plt.tight_layout()
    safe = block_name.lower().replace(' ', '_').replace('×', 'x')
    fname = os.path.join(res_dir, f'kernel_pca_{preprocess}_{safe}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {fname} 저장 완료')


def analyze_pca(model: KANHyperPDKNN, res_dir: str,
                preprocess: str, try_symbolic: bool = False):
    print('\n' + '='*60)
    print('Kernel PCA 분석')
    print('='*60)

    for block_name, block in [('Block1 (32x32)', model.block1),
                               ('Block2 (16x16)', model.block2)]:
        print(f'\n[{block_name}]')
        kernels = extract_kernels(block)
        print(f'  kernel matrix shape : {kernels.shape}')

        pca_vals, explained = run_pca(kernels, n_components=3)
        print(f'  PC1 explained var   : {explained[0]*100:.1f}%')
        print(f'  PC2 explained var   : {explained[1]*100:.1f}%')
        print(f'  PC3 explained var   : {explained[2]*100:.1f}%')

        visualize_kernel_pca(
            pca_vals, block.H_out, block.W_out,
            block_name, explained, res_dir, preprocess
        )

        if try_symbolic:
            _fit_kan_symbolic(block.coords.cpu().numpy(), pca_vals[:, 0],
                              block_name, res_dir, preprocess)


def _fit_kan_symbolic(coords_np: np.ndarray, pc1: np.ndarray,
                      block_name: str, res_dir: str, preprocess: str):
    try:
        from kan import KAN as PyKAN

        coords_t = torch.tensor(coords_np, dtype=torch.float32)
        pc1_t    = torch.tensor(pc1,       dtype=torch.float32).unsqueeze(-1)
        n        = len(coords_t)
        split    = int(n * 0.8)
        dataset  = {
            'train_input': coords_t[:split], 'train_label': pc1_t[:split],
            'test_input' : coords_t[split:], 'test_label' : pc1_t[split:],
        }

        print(f'\n  [{block_name}] pykan KAN([2, 5, 1]) fitting on (x,y) → PC1 ...')
        kan_model = PyKAN(width=[2, 5, 1], grid=5, k=3, seed=42)
        results   = kan_model.fit(dataset, opt='LBFGS', steps=50, lamb=0.001)
        print(f'  train RMSE: {results["train_loss"][-1]:.5f} | '
              f'test RMSE: {results["test_loss"][-1]:.5f}')

        kan_model.plot(beta=3, title=f'KAN: (x,y)→PC1 [{preprocess}] {block_name}')
        safe  = block_name.lower()[:6].replace(' ', '_')
        fname = os.path.join(res_dir, f'kan_pc1_{preprocess}_{safe}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  → {fname} 저장 완료')

        try:
            kan_model.auto_symbolic(lib=['sin', 'cos', 'x^2', 'x', 'exp'])
            print(f'  symbolic formula: {kan_model.symbolic_formula()[0][0]}')
        except Exception as e:
            print(f'  symbolic 추론 실패: {e}')

    except ImportError:
        print('  pykan 미설치 → symbolic 분석 생략 (pip install pykan)')


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

import os
import csv

def count_params(m): return sum(p.numel() for p in m.parameters())


def run_one(preprocess: str, args, device, res_dir: str):
    """단일 preprocess 모드로 학습 + CSV 저장 + PCA 분석."""
    print(f'\n{"="*60}')
    print(f'preprocess = {preprocess}')
    print(f'{"="*60}')

    # 모델 초기화 (매 run마다 새로 생성)
    model = KANHyperPDKNN(num_classes=10).to(device) \
            if args.model == 'KANHyperPDKNN' \
            else BaselineCNN(num_classes=10).to(device)
    print(f'Model      : {model.__class__.__name__}  '
          f'({count_params(model):,} params)')

    if args.load_path:
        model.load_state_dict(torch.load(args.load_path, map_location=device))
        print(f'Loaded     : {args.load_path}')

    # CSV 준비
    csv_path = os.path.join(res_dir, f'{args.model}_{preprocess}.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    if not args.pca_only:
        train_loader, test_loader = get_cifar10_loaders(
            args.batch_size, args.num_workers, args.data_root)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        best_acc  = 0.0

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc, tr_zone = train_one_epoch(
                model, train_loader, optimizer, criterion, device, preprocess)
            te_loss, te_acc, _       = evaluate(
                model, test_loader,  criterion, device, preprocess)
            scheduler.step()
            best_acc = max(best_acc, te_acc)

            writer.writerow([epoch,
                             f'{tr_loss:.6f}', f'{tr_acc:.6f}',
                             f'{te_loss:.6f}', f'{te_acc:.6f}'])
            csv_file.flush()

            msg = (f'[{epoch:03d}/{args.epochs}] '
                   f'train loss={tr_loss:.4f} acc={tr_acc:.3f} | '
                   f'test loss={te_loss:.4f} acc={te_acc:.3f} | '
                   f'best={best_acc:.3f}')
            if preprocess == 'pdk' and tr_zone:
                msg += (f" | zones: R={tr_zone['recover']:.2f}"
                        f" P={tr_zone['preserve']:.2f}"
                        f" S={tr_zone['suppress']:.2f}")
            print(msg)

        pth_path = os.path.join(res_dir, f'{args.model}_{preprocess}.pth')
        torch.save(model.state_dict(), pth_path)
        print(f'Saved      : {pth_path}')

    csv_file.close()
    print(f'CSV        : {csv_path}')

    # PCA 분석
    if isinstance(model, KANHyperPDKNN):
        model.eval()
        analyze_pca(model, res_dir=res_dir, preprocess=preprocess,
                    try_symbolic=args.symbolic)
    else:
        print('BaselineCNN 은 PCA 분석 불가.')


def main():
    parser = argparse.ArgumentParser(description='Classification + Kernel PCA Analysis')
    parser.add_argument('--model',       type=str, default='KANHyperPDKNN',
                        choices=['KANHyperPDKNN', 'BaselineCNN'])
    parser.add_argument('--preprocess',  type=str, default='raw',
                        choices=['raw', 'global', 'pdk'])
    parser.add_argument('--all',         action='store_true',
                        help='raw / global / pdk 세 가지 모두 순차 실행')
    parser.add_argument('--epochs',      type=int,   default=20)
    parser.add_argument('--batch-size',  type=int,   default=128)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int,   default=2)
    parser.add_argument('--data-root',   type=str,   default='./data')
    parser.add_argument('--load-path',   type=str,   default='')
    parser.add_argument('--pca-only',    action='store_true')
    parser.add_argument('--symbolic',    action='store_true')
    parser.add_argument('--res-dir',     type=str,   default='./res')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res_dir = args.res_dir
    os.makedirs(res_dir, exist_ok=True)
    print(f'Device   : {device}')
    print(f'Results  → {res_dir}')

    targets = ['raw', 'global', 'pdk'] if args.all else [args.preprocess]
    for preprocess in targets:
        run_one(preprocess, args, device, res_dir)


if __name__ == '__main__':
    main()
