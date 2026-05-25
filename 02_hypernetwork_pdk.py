import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# 1. B-Spline KAN Layer
# ---------------------------------------------------------------------------

class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3,
                 grid_range: tuple = (-1.0, 1.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1
        )
        self.register_buffer('grid', grid)

        self.spline_weight = nn.Parameter(
            torch.randn(in_features, out_features, self.n_basis) * 0.1
        )
        self.residual_scale = nn.Parameter(
            torch.ones(in_features, out_features) * 0.1
        )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()

        for k in range(1, self.spline_order + 1):
            denom_left = grid[k:-1] - grid[:-(k+1)]
            denom_right = grid[k+1:] - grid[1:-k]

            def safe_div(num, den):
                return num / den.clamp(min=1e-8)

            left = safe_div(x - grid[:-(k+1)], denom_left) * bases[..., :-1]
            right = safe_div(grid[k+1:] - x, denom_right) * bases[..., 1:]
            bases = left + right

        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = self.b_splines(x)
        spline_out = torch.einsum('bin,ion->bo', bs, self.spline_weight)
        silu_out = torch.einsum('bi,io->bo', F.silu(x), self.residual_scale)
        return spline_out + silu_out


class KAN(nn.Module):
    def __init__(self, dims: list[int], grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLayer(dims[i], dims[i + 1], grid_size, spline_order)
            for i in range(len(dims) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# 2. Sinusoidal Position Encoding
# ---------------------------------------------------------------------------

def sinusoidal_pos_encoding(coords: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 4 == 0, "dim must be divisible by 4 (2 coords × 2 sin/cos)"
    half = dim // 2
    freqs = torch.arange(half // 2, device=coords.device).float()
    freqs = 1.0 / (10000 ** (2 * freqs / half))

    enc_list = []
    for c in range(2):
        v = coords[:, c:c+1] * freqs.unsqueeze(0)
        enc_list += [torch.sin(v), torch.cos(v)]

    return torch.cat(enc_list, dim=-1)


# ---------------------------------------------------------------------------
# 3. KAN Hypernetwork Conv Layer
# ---------------------------------------------------------------------------

class KANHyperPDKConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 input_height: int, input_width: int,
                 stride: int = 1, padding: int = 0,
                 pos_enc_dim: int = 16,
                 kan_hidden: int = 32,
                 kan_latent: int = 32,
                 grid_size: int = 5,
                 spline_order: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pos_enc_dim = pos_enc_dim

        kH = kW = kernel_size
        self.H_out = (input_height + 2 * padding - kH) // stride + 1
        self.W_out = (input_width + 2 * padding - kW) // stride + 1
        self.kernel_flat = out_channels * in_channels * kH * kW

        self.kan = KAN([pos_enc_dim, kan_hidden, kan_latent], grid_size, spline_order)
        self.kernel_decoder = nn.Linear(kan_latent, self.kernel_flat)
        self.bias_decoder = nn.Linear(kan_latent, out_channels)

        ys = torch.linspace(0, 1, self.H_out)
        xs = torch.linspace(0, 1, self.W_out)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
        self.register_buffer('coords', coords)

        enc = sinusoidal_pos_encoding(coords, pos_enc_dim)
        enc = (enc - enc.mean()) / (enc.std() + 1e-6)
        self.register_buffer('pos_enc', enc)

    def _generate_kernels(self):
        P = self.H_out * self.W_out
        latent = self.kan(self.pos_enc)
        kernels = self.kernel_decoder(latent)
        kernels = kernels.view(P, self.out_channels, -1)
        biases = self.bias_decoder(latent)
        return kernels, biases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        out = out + biases.unsqueeze(0)
        out = out.permute(0, 2, 1).view(B, self.out_channels, self.H_out, self.W_out)
        return out


# ---------------------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------------------

class KANHyperPDKNN(nn.Module):
    def __init__(self, num_classes: int = 10, img_h: int = 32, img_w: int = 32,
                 pos_enc_dim: int = 16, kan_hidden: int = 32, kan_latent: int = 32):
        super().__init__()
        self.block1 = KANHyperPDKConv2d(
            in_channels=3, out_channels=32, kernel_size=3,
            input_height=img_h, input_width=img_w, padding=1,
            pos_enc_dim=pos_enc_dim, kan_hidden=kan_hidden, kan_latent=kan_latent,
        )
        self.bn1 = nn.BatchNorm2d(32)

        h1, w1 = img_h // 2, img_w // 2

        self.block2 = KANHyperPDKConv2d(
            in_channels=32, out_channels=64, kernel_size=3,
            input_height=h1, input_width=w1, padding=1,
            pos_enc_dim=pos_enc_dim, kan_hidden=kan_hidden, kan_latent=kan_latent,
        )
        self.bn2 = nn.BatchNorm2d(64)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.bn1(self.block1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.block2(x))), 2)
        x = self.gap(x).flatten(1)
        return self.fc(x)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        return self.fc(self.gap(x).flatten(1))


# ---------------------------------------------------------------------------
# 5. PDK preprocessing
# ---------------------------------------------------------------------------

def _make_kernel(kernel_type: str) -> torch.Tensor:
    if kernel_type == 'identity':
        k = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    elif kernel_type == 'binomial':
        k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0
    elif kernel_type == 'gaussian':
        k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0
    elif kernel_type == 'highboost':
        k = torch.tensor([[0., -0.25, 0.], [-0.25, 2.0, -0.25], [0., -0.25, 0.]])
    else:
        raise ValueError(f'Unsupported kernel_type: {kernel_type}')
    return k.float()


def _apply_kernel_rgb(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    w = kernel.to(x.device, x.dtype).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    y = F.conv2d(x, w, padding=1, groups=c)
    return torch.clamp(y, -3.0, 3.0)


def _sobel_mag(gray: torch.Tensor) -> torch.Tensor:
    sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(gray, sx, padding=1)
    gy = F.conv2d(gray, sy, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    return 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]


@torch.no_grad()
def pdk_preprocess_batch(x: torch.Tensor):
    raw_gray = _to_gray(x)
    global_k = _make_kernel('binomial')
    recover_k = _make_kernel('highboost')
    suppress_k = _make_kernel('gaussian')

    global_out = _apply_kernel_rgb(x, global_k)
    global_gray = _to_gray(global_out)

    edge_raw = _sobel_mag(raw_gray)
    edge_global = _sobel_mag(global_gray)

    missed = torch.relu(edge_raw - edge_global)
    flat = missed.flatten(1)
    maxv = flat.max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8
    missed_n = missed / maxv

    recover_mask = (missed_n > 0.35).float()
    median_edge = edge_global.flatten(1).median(dim=1)[0].view(-1, 1, 1, 1)
    preserve_mask = ((edge_global > median_edge) * (1 - recover_mask)).float()
    suppress_mask = 1.0 - torch.clamp(recover_mask + preserve_mask, 0, 1)

    recover_out = _apply_kernel_rgb(x, recover_k)
    suppress_out = _apply_kernel_rgb(x, suppress_k)
    preserve_out = x

    out = preserve_mask * preserve_out + recover_mask * recover_out + suppress_mask * suppress_out
    out = torch.clamp(out, -3.0, 3.0)

    zone_stats = {
        'recover': recover_mask.mean().item(),
        'preserve': preserve_mask.mean().item(),
        'suppress': suppress_mask.mean().item(),
    }
    return out, zone_stats


# ---------------------------------------------------------------------------
# 6. Data
# ---------------------------------------------------------------------------

def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 2, data_root: str = './data'):
    T_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    T_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=T_train)
    test = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=T_test)

    return (
        DataLoader(train, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(test, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


# ---------------------------------------------------------------------------
# 7. Train / Eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, preprocess: str):
    model.train()
    total_loss = correct = total = 0
    zone_accum = {'recover': 0.0, 'preserve': 0.0, 'suppress': 0.0}
    zone_batches = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if preprocess == 'pdk':
            imgs, zone_stats = pdk_preprocess_batch(imgs)
            for k in zone_accum:
                zone_accum[k] += zone_stats[k]
            zone_batches += 1

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    zone_stats = None
    if zone_batches > 0:
        zone_stats = {k: v / zone_batches for k, v in zone_accum.items()}

    return total_loss / total, correct / total, zone_stats


@torch.no_grad()
def evaluate(model, loader, criterion, device, preprocess: str):
    model.eval()
    total_loss = correct = total = 0
    zone_accum = {'recover': 0.0, 'preserve': 0.0, 'suppress': 0.0}
    zone_batches = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if preprocess == 'pdk':
            imgs, zone_stats = pdk_preprocess_batch(imgs)
            for k in zone_accum:
                zone_accum[k] += zone_stats[k]
            zone_batches += 1

        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    zone_stats = None
    if zone_batches > 0:
        zone_stats = {k: v / zone_batches for k, v in zone_accum.items()}
    return total_loss / total, correct / total, zone_stats


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='KANHyperPDKNN', choices=['KANHyperPDKNN', 'BaselineCNN'])
    parser.add_argument('--preprocess', type=str, default='raw', choices=['raw', 'pdk'])
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--save-path', type=str, default='')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"Preprocess mode : {args.preprocess}")

    models = {
        'KANHyperPDKNN': KANHyperPDKNN(num_classes=10).to(device),
        'BaselineCNN': BaselineCNN(num_classes=10).to(device),
    }
    for name, m in models.items():
        print(f"{name:20s} | params: {count_params(m):>10,}")

    model = models[args.model]
    print(f"\nTraining: {model.__class__.__name__}")

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    prev_lr = optimizer.param_groups[0]['lr']

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, train_zone = train_one_epoch(model, train_loader, optimizer, criterion, device, args.preprocess)
        te_loss, te_acc, test_zone = evaluate(model, test_loader, criterion, device, args.preprocess)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr_changed = abs(current_lr - prev_lr) > 1e-12
        best_acc = max(best_acc, te_acc)

        msg = (
            f"[{epoch:03d}/{args.epochs}] train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"test loss={te_loss:.4f} acc={te_acc:.3f} | best={best_acc:.3f} | lr={current_lr:.6e}"
        )
        if lr_changed:
            msg += f" | LR changed: {prev_lr:.6e} -> {current_lr:.6e}"
        if args.preprocess == 'pdk' and train_zone is not None:
            msg += (
                f" | train zones: recover={train_zone['recover']:.3f},"
                f" preserve={train_zone['preserve']:.3f},"
                f" suppress={train_zone['suppress']:.3f}"
            )
        print(msg)
        prev_lr = current_lr

    save_path = args.save_path or f"{args.model}_{args.preprocess}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()
