"""
01_hypernetwork_kan.py
KAN-based Hypernetwork for Position-Dependent Kernel CNN

핵심 아이디어:
  - 소형 KAN 이 position (i/H, j/W) → kernel weights 를 *생성* (Hypernetwork)
  - KAN 은 B-spline 기반 학습 가능한 단변수 함수의 합 → smooth positional variation 포착에 유리
  - MLP hypernetwork 대비 동일 표현력에서 파라미터 효율적

전체 구조:
  [pos (2D)] → sinusoidal encoding → KAN → latent → Linear decoder → kernel(C_out, C_in, kH, kW)

파라미터 절감:
  naive PDKConv : 5.65M
  KAN hypernet  : ~700K  (≈8× 절감)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# 1. B-Spline KAN Layer
# ---------------------------------------------------------------------------

class KANLayer(nn.Module):
    """
    단일 KAN layer.

    y_j = Σ_i  [ w_b_{ij} · silu(x_i)  +  Σ_k c_{ijk} · B_k(x_i) ]
          ↑ residual (linear-like)            ↑ B-spline (learnable nonlinear)

    파라미터:
      spline_weight : (in_features, out_features, grid_size + spline_order)
      residual_scale: (in_features, out_features)
    """

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3,
                 grid_range: tuple = (-1.0, 1.0)):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        self.n_basis      = grid_size + spline_order  # number of B-spline basis funcs

        # Extended uniform grid (fixed, not learned)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1
        )
        self.register_buffer('grid', grid)

        # Learnable B-spline coefficients
        self.spline_weight = nn.Parameter(
            torch.randn(in_features, out_features, self.n_basis) * 0.1
        )
        # Residual (SiLU) scale
        self.residual_scale = nn.Parameter(
            torch.ones(in_features, out_features) * 0.1
        )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Cox-de Boor 재귀로 B-spline basis 계산.

        x    : (batch, in_features)
        return: (batch, in_features, n_basis)
        """
        x = x.unsqueeze(-1)                        # (B, in, 1)
        grid = self.grid                            # (G + 2k + 1,)

        # order-1 basis (indicator)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()   # (B, in, G+2k)

        for k in range(1, self.spline_order + 1):
            denom_left  = grid[k:-1]     - grid[:-(k+1)]        # (n,)
            denom_right = grid[k+1:]     - grid[1:-k]           # (n,)

            safe_div = lambda num, den: num / den.clamp(min=1e-8)

            left  = safe_div(x - grid[:-(k+1)], denom_left)  * bases[..., :-1]
            right = safe_div(grid[k+1:] - x,    denom_right) * bases[..., 1:]
            bases = left + right                               # (B, in, n_basis)

        return bases                                           # (B, in, n_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x     : (batch, in_features)  — 정규화된 값 (grid_range 내)
        return: (batch, out_features)
        """
        # B-spline contribution
        bs = self.b_splines(x)                      # (B, in, n_basis)
        # spline_weight: (in, out, n_basis)
        spline_out = torch.einsum('bin,ion->bo', bs, self.spline_weight)

        # Residual SiLU contribution
        # residual_scale: (in, out)
        silu_out = torch.einsum('bi,io->bo', F.silu(x), self.residual_scale)

        return spline_out + silu_out


class KAN(nn.Module):
    """KAN layer 를 쌓은 소형 네트워크."""

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
    """
    coords : (N, 2)  — 정규화된 (i/H, j/W) ∈ [0,1]
    dim    : 출력 차원 (짝수)
    return : (N, dim)
    """
    assert dim % 4 == 0, "dim must be divisible by 4 (2 coords × 2 sin/cos)"
    half = dim // 2
    freqs = torch.arange(half // 2, device=coords.device).float()
    freqs = 1.0 / (10000 ** (2 * freqs / half))   # (half//2,)

    enc_list = []
    for c in range(2):                             # x coord, y coord
        v = coords[:, c:c+1] * freqs.unsqueeze(0) # (N, half//2)
        enc_list += [torch.sin(v), torch.cos(v)]

    return torch.cat(enc_list, dim=-1)             # (N, dim)


# ---------------------------------------------------------------------------
# 3. KAN Hypernetwork Conv Layer
# ---------------------------------------------------------------------------

class KANHyperPDKConv2d(nn.Module):
    """
    KAN Hypernetwork 기반 Position-Dependent Kernel Convolution.

    kernel_weight(i, j) = LinearDecoder( KAN( sinusoidal_enc(i/H, j/W) ) )

    forward 에서:
      1. 모든 위치 (i,j)의 좌표를 생성
      2. KAN 을 한 번에 통과 → 모든 위치의 kernel 생성
      3. F.unfold 로 패치 추출 → 위치별 batched matmul
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 input_height: int, input_width: int,
                 stride: int = 1, padding: int = 0,
                 pos_enc_dim: int = 16,
                 kan_hidden: int = 32,
                 kan_latent: int = 32,
                 grid_size: int = 5,
                 spline_order: int = 3):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.pos_enc_dim  = pos_enc_dim

        kH = kW = kernel_size
        self.H_out = (input_height + 2 * padding - kH) // stride + 1
        self.W_out = (input_width  + 2 * padding - kW) // stride + 1
        self.kernel_flat = out_channels * in_channels * kH * kW

        # KAN: pos_enc_dim → kan_hidden → kan_latent
        self.kan = KAN([pos_enc_dim, kan_hidden, kan_latent], grid_size, spline_order)

        # Linear decoder: latent → full kernel weights
        self.kernel_decoder = nn.Linear(kan_latent, self.kernel_flat)
        # Linear decoder: latent → bias
        self.bias_decoder   = nn.Linear(kan_latent, out_channels)

        # Pre-generate position grid (fixed)
        ys = torch.linspace(0, 1, self.H_out)
        xs = torch.linspace(0, 1, self.W_out)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        # (H*W, 2) — normalized coords ∈ [0,1]
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
        self.register_buffer('coords', coords)

        # Sinusoidal encoding of coords is position-fixed → cache it
        enc = sinusoidal_pos_encoding(coords, pos_enc_dim)  # (H*W, pos_enc_dim)
        # Rescale to [-1, 1] for KAN grid
        enc = (enc - enc.mean()) / (enc.std() + 1e-6)
        self.register_buffer('pos_enc', enc)

    def _generate_kernels(self):
        """
        KAN 을 통과해 모든 위치의 kernel weight 를 생성.
        return:
          kernels: (P, C_out, C_in*kH*kW)
          biases : (P, C_out)
          P = H_out * W_out
        """
        P = self.H_out * self.W_out
        latent  = self.kan(self.pos_enc)               # (P, latent)
        kernels = self.kernel_decoder(latent)           # (P, C_out*C_in*kH*kW)
        kernels = kernels.view(P, self.out_channels, -1)  # (P, C_out, C_in*kH*kW)
        biases  = self.bias_decoder(latent)             # (P, C_out)
        return kernels, biases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, C_in, H_in, W_in)
        return: (B, C_out, H_out, W_out)
        """
        B = x.size(0)
        kH = kW = self.kernel_size

        # 1. 모든 위치의 kernel 생성
        kernels, biases = self._generate_kernels()     # (P, C_out, k), (P, C_out)
        # k = C_in * kH * kW

        # 2. unfold 로 모든 패치 한번에 추출
        patches = F.unfold(x, kernel_size=kH,
                           padding=self.padding,
                           stride=self.stride)          # (B, C_in*kH*kW, P)

        # 3. 위치별 matmul: out[b, c, p] = Σ_k kernels[p,c,k] * patches[b,k,p]
        #    patches: (B, k, P) → (B, P, k)
        patches = patches.permute(0, 2, 1)              # (B, P, k)
        # kernels: (P, C_out, k) → (P, k, C_out)
        kernels = kernels.permute(0, 2, 1)              # (P, k, C_out)
        # bmm over P positions (treat B and P together)
        # out: (B, P, C_out)
        out = torch.bmm(
            patches.reshape(B * self.H_out * self.W_out, 1, -1),          # (B*P, 1, k)
            kernels.unsqueeze(0).expand(B, -1, -1, -1).reshape(
                B * self.H_out * self.W_out, -1, self.out_channels)        # (B*P, k, C_out)
        ).squeeze(1)                                                        # (B*P, C_out)

        out = out.view(B, self.H_out * self.W_out, self.out_channels)      # (B, P, C_out)

        # bias 추가: biases (P, C_out)
        out = out + biases.unsqueeze(0)                                     # (B, P, C_out)

        # 4. reshape to (B, C_out, H_out, W_out)
        out = out.permute(0, 2, 1).view(B, self.out_channels,
                                        self.H_out, self.W_out)
        return out


# ---------------------------------------------------------------------------
# 4. KAN Hypernetwork PDKNN Model
# ---------------------------------------------------------------------------

class KANHyperPDKNN(nn.Module):
    """
    Block 1: KANHyperPDKConv(3→32, k=3) → BN → ReLU → MaxPool(2)  [32→16]
    Block 2: KANHyperPDKConv(32→64, k=3) → BN → ReLU → MaxPool(2) [16→8]
    Head   : AdaptiveAvgPool → Flatten → FC(64→num_classes)
    """

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
        self.fc  = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.bn1(self.block1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.block2(x))), 2)
        x = self.gap(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 5. Baseline CNN (비교용)
# ---------------------------------------------------------------------------

class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        return self.fc(self.gap(x).flatten(1))


# ---------------------------------------------------------------------------
# 6. Data
# ---------------------------------------------------------------------------

def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 2):
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
    train = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=T_train)
    test  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=T_test)
    return (DataLoader(train, batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
            DataLoader(test,  batch_size, shuffle=False, num_workers=num_workers, pin_memory=True))


# ---------------------------------------------------------------------------
# 7. Train / Eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # --- 하이퍼파라미터 ---
    num_epochs = 160
    batch_size = 128
    lr         = 1e-3

    # --- 모델 비교 ---
    models = {
        'KANHyperPDKNN': KANHyperPDKNN(num_classes=10).to(device),
        'BaselineCNN'  : BaselineCNN(num_classes=10).to(device),
    }
    for name, m in models.items():
        print(f"{name:20s} | params: {count_params(m):>10,}")

    # --- 학습할 모델 선택 ---
    model = models['KANHyperPDKNN']
    print(f"\nTraining: {model.__class__.__name__}")

    train_loader, test_loader = get_cifar10_loaders(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        print(f"[{epoch:02d}/{num_epochs}] "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"test  loss={te_loss:.4f} acc={te_acc:.3f}")

    torch.save(model.state_dict(), 'kan_hyper_pdknn.pth')
    print("Saved: kan_hyper_pdknn.pth")


if __name__ == '__main__':
    main()
