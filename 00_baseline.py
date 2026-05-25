"""
00_baseline.py
Position-Dependent Kernel Neural Network (PDKNN) - Baseline

핵심 아이디어:
  기존 CNN은 spatially shared kernel을 사용 (위치에 무관하게 동일한 필터 적용).
  PDKNN은 feature map의 각 spatial position (i, j)마다 다른 kernel을 학습·적용함.
  → local structure + positional context를 동시에 포착 가능.

구조:
  Input → PDKConv layers → BN + ReLU → GlobalAvgPool → FC → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# 1. Position-Dependent Kernel Convolution Layer
# ---------------------------------------------------------------------------

class PDKConv2d(nn.Module):
    """
    Position-Dependent Kernel Convolution.

    각 spatial location (h, w)에 대해 독립적인 kernel weight를 가짐.
    weight shape: (H_out, W_out, C_out, C_in, kH, kW)

    Note:
      - 메모리/연산량이 표준 Conv 대비 H*W 배 증가함.
      - 소규모 feature map 또는 연구 목적에 적합.
      - 실용적 확장: low-rank factorization, kernel interpolation 등으로 경량화 가능.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 input_height, input_width,
                 stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        kH, kW = self.kernel_size

        # 출력 feature map 크기 계산
        self.H_out = (input_height + 2 * padding - kH) // stride + 1
        self.W_out = (input_width  + 2 * padding - kW) // stride + 1

        # position마다 고유한 kernel: (H_out * W_out, C_out, C_in, kH, kW)
        self.weight = nn.Parameter(
            torch.randn(self.H_out * self.W_out, out_channels, in_channels, kH, kW)
            * (2.0 / (in_channels * kH * kW)) ** 0.5  # He init
        )
        self.bias = nn.Parameter(torch.zeros(self.H_out * self.W_out, out_channels))

    def forward(self, x):
        """
        x: (B, C_in, H_in, W_in)
        return: (B, C_out, H_out, W_out)
        """
        B, C_in, H_in, W_in = x.shape
        kH, kW = self.kernel_size

        # padding 적용
        if self.padding > 0:
            x = F.pad(x, [self.padding] * 4)

        H_out, W_out = self.H_out, self.W_out
        out = torch.zeros(B, self.out_channels, H_out, W_out, device=x.device, dtype=x.dtype)

        pos = 0
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                w_start = j * self.stride
                # patch: (B, C_in, kH, kW)
                patch = x[:, :, h_start:h_start + kH, w_start:w_start + kW]
                # kernel: (C_out, C_in, kH, kW)
                kernel = self.weight[pos]
                bias   = self.bias[pos]
                # (B, C_out) = einsum(B C_in kH kW, C_out C_in kH kW)
                out[:, :, i, j] = torch.einsum('bckh,ockh->bo', patch, kernel) + bias
                pos += 1

        return out


# ---------------------------------------------------------------------------
# 2. PDKNN Model
# ---------------------------------------------------------------------------

class PDKNN(nn.Module):
    """
    간단한 2-block PDKNN.
    CIFAR-10 (32x32) 기준으로 설계.

    Block 1: PDKConv(3→32, k=3) → BN → ReLU → MaxPool(2x2)   [32→16]
    Block 2: PDKConv(32→64, k=3) → BN → ReLU → MaxPool(2x2)  [16→8]
    Head:    AdaptiveAvgPool → Flatten → FC(64→num_classes)
    """

    def __init__(self, num_classes=10, img_h=32, img_w=32):
        super().__init__()

        # Block 1
        self.pdk1 = PDKConv2d(3, 32, kernel_size=3,
                               input_height=img_h, input_width=img_w,
                               padding=1)
        self.bn1  = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)  # 32→16

        h1 = img_h // 2
        w1 = img_w // 2

        # Block 2
        self.pdk2 = PDKConv2d(32, 64, kernel_size=3,
                               input_height=h1, input_width=w1,
                               padding=1)
        self.bn2  = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)  # 16→8

        # Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.pdk1(x))))
        x = self.pool2(F.relu(self.bn2(self.pdk2(x))))
        x = self.gap(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 3. Baseline CNN (비교용)
# ---------------------------------------------------------------------------

class BaselineCNN(nn.Module):
    """일반 spatially-shared Conv2d를 사용하는 CNN (PDKNN과 비교용)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = self.gap(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 4. Data
# ---------------------------------------------------------------------------

def get_cifar10_loaders(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 5. Train / Eval loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
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
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- 하이퍼파라미터 ---
    num_epochs  = 10
    batch_size  = 64   # PDKConv은 메모리 많이 쓰므로 작게 설정
    lr          = 1e-3
    num_classes = 10

    # --- 데이터 ---
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

    # --- 모델 선택 ---
    # model = BaselineCNN(num_classes).to(device)
    model = PDKNN(num_classes, img_h=32, img_w=32).to(device)
    print(f"Model: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # --- 학습 ---
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        print(f"[{epoch:02d}/{num_epochs}] "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"test  loss={te_loss:.4f} acc={te_acc:.4f}")

    torch.save(model.state_dict(), 'pdknn_baseline.pth')
    print("Saved: pdknn_baseline.pth")


if __name__ == '__main__':
    main()
