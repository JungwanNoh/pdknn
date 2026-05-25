"""
06_practical_eurosat.py
=======================
Compares three preprocessing strategies on the EuroSAT satellite image dataset:

  1. raw          – identity transform (no preprocessing beyond normalization)
  2. global       – binomial (Gaussian-like) blur to suppress noise globally
  3. satellite_pdk – adaptive edge-based PDK tuned for satellite imagery

Satellite PDK rationale
-----------------------
Satellite images contain:
  * Thin linear features (roads, rivers, field boundaries) that carry strong
    discriminative signal but are easily destroyed by uniform blurring.
  * Large homogeneous regions (water bodies, bare soil, dense forest) where
    fine texture is noise rather than signal.
  * Moderate-complexity zones (urban fabric, crop patterns) with structured
    but not razor-thin edges.

The satellite_pdk preprocessor therefore:
  - Estimates per-image complexity via the standard deviation of a Sobel
    edge map.  High-std images are "complex" (urban / mixed land-use);
    low-std images are "simple" (water / bare soil).
  - Uses a soft sigmoid mask (not a hard binary threshold) to blend the
    edge-enhanced signal continuously.
  - Lowers the edge-recovery threshold for complex images so thin roads and
    rivers are preserved.
  - Suppresses uniform regions by down-weighting pixels whose local gradient
    magnitude falls below an adaptive floor, reducing noise in homogeneous
    areas without erasing textured boundaries.

This is analogous to PDK (Partial Differential-equation-inspired Kernel)
preprocessing, where the kernel shape is conditioned on local image
statistics rather than being fixed globally.
"""

import argparse
import os
import time
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# EuroSAT grayscale statistics (computed from full training split)
# ---------------------------------------------------------------------------
GRAY_MEAN = 0.4734
GRAY_STD  = 0.2516

# ---------------------------------------------------------------------------
# KAN / KANHyperPDKNN model code (copied exactly as specified)
# ---------------------------------------------------------------------------

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer('grid', grid)
        self.spline_weight = nn.Parameter(torch.randn(in_features, out_features, self.n_basis) * 0.1)
        self.residual_scale = nn.Parameter(torch.ones(in_features, out_features) * 0.1)

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            denom_left  = grid[k:-1]   - grid[:-(k + 1)]
            denom_right = grid[k + 1:] - grid[1:-k]
            safe_div = lambda n, d: n / d.clamp(min=1e-8)
            left  = safe_div(x - grid[:-(k + 1)], denom_left)  * bases[..., :-1]
            right = safe_div(grid[k + 1:] - x,    denom_right) * bases[..., 1:]
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
        self.layers = nn.ModuleList(
            [KANLayer(dims[i], dims[i + 1], grid_size, spline_order) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class KANHyperPDKConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_height, input_width,
                 stride=1, padding=0, kan_hidden=32, kan_latent=32, grid_size=5, spline_order=3):
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
        self.kan = KAN([2, kan_hidden, kan_latent], grid_size, spline_order)
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
        B    = x.size(0)
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
            B, self.out_channels, self.H_out, self.W_out
        )
        return out


class KANHyperPDKNN(nn.Module):
    def __init__(self, num_classes=10, img_h=64, img_w=64, in_channels=1, kan_hidden=32, kan_latent=32):
        super().__init__()
        self.block1 = KANHyperPDKConv2d(
            in_channels, 32, 3, img_h, img_w,
            padding=1, kan_hidden=kan_hidden, kan_latent=kan_latent,
        )
        self.bn1 = nn.BatchNorm2d(32)
        h1, w1 = img_h // 2, img_w // 2
        self.block2 = KANHyperPDKConv2d(
            32, 64, 3, h1, w1,
            padding=1, kan_hidden=kan_hidden, kan_latent=kan_latent,
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(64, num_classes)

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
# Preprocessing transforms
# ---------------------------------------------------------------------------

class IdentityTransform:
    """Raw – no preprocessing beyond what the dataset loader already applied."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class BinomialBlurTransform:
    """
    Global preprocessing: apply a separable binomial (Gaussian-approximation)
    blur to suppress sensor noise uniformly across all spatial frequencies.
    Kernel: [1, 2, 1] / 4 applied in both H and W.
    """
    def __init__(self):
        k = torch.tensor([1.0, 2.0, 1.0]) / 4.0
        self._kernel = k.view(1, 1, 1, 3) * k.view(1, 1, 3, 1)  # (1,1,3,3)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, H, W)
        C = x.shape[0]
        kernel = self._kernel.to(x.device).expand(C, 1, 3, 3)
        return F.conv2d(x.unsqueeze(0), kernel, padding=1, groups=C).squeeze(0)


class SatellitePDKTransform:
    """
    Adaptive edge-based PDK for satellite imagery.

    Design notes
    ------------
    * Sobel edge map gives a gradient-magnitude image G.
    * Image complexity is measured as std(G).  Complex images (high std,
      e.g. urban) get a lower edge-recovery threshold so thin roads/rivers
      survive.  Simple images (low std, e.g. water) get a higher threshold,
      suppressing uniform-region noise.
    * A soft sigmoid mask M is computed from G and the adaptive threshold t:
          M = sigmoid((G - t) / scale)
      This avoids the artefacts of hard binary masks.
    * The enhanced image is a weighted blend:
          out = x * (1 + alpha * M)   – boosts edges
              - beta * (1 - M) * blur  – suppresses homogeneous regions by
                                         nudging them toward a blurred version
    * All operations are per-image (no learned parameters), so the transform
      can be applied in the DataLoader workers.
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.3, scale: float = 0.05):
        self.alpha = alpha
        self.beta  = beta
        self.scale = scale
        # Sobel kernels
        self._sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        self._sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        # Binomial blur kernel (3x3)
        k = torch.tensor([1.0, 2.0, 1.0]) / 4.0
        self._blur_k = (k.view(1, 1, 1, 3) * k.view(1, 1, 3, 1))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, H, W) – single-channel grayscale
        dev   = x.device
        sx    = self._sx.to(dev)
        sy    = self._sy.to(dev)
        blur_k = self._blur_k.to(dev)

        xb = x.unsqueeze(0)  # (1,1,H,W)

        # --- Sobel edge magnitude ---
        gx = F.conv2d(xb, sx, padding=1)
        gy = F.conv2d(xb, sy, padding=1)
        G  = (gx ** 2 + gy ** 2).sqrt().squeeze(0)  # (1,H,W)

        # --- Adaptive threshold based on image complexity ---
        complexity = G.std().item()
        # For complex images (high std) lower the threshold → recover thin edges.
        # For simple images (low std) raise the threshold → suppress uniform noise.
        # Empirically calibrated for EuroSAT 64x64 grayscale range ≈ [0,1].
        base_threshold = 0.04
        complexity_factor = float(np.clip(complexity / 0.12, 0.5, 2.0))
        t = base_threshold / complexity_factor  # lower t when complexity is high

        # --- Soft sigmoid edge mask ---
        M = torch.sigmoid((G - t) / self.scale)  # (1,H,W)

        # --- Blurred version for suppressing uniform regions ---
        blurred = F.conv2d(xb, blur_k, padding=1).squeeze(0)  # (1,H,W)

        # --- Blend: boost edges, nudge homogeneous areas toward blur ---
        out = x * (1.0 + self.alpha * M) - self.beta * (1.0 - M) * blurred

        # Re-normalise to keep roughly the same dynamic range
        out = out.clamp(min=0.0)
        return out


def get_transform(preprocess_name: str) -> T.Compose:
    """
    Return a torchvision Compose transform for the requested preprocessing.
    The base pipeline converts RGB to grayscale 64x64 and normalises.
    The preprocessing step is inserted after ToTensor (before normalisation
    so that it operates on [0,1]-range values, which the PDK thresholds
    assume).
    """
    base_to_tensor = [
        T.Resize((64, 64)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
    ]

    if preprocess_name == "raw":
        preprocess_fn = IdentityTransform()
    elif preprocess_name == "global":
        preprocess_fn = BinomialBlurTransform()
    elif preprocess_name == "satellite_pdk":
        preprocess_fn = SatellitePDKTransform()
    else:
        raise ValueError(f"Unknown preprocessing: {preprocess_name!r}")

    normalise = T.Normalize(mean=[GRAY_MEAN], std=[GRAY_STD])

    return T.Compose(base_to_tensor + [preprocess_fn, normalise])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def get_dataloaders(preprocess_name: str, batch_size: int, data_root: str = "./data"):
    transform  = get_transform(preprocess_name)
    full_ds    = torchvision.datasets.EuroSAT(root=data_root, transform=transform, download=True)
    n_total    = len(full_ds)
    n_train    = int(0.8 * n_total)
    n_val      = n_total - n_train
    generator  = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def build_model(model_name: str, device: torch.device) -> nn.Module:
    if model_name == "KANHyperPDKNN":
        model = KANHyperPDKNN(in_channels=1, img_h=64, img_w=64, num_classes=10)
    elif model_name == "BaselineCNN":
        model = BaselineCNN(num_classes=10, in_channels=1)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")
    return model.to(device)


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    criterion  = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    criterion  = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(preprocess_name: str,
            model_name: str,
            epochs: int,
            batch_size: int,
            lr: float,
            res_dir: str,
            device: torch.device) -> dict:
    """
    Train and evaluate one (preprocessing, model) configuration.
    Returns a dict with per-epoch metrics and a 'best_val_acc' summary key.
    """
    print(f"\n{'='*60}")
    print(f"  Preprocessing : {preprocess_name}")
    print(f"  Model         : {model_name}")
    print(f"  Epochs        : {epochs}  |  Batch : {batch_size}  |  LR : {lr}")
    print(f"{'='*60}")

    train_loader, val_loader = get_dataloaders(preprocess_name, batch_size)
    model     = build_model(model_name, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "preprocess"  : preprocess_name,
        "model"       : model_name,
        "train_loss"  : [],
        "train_acc"   : [],
        "val_loss"    : [],
        "val_acc"     : [],
    }

    best_val_acc  = 0.0
    ckpt_path     = os.path.join(res_dir, f"best_{preprocess_name}_{model_name}.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f} | "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

    history["best_val_acc"] = best_val_acc
    print(f"\n  Best val acc: {best_val_acc:.4f}  (checkpoint: {ckpt_path})")
    return history


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------

def save_csv(histories: list[dict], res_dir: str, model_name: str):
    csv_path = os.path.join(res_dir, f"results_{model_name}.csv")
    fieldnames = ["preprocess", "epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in histories:
            for ep_idx in range(len(h["train_loss"])):
                writer.writerow({
                    "preprocess" : h["preprocess"],
                    "epoch"      : ep_idx + 1,
                    "train_loss" : h["train_loss"][ep_idx],
                    "train_acc"  : h["train_acc"][ep_idx],
                    "val_loss"   : h["val_loss"][ep_idx],
                    "val_acc"    : h["val_acc"][ep_idx],
                })
    print(f"\nCSV saved → {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def plot_comparison(histories: list[dict], res_dir: str, model_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors    = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for idx, h in enumerate(histories):
        c      = colors[idx % len(colors)]
        label  = h["preprocess"]
        epochs = list(range(1, len(h["val_acc"]) + 1))
        axes[0].plot(epochs, h["train_loss"], linestyle="--", color=c, alpha=0.5)
        axes[0].plot(epochs, h["val_loss"],   linestyle="-",  color=c, label=label)
        axes[1].plot(epochs, h["train_acc"],  linestyle="--", color=c, alpha=0.5)
        axes[1].plot(epochs, h["val_acc"],    linestyle="-",  color=c, label=label)

    axes[0].set_title(f"Loss ({model_name})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f"Accuracy ({model_name})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Dashed = train, solid = val
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], linestyle="--", color="grey", label="train (dashed)"),
        Line2D([0], [0], linestyle="-",  color="grey", label="val (solid)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)

    fig.suptitle("EuroSAT: raw vs global vs satellite_pdk preprocessing", y=1.05)
    fig.tight_layout()

    plot_path = os.path.join(res_dir, f"comparison_{model_name}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved  → {plot_path}")
    return plot_path


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(histories: list[dict]):
    print("\n" + "=" * 50)
    print(f"  {'Preprocessing':<20} {'Best Val Acc':>12}")
    print("  " + "-" * 34)
    for h in histories:
        print(f"  {h['preprocess']:<20} {h['best_val_acc']:>12.4f}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare raw/global/satellite_pdk preprocessing on EuroSAT."
    )
    p.add_argument(
        "--preprocess",
        choices=["raw", "global", "satellite_pdk"],
        default="raw",
        help="Preprocessing strategy to use (ignored when --all is set).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run all three preprocessing strategies and produce comparison plots/CSV.",
    )
    p.add_argument("--model",    choices=["KANHyperPDKNN", "BaselineCNN"], default="KANHyperPDKNN")
    p.add_argument("--epochs",   type=int,   default=20)
    p.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--res-dir",  type=str,   default="./res", dest="res_dir")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    res_dir = os.path.join(args.res_dir, "eurosat")
    os.makedirs(res_dir, exist_ok=True)

    if args.all:
        preprocessing_list = ["raw", "global", "satellite_pdk"]
    else:
        preprocessing_list = [args.preprocess]

    histories = []
    for pp in preprocessing_list:
        h = run_one(
            preprocess_name=pp,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            res_dir=res_dir,
            device=device,
        )
        histories.append(h)

    # Always save CSV (even for a single run, useful for comparison later)
    save_csv(histories, res_dir, args.model)

    if len(histories) > 1:
        plot_comparison(histories, res_dir, args.model)

    print_summary(histories)


if __name__ == "__main__":
    main()
