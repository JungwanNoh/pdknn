"""
28_save_bsd_grayscale.py
=================================================
BSDS300 train 이미지를 grayscale로 저장.
- PDK simulation(27번)과 동일한 포맷
  · center crop → square
  · resize to img_size
  · matplotlib 저장 (dpi 600+)
- 출력: <out-dir>/<original_name>.png

Run:
  python 28_save_bsd_grayscale.py
  python 28_save_bsd_grayscale.py \\
      --src ./data/BSD300/images/train \\
      --out ./res/bsd_gray \\
      --size 256 --dpi 600 --n -1
"""

import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def save_gray(arr, path, dpi):
    """27번 save_img와 동일한 매개변수 (dpi만 인자화)"""
    arr = np.clip(arr, 0, 1)
    fig, ax = plt.subplots(
        1, 1,
        figsize=(arr.shape[1] / 100, arr.shape[0] / 100),
        dpi=dpi,
    )
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def collect_paths(src):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(src, "**", ext), recursive=True)
        paths += glob.glob(os.path.join(src, ext))
    return sorted(set(paths))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src",  type=str, default="./data/BSD300/images/train",
                   help="BSDS300 train 폴더")
    p.add_argument("--out",  type=str, default="./res/bsd_gray",
                   help="출력 폴더")
    p.add_argument("--size", type=int, default=256,
                   help="정사각 출력 크기 (PDK simulation과 맞추기)")
    p.add_argument("--dpi",  type=int, default=600,
                   help="저장 dpi (>=600)")
    p.add_argument("--n",    type=int, default=-1,
                   help="저장할 이미지 수 (-1=전부)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = collect_paths(args.src)
    if args.n > 0:
        paths = paths[:args.n]

    print(f"[Found] {len(paths)} images in {args.src}")
    print(f"[Saving] grayscale {args.size}×{args.size} @ dpi={args.dpi}")
    print(f"[Out]    {args.out}")

    saved = 0
    for src in paths:
        try:
            img = Image.open(src).convert("L")
            w, h = img.size
            s = min(w, h)
            img = img.crop(((w - s) // 2, (h - s) // 2,
                           (w + s) // 2, (h + s) // 2))
            img = img.resize((args.size, args.size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.

            name = os.path.splitext(os.path.basename(src))[0] + ".png"
            out_path = os.path.join(args.out, name)
            save_gray(arr, out_path, args.dpi)
            saved += 1
            if saved % 50 == 0:
                print(f"  ...saved {saved}/{len(paths)}")
        except Exception as e:
            print(f"  [Skip] {src} → {e}")

    print(f"\n[Done] {saved}/{len(paths)} images saved to {args.out}")


if __name__ == "__main__":
    main()