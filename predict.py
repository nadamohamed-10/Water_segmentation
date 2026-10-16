#!/usr/bin/env python
"""Command-line inference.

    python predict.py samples/tile_low_water.tif --out mask.png --overlay overlay.png

Reads a 12-band image and writes a binary mask (0 = land, 255 = water) as PNG.
"""
import argparse
import sys

import numpy as np
from PIL import Image

from water_seg.inference import DEFAULT_WEIGHTS, InputError, WaterSegmenter, overlay, read_image, rgb_preview


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="12-band .tif/.tiff (or .npy shaped (12,H,W))")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--out", default="mask.png", help="binary mask PNG (0/255)")
    ap.add_argument("--overlay", help="optional PNG with the water drawn over the true-colour image")
    ap.add_argument("--threshold", type=float, help="probability threshold (default: value stored in the checkpoint)")
    ap.add_argument("--tta", action="store_true", help="average 8 rotations/flips (slower, slightly smoother)")
    a = ap.parse_args()

    try:
        img = read_image(a.image)
    except InputError as e:
        sys.exit(f"error: {e}")
    seg = WaterSegmenter(a.weights)
    res = seg.predict(img, a.threshold, a.tta)
    Image.fromarray(res["mask"] * 255).save(a.out)
    if a.overlay:
        Image.fromarray(overlay(rgb_preview(img), res["mask"])).save(a.overlay)
    print(f"{a.image}: {img.shape[1]}x{img.shape[2]} px | water {100 * res['mask'].mean():.1f}% | "
          f"threshold {res['threshold']:.2f} | {res['ms']:.0f} ms -> {a.out}")


if __name__ == "__main__":
    main()
