#!/usr/bin/env python
"""Generate SYNTHETIC 12-band tiles + masks to smoke-test the pipeline when the real dataset is not at hand.

Band means loosely follow the real dataset statistics reported in the notebook (water is dark in NIR/SWIR).
Models trained on this data are for plumbing tests only - never report their scores.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import gaussian_filter

# per band: (water mean, land mean)
SIG = [(363, 408), (479, 500), (700, 640), (951, 900), (962, 2480), (818, 2365), (596, 1615),
       (126, 95), (-100, 350), (155, 352), (51, 30), (37, 0.3)]


def make_tile(rng, size, water_frac):
    field = gaussian_filter(rng.standard_normal((size, size)), rng.uniform(6, 14))
    thr = np.quantile(field, 1 - water_frac) if 0 < water_frac < 1 else (np.inf if water_frac <= 0 else -np.inf)
    mask = (field >= thr).astype(np.uint8)
    soft = gaussian_filter(mask.astype(np.float32), 1.0)
    tex = [gaussian_filter(rng.standard_normal((size, size)), 2.5) for _ in range(3)]
    img = np.empty((12, size, size), np.float32)
    for b, (w, l) in enumerate(SIG):
        base = soft * w + (1 - soft) * l
        spread = 0.12 * max(abs(w), abs(l), 50)
        img[b] = base + spread * (0.8 * tex[b % 3] + 0.4 * rng.standard_normal((size, size)))
    return img, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_data"); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    fracs = np.concatenate([[0.0, 1.0], rng.uniform(0.02, 0.85, a.n - 2)])
    for i, f in enumerate(fracs):
        img, mask = make_tile(rng, a.size, f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(out / "images" / f"{i}.tif", "w", driver="GTiff", height=a.size, width=a.size,
                               count=12, dtype="float32") as dst:
                dst.write(img)
        Image.fromarray(mask).save(out / "labels" / f"{i}.png")
    print(f"wrote {a.n} synthetic tiles to {out}/")


if __name__ == "__main__":
    main()
