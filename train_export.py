#!/usr/bin/env python
"""Train the final model on ALL labelled tiles and export a self-contained deployment checkpoint.

    python train_export.py --data-dir /path/to/data            # data/images/*.tif + data/labels/*.png

The checkpoint stores the weights, the model spec, the dataset-wise normaliser statistics and the decision
threshold, so ``predict.py`` / ``app.py`` need nothing else. It also copies a few sample tiles into samples/
for testing the API. Defaults reproduce the Week 1 model that won the cross-validation
(U-Net from scratch, 12 bands + 5 water indices = 17 channels, base 32).

The training-set score printed at the end is NOT the expected accuracy - quote the cross-validation numbers
(``--cv-iou/--cv-f1``) from the notebook in the README instead.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from water_seg.features import ChannelNormalizer, build_features
from water_seg.models import UNet, build_smp_model
from water_seg.training import fit, iou_f1, load_dataset, set_seed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="models/water_unet17.pt")
    ap.add_argument("--model", choices=["unet_scratch", "smp"], default="unet_scratch")
    ap.add_argument("--features", choices=["engineered", "raw"], default="engineered")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--arch", default="unet"); ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--init-strategy", default="rgb_random", help="smp only: rgb_random | random | adapter | rgb_zero")
    ap.add_argument("--no-pretrained", action="store_true", help="smp only: skip ImageNet weights")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None); ap.add_argument("--trained-on", default="real dataset (all labelled tiles)")
    ap.add_argument("--cv-iou", type=float); ap.add_argument("--cv-f1", type=float)
    ap.add_argument("--samples-dir", default="samples"); ap.add_argument("--n-samples", type=int, default=3)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y, ids = load_dataset(a.data_dir)
    print(f"{len(ids)} tiles, X {X.shape}, water {100 * Y.mean():.1f}% | device {device}")
    F = build_features(X) if a.features == "engineered" else X
    norm = ChannelNormalizer().fit(F)  # dataset-wise statistics, saved with the weights
    Z = norm.transform(F)

    set_seed(a.seed)
    if a.model == "unet_scratch":
        spec = dict(type="unet_scratch", base=a.base, dropout=0.0)
        model, enc_factor = UNet(Z.shape[1], base=a.base), None
    else:
        spec = dict(type="smp", arch=a.arch, encoder=a.encoder, init_strategy=a.init_strategy)
        model, enc_factor = build_smp_model(a.arch, a.encoder, Z.shape[1], not a.no_pretrained, a.init_strategy), 0.3
    fit(model, Z, Y, device, a.epochs, a.steps, a.batch, a.lr, seed=a.seed, enc_lr_factor=enc_factor)

    model.eval()
    with torch.no_grad():
        prob = np.concatenate([torch.sigmoid(model(torch.from_numpy(Z[i:i + 16]).to(device)))[:, 0].cpu().numpy()
                               for i in range(0, len(Z), 16)])
    iou, f1 = iou_f1(prob, Y, a.threshold)
    print(f"training-set IoU {iou:.3f} | F1 {f1:.3f}  (optimistic - do not report as accuracy)")

    meta = dict(name=a.name or f"{a.model}-{Z.shape[1]}ch", trained_on=a.trained_on, n_train_tiles=len(ids),
                epochs=a.epochs, train_set_iou=round(iou, 4))
    if a.cv_iou is not None: meta["cv_iou"] = a.cv_iou
    if a.cv_f1 is not None: meta["cv_f1"] = a.cv_f1
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(version=1, model=spec, in_channels=Z.shape[1], features=a.features, norm=norm.to_dict(),
                    threshold=a.threshold, meta=meta, state={k: v.cpu() for k, v in model.state_dict().items()}), out)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # copy a low / medium / high water-coverage tile into samples/ for API testing
    frac = Y.mean(axis=(1, 2)); order = np.argsort(frac)
    picks = sorted({int(order[int(q * (len(order) - 1))]) for q in np.linspace(0.1, 0.9, a.n_samples)})
    sd = Path(a.samples_dir); sd.mkdir(parents=True, exist_ok=True)
    for j in picks:
        src = Path(a.data_dir) / "images" / f"{ids[j]}.tif"
        if src.exists():
            stem = f"tile_{ids[j]}_water{int(100 * frac[j]):02d}pct"
            shutil.copy(src, sd / f"{stem}.tif")
            for lab in (Path(a.data_dir) / "labels").glob(f"{ids[j]}.png"):  # ground truth for test_api.py
                shutil.copy(lab, sd / f"{stem}_truth.png")
    print(f"copied {len(picks)} sample tiles to {sd}/")


if __name__ == "__main__":
    main()
