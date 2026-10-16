"""Training / export helpers. Recipe mirrors the notebook (BCE+Dice, AdamW, OneCycle, rot90/flip augmentation)."""
import random
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .features import ChannelNormalizer, build_features
from .models import param_groups


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _id(p: Path):
    m = re.match(r"(\d+)", p.stem)
    return int(m.group(1)) if m else None


def load_dataset(data_dir):
    """Pair images/<id>.tif with labels/<id>.png (exact name preferred, like the notebook). -> X (N,12,H,W), Y (N,H,W), ids"""
    import rasterio
    data_dir = Path(data_dir)
    images = {_id(p): p for p in sorted((data_dir / "images").glob("*.tif")) if _id(p) is not None}
    labels = defaultdict(list)
    for p in sorted((data_dir / "labels").glob("*.png")):
        if _id(p) is not None:
            labels[_id(p)].append(p)
    ids = sorted(i for i in images if i in labels)
    if not ids:
        raise SystemExit(f"No image/label pairs found under {data_dir} (expected images/*.tif and labels/*.png)")
    def primary(i):
        exact = [p for p in labels[i] if p.stem == str(i)]
        return exact[0] if exact else sorted(labels[i])[0]
    def mask(p):
        m = np.array(Image.open(p))
        if m.ndim == 3:
            m = m.max(axis=-1)
        return (m > 0).astype(np.float32)
    X, Y = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in ids:
            with rasterio.open(images[i]) as src:
                X.append(src.read().astype(np.float32))
            Y.append(mask(primary(i)))
    return np.nan_to_num(np.stack(X)), np.stack(Y), ids


class BCEDiceLoss(nn.Module):
    def __init__(self, w_bce=0.5, w_dice=0.5, smooth=1.0):
        super().__init__()
        self.bce, self.w_bce, self.w_dice, self.smooth = nn.BCEWithLogitsLoss(), w_bce, w_dice, smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        dice = 1 - (2 * (p * targets).sum() + self.smooth) / (p.sum() + targets.sum() + self.smooth)
        return self.w_bce * self.bce(logits, targets) + self.w_dice * dice


class SegDataset(Dataset):
    def __init__(self, X, Y, augment=True):
        self.X, self.Y, self.augment = torch.from_numpy(X), torch.from_numpy(Y)[:, None], augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i], self.Y[i]
        if self.augment:
            k = random.randint(0, 3)
            x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2))
            if random.random() < 0.5:
                x, y = torch.flip(x, (2,)), torch.flip(y, (2,))
        return x.contiguous(), y.contiguous()


def iou_f1(prob, Y, thr=0.5):
    pred, gt = prob >= thr, Y > 0.5
    tp, fp, fn = (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()
    iou = tp / (tp + fp + fn + 1e-9)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return float(iou), float(f1)


def fit(model, X, Y, device, epochs=15, steps=32, batch=8, lr=1e-3, wd=1e-4, seed=42, enc_lr_factor=None, log=print):
    """Train on (X, Y) (already normalised). enc_lr_factor!=None -> separate encoder lr (pretrained models)."""
    set_seed(seed)
    model.to(device)
    ds = SegDataset(X, Y)
    loader = DataLoader(ds, batch_size=batch, sampler=RandomSampler(ds, replacement=True, num_samples=steps * batch),
                        num_workers=0, drop_last=True)
    crit = BCEDiceLoss()
    if enc_lr_factor is None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd); max_lr = lr
    else:
        opt = torch.optim.AdamW(param_groups(model, lr * enc_lr_factor, lr, wd)); max_lr = [lr * enc_lr_factor, lr]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=epochs * len(loader), pct_start=0.2)
    for ep in range(1, epochs + 1):
        model.train(); run = n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item() * len(xb); n += len(xb)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            log(f"  epoch {ep:02d}/{epochs} | train loss {run / n:.4f}")
    return model
