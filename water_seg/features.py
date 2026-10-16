"""Spectral indices and dataset-wise channel normalisation (identical to the Week 1/2 notebook)."""
import numpy as np

EPS = 1e-6
B_BLUE, B_GREEN, B_RED, B_NIR, B_SWIR1, B_SWIR2 = 1, 2, 3, 4, 5, 6
RGB_BANDS = (B_RED, B_GREEN, B_BLUE)  # band index for R, G, B
INDEX_NAMES = ["NDWI", "MNDWI", "NDVI", "AWEInsh", "AWEIsh"]
NUM_RAW_CHANNELS = 12


def normalized_difference(a, b):
    den = a + b
    out = np.zeros_like(a, dtype=np.float32)
    np.divide(a - b, den, out=out, where=np.abs(den) > EPS)
    return np.clip(out, -1.0, 1.0)


def compute_indices(img):
    band = lambda i: img[..., i, :, :]
    blue, green, red, nir, swir1, swir2 = (band(i) for i in (B_BLUE, B_GREEN, B_RED, B_NIR, B_SWIR1, B_SWIR2))
    return {
        "NDWI": normalized_difference(green, nir),
        "MNDWI": normalized_difference(green, swir1),
        "NDVI": normalized_difference(nir, red),
        "AWEInsh": 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2),
        "AWEIsh": blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2,
    }


def build_features(X):
    """(..., 12, H, W) -> (..., 17, H, W): 12 raw bands + 5 water indices."""
    idx = compute_indices(X)
    eng = np.stack([idx[n] for n in INDEX_NAMES], axis=-3).astype(np.float32)
    return np.concatenate([X, eng], axis=-3).astype(np.float32)


class ChannelNormalizer:
    """Per-channel clip to [q_lo, q_hi] percentiles, then z-score. Fit on the TRAINING set only."""

    def __init__(self, clip_q=(0.5, 99.5)):
        self.clip_q = clip_q

    def fit(self, X):
        C = X.shape[1]
        flat = X.transpose(1, 0, 2, 3).reshape(C, -1)
        q = np.percentile(flat, self.clip_q, axis=1)
        self.lo, self.hi = q[0], q[1]
        clipped = np.clip(flat, self.lo[:, None], self.hi[:, None])
        self.mean = clipped.mean(1)
        self.std = clipped.std(1)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, X):
        v = lambda a: np.asarray(a).reshape(1, -1, 1, 1)
        return ((np.clip(X, v(self.lo), v(self.hi)) - v(self.mean)) / v(self.std)).astype(np.float32)

    def to_dict(self):
        return dict(lo=np.asarray(self.lo), hi=np.asarray(self.hi), mean=np.asarray(self.mean),
                    std=np.asarray(self.std), clip_q=tuple(self.clip_q))

    @classmethod
    def from_dict(cls, d):
        o = cls(tuple(d["clip_q"]))
        o.lo, o.hi, o.mean, o.std = (np.asarray(d[k], dtype=np.float32) for k in ("lo", "hi", "mean", "std"))
        return o
