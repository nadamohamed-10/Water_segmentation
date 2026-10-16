"""Inference: 12-channel image in, binary water mask out."""
import io
import time
from pathlib import Path

import numpy as np
import torch

from .features import NUM_RAW_CHANNELS, RGB_BANDS, ChannelNormalizer, build_features
from .models import build_from_spec

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "water_unet17.pt"


class InputError(ValueError):
    """The uploaded image cannot be used (wrong type, wrong number of channels, ...)."""


def read_image(source, filename: str = "") -> np.ndarray:
    """Read a 12-band image as float32 (12, H, W).

    ``source`` is a path or raw bytes. Supported: GeoTIFF/TIFF (bands first, like rasterio) and .npy.
    """
    name = (filename or (str(source) if isinstance(source, (str, Path)) else "")).lower()
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
    try:
        if name.endswith(".npy"):
            arr = np.load(io.BytesIO(data), allow_pickle=False)
        elif name.endswith((".tif", ".tiff")):
            arr = _read_tiff(data)
        else:
            raise InputError("Unsupported file type. Upload a 12-band .tif/.tiff or a .npy array shaped (12, H, W).")
    except InputError:
        raise
    except Exception as exc:  # corrupt files, odd dtypes, ...
        raise InputError(f"Could not read the image: {exc}") from exc
    if arr.ndim != 3:
        raise InputError(f"Expected a 3-D image (12, H, W), got an array with shape {arr.shape}.")
    if arr.shape[0] != NUM_RAW_CHANNELS and arr.shape[-1] == NUM_RAW_CHANNELS:
        arr = np.moveaxis(arr, -1, 0)  # (H, W, 12) -> (12, H, W)
    if arr.shape[0] != NUM_RAW_CHANNELS:
        raise InputError(f"Expected {NUM_RAW_CHANNELS} bands, but the image has {arr.shape[0]} (shape {arr.shape}).")
    arr = np.nan_to_num(arr.astype(np.float32))
    if min(arr.shape[1:]) < 16:
        raise InputError(f"Image is too small ({arr.shape[1]}x{arr.shape[2]}); minimum is 16x16.")
    return arr


def _read_tiff(data: bytes) -> np.ndarray:
    try:
        import rasterio
        from rasterio.io import MemoryFile
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # NotGeoreferencedWarning: tiles have no geotransform
            with MemoryFile(data) as mem, mem.open() as src:
                return src.read()
    except ImportError:  # rasterio missing -> tifffile fallback
        import tifffile
        return tifffile.imread(io.BytesIO(data))


def _stretch(a, lo=2, hi=98):
    l, h = np.percentile(a, [lo, hi])
    return np.clip((a - l) / (h - l + 1e-6), 0, 1)


def rgb_preview(img: np.ndarray) -> np.ndarray:
    """True-colour preview (uint8, H, W, 3) using the Red/Green/Blue bands with a 2-98 % stretch."""
    return (np.stack([_stretch(img[b]) for b in RGB_BANDS], axis=-1) * 255).astype(np.uint8)


def overlay(rgb: np.ndarray, mask: np.ndarray, color=(31, 111, 235), alpha=0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    return out.astype(np.uint8)


class WaterSegmenter:
    def __init__(self, weights_path=DEFAULT_WEIGHTS, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.in_channels = ckpt["in_channels"]
        self.features = ckpt["features"]  # "engineered" (12 bands + 5 indices) or "raw"
        self.threshold = float(ckpt.get("threshold", 0.5))
        self.meta = ckpt.get("meta", {})
        self.norm = ChannelNormalizer.from_dict(ckpt["norm"])
        self.model = build_from_spec(ckpt["model"], self.in_channels)
        self.model.load_state_dict(ckpt["state"])
        self.model.to(self.device).eval()

    # ---- preprocessing identical to training: indices -> dataset-wise clip + z-score
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        x = build_features(img) if self.features == "engineered" else img
        return self.norm.transform(x[None])[0]

    @torch.no_grad()
    def _prob(self, x: torch.Tensor, tta: bool) -> torch.Tensor:
        if not tta:
            return torch.sigmoid(self.model(x))
        acc = 0
        for k in range(4):
            for flip in (False, True):
                v = torch.rot90(x, k, (2, 3))
                if flip:
                    v = torch.flip(v, (3,))
                p = torch.sigmoid(self.model(v))
                if flip:
                    p = torch.flip(p, (3,))
                acc = acc + torch.rot90(p, -k, (2, 3))
        return acc / 8

    def predict(self, img: np.ndarray, threshold: float | None = None, tta: bool = False) -> dict:
        """img: float32 (12, H, W). Returns probability map, binary mask (uint8 0/1) and timing."""
        t0 = time.perf_counter()
        thr = self.threshold if threshold is None else float(threshold)
        x = self.preprocess(img)
        h, w = x.shape[1:]
        ph, pw = (-h) % 16, (-w) % 16  # U-Net pools 4x -> sides must be divisible by 16
        if ph or pw:
            mode = "reflect" if ph < h and pw < w else "edge"
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode=mode)
        xt = torch.from_numpy(np.ascontiguousarray(x))[None].to(self.device)
        prob = self._prob(xt, tta)[0, 0, :h, :w].cpu().numpy()
        return dict(prob=prob, mask=(prob >= thr).astype(np.uint8), threshold=thr,
                    ms=(time.perf_counter() - t0) * 1000)

    def info(self) -> dict:
        return dict(model=self.meta.get("name", "water-unet"), in_channels=self.in_channels,
                    features=self.features, threshold=self.threshold, device=str(self.device),
                    **{k: v for k, v in self.meta.items() if k != "name"})
