#!/usr/bin/env python
"""Black-box test of the running API. Start the server first (python app.py), then:

    python test_api.py                      # tests every samples/*.tif against http://127.0.0.1:5000
    python test_api.py --url http://host:5000 --samples samples

For each sample it checks status, mask size, that the mask is strictly binary, the PNG endpoint and (if a
<name>_truth.png sits next to the tile) IoU / F1 against the ground truth. Edge cases (bad input) are tested too.
Results are written to docs/test_results.md and the returned masks to docs/results/.
"""
import argparse
import base64
import io
import sys
from datetime import date
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from water_seg.inference import read_image  # only used to know the expected tile size


def png_to_arr(b: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000"); ap.add_argument("--samples", default="samples")
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    out = Path(a.out); (out / "results").mkdir(parents=True, exist_ok=True)
    tiles = sorted(Path(a.samples).glob("*.tif"))
    if len(tiles) < 3:
        sys.exit(f"Need at least 3 sample tiles in {a.samples}/ (found {len(tiles)}).")

    health = requests.get(f"{a.url}/health", timeout=10).json()
    rows, failures = [], []

    def check(name, cond, detail=""):
        if not cond:
            failures.append(f"{name}: {detail}")
        return cond

    for t in tiles:
        img = read_image(t)
        with open(t, "rb") as f:
            r = requests.post(f"{a.url}/predict", files={"file": (t.name, f)}, timeout=120)
        ok = check(f"{t.name} status", r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}")
        if not ok:
            rows.append((t.name, "FAIL", "", "", "", "")); continue
        d = r.json()
        mask = png_to_arr(base64.b64decode(d["mask_png_base64"]))
        binary = set(np.unique(mask).tolist()) <= {0, 255}
        check(f"{t.name} size", mask.shape == img.shape[1:], f"{mask.shape} vs {img.shape[1:]}")
        check(f"{t.name} binary", binary, f"values {np.unique(mask)}")
        check(f"{t.name} fraction", abs(d["water_fraction"] - (mask > 0).mean()) < 1e-6)
        with open(t, "rb") as f:  # raw-image endpoint
            rp = requests.post(f"{a.url}/predict?format=png", files={"file": (t.name, f)}, timeout=120)
        check(f"{t.name} png endpoint", rp.status_code == 200 and rp.headers["Content-Type"] == "image/png"
              and np.array_equal(png_to_arr(rp.content), mask), "png differs from JSON mask")
        Image.fromarray(mask).save(out / "results" / f"{t.stem}_mask.png")
        Image.open(io.BytesIO(base64.b64decode(d["overlay_png_base64"]))).save(out / "results" / f"{t.stem}_overlay.png")
        score = ""
        truth = t.with_name(t.stem + "_truth.png")
        if truth.exists():
            gt = np.array(Image.open(truth)) > 0
            pr = mask > 0
            tp, fp, fn = (pr & gt).sum(), (pr & ~gt).sum(), (~pr & gt).sum()
            score = f"{tp / (tp + fp + fn + 1e-9):.3f} / {2 * tp / (2 * tp + fp + fn + 1e-9):.3f}"
        rows.append((t.name, "PASS" if ok and binary else "FAIL", f"{d['width']}x{d['height']}",
                     f"{100 * d['water_fraction']:.1f}%", f"{d['inference_ms']:.0f} ms", score))

    # ---- edge cases
    base = read_image(tiles[0])
    def post(name, data, **kw):
        return requests.post(f"{a.url}/predict", files={"file": (name, data)}, data=kw, timeout=60)
    def npy(arr):
        b = io.BytesIO(); np.save(b, arr); return b.getvalue()

    edge = []
    r = post("odd.npy", npy(base[:, :100, :90])); d = r.json() if r.ok else {}
    edge.append(("Tile of 100x90 px (not a multiple of 16)", "200 and mask 100x90",
                 r.status_code == 200 and (d.get("height"), d.get("width")) == (100, 90)))
    r = post("hwc.npy", npy(np.moveaxis(base, 0, -1)))
    edge.append(("Channels-last (H, W, 12) array", "200", r.status_code == 200))
    r = requests.post(f"{a.url}/predict", timeout=30)
    edge.append(("No file attached", "400 with an error message", r.status_code == 400 and "error" in r.json()))
    r = post("notes.txt", b"hello")
    edge.append(("Unsupported file type (.txt)", "422 with an error message", r.status_code == 422 and "error" in r.json()))
    r = post("rgb.npy", npy(base[:3]))
    edge.append(("Image with 3 bands instead of 12", "422 naming the band count", r.status_code == 422 and "3" in r.json().get("error", "")))
    r = post("broken.tif", b"not really a tiff")
    edge.append(("Corrupt .tif", "422", r.status_code == 422))
    with open(tiles[0], "rb") as f:
        r = requests.post(f"{a.url}/predict", files={"file": (tiles[0].name, f)}, data={"threshold": "7"}, timeout=30)
    edge.append(("threshold=7 (out of range)", "400", r.status_code == 400))
    with open(tiles[0], "rb") as f:
        lo = requests.post(f"{a.url}/predict", files={"file": (tiles[0].name, f)}, data={"threshold": "0.05"}, timeout=60).json()
    with open(tiles[0], "rb") as f:
        hi = requests.post(f"{a.url}/predict", files={"file": (tiles[0].name, f)}, data={"threshold": "0.95"}, timeout=60).json()
    edge.append(("Lower threshold never finds less water than a higher one", "water(0.05) >= water(0.95)",
                 lo["water_pixels"] >= hi["water_pixels"]))
    for n, exp, ok in edge:
        check(n, ok, exp)

    md = [f"# API test results\n", f"Run on {date.today().isoformat()} against `{a.url}`.",
          f"Model: **{health.get('model')}** ({health.get('in_channels')} input channels, device {health.get('device')}). "
          f"Trained on: {health.get('trained_on', 'n/a')}.\n"]
    if score_note := any(r[5] for r in rows):
        md.append("IoU / F1 are measured against the ground-truth mask copied next to each tile. "
                  "**These tiles were part of the training set**, so the scores are optimistic and are not the model's "
                  "expected accuracy (see the cross-validation numbers in the README).\n")
    md += ["## Sample inputs\n", "| Tile | Result | Size | Water | Time | IoU / F1 |", "|---|---|---|---|---|---|"]
    md += [f"| {' | '.join(r)} |" for r in rows]
    md += ["\nMasks and overlays: `docs/results/`.\n", "## Edge cases\n", "| Case | Expected | Result |", "|---|---|---|"]
    md += [f"| {n} | {e} | {'PASS' if ok else 'FAIL'} |" for n, e, ok in edge]
    (out / "test_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if failures:
        print("\nFAILURES:\n- " + "\n- ".join(failures)); sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
