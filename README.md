# Water segmentation: model, REST API and web UI

Segments water in 12-band multispectral satellite tiles (128x128 px in the training data).
A tile goes in, a binary mask (0 = land, 255 = water) comes out, through a Python script, a Flask REST API
or a small web page.

> **Weights in this repo.** `models/water_unet17.pt` is a *demo* checkpoint trained on synthetic tiles so the
> code runs out of the box. Replace it with the real model before submitting:
> `python train_export.py --data-dir <data> --cv-iou <x> --cv-f1 <y>` (see "Export the real model").
> If you only want to try the app first, create the demo checkpoint with
> `python tools/make_synthetic_data.py --out synthetic_data && python train_export.py --data-dir synthetic_data --epochs 4 --steps 12 --name "DEMO unet-17ch (synthetic data)" --trained-on "SYNTHETIC tiles - plumbing test only"` (about 1 minute on CPU; needs `scipy`).
> Then delete `samples/synthetic_*` and re-run `python test_api.py`.

## Which model is deployed

Selected by out-of-fold (5-fold CV) water IoU from the Week 1-2 comparison:

| Model | Channels | IoU | F1 |
|---|---|---|---|
| **U-Net from scratch + water indices (deployed)** | 17 | **0.7102** | **0.8306** |
| UNet-ResNet34, random init | 17 | 0.7037 | 0.8261 |
| UNet-EffNetB0 pretrained | 17 | 0.7012 | 0.8244 |
| UNet-ResNet34 pretrained | 12 | 0.7007 | 0.8240 |
| UNet-ResNet34 pretrained | 17 | 0.7005 | 0.8238 |
| U-Net from scratch (baseline) | 12 | 0.6848 | 0.8130 |
| UNet-EffNetB0 pretrained | 12 | 0.6590 | 0.7945 |

These numbers were produced with the old first-layer initialisation for the pretrained models. The fixed notebook
(`docs/Water_Segmentation_Week2_fixed.ipynb`, section 10) re-runs them and picks the deployment model again.
Input = 12 bands + NDWI, MNDWI, NDVI, AWEInsh, AWEIsh, normalised with dataset-wise per-channel statistics that are
stored inside the checkpoint.

## First-conv fix (Week 2 feedback)

`adapt_first_conv(strategy="mean")` repeated one averaged RGB filter on every band, so all bands started identical.
`water_seg/models.py` now provides `rgb_random` (default: pretrained filters on R/G/B, independent random filters on
the other bands), `random`, `adapter` (a learnable 1x1 conv N -> 3 in front of an unmodified pretrained encoder)
and the legacy `mean` for reproducing the old numbers.

## Run it

```bash
pip install -r requirements.txt
python app.py                 # http://127.0.0.1:5000  (UI + API)
python predict.py samples/<tile>.tif --out mask.png --overlay overlay.png
python test_api.py            # with the server running
```

Environment variables: `PORT`, `HOST`, `WEIGHTS` (checkpoint path), `MAX_UPLOAD_MB` (default 64).

## API

`POST /predict`: multipart/form-data, field `file` = 12-band `.tif`/`.tiff` (bands first) or `.npy` shaped (12,H,W) / (H,W,12).

| Option | Meaning |
|---|---|
| `format=json` (default) | JSON with `mask_png_base64` (0/255), `overlay_png_base64`, `rgb_png_base64`, `probability_png_base64`, `water_fraction`, `water_pixels`, `width`, `height`, `threshold`, `inference_ms` |
| `format=png` | the binary mask as `image/png` (header `X-Water-Fraction`) |
| `threshold=0..1` | probability cut-off (default stored in the checkpoint, 0.5) |
| `tta=1` | average 8 rotations/flips |
| `include_mask=1` | add the mask as a nested 0/1 list under `mask` |

Errors are JSON `{"error": "..."}`: 400 missing file / bad threshold, 413 file too large, 422 unreadable image,
wrong file type or wrong number of bands. `GET /health` returns the model info, `GET /samples` lists bundled tiles.

```bash
curl -F "file=@samples/tile.tif" http://127.0.0.1:5000/predict
curl -F "file=@samples/tile.tif" "http://127.0.0.1:5000/predict?format=png" -o mask.png
```

## Tests

`python test_api.py` posts every `samples/*.tif` (at least 3 required), checks size, binary values and the PNG
endpoint, computes IoU/F1 when a `_truth.png` exists, runs 8 edge cases and writes `docs/test_results.md` and
`docs/results/`. See that file for the latest run.

## Export the real model

```bash
# 1) in the notebook, section 11 does this for you; or by hand:
python train_export.py --data-dir /path/to/data --cv-iou 0.7102 --cv-f1 0.8306
```
`data/` must contain `images/<id>.tif` and `labels/<id>.png`. The script trains on all tiles with the notebook recipe,
saves `models/water_unet17.pt` and copies 3 sample tiles (+ ground truth) to `samples/`. The sample tiles were in the
training set, so IoU/F1 from `test_api.py` on them are optimistic; the CV numbers above are the honest estimate.

## Layout

```
app.py  predict.py  train_export.py  test_api.py
water_seg/   features.py (indices, normaliser)  models.py (U-Net, first-conv fixes)  inference.py  training.py
templates/index.html   models/   samples/   docs/   tools/make_synthetic_data.py
```
