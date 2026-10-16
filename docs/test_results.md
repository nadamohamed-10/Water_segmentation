# API test results

Run on 2026-10-16 against `http://127.0.0.1:5000`.
Model: **DEMO unet-17ch (synthetic data)** (17 input channels, device cpu). Trained on: SYNTHETIC tiles - plumbing test only.

IoU / F1 are measured against the ground-truth mask copied next to each tile. **These tiles were part of the training set**, so the scores are optimistic and are not the model's expected accuracy (see the cross-validation numbers in the README).

## Sample inputs

| Tile | Result | Size | Water | Time | IoU / F1 |
|---|---|---|---|---|---|
| synthetic_tile_10_water47pct.tif | PASS | 128x128 | 47.2% | 64 ms | 0.998 / 0.999 |
| synthetic_tile_47_water75pct.tif | PASS | 128x128 | 75.9% | 40 ms | 0.999 / 1.000 |
| synthetic_tile_4_water05pct.tif | PASS | 128x128 | 5.3% | 47 ms | 0.983 / 0.991 |
| tile_10_water47pct.tif | PASS | 128x128 | 47.2% | 52 ms | 0.998 / 0.999 |
| tile_47_water75pct.tif | PASS | 128x128 | 75.9% | 48 ms | 0.999 / 1.000 |
| tile_4_water05pct.tif | PASS | 128x128 | 5.3% | 48 ms | 0.983 / 0.991 |

Masks and overlays: `docs/results/`.

## Edge cases

| Case | Expected | Result |
|---|---|---|
| Tile of 100x90 px (not a multiple of 16) | 200 and mask 100x90 | PASS |
| Channels-last (H, W, 12) array | 200 | PASS |
| No file attached | 400 with an error message | PASS |
| Unsupported file type (.txt) | 422 with an error message | PASS |
| Image with 3 bands instead of 12 | 422 naming the band count | PASS |
| Corrupt .tif | 422 | PASS |
| threshold=7 (out of range) | 400 | PASS |
| Lower threshold never finds less water than a higher one | water(0.05) >= water(0.95) | PASS |
