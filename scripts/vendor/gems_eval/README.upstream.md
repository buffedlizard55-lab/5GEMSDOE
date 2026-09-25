# gems-eval

Local evaluation for the DOE **GEMS Prize Challenge** on DrivenData (competition 306): fault-probability rasters over western Nevada, scored with a distance-weighted Tversky index on faults that are *not* in the training labels.

Three things this package gives you:

1. **`dti`** — the distance-weighted Tversky index exactly as the problem page defines it: α = 0.2, β = 0.8, 300 m triangular kernel, TP as a *max per truth pixel*, continuous predictions, no threshold.
2. **A held-out-segment protocol** that imitates how the board scores: hold out whole fault segments from training, score only on those, exclude a buffer around the remaining known faults.
3. **A submission validator** and a **top-fraction trim** post-processor.

Everything works on numpy arrays; `rasterio` is only needed for the GeoTIFF helpers and the CLI.

## What this is not

It is **not the organizer's code**. It implements the formula published on the problem page (TP_w = Σ_g max_x p(x)·k(d), FP_w = Σ_x p(x)·[1 − max_g k(d)], FN_w = |G| − TP_w). One data point of calibration so far: the organizer's reference model scores 0.0435 on our held-out segments and 0.0387 on the leaderboard. Treat numbers from this package as internal until you have your own leaderboard anchor.

## Why the held-out-segment protocol

The training labels are the known (USGS Quaternary + INGENIOUS) faults. The leaderboard scores newly identified faults and excludes the known ones. A model that re-draws the known faults therefore looks excellent under any local split that contains them and scores near zero on the board. Our first submission scored 0.305 locally under a random patch split and **0.0387** on the leaderboard.

The protocol: label the connected fault segments, hold out a fraction (default 30%) from training, and score predictions on the held-out segments only, with a 3-pixel buffer around the training faults excluded from scoring. Evaluation raster codes: `1` held-out truth, `2` excluded buffer, `0` scoreable background, `-1` nodata.

Three properties of the score worth knowing before you tune anything (measured on one U-Net trained on the held-out split):

- **TP saturates per truth pixel.** Writing r = TP_w/|G| and φ = FP_w/|G|, DTI = r / (0.2·r + 0.8 + 0.2·φ): a thick band of predictions earns nothing beyond its best pixel per truth pixel and pays for every off-line pixel.
- **Thin lines win.** Raw probabilities 0.088 → binarised top-2% 0.106 → **skeleton of the top-5% 0.132** (+49%). Adding a 3-px band back around the skeleton loses.
- **It is dominated by the false-positive term when the scored faults are sparse.** Thinning the truth to a third of the held-out segments cuts the same prediction's score roughly threefold; under sparse truth a tighter cut (top 2–3%) is safer than top 5%.

## Install

```bash
pip install -e ".[io]"        # numpy, scipy, rasterio
pip install -e ".[dev]" && pytest -q
```

## CLI

```bash
# 1. build the held-out split from the competition labels
gems-eval holdout --labels existing_faults.tif --out seghold/ --frac 0.3 --seed 0 --buffer 3

# 2. train on seghold/train_labels.tif, then score a prediction on the held-out segments
gems-eval score --pred pred.tif --eval-labels seghold/eval_labels.tif --top 0.02
#   dti: 0.0883   coverage_ge_0.5: 0.0298   mean_pred: 0.0349   dti_top0.02: 0.1056   dti_skel_top0.02: 0.1122

# 3. turn a full-label prediction into the file we actually submit: known faults masked (3 px),
#    top 5% of the remaining study-area pixels, thinned to one-pixel lines at 1.0, NaN outside the polygon
gems-eval trim --pred pred_all_labels.tif --template existing_faults.tif --out submission.tif --top 0.05 --skeleton --mask-known 3

# 4. check the file against the published format before uploading
gems-eval validate --sub submission.tif --template existing_faults.tif
```

## Python

```python
import numpy as np
from gems_eval import dti, split_segments, holdout_dti, keep_top_fraction

train, ev = split_segments(labels, frac=0.3, seed=0, exclude_buffer_px=3)   # labels: int array, >=1 fault, <0 nodata
score = holdout_dti(pred, ev)                                              # pred: float array in [0, 1]
score_top2 = holdout_dti(keep_top_fraction(pred, 0.02, valid=(ev == 0) | (ev == 1)), ev)
```

`dti(pred, truth, valid=...)` is the bare metric if you have your own split; `block_folds(H, W, block_px=512)` gives spatial block folds for a stricter, region-level check.

## Reference numbers (ours, 2026-09-18)

| model (trained on train_labels only) | held-out DTI | coverage ≥ 0.5 |
|---|---|---|
| organizer reference recipe (ResNet18 U-Net) | 0.044* | 3.8% |
| U-Net, ResNet34 encoder, 25 epochs | 0.040 | 8.8% |
| U-Net, ConvNeXt-tiny encoder, 25 epochs | 0.056 | 3.6% |
| U-Net, EfficientNet-B3 encoder, 25 epochs | 0.088 | 3.0% |
| same, 40 epochs | 0.099 | 1.9% |
| EfficientNet-B3 25 ep + skeleton of top-5% | 0.132 | — |

\* the reference model was trained on all labels (leaky); the others on `train_labels.tif` only. The reference's leaderboard score was 0.0387.

## License

MIT — see `LICENSE`. This is an independent tool; it is not affiliated with DrivenData or the U.S. Department of Energy.
