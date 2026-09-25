"""Distance-weighted Tversky index (DTI) as defined on the GEMS problem page:

    k(d)  = max(1 - d/R, 0),  R = 300 m = 3 pixels at 100 m
    TP_w  = sum over truth pixels g of   max over x within R of  p(x) * k(d(x, g))
    FP_w  = sum over pixels x of         p(x) * [1 - max over truth g of k(d(x, g))]
    FN_w  = sum over truth pixels g of   [1 - max over x within R of p(x) * k(d(x, g))]   (= |G| - TP_w)
    DTI   = TP_w / (TP_w + alpha * FP_w + beta * FN_w + eps),   alpha = 0.2, beta = 0.8

The prediction p is continuous in [0, 1]; nothing is thresholded. TP saturates at one per
truth pixel (a max, not a sum), so a thick band earns no more credit than its best pixel per
truth pixel while all of its off-line mass is charged as a false positive.

This is a re-implementation of the published formula, not the organizer's code; treat scores
as internal until one has been matched against a leaderboard value.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage

R_PX = 3.0
_r = int(np.ceil(R_PX))
_yy, _xx = np.mgrid[-_r:_r + 1, -_r:_r + 1]
_CONE = np.clip(1.0 - np.hypot(_yy, _xx) / R_PX, 0.0, 1.0).astype(np.float32)
_LOG_CONE = np.where(_CONE > 0, np.log(np.clip(_CONE, 1e-6, 1.0)), -np.inf).astype(np.float32)


def cone_max(p: np.ndarray) -> np.ndarray:
    """m(g) = max over x of p(x) * k(d(x, g)) for every pixel g: a grey dilation in the log domain."""
    lp = np.log(np.clip(p, 1e-6, 1.0)).astype(np.float32)
    m = np.exp(ndimage.grey_dilation(lp, structure=_LOG_CONE)).astype(np.float32)
    m[m <= 1.0001e-6] = 0.0
    return m


def dti(pred: np.ndarray, truth: np.ndarray, alpha: float = 0.2, beta: float = 0.8,
        eps: float = 1e-7, valid: np.ndarray | None = None, return_terms: bool = False):
    """Score `pred` (float, 0..1, NaN allowed) against `truth` (0/1) on the pixels in `valid`.
    Pixels outside `valid` (nodata, faults excluded from scoring) contribute to nothing."""
    p = np.clip(np.nan_to_num(pred, nan=0.0), 0.0, 1.0).astype(np.float32)
    g = truth > 0
    if valid is not None:
        p = np.where(valid, p, 0.0).astype(np.float32)
        g = g & valid
    d_true = ndimage.distance_transform_edt(~g).astype(np.float32) if g.any() else np.full(g.shape, np.inf, np.float32)
    w_true = np.clip(1.0 - d_true / R_PX, 0.0, 1.0)
    tp = float(cone_max(p)[g].sum())
    fp = float((p * (1.0 - w_true)).sum())
    fn = float(g.sum()) - tp
    score = tp / (tp + alpha * fp + beta * fn + eps)
    if return_terms:
        return score, {"tp": tp, "fp": fp, "fn": fn, "n_truth": int(g.sum())}
    return score
