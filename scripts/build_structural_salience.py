#!/usr/bin/env python3
"""Unsupervised structural lineament salience from the 19 official bands.

WHY THIS EXISTS
---------------
Every detector this project family has uploaded was trained on the Quaternary catalogue
(`labels.tif`), and the scored population is the faults that catalogue does NOT contain
(rules 1.1/3.3, quoted verbatim in scripts/verify_rules_quotes.py). A supervised model can only
re-derive what the catalogue already says, and docs/STRATEGY.md D1 measured the consequence: the
best-scoring file we own earned its score on ground 78 % of which is more than 300 m from any
catalogue pixel. So a detector that never sees the catalogue is not a curiosity here - it is the
only kind of detector that is not fitting the wrong population.

WHAT IT COMPUTES
----------------
For each source band, independently:

1. robust normalisation over the valid footprint (1st/99th percentile, NaN-aware);
2. gradient magnitude, which turns a STEP (a fault juxtaposing two blocks, the classic expression
   of a Basin-and-Range normal fault in a potential field) into a RIDGE;
3. multi-scale Frangi vesselness (sigma = 1, 2, 4 px) on that gradient magnitude - the standard
   line-enhancement filter, which rewards pixels that are locally ridge-like (one strongly
   negative Hessian eigenvalue) and locally LINEAR (low blobness) over a linear structure's own
   scale, and suppresses both noise and blobs;
4. max over scales, then 0-1 rescaling of the band's own salience.

Bands are then combined two ways: `mean` (how many independent sources agree a pixel is on a
lineament) and `max` (the single strongest source). `mean` is the default covariate: a lineament
seen by magnetics AND gravity AND topography is a structure, a lineament seen by one band is a
band artefact, and this combination is exactly what a geophysicist does when mapping.

WHAT IT IS NOT
--------------
It is not a probability, it was not fitted to any label, and it is not a submission. It is a
spatial covariate and a pixel ranking, both of which are scored downstream
(scripts/fit_hidden_prior.py, scripts/rank_instruments.py) rather than asserted here.

USAGE
    python scripts/build_structural_salience.py --out data/derived/salience.tif
    python scripts/build_structural_salience.py --bands rtp,iso_grav_anom,det_elev --quick
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, median_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_BANDS = ["rtp", "tmi", "iso_grav_anom", "det_elev", "det_elev_slope",
                 "geod_2ndinv", "geod_shearrate", "deq_n100a15", "cond_surf",
                 "depth_to_base_surf"]
SIGMAS = (1.0, 2.0, 4.0)
FRANGI_BETA = 0.5      # blobness sensitivity: low => only elongated structures survive
FRANGI_C = 0.3         # structureness sensitivity (fraction of the max Hessian norm)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def band_index_map(features: Path) -> dict[str, int]:
    with rasterio.open(features) as src:
        out = {}
        for i in range(1, src.count + 1):
            name = (src.tags(i).get("band_name") or f"band{i}").strip().lower()
            out[name] = i
        return out


def robust_normalise(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    v = a[valid]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros(a.shape, dtype=np.float32)
    lo, hi = np.percentile(v, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.float32)
    out = (a - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def frangi_on_gradient(field: np.ndarray, sigmas=SIGMAS, beta=FRANGI_BETA, c=FRANGI_C):
    """Multi-scale line enhancement of |grad(field)|; returns the max over scales, in [0, 1]."""
    # gradient magnitude at the finest scale, then smoothed at each scale: differentiating after
    # smoothing is the numerically stable order (the derivative of a Gaussian is exact).
    gy, gx = np.gradient(gaussian_filter(field.astype(np.float32), SIGMAS[0], mode="nearest"))
    grad_mag = np.hypot(gy, gx).astype(np.float32)
    best = np.zeros(field.shape, dtype=np.float32)
    norm_ref = None
    for s in sigmas:
        sm = gaussian_filter(grad_mag, s, mode="nearest")
        hyy = gaussian_filter(sm, s, order=[0, 2], mode="nearest")
        hxx = gaussian_filter(sm, s, order=[2, 0], mode="nearest")
        hxy = gaussian_filter(sm, s, order=[1, 1], mode="nearest")
        # eigenvalues of the symmetric 2x2 Hessian, closed form
        tmp = np.sqrt((hxx - hyy) ** 2 + 4.0 * hxy ** 2)
        lam1 = 0.5 * (hxx + hyy + tmp)      # larger
        lam2 = 0.5 * (hxx + hyy - tmp)      # smaller
        # ridges in the gradient-magnitude image are where lam2 is strongly NEGATIVE
        rb = np.abs(lam2) / (np.abs(lam1) + 1e-12)
        s_norm = np.sqrt(lam1 ** 2 + lam2 ** 2)
        scale_ref = float(np.nanpercentile(s_norm, 99.0)) if norm_ref is None else norm_ref
        if scale_ref <= 0:
            continue
        vessel = np.exp(-(rb ** 2) / (2.0 * beta ** 2)) * (1.0 - np.exp(-(s_norm ** 2) / (2.0 * (c * scale_ref) ** 2)))
        vessel = np.where(lam2 < 0, vessel, 0.0).astype(np.float32)
        best = np.maximum(best, vessel)
        del hxx, hyy, hxy, tmp, lam1, lam2, rb, s_norm, vessel, sm
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/training_features.tif")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--bands", default=",".join(DEFAULT_BANDS))
    ap.add_argument("--sigmas", default=",".join(str(s) for s in SIGMAS))
    ap.add_argument("--out", default="data/derived/salience.tif")
    ap.add_argument("--out-mean", default=None,
                    help="path for the cross-source MEAN salience (default: <out> with _mean)")
    ap.add_argument("--report", default="data/evidence/salience/build.json")
    ap.add_argument("--median-3", action="store_true",
                    help="3x3 median filter on each band's salience (kills single-pixel spikes)")
    a = ap.parse_args()

    sigmas = tuple(float(s) for s in a.sigmas.split(","))
    with rasterio.open(a.template) as src:
        valid = np.isfinite(src.read(1))
        profile = src.profile.copy()
    idx = band_index_map(Path(a.features))
    wanted = [b.strip() for b in a.bands.split(",") if b.strip()]
    unknown = [b for b in wanted if b not in idx]
    if unknown:
        raise SystemExit(f"unknown band(s) {unknown}; available: {sorted(idx)}")

    per_band: dict[str, np.ndarray] = {}
    rows = []
    with rasterio.open(a.features) as src:
        for name in wanted:
            arr = src.read(idx[name]).astype(np.float32)
            arr[~np.isfinite(arr)] = np.nan
            arr[arr < -1e30] = np.nan
            n = robust_normalise(arr, valid)
            sal = frangi_on_gradient(n, sigmas=sigmas)
            if a.median_3:
                sal = median_filter(sal, size=3, mode="nearest").astype(np.float32)
            v = sal[valid]
            p999 = float(np.percentile(v, 99.9)) if v.size else 0.0
            if p999 > 0:
                sal = np.clip(sal / p999, 0.0, 1.0)
            sal[~valid] = 0.0
            per_band[name] = sal.astype(np.float32)
            rows.append(dict(band=name, mean=round(float(sal[valid].mean()), 6),
                             p99=round(float(np.percentile(sal[valid], 99.0)), 6),
                             max=round(float(sal[valid].max()), 6),
                             frac_above_half=round(float((sal[valid] > 0.5).mean()), 6)))
            del arr, n, v

    stack = np.stack([per_band[b] for b in wanted])
    sal_mean = stack.mean(axis=0).astype(np.float32)
    sal_max = stack.max(axis=0).astype(np.float32)
    # quantile-transform the mean so it is comparable with other covariates on [0, 1]
    v = sal_mean[valid]
    if v.size and float(np.nanmax(v)) > 0:
        sal_mean = np.clip(sal_mean / float(np.nanmax(v)), 0.0, 1.0).astype(np.float32)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=float("nan"), compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.where(valid, sal_max, np.nan).astype(np.float32), 1)
        dst.set_band_description(1, "structural lineament salience: max over "
                                    f"{len(wanted)} bands, Frangi on |grad|, sigmas {sigmas}")
    mean_path = Path(a.out_mean) if a.out_mean else out_path.with_name(out_path.stem + "_mean.tif")
    with rasterio.open(mean_path, "w", **profile) as dst:
        dst.write(np.where(valid, sal_mean, np.nan).astype(np.float32), 1)
        dst.set_band_description(1, "structural lineament salience: mean over "
                                    f"{len(wanted)} bands, Frangi on |grad|, sigmas {sigmas}")

    rep = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/build_structural_salience.py",
        "features_sha256": sha256(Path(a.features)),
        "bands": wanted, "sigmas": list(sigmas),
        "frangi": dict(beta=FRANGI_BETA, c=FRANGI_C),
        "per_band": rows,
        "outputs": {
            "max": dict(path=str(out_path), sha256=sha256(out_path),
                        size=out_path.stat().st_size),
            "mean": dict(path=str(mean_path), sha256=sha256(mean_path),
                         size=mean_path.stat().st_size),
        },
        "note": ("unsupervised: no label, no catalogue and no score entered this computation; "
                 "it is a spatial covariate to be scored downstream, not a prediction"),
    }
    rep_path = Path(a.report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(rep, indent=1) + "\n")

    print(f"{'band':<22}{'mean':>10}{'p99':>10}{'max':>10}{'frac>0.5':>10}")
    for r in rows:
        print(f"{r['band']:<22}{r['mean']:>10.4f}{r['p99']:>10.4f}{r['max']:>10.4f}"
              f"{r['frac_above_half']:>10.4f}")
    print(f"\nwrote {out_path} (max) and {mean_path} (mean); report {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
