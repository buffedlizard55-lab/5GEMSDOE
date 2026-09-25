#!/usr/bin/env python3
"""A detector that runs on CPU, is scored by SPATIAL hold-out, and stacks any aux channel.

WHY A SECOND DETECTOR FAMILY
----------------------------
Every field this project family has uploaded came from one of two detectors: an 11-fold CNN
ensemble (our best, 0.1563) or a pixel-wise gradient-boosted model (GEMSDOE3's "Pindrop",
0.1193).  Both were trained on the same 60,988 catalogue pixels and both are, in
docs/STRATEGY.md D5's measurement, nearly disjoint supports (Jaccard 0.066).  D5 also measured
that the CNN beats pixel-wise boosting by ~+0.04 at equal budget - so a third pixel-wise booster
is not the point.

The point is the CHANNELS.  The 19 official bands carry no radiometrics (STRATEGY.md F8) and only
100 m topography, while the expert test faults were mapped from high-resolution topography
(STRATEGY.md F9/F11, and 39 % of the 426 catalogued Great Basin systems are BLIND with no
surface expression - Faulds et al., DE-EE0002748, gbcge.org).  `scripts/build_topo_features.py`
builds a 9-band 10 m -> 100 m scarp channel from the public 3DEP DEM; this script is what trains
on it.  It is deliberately classical (HistGradientBoosting) so that it runs on 2 vCPU / 3 GB -
i.e. in this sandbox, on a runner, anywhere - with no GPU and no torch, which is the difference
between a hypothesis that can be tested this week and one that cannot.

WHAT MAKES ITS NUMBER HONEST
----------------------------
A random pixel split leaks: a fault trace is kilometres long, so a random split puts the same
trace on both sides of the split and the model is scored on pixels it effectively trained on.
This script therefore uses `src/blocks.py` - contiguous 512 px (51.2 km) super-regions with a
3 px (= the metric's R) training collar - and reports the distance-weighted Tversky index
computed by `src.metrics.score_within_mask` with GLOBAL geometry, exactly as the platform's own
masking rule implies.  The held-out geography is never trained on, collar included.

WHAT IT WRITES
--------------
    data/derived/context_detector_prob.tif   float32 probability field on the competition grid
    data/evidence/context_detector.json      per-fold DTI, feature gain, calibration, provenance

USAGE
    python scripts/train_context_detector.py                       # 19 bands, 4 spatial folds
    python scripts/train_context_detector.py --aux data/external/topo_features_100m.tif
    python scripts/train_context_detector.py --folds 2 --neg 150000 --quick   # smoke run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocks import (DEFAULT_BLOCK_PX, DEFAULT_BUFFER_PX, DEFAULT_N_FOLDS,  # noqa: E402
                        assign_folds, block_table, held_out_mask, scored_mask)
from src.metrics import DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_R_PIXELS, GtContext, score_within_mask  # noqa: E402
from src.submission_io import conform_to_template, sha256_file  # noqa: E402

# bands whose horizontal gradient is a fault-candidate edge in a potential field.  Chosen from
# the band descriptions in data/evidence/inventory.json, not from a fit: a STEP in a potential
# field is what a normal fault looks like from the air (STRATEGY.md, the salience script's
# rationale).  Everything is reported, so the choice is auditable and reversible.
EDGE_BANDS = ["rtp", "tmi", "mag_anom", "iso_grav_anom", "det_elev", "det_elev_slope",
              "geod_2ndinv", "geod_shearrate"]
SMOOTH_PX = 5          # local mean/std window for texture context (500 m at 100 m)
NODATA_SENTINEL = -1e30  # anything below this is the stack's -3.4e38 nodata, not data
ROW_BLOCK = 256        # rows per streaming block: 256 x 3292 x 19 x 4 B = 64 MB


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def band_names(path: Path) -> list[str]:
    with rasterio.open(path) as src:
        return [src.tags(i + 1).get("band_name", f"band{i + 1}") for i in range(src.count)]


def _percentiles(path: Path, valid: np.ndarray, aux: list[Path]) -> list[tuple[float, float]]:
    """Robust (p1, p99) per band over the valid footprint, in one streaming pass."""
    stats = []
    with rasterio.open(path) as src:
        for b in range(1, src.count + 1):
            acc = []
            for r0 in range(0, src.height, ROW_BLOCK):
                r1 = min(src.height, r0 + ROW_BLOCK)
                blk = src.read(b, window=((r0, r1), (0, src.width)))
                m = valid[r0:r1] & np.isfinite(blk)
                if m.any():
                    acc.append(blk[m])
            v = np.concatenate(acc) if acc else np.zeros(1)
            stats.append((float(np.percentile(v, 1.0)), float(np.percentile(v, 99.0))))
    for p in aux:
        with rasterio.open(p) as src:
            for b in range(1, src.count + 1):
                acc = []
                for r0 in range(0, src.height, ROW_BLOCK):
                    r1 = min(src.height, r0 + ROW_BLOCK)
                    blk = src.read(b, window=((r0, r1), (0, src.width)))
                    m = valid[r0:r1] & np.isfinite(blk)
                    if m.any():
                        acc.append(blk[m])
                v = np.concatenate(acc) if acc else np.zeros(1)
                stats.append((float(np.percentile(v, 1.0)), float(np.percentile(v, 99.0))))
    return stats


def _norm(blk: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Percentile-clip -> [0,1].

    The official stack carries nodata = -3.4e38 (float32 min).  Subtracting from that overflows
    float32, so invalid pixels are set to the low percentile FIRST and then clipped: the result is
    0.0 where there is no data, which is also what every scorer in this repo already counted
    (``np.nan_to_num(pred, nan=0.0)``).  Measured: this removes the overflow RuntimeWarnings the
    first version of this function emitted on every band of every block.
    """
    # The official stack's nodata is -3.4e38, which IS a finite float32, so isfinite() does not
    # catch it and the subtraction overflows.  Measured on data/training_features.tif: every one
    # of the 19 bands carries that sentinel outside the survey footprint.
    bad = ~np.isfinite(blk) | (blk < NODATA_SENTINEL)
    blk = np.where(bad, np.float32(lo), blk).astype(np.float32)
    out = (blk - lo) / (hi - lo if hi > lo else 1.0)
    return np.clip(out, 0.0, 1.0)


def _feature_block(r0: int, r1: int, src, aux_srcs, stats, names, valid_blk):
    """Build the feature matrix for rows [r0, r1): bands + |grad| of the edge bands + texture."""
    H, W = src.height, src.width
    rows = []
    cols = []
    nb = 0
    for b in range(1, src.count + 1):
        blk = src.read(b, window=((r0, r1), (0, W))).astype(np.float32)
        lo, hi = stats[nb]
        n = _norm(blk, lo, hi)
        rows.append(n.ravel())
        nm = names[nb]
        cols.append(nm)
        if nm in EDGE_BANDS:
            gy, gx = np.gradient(np.nan_to_num(n, nan=0.0))
            rows.append(np.hypot(gx, gy).astype(np.float32).ravel())
            cols.append(f"|grad|_{nm}")
        if nm in ("rtp", "iso_grav_anom", "det_elev"):
            z = np.nan_to_num(n, nan=0.0)
            rows.append(uniform_filter(z, SMOOTH_PX).astype(np.float32).ravel())
            cols.append(f"mean{SMOOTH_PX}_{nm}")
        nb += 1
    for asrc in aux_srcs:
        for b in range(1, asrc.count + 1):
            blk = asrc.read(b, window=((r0, r1), (0, W))).astype(np.float32)
            lo, hi = stats[nb]
            n = _norm(blk, lo, hi)
            rows.append(n.ravel())
            cols.append(asrc.tags(b).get("band_name", f"aux{b}"))
            nb += 1
    X = np.empty((rows[0].size, len(rows)), dtype=np.float32, order="F")
    for i, r in enumerate(rows):
        X[:, i] = r
    X[~np.isfinite(X)] = 0.0
    return X, cols


def _class_weight(spec):
    if not spec:
        return None
    out = {}
    for part in spec.split(","):
        k, v = part.split(":")
        out[int(k)] = float(v)
    return out


def _make_clf(a):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=a.max_iter, learning_rate=a.learning_rate, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=15, random_state=a.seed,
        class_weight=_class_weight(a.class_weight))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/training_features.tif")
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--aux", action="append", default=[],
                    help="grid-aligned auxiliary rasters (e.g. the 9-band scarp channel)")
    ap.add_argument("--out", default="data/derived/context_detector_prob.tif")
    ap.add_argument("--dtype", choices=("float32", "uint8"), default="uint8",
                    help="uint8 (probabilities x 255) keeps the field small enough to commit and "
                         "to hand back from a runner; float32 is the lossless form")
    ap.add_argument("--report", default="data/evidence/context_detector.json")
    ap.add_argument("--folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--block-px", type=int, default=DEFAULT_BLOCK_PX)
    ap.add_argument("--buffer-px", type=int, default=DEFAULT_BUFFER_PX)
    ap.add_argument("--neg", type=int, default=400_000,
                    help="negative (non-fault) training pixels sampled per fold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--class-weight", default=None,
                    help='e.g. "0:1,1:3" to re-weight the 1:6.6 positive:negative '
                         'prior toward the 1:85 true prior')
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--quick", action="store_true", help="1 fold, 40k negatives, 40 iterations")
    a = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    if a.quick:
        a.folds, a.neg, a.max_iter = 2, 40_000, 40
    if a.folds < 2:
        # one fold holds out the ENTIRE grid: nothing is left to train on, and the fold would be
        # silently skipped.  Say so instead of producing a report with no measurement in it.
        sys.exit("FAIL: --folds must be >= 2; a single fold holds out the whole grid")

    with rasterio.open(a.template) as src:
        tmpl = src.read(1)
        profile = src.profile.copy()
    valid = np.isfinite(tmpl)
    with rasterio.open(a.labels) as src:
        labels = np.nan_to_num(src.read(1), nan=0.0) > 0.5
    labels &= valid
    H, W = labels.shape
    names = band_names(Path(a.features))
    aux_paths = [Path(p) for p in a.aux]
    aux_names = []
    for p in aux_paths:
        aux_names += band_names(p)

    print(f"grid {H}x{W}; valid {int(valid.sum()):,}; catalogue {int(labels.sum()):,}; "
          f"bands {len(names)} + aux {len(aux_names)}")
    stats = _percentiles(Path(a.features), valid, aux_paths)

    # ---- spatial partition -------------------------------------------------------------
    table = block_table((H, W), a.block_px, valid=valid, labels=labels)
    fold_of = assign_folds(table, a.folds, seed=a.seed)
    print(f"{a.folds} spatial folds over {a.block_px} px blocks, collar {a.buffer_px} px")
    n_neg_total = int((valid & ~labels).sum())
    neg_frac = min(1.0, a.neg / max(1, n_neg_total))
    print(f"negative sampling: {neg_frac:.5f} of {n_neg_total:,} background px "
          f"(target {a.neg:,} total)")

    from sklearn.ensemble import HistGradientBoostingClassifier

    feat_src = rasterio.open(a.features)
    aux_srcs = [rasterio.open(p) for p in aux_paths]
    try:
        # ---- training pool (streamed once; the same pixels serve every fold) ------------
        # X is (n_px, n_features); pool_rc is (n_px, 2) pixel coordinates, so a fold's training
        # mask can be applied to the pool without rebuilding it.  Positives are ALL catalogue
        # pixels (60,988 - cheap); negatives are a fixed random sample per row-block.
        X_parts, y_parts, rc_parts = [], [], []
        for r0 in range(0, H, ROW_BLOCK):
            r1 = min(H, r0 + ROW_BLOCK)
            Xb, cols = _feature_block(r0, r1, feat_src, aux_srcs, stats, names, valid[r0:r1])
            yb = labels[r0:r1].ravel()
            vb = valid[r0:r1].ravel()
            negpool = np.nonzero(~yb & vb)[0]
            # --neg is the TOTAL over the grid, so each row-block contributes its share; a
            # per-block budget would multiply by the number of blocks (15) and silently train
            # on millions of background pixels.
            n_neg_block = int(round(neg_frac * negpool.size))
            neg_sel = (rng.choice(negpool, size=n_neg_block, replace=False)
                       if n_neg_block > 0 else np.zeros(0, dtype=np.int64))
            for sel, lab in ((np.nonzero(yb & vb)[0], 1), (neg_sel, 0)):
                if sel.size == 0:
                    continue
                X_parts.append(Xb[sel])
                y_parts.append(np.full(sel.size, lab, dtype=np.int8))
                rc_parts.append(np.stack([r0 + sel // W, sel % W], axis=1))
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        pool_rc = np.concatenate(rc_parts, axis=0)
        del X_parts, rc_parts
        print(f"training pool {X.shape[0]:,} px x {X.shape[1]} features "
              f"({int(y.sum()):,} positive)")

        ctx = GtContext(labels.astype(np.float32), DEFAULT_R_PIXELS)
        per_fold = []
        prob_full = np.zeros((H, W), dtype=np.float32)

        for fold in range(a.folds):
            train_ok = ~held_out_mask((H, W), a.block_px, fold_of, fold, a.buffer_px)
            in_train = train_ok[pool_rc[:, 0], pool_rc[:, 1]]
            Xtr, ytr = X[in_train], y[in_train]
            if ytr.sum() == 0 or (ytr == 0).all():
                print(f"fold {fold}: degenerate split, skipped")
                continue
            clf = _make_clf(a)
            clf.fit(Xtr, ytr)
            # score on the held-out blocks with GLOBAL geometry
            prob = np.zeros((H, W), dtype=np.float32)
            for r0 in range(0, H, ROW_BLOCK):
                r1 = min(H, r0 + ROW_BLOCK)
                Xb, _ = _feature_block(r0, r1, feat_src, aux_srcs, stats, names, valid[r0:r1])
                prob[r0:r1] = clf.predict_proba(Xb)[:, 1].reshape(r1 - r0, W)
            sc_mask = scored_mask((H, W), a.block_px, fold_of, fold)
            r = score_within_mask(prob, ctx, sc_mask)
            # The continuous field is scored as-is, so its FP mass is the sum of every small
            # probability over 5.1 M px and its DTI says almost nothing.  The decision variable
            # is the BINARY emission, so sweep it here; the budget itself is set by the metric's
            # marginal rule in scripts/emit_by_marginal_rule.py, not by this sweep.
            sweep = {}
            for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
                rt = score_within_mask((prob >= t).astype(np.float32), ctx, sc_mask)
                sweep[f"{t:.2f}"] = dict(dti=(None if rt["dti"] is None else round(rt["dti"], 5)),
                                         TP_w=round(rt["TP_w"], 1), FP_w=round(rt["FP_w"], 1),
                                         pred_px=rt["pred_px"])
            per_fold.append(dict(fold=fold, held_out_px=int(sc_mask.sum()),
                                 held_out_fault_px=int((labels & sc_mask).sum()),
                                 train_px=int(in_train.sum()),
                                 dti=r["dti"], TP_w=round(r["TP_w"], 2),
                                 FP_w=round(r["FP_w"], 2), FN_w=round(r["FN_w"], 2),
                                 n_gt=r["n_gt"], pred_mass=round(r["pred_mass"], 1),
                                 binary_threshold_sweep=sweep))
            print(f"fold {fold}: DTI {r['dti']:.4f} on {r['n_gt']:,} held-out fault px "
                  f"(T={r['TP_w']:,.0f} F={r['FP_w']:,.0f})")
            prob_full += prob / a.folds

        # ---- final model on everything --------------------------------------------------
        clf = _make_clf(a)
        clf.fit(X, y)
        gain = getattr(clf, "train_score_", None)
        prob = np.zeros((H, W), dtype=np.float32)
        for r0 in range(0, H, ROW_BLOCK):
            r1 = min(H, r0 + ROW_BLOCK)
            Xb, _ = _feature_block(r0, r1, feat_src, aux_srcs, stats, names, valid[r0:r1])
            prob[r0:r1] = clf.predict_proba(Xb)[:, 1].reshape(r1 - r0, W)
        prob = np.where(valid, prob, 0.0).astype(np.float32)

        if a.dtype == "uint8":
            # 1/255 resolution is far below anything the marginal rule or a threshold sweep can
            # resolve (the operating break-even is 0.032 = 8/255), and it turns a 23 MB float32
            # raster into a ~2 MB one that can be committed and handed back from a runner.
            stored = np.rint(np.clip(prob, 0.0, 1.0) * 255.0).astype(np.uint8)
            profile.update(dtype="uint8", count=1, nodata=0, compress="lzw",
                           tiled=True, blockxsize=256, blockysize=256)
            desc = ("5GEMSDOE context detector probability x 255 (uint8): HistGradientBoosting "
                    "on the official bands + aux channels, spatially blocked CV")
        else:
            stored = np.where(valid, prob, np.nan).astype(np.float32)
            profile.update(dtype="float32", count=1, nodata=float("nan"), compress="lzw",
                           tiled=True, blockxsize=256, blockysize=256)
            desc = ("5GEMSDOE context detector: HistGradientBoosting on the official bands + "
                    "aux channels, spatially blocked CV")
        out_path = Path(a.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(stored, 1)
            dst.set_band_description(1, desc)

        dti_all, terms = ctx.score(prob, return_components=True)
        # in-domain reference for the whole field, and the same field thresholded
        report = {
            "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generated_by": "scripts/train_context_detector.py",
            "detector": "sklearn HistGradientBoostingClassifier (CPU, no torch, no GPU)",
            "inputs": dict(features=a.features, labels=a.labels, template=a.template,
                           aux=[str(p) for p in aux_paths],
                           feature_names=cols,
                           band_percentiles={cols[i]: [round(stats[i][0], 4), round(stats[i][1], 4)]
                                             for i in range(len(stats))}),
            "spatial_cv": dict(block_px=a.block_px, collar_px=a.buffer_px, folds=a.folds,
                               rule="contiguous 512 px super-regions; the collar is excluded from "
                                    "training, never from scoring (src/blocks.py)"),
            "per_fold": per_fold,
            "full_field_in_domain": dict(dti=round(dti_all, 6), T=round(terms[0], 2),
                                         F=round(terms[1], 2), FN_w=round(terms[2], 2),
                                         note="scored against the catalogue the model trained on: "
                                              "an in-domain monitor, NOT the competition test set"),
            "train_score_head": (None if gain is None else [round(float(v), 5)
                                                            for v in gain[:5]]),
            "feature_importance_permutation": None,
            "output": dict(path=str(out_path), sha256=sha256(out_path),
                           bytes=out_path.stat().st_size,
                           valid_px=int(valid.sum()),
                           dtype=a.dtype,
                           mean=round(float(prob[valid].mean()), 6),
                           p99=round(float(np.percentile(prob[valid], 99)), 6),
                           frac_gt_0_5=round(float((prob[valid] > 0.5).mean()), 6),
                           note=("uint8 stores probability x 255 with 0 outside the footprint; "
                                 "readers must divide by 255 and mask with the template")
                                 if a.dtype == "uint8" else None),
            "caveat": ("a second detector family, not a better one: STRATEGY.md D5 measured "
                       "pixel-wise boosting ~+0.04 BELOW the CNN at equal budget.  Its value is "
                       "the channels it can carry (the 10 m scarp channel) and that it can be "
                       "trained and scored anywhere, on CPU, in minutes"),
            "sources": [
                "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/",
                "https://www.osti.gov/servlets/purl/1724082",
                "https://gbcge.org/recent-projects/characterizing-structural-controls/",
            ],
            "seconds": round(time.time() - t0, 1),
        }
        rep = Path(a.report)
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {out_path} ({out_path.stat().st_size:,} B) and {rep}")
    finally:
        feat_src.close()
        for s in aux_srcs:
            s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
