#!/usr/bin/env python3
"""The literature prior, priced: dilational settings of the catalogue, kept only where a detector
already believes there is a fault.

THE HYPOTHESIS (STRATEGY.md H3 + H6)
------------------------------------
Two verified statements point at the same place:

* the platform's own staff define a scored "new fault" as *"any fault pixel not already captured
  by USGS/INGENIOUS"*, which *"can include newly mapped geometry of an existing fault system"* -
  continuations, splays, parallel strands (forum 11536); and
* forty years of Great Basin geothermal exploration says the faults that matter sit in dilational
  settings, not on the main range-front break: step-overs/relay ramps ~32 %, terminations 25 %,
  intersections 22 %, bends 2 %, major range-front faults 1 % (Faulds & Hinz, WGC 2015,
  OSTI 1724082), with tips horse-tailing into *"a myriad of closely-spaced faults"*.

So the highest-prior ground for a NEW fault pixel is the thin neighbourhood of a catalogue
tip / branch / bend / relay ramp.  Session 2 shipped that idea as the top-N pixels of a smoothed
favourability field (S5-B), which is a DENSE patch and therefore expensive in false-positive
mass.  This script ships the same prior as a THIN ANNULUS - one halo pixel around each setting -
and then keeps only the part of it a detector already agrees with, which is the evidence-layer
combination the exploration literature uses (Faulds et al., DE-EE0002748: permeability proxies
weighted per evidence layer).

WHY THE INTERSECTION MATTERS
---------------------------
The annulus alone is cheap but blind; the detector alone is what it is.  Their product is a small
set of pixels that are (a) in the setting the literature says hosts systems and (b) on a
lineament the data see.  Both ingredients are already measured in this repository
(``scripts/build_structural_targets.py`` for the geometry, ``scripts/train_context_detector.py``
for the detector), so nothing here is fitted to a score.

WHAT IT IS NOT
--------------
It is not priced against the hidden truth - nothing in this repository is.  What it IS priced
against is the metric's algebra: every chargeable pixel it adds must earn the marginal hit rate
``scripts/emit_by_marginal_rule.py`` computes at the operating point (0.0323 at DTI 0.1563).  The
report prints that number next to the pixel count so the trade is explicit.

USAGE
    python scripts/build_dilational_annulus.py                 # sweep + write the chosen candidate
    python scripts/build_dilational_annulus.py --halo 2 --min-prob 0.10
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import DEFAULT_ALPHA, DEFAULT_BETA, GtContext, score_within_mask  # noqa: E402
from src.submission_io import conform_to_template, sha256_file  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--detector", default="data/derived/context_detector_prob.tif",
                    help="probability field whose agreement is required")
    ap.add_argument("--base", default="data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif",
                    help="the 0.1563 field this candidate must be a superset of")
    ap.add_argument("--halo", type=int, default=1, help="annulus width in pixels around a setting")
    ap.add_argument("--quantile", type=float, default=0.9,
                    help="keep annulus pixels whose detector probability is at or above this "
                         "quantile OF THE ANNULUS ITSELF (0.9 = the most confident tenth)")
    ap.add_argument("--min-prob", type=float, default=None,
                    help="absolute detector probability instead of --quantile")
    ap.add_argument("--out", default="data/derived/candidate_s5_dilational_annulus.tif")
    ap.add_argument("--report", default="data/evidence/emission/dilational_annulus.json")
    ap.add_argument("--downloads", default="docs/downloads")
    a = ap.parse_args()

    bst = _load("build_structural_targets")
    emr = _load("emit_by_marginal_rule")

    with rasterio.open(a.template) as src:
        tmpl = src.read(1)
        profile = src.profile.copy()
    valid = np.isfinite(tmpl)
    with rasterio.open(a.labels) as src:
        labels = np.nan_to_num(src.read(1), nan=0.0) > 0.5
    labels &= valid
    with rasterio.open(a.detector) as src:
        prob = np.where(valid, np.nan_to_num(src.read(1), nan=0.0), 0.0).astype(np.float32)
    with rasterio.open(a.base) as src:
        bf = src.read(1)
    base = np.isfinite(bf) & (bf > 0) & valid

    mk = bst.structural_marks(labels, valid)
    prior = (mk["ends"] | mk["joins"] | mk["bends"] | mk["ramps"])
    struct = np.ones((2 * a.halo + 1,) * 2, bool)
    annulus = binary_dilation(prior, structure=struct) & valid & ~labels
    if a.min_prob is not None:
        min_prob = a.min_prob
    elif annulus.any():
        min_prob = float(np.quantile(prob[annulus], a.quantile))
    else:
        min_prob = 0.0

    ctx = GtContext(labels.astype(np.float32), 3)
    sweep = []
    chosen = None
    for halo in (0, 1, 2):
        an = binary_dilation(prior, structure=np.ones((2 * halo + 1,) * 2, bool)) & valid & ~labels
        for q in (0.25, 0.5, 0.75, 0.9):
            if not an.any():
                continue
            thr = float(np.quantile(prob[an], q))
            keep = an & (prob >= thr)
            emit = base | labels | keep
            r = score_within_mask(emit.astype(np.float32), ctx, valid)
            sweep.append(dict(halo=halo, quantile=q, threshold=round(thr, 5),
                              annulus_px=int(an.sum()), kept_px=int(keep.sum()),
                              emitted_px=int(emit.sum()),
                              chargeable_px=int((emit & ~labels).sum()),
                              required_marginal_hit_rate_at_0_1563=round(
                                  emr.breakeven_marginal_hit_rate(0.1563), 5),
                              in_domain_dti=round(r["dti"], 5) if r["dti"] is not None else None))
    # the shipped setting is the one asked for on the command line
    keep = annulus & (prob >= min_prob)
    emit = base | labels | keep
    chosen = dict(halo=a.halo, quantile=a.quantile, min_prob=round(min_prob, 5),
                  annulus_px=int(annulus.sum()), kept_px=int(keep.sum()),
                  emitted_px=int(emit.sum()),
                  chargeable_px=int((emit & ~labels).sum()),
                  on_masked_catalogue_px=int((emit & labels).sum()),
                  added_over_base_px=int((emit & ~base).sum()),
                  required_marginal_hit_rate_at_0_1563=round(
                      emr.breakeven_marginal_hit_rate(0.1563), 5))

    field = np.where(emit, np.float32(1.0), np.float32(0.0))
    field, conformance = conform_to_template(field, tmpl)
    profile.update(dtype="float32", count=1, nodata=float("nan"), compress="lzw",
                   tiled=True, blockxsize=256, blockysize=256)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(field, 1)
        dst.set_band_description(
            1, "5GEMSDOE: 0.1563 field + masked catalogue + dilational settings of the catalogue "
               "where the CPU context detector agrees (Faulds & Hinz 2015 prior)")
    dl = Path(a.downloads)
    dl.mkdir(parents=True, exist_ok=True)
    import shutil
    dl_path = dl / out_path.name
    shutil.copyfile(out_path, dl_path)

    dti_cal, terms = ctx.score(field, return_components=True)
    rep = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/build_dilational_annulus.py",
        "prior": dict(
            source=("Faulds & Hinz, World Geothermal Congress 2015, OSTI 1724082 "
                    "(https://www.osti.gov/servlets/purl/1724082); dataset GDR 355 "
                    "(https://gdr.openei.org/submissions/355, DOI 10.15121/1148722)"),
            weights=bst.WEIGHTS,
            settings_used=["termination", "intersection", "bend", "step_over"],
            settings_not_computable=bst.NOT_COMPUTED,
            counts=mk["counts"]),
        "detector": dict(path=a.detector,
                         note="CPU HistGradientBoosting on the official bands "
                              "(scripts/train_context_detector.py), spatially blocked CV"),
        "chosen": chosen,
        "sweep": sweep,
        "in_domain_score": dict(dti=round(dti_cal, 6), T=round(terms[0], 2), F=round(terms[1], 2),
                                note="against the catalogue: an in-domain monitor, NOT the test set"),
        "output": dict(path=str(out_path), download=str(dl_path), sha256=sha256_file(out_path),
                       bytes=out_path.stat().st_size),
        "conformance": conformance,
        "note": ("a prior intersected with a detector, not a fitted model; its cost is the "
                 "chargeable pixel count above and its required marginal hit rate is printed "
                 "beside it"),
        "sources": [
            "https://www.osti.gov/servlets/purl/1724082",
            "https://gbcge.org/recent-projects/characterizing-structural-controls/",
            "https://community.drivendata.org/t/where-do-you-draw-the-line/11536",
            "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/",
        ],
    }
    rep_path = Path(a.report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(rep, indent=1) + "\n")
    print(f"annulus {chosen['annulus_px']:,} px -> kept {chosen['kept_px']:,} at p>="
          f"{chosen['min_prob']}")
    print(f"emitted {chosen['emitted_px']:,} px ({chosen['chargeable_px']:,} chargeable, "
          f"{chosen['added_over_base_px']:,} added over the 0.1563 base)")
    print(f"required marginal hit rate at DTI 0.1563: "
          f"{chosen['required_marginal_hit_rate_at_0_1563']}")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} B) and {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
