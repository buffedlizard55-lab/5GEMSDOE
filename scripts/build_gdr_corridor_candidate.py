#!/usr/bin/env python3
"""S5-D: the geothermal-corridor candidate - new fault pixels where the GDR 355 systems sit and
the fault detector already believes.

THE HYPOTHESIS (STRATEGY.md H8, measured in this repository)
------------------------------------------------------------
* The hidden truth is NEW faults; the supplied catalogue is what the platform MASKS (forum 11516).
* 117 geothermal systems from the Faulds (2013) Great Basin structural inventory (GDR 355,
  CC BY 4.0) fall inside the scored footprint, and 91 of them are more than 1 km from the
  nearest supplied-catalogue pixel (median 7.07 km; scripts/build_gdr355_prior.py).
* A geothermal system marks a fluid-conducting structural corridor; the new faults the experts
  mapped are most plausibly in the same corridors - step-overs, terminations, intersections
  (the settings GDR 355 itself codes for each system).
* This file adds, over the S5-A base (0.1563 field + masked catalogue), exactly those pixels
  that are (a) inside the R20 m corridor of a known system, (b) NOT on the masked catalogue,
  and (c) above a quantile of the CPU context detector's fault probability.  It is the same
  evidence-layer combination (prior x detector) that build_dilational_annulus.py uses for the
  catalogue geometry, but anchored on geothermal systems instead of catalogue tips - ground the
  catalogue-anchored policies can never reach.

WHAT THIS IS NOT
----------------
It is not priced against the hidden truth - nothing in this repository is.  Its price is the
metric's algebra: every added chargeable pixel must earn the marginal hit rate the rule computes
at the operating point (0.0323 at DTI 0.1563; scripts/emit_by_marginal_rule.py).  The report
prints the added pixel count next to that number so the trade is explicit.  The corridor prior
FAILED the instrument test (scripts/rank_instruments.py --instrument-tif, rho = -0.2..-0.6 vs
the board, data/evidence/rank_instruments_gdr355.json), so this file is a measured, budgeted
BET, not a validated choice - the upload note says exactly that.

USAGE
    python scripts/build_gdr_corridor_candidate.py                # R20 corridor, q=0.9
    python scripts/build_gdr_corridor_candidate.py --corridor data/derived/gdr355_prior_R30.tif --quantile 0.95
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import GtContext, score_within_mask                     # noqa: E402
from src.submission_io import conform_to_template, sha256_file          # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--detector", default="data/derived/context_detector_prob_topo_rad.tif",
                    help="CPU context-detector probability field (0..1; uint8 = probability x 255). "
                         "Default is the H1+H2 field the committed candidate was built with "
                         "(see data/evidence/emission/gdr_corridor.json).")
    ap.add_argument("--base", default="docs/downloads/candidate_s5_catalogue_hedge.tif",
                    help="the S5-A field (0.1563 field + masked catalogue)")
    ap.add_argument("--corridor", default="data/derived/gdr355_prior_R20.tif",
                    help="GDR 355 system-corridor raster (1 inside)")
    ap.add_argument("--quantile", type=float, default=0.9,
                    help="detector-probability quantile INSIDE the eligible corridor region")
    ap.add_argument("--out", default="data/derived/candidate_s5d_gdr_corridor.tif")
    ap.add_argument("--report", default="data/evidence/emission/gdr_corridor.json")
    ap.add_argument("--downloads", default="docs/downloads")
    a = ap.parse_args()

    emr = _load("emit_by_marginal_rule")

    with rasterio.open(a.template) as src:
        tmpl = src.read(1)
        profile = src.profile.copy()
    valid = np.isfinite(tmpl)
    with rasterio.open(a.labels) as src:
        labels = np.nan_to_num(src.read(1), nan=0.0) > 0.5
    labels &= valid
    from scipy.ndimage import distance_transform_edt
    dist_known = distance_transform_edt(~labels)  # px to the nearest supplied-catalogue pixel
    with rasterio.open(a.detector) as src:
        d = src.read(1).astype(np.float64)
        if src.dtypes[0] == "uint8":
            d = d / 255.0  # the committed field is probability x 255 (train script --dtype uint8)
        prob = np.where(valid, np.nan_to_num(d, nan=0.0), 0.0).astype(np.float32)
    with rasterio.open(a.base) as src:
        bf = src.read(1)
    base = np.isfinite(bf) & (bf > 0) & valid
    with rasterio.open(a.corridor) as src:
        corridor = (np.nan_to_num(src.read(1), nan=0.0) > 0.5) & valid

    eligible = corridor & ~labels & ~base   # where a NEW pixel could be added
    min_prob = float(np.quantile(prob[eligible], a.quantile)) if eligible.any() else 0.0
    keep = eligible & (prob >= min_prob)
    emit = base | keep

    # sweep so the chosen budget is visible against its neighbours
    sweep = []
    for q in (0.5, 0.75, 0.9, 0.95, 0.99):
        if not eligible.any():
            break
        thr = float(np.quantile(prob[eligible], q))
        k = eligible & (prob >= thr)
        e = base | k
        r = score_within_mask(e.astype(np.float32), GtContext(labels.astype(np.float32), 3), valid)
        sweep.append(dict(quantile=q, threshold=round(thr, 5), added_px=int(k.sum()),
                          emitted_px=int(e.sum()),
                          chargeable_px=int((e & ~labels).sum()),
                          in_domain_dti=round(r["dti"], 5) if r["dti"] is not None else None))

    field = np.where(emit, np.float32(1.0), np.float32(0.0))
    field, conformance = conform_to_template(field, tmpl)
    profile.update(dtype="float32", count=1, nodata=float("nan"), compress="lzw",
                   tiled=True, blockxsize=256, blockysize=256)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(field, 1)
        dst.set_band_description(
            1, "5GEMSDOE S5-D: S5-A base + GDR 355 geothermal-corridor pixels (R20) where the "
               "CPU context detector agrees - a budgeted bet on systems-anchored new faults")
    dl = Path(a.downloads)
    dl.mkdir(parents=True, exist_ok=True)
    dl_path = dl / out_path.name
    shutil.copyfile(out_path, dl_path)

    ctx = GtContext(labels.astype(np.float32), 3)
    dti_cal, terms = ctx.score(field, return_components=True)
    sha = sha256_file(out_path)
    rep = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/build_gdr_corridor_candidate.py",
        "hypothesis": ("new faults cluster in the structural corridors of known geothermal "
                       "systems (GDR 355); 91 of 117 in-footprint systems are >1 km off the "
                       "supplied catalogue (median 7.07 km)"),
        "prior": dict(corridor=a.corridor,
                      provenance="scripts/build_gdr355_prior.py on GDR 355 "
                                 "(https://gdr.openei.org/submissions/355, DOI 10.15121/1148722, "
                                 "CC BY 4.0)",
                      instrument_test=("FAILED as a ranking instrument: Spearman rho vs the "
                                       "leaderboard -0.2..-0.6 (data/evidence/rank_instruments_gdr355.json) "
                                       "- this file is therefore a budgeted bet, not a validated choice")),
        "detector": a.detector,
        "base": a.base,
        "chosen": dict(
            corridor_px=int((corridor & ~base).sum()),
            eligible_px=int(eligible.sum()),
            quantile=a.quantile,
            threshold=round(min_prob, 5),
            added_px=int(keep.sum()),
            added_over_base_px=int((emit & ~base).sum()),
            emitted_px=int(emit.sum()),
            chargeable_px=int((emit & ~labels).sum()),
            on_masked_catalogue_px=int((emit & labels).sum()),
            far_from_catalogue_px=(int((keep & (dist_known > 3)).sum()) if keep.sum() else 0),
            far_from_catalogue_fraction=(round(float((keep & (dist_known > 3)).sum() / keep.sum()), 4)
                                         if keep.sum() else None),
            required_marginal_hit_rate_at_0_1563=round(emr.breakeven_marginal_hit_rate(0.1563), 5)),
        "sweep": sweep,
        "in_domain_score": dict(dti=round(dti_cal, 6), T=round(terms[0], 2), F=round(terms[1], 2),
                                note=("against the supplied catalogue - an in-domain monitor; the "
                                      "added pixels are mostly OFF-catalogue where this monitor "
                                      "is blind (the measured H7 degeneracy)")),
        "output": dict(path=str(out_path), download=str(dl_path), sha256=sha,
                       bytes=out_path.stat().st_size),
        "suggested_upload_name": f"gems-submission-s5d-gdr-corridor-{sha[:8]}.tif",
        "suggested_note": (f"S5-D · {sha[:8]} · S5-A + GDR355 R20 corridor ∩ context detector "
                           f"(q{a.quantile:g}) · +{int(keep.sum()):,} off-catalogue px · "
                           "budgeted bet on systems-anchored new faults (H8)"),
        "conformance": conformance,
    }
    out_report = Path(a.report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(rep, indent=1) + "\n")
    print(f"corridor eligible px: {int(eligible.sum()):,}  threshold p{a.quantile:g} = {min_prob:.4f}")
    print(f"added px: {int(keep.sum()):,}  emitted px: {int(emit.sum()):,}  "
          f"chargeable px: {int((emit & ~labels).sum()):,}")
    print(f"required marginal hit rate @ DTI 0.1563: "
          f"{rep['chosen']['required_marginal_hit_rate_at_0_1563']}")
    print(f"wrote {out_path.resolve().relative_to(ROOT)}  sha256 {sha[:16]}…  ({out_path.stat().st_size:,} B)")
    print(f"download: {dl_path.resolve().relative_to(ROOT)}")
    print(f"note: {rep['suggested_note']}")
    print(f"wrote {out_report.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
