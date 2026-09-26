#!/usr/bin/env python3
"""S5-F, the flagship: the proven field + the best detector this family has, at a measured budget.

DESIGN (each number is a measurement in this repository, not an argument)
-------------------------------------------------------------------------
* Base: the S5-A field (the 0.1563 CNN ensemble's support + every masked catalogue pixel).
  The support is what the board already paid 0.1563 for; the catalogue pixels are charge-free
  by the platform's written rule (forum 11516).
* Addition: the top-N pixels OFF the base of the H1+H2 context detector - HistGradientBoosting
  on the 19 official bands + the 9-band 10 m 3DEP scarp channel + the 7-band GeoDAWN radiometric
  channel (held-out fold peaks 0.131-0.182 vs 0.106-0.146 for the 19-band baseline; the scarp
  channel is D7's missing channel, measured live in session 4).
* Why a union at all: D5 measured that the CNN and HGB families find almost disjoint things
  (Jaccard 0.066) with the CNN +0.04 ahead - "they find different things". The HGB top pixels
  off the CNN field are the complementary support.
* Why the budget is N = 10,000 by default: D2 measured a strict superset of +9,430 unmasked px
  from the SAME detector family scoring -0.0003 (break-even). The HGB addition is a DIFFERENT
  family, so it is at least as good as that measured break-even line; it is priced no higher.
  The sweep reports N = 5k / 10k / 20k / 50k so the Phase-2 decision (expanded labels reward
  verified discoveries - the aggressive end is the discovery play) is a choice with numbers,
  not a guess.
* Every added pixel must still earn the marginal hit rate the metric's own algebra computes at
  the operating point (0.0323 at DTI 0.1563; scripts/emit_by_marginal_rule.py). The rule
  structurally cannot price off-catalogue additions (the measured H7 degeneracy), which is why
  the budget comes from the D2 parity line and is printed, not hidden.

USAGE
    python scripts/build_flagship_candidate.py                        # N = 10,000 (default)
    python scripts/build_flagship_candidate.py --n-add 50000          # the Phase-2 discovery end
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
                    help="H1+H2 detector probability field (uint8 = probability x 255)")
    ap.add_argument("--base", default="docs/downloads/candidate_s5_catalogue_hedge.tif")
    ap.add_argument("--n-add", type=int, default=10_000,
                    help="number of top detector pixels added off the base (D2 parity line: "
                         "+9,430 same-family px measured break-even)")
    ap.add_argument("--out", default="data/derived/candidate_s5f_flagship.tif")
    ap.add_argument("--report", default="data/evidence/emission/flagship.json")
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
    with rasterio.open(a.detector) as src:
        d = src.read(1).astype(np.float64)
        if src.dtypes[0] == "uint8":
            d = d / 255.0
        prob = np.where(valid, np.nan_to_num(d, nan=0.0), 0.0).astype(np.float32)
    with rasterio.open(a.base) as src:
        bf = src.read(1)
    base = np.isfinite(bf) & (bf > 0) & valid

    eligible = valid & ~labels & ~base
    order = np.argsort(-prob[eligible], kind="stable")

    def build(n: int) -> dict:
        idx = np.flatnonzero(eligible)[order[:n]]
        keep = np.zeros((prob.shape[0], prob.shape[1]), dtype=bool)
        keep[idx // prob.shape[1], idx % prob.shape[1]] = True
        emit = base | keep
        r = score_within_mask(emit.astype(np.float32), GtContext(labels.astype(np.float32), 3), valid)
        thr = float(prob[keep].min()) if keep.any() else None
        return dict(n_add=n, added_px=int(keep.sum()), threshold_off_base=thr,
                    emitted_px=int(emit.sum()), chargeable_px=int((emit & ~labels).sum()),
                    in_domain_dti=round(r["dti"], 5) if r["dti"] is not None else None)

    sweep = [build(n) for n in (5_000, 10_000, 20_000, 50_000)]
    if a.n_add in (5_000, 10_000, 20_000, 50_000):
        chosen = next(s for s in sweep if s["n_add"] == a.n_add)
    else:  # a custom budget: price it the same way, keep the standard sweep for reference
        chosen = build(a.n_add)
        sweep.append(chosen)
    # the exact keep-mask for the shipped N
    idx = np.flatnonzero(eligible)[order[:a.n_add]]
    keep = np.zeros((prob.shape[0], prob.shape[1]), dtype=bool)
    keep[idx // prob.shape[1], idx % prob.shape[1]] = True
    emit = base | keep

    field = np.where(emit, np.float32(1.0), np.float32(0.0))
    field, conformance = conform_to_template(field, tmpl)
    profile.update(dtype="float32", count=1, nodata=float("nan"), compress="lzw",
                   tiled=True, blockxsize=256, blockysize=256)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(field, 1)
        dst.set_band_description(
            1, "5GEMSDOE S5-F flagship: S5-A base + top-N off-base pixels of the H1+H2 "
               "(19 bands + scarp + radiometric) context detector, N priced at the D2 break-even line")
    dl = Path(a.downloads)
    dl.mkdir(parents=True, exist_ok=True)
    dl_path = dl / out_path.name
    shutil.copyfile(out_path, dl_path)

    ctx = GtContext(labels.astype(np.float32), 3)
    dti_cal, terms = ctx.score(field, return_components=True)
    sha = sha256_file(out_path)
    rep = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/build_flagship_candidate.py",
        "base": a.base,
        "detector": a.detector,
        "budget_basis": ("D2 (STRATEGY.md): a strict superset of +9,430 unmasked px from the SAME "
                         "detector family scored -0.0003 (break-even) at DTI 0.1563; N = 10,000 is "
                         "parity with that measured line for a DIFFERENT detector family (D5: "
                         "Jaccard 0.066, near-disjoint supports)"),
        "sweep": sweep,
        "chosen": {**chosen,
                   "on_masked_catalogue_px": int((emit & labels).sum()),
                   "required_marginal_hit_rate_at_0_1563": round(emr.breakeven_marginal_hit_rate(0.1563), 5)},
        "in_domain_score": dict(dti=round(dti_cal, 6), T=round(terms[0], 2), F=round(terms[1], 2),
                                note="against the supplied catalogue: in-domain monitor, NOT the test set"),
        "output": dict(path=str(out_path), download=str(dl_path), sha256=sha,
                       bytes=out_path.stat().st_size),
        "suggested_upload_name": f"gems-submission-s5f-flagship-{sha[:8]}.tif",
        "suggested_note": (f"S5-F · {sha[:8]} · S5-A + top {a.n_add:,} off-base px of H1+H2 detector "
                           f"(scarp + radiometric) · D2-parity budget · flagship for final selection"),
        "conformance": conformance,
        "phase2_note": ("the same file is scored in both prize rounds; the 50k row of the sweep is "
                        "the aggressive end (expanded labels reward verified discoveries) if the "
                        "owner wants the discovery play at selection time"),
    }
    # measurable off-catalogue share of the addition
    from scipy.ndimage import distance_transform_edt
    dist_known = distance_transform_edt(~labels)
    rep["chosen"]["far_from_catalogue_px"] = int((keep & (dist_known > 3)).sum())
    rep["chosen"]["far_from_catalogue_fraction"] = (
        round(float((keep & (dist_known > 3)).sum() / keep.sum()), 4) if keep.sum() else None)
    out_report = Path(a.report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(rep, indent=1) + "\n")
    for s in sweep:
        print(f"  N={s['n_add']:>6,}  added={s['added_px']:>6,}  thr={s['threshold_off_base']}  "
              f"chargeable={s['chargeable_px']:>7,}  in_domain_dti={s['in_domain_dti']}")
    print(f"shipped N={a.n_add:,}: emitted {chosen['emitted_px']:,} px, chargeable "
          f"{chosen['chargeable_px']:,}, required marginal hit rate "
          f"{rep['chosen']['required_marginal_hit_rate_at_0_1563']}")
    print(f"wrote {out_path.resolve().relative_to(ROOT)}  sha256 {sha[:16]}…  ({out_path.stat().st_size:,} B)")
    print(f"download: {dl_path.resolve().relative_to(ROOT)}")
    print(f"note: {rep['suggested_note']}")
    print(f"wrote {out_report.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
