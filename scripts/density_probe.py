#!/usr/bin/env python3
"""Density probe: one format-valid upload that measures the size of the hidden truth.

WHY
---
Every emission decision under the official metric depends on ONE unknown number: the kernel
mass |G| of the scored new-fault pixels (scripts/leaderboard_anchor.py, "algebra").  Nobody
outside the organizers knows it, and the organizers will not describe the test faults
(forum 11527/7).  But the metric leaks it: a submission that emits p = 1.0 on every unmasked
valid pixel covers every truth pixel with k = 1, so

    TP_w = |G|,   FN_w = 0,   FP_w = N_unmasked - K,   K = sum over emitted px of max_g k(d) ≈ kappa*|G|

    DTI_blanket = |G| / (|G| + 0.2*(N_unmasked - kappa*|G|))
    =>  |G| = 0.2*N_unmasked*D / (1 - D + 0.2*kappa*D)        (D = the returned public score)

kappa is the kernel mass a truth pixel spreads over its R-neighbourhood: 3.0 for an isolated
1-px straight line (1 + 2*(2/3 + 1/3)), less where truth lines are close together.  The inversion is
insensitive to it (kappa 2 vs 3 changes |G| by < 2 %, table in leaderboard_anchor.json).

WHAT IT BUYS
------------
With |G| known, every scored file's recall and hit-rate become POINTS instead of lines, the
break-even threshold for adding pixels becomes a number we can tune to, and a local proxy can be
rejected the moment its truth density disagrees with the board.  That is worth one of the three
weekly slots on one account once.  It is not a competitive entry and the site says so.

WHAT IT DOES NOT DO
-------------------
It does not locate the public/private chunks and it does not probe them; a region-split probe
would be gaming the split, so this repository does not build one.  If FP is counted only inside
the public chunks, replace N_unmasked by the (unknown) public-chunk area: the returned D is then
larger and the inversion gives an UPPER bound on |G_public|.  Both readings are written next to
the file.

Sources:
  metric   https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric
  masking  https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516
  format   https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.submission_io import clean_profile, conform_to_template, conformance_findings, write_submission  # noqa: E402

ALPHA = 0.2


def invert(D: float, n_unmasked: int, kappa: float) -> float:
    return ALPHA * n_unmasked * D / (1.0 - D + ALPHA * kappa * D)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", default=str(ROOT / "data/sample_submission.tif"))
    ap.add_argument("--labels", default=str(ROOT / "data/labels.tif"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/density_probe"))
    ap.add_argument("--value", type=float, default=1.0, help="constant to emit (1.0 is the informative choice)")
    ap.add_argument("--include-known", action="store_true",
                    help="also emit on supplied-catalogue pixels (they are masked; default leaves them 0 so the "
                         "file is visibly a probe of NEW ground)")
    ap.add_argument("--invert", type=float, default=None, metavar="SCORE",
                    help="do not write a file; print |G| implied by a returned public score")
    a = ap.parse_args(argv)

    with rasterio.open(a.sample) as ds:
        samp = ds.read(1)
        prof = ds.profile
    with rasterio.open(a.labels) as ds:
        lab = ds.read(1)
    valid = np.isfinite(samp)
    known = lab == 1
    n_valid, n_known = int(valid.sum()), int(known.sum())
    n_unmasked = n_valid - n_known

    if a.invert is not None:
        D = a.invert
        print(f"blanket score D={D}: |G| ≈ {invert(D, n_unmasked, 3.0):,.0f} (kappa=3) … {invert(D, n_unmasked, 2.0):,.0f} (kappa=2) "
              f"of {n_unmasked:,} unmasked px = {invert(D, n_unmasked, 3.0) / n_unmasked:.3%} truth density")
        return 0

    field = np.where(valid, np.float32(a.value), np.float32(np.nan)).astype(np.float32)
    if not a.include_known:
        field[known] = 0.0
    field, conf = conform_to_template(field, samp)
    findings = conformance_findings(field, samp)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    profile = clean_profile(prof, nodata=np.nan)
    tmp = out_dir / "probe.tif"
    ver = write_submission(tmp, field, profile,
                           band_description="density probe: constant on every unmasked valid pixel (not a fault map)",
                           tags={"purpose": "density-probe", "value": str(a.value), "emits_on_known": str(a.include_known)})
    sha8 = ver["sha256"][:8]
    final = out_dir / f"gems-density-probe-{stamp}-{sha8}.tif"
    tmp.rename(final)
    note = f"density probe · p={a.value:g} on all unmasked valid px ({n_unmasked:,}) · inverts to |G| · {sha8}"
    meta = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/density_probe.py",
        "file": final.name, "sha256": ver["sha256"], "bytes": ver.get("bytes"),
        "suggested_note": note[:120],
        "emitted_px": int(np.sum(np.isfinite(field) & (field > 0))),
        "valid_px": n_valid, "known_fault_px": n_known, "N_unmasked": n_unmasked,
        "conformance": {"template_adjustments": conf, "findings": findings},
        "inversion": {"formula": "|G| = 0.2*N_unmasked*D/(1 - D + 0.2*kappa*D)",
                      "table": [{"D": D, "G_kappa3": invert(D, n_unmasked, 3.0), "G_kappa2": invert(D, n_unmasked, 2.0),
                                 "truth_density_kappa3": invert(D, n_unmasked, 3.0) / n_unmasked}
                                for D in (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15)],
                      "reading_if_FP_counted_only_in_public_chunks": "the inversion then yields an UPPER bound on |G_public|"},
        "expected_score_is_low_by_design": True,
        "sources": {
            "metric": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric",
            "masking": "https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516",
        },
    }
    (out_dir / "density_probe.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {final} ({ver.get('bytes'):,} B, sha256 {ver['sha256']})")
    print(f"note : {meta['suggested_note']}")
    print(f"emits {meta['emitted_px']:,} px = every unmasked valid pixel; findings: {findings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
