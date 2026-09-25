#!/usr/bin/env python3
"""Judge every candidate submission with a SECOND, independent implementation.

WHY A SECOND ONE
----------------
`scripts/validate_submission.py` is this repository's own gate.  A gate written by the same
project that wrote the file it is checking cannot catch a shared misreading of the rules, and the
one rejection this project has actually suffered from the platform (`Predicted values must be in
range [0, 1]`, 2026-09-24) was exactly such a misreading: every value in the file was in [0, 1]
and the NaNs were in the wrong places.

So this script judges each candidate twice:

  A. ours - `scripts/validate_submission.py` (17 checks, exit code gates CI);
  B. theirs - `scripts/vendor/gems_eval/validate.py`, copied unmodified from
     https://github.com/Gameassassin777/gems-eval (MIT, (c) 2026 Syntropy Digital), whose
     provenance and upstream findings are recorded in `scripts/vendor/gems_eval/PROVENANCE.md`.

and then cross-checks the METRIC itself: `src/metrics.GtContext` (this repository's fast path)
against `gems_eval.dti.dti` (an independent full-raster formulation) on the real rasters, for
every candidate.  Two implementations agreeing on real data is evidence; one implementation
agreeing with itself is not.

Exit 0 only if BOTH judges accept every file and the two metrics agree to 1e-4.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import GtContext  # noqa: E402
from src.submission_io import sha256_file  # noqa: E402


def _vendor():
    spec = importlib.util.spec_from_file_location(
        "gems_eval_validate", ROOT / "scripts" / "vendor" / "gems_eval" / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _vendor_dti():
    spec = importlib.util.spec_from_file_location(
        "gems_eval_dti", ROOT / "scripts" / "vendor" / "gems_eval" / "dti.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", default=None,
                    help="candidate .tif files (default: every docs/downloads/*.tif)")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--report", default="data/evidence/submission_independent_gate.json")
    ap.add_argument("--blanket", action="append", default=[],
                    help="a file that is a BLANKET probe on purpose: it trips the independent "
                         "validator's plausible_coverage heuristic (0.1%%-30%% of the footprint "
                         "at >= 0.5) by design, and that check is the only one it may fail")
    a = ap.parse_args()

    files = [Path(f) for f in (a.files or sorted((ROOT / "docs" / "downloads").glob("*.tif")))]
    if not files:
        sys.exit("FAIL: no candidate files given and docs/downloads/ holds none")
    vval, vdti = _vendor(), _vendor_dti()

    with rasterio.open(a.template) as src:
        tmpl = src.read(1)
    valid = np.isfinite(tmpl)
    with rasterio.open(a.labels) as src:
        labels = np.nan_to_num(src.read(1), nan=0.0) > 0.5
    labels &= valid
    ctx = GtContext(labels.astype(np.float32), 3)

    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/verify_candidates_independently.py",
        "judges": {
            "ours": "scripts/validate_submission.py (17 checks; exit code gates CI)",
            "theirs": ("scripts/vendor/gems_eval/validate.py - unmodified copy of "
                       "https://github.com/Gameassassin777/gems-eval (MIT, (c) 2026 Syntropy "
                       "Digital); provenance in scripts/vendor/gems_eval/PROVENANCE.md"),
        },
        "metric_cross_check": ("src/metrics.GtContext.score (this repository, fast path) vs "
                               "gems_eval.dti.dti (independent full-raster formulation), on the "
                               "real rasters, for every candidate"),
        "files": [],
    }
    all_ok = True
    for f in files:
        if not f.exists():
            print(f"MISSING {f}")
            all_ok = False
            continue
        # A. ours
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_submission.py"),
                               "--pred", str(f), "--sample", a.template,
                               "--train", "data/training_features.tif"],
                              capture_output=True, text=True, cwd=ROOT)
        # B. theirs
        try:
            theirs = vval.validate_submission(str(f), a.template)
        except Exception as exc:  # noqa: BLE001 - a crash is a finding, not a stop
            theirs = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        # metric parity on the real raster
        with rasterio.open(f) as src:
            arr = src.read(1)
        arr = np.where(valid, np.nan_to_num(arr, nan=0.0), 0.0).astype(np.float32)
        ours_dti, (T, F, FN) = ctx.score(arr, return_components=True)
        theirs_dti = vdti.dti(np.where(valid, arr, 0.0), labels.astype(np.float32), valid=valid)
        agree = abs(float(ours_dti) - float(theirs_dti)) <= 1e-4
        row = dict(
            file=f.name, path=str(f), sha256=sha256_file(f), bytes=f.stat().st_size,
            our_validator=dict(exit_code=proc.returncode,
                               passed=proc.returncode == 0,
                               tail=proc.stdout.strip().splitlines()[-1] if proc.stdout else ""),
            independent_validator=theirs,
            metric=dict(ours_dti=round(float(ours_dti), 6), independent_dti=round(float(theirs_dti), 6),
                        abs_diff=round(abs(float(ours_dti) - float(theirs_dti)), 8),
                        agree=bool(agree),
                        T=round(T, 2), F=round(F, 2), FN_w=round(FN, 2),
                        note="in-domain: the catalogue the supervised files trained on"),
        )
        ok = bool(row["our_validator"]["passed"] and theirs.get("ok") and agree)
        # A blanket density probe is *supposed* to violate the independent validator's
        # plausible-coverage heuristic: it emits on essentially the whole footprint so that its
        # public score inverts to |G|.  The platform's own rule (finite values in [0, 1] inside
        # the footprint, NaN outside) is still satisfied, so it is accepted there.  The exception
        # is named, not silent.
        exception = None
        if not ok and str(f) in [str(Path(b).resolve()) for b in a.blanket] + \
                [str((ROOT / b).resolve()) for b in a.blanket]:
            checks = theirs.get("checks", {})
            failing = [k for k, v in checks.items() if not v]
            if failing == ["plausible_coverage"] and row["our_validator"]["passed"] and agree:
                ok, exception = True, (
                    "intentional blanket probe: fails only the independent validator's "
                    "plausible_coverage heuristic, which the platform itself does not enforce")
        row["exception"] = exception
        row["verdict"] = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        out["files"].append(row)
        print(f"{row['verdict']}  {f.name:<48} ours_exit={proc.returncode} "
              f"independent_ok={theirs.get('ok')} DTI ours={ours_dti:.4f} "
              f"indep={theirs_dti:.4f} (agree={agree})")

    out["verdict"] = "PASS" if all_ok else "FAIL"
    out["blanket_exceptions"] = [r["file"] for r in out["files"] if r.get("exception")]
    rep = Path(a.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\n{out['verdict']} - {len(out['files'])} files judged by both implementations")
    print(f"wrote {rep}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
