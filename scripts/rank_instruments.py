#!/usr/bin/env python3
"""Does a LOCAL instrument rank the files the way the public leaderboard ranks them?

WHY THIS EXISTS
---------------
This repository holds the bytes of five files the platform has already scored
(`data/evidence/leaderboard_anchor/`, sha256-verified in `scripts/leaderboard_anchor.py`), and it
holds two candidate stand-ins for the hidden "new fault" truth:

* `data/evidence/proxy/proxy_catalogue.tif` code 2 - USGS SGMC faults (Horton et al. 2017,
  DOI 10.3133/ds1052) that the training labels do NOT contain, and
* `data/labels.tif` itself - the Quaternary catalogue the models were trained on.

Every policy decision so far has been argued on one of them. What has never been measured is the
only thing that would justify arguing on either: **if an instrument cannot order five files whose
true order we know, it cannot choose the sixth.** This script performs that measurement. It scores
each file with the OFFICIAL metric (same R, alpha, beta, kernel; `src.metrics.GtContext`) on each
population, with the pixels the platform masks excluded from the FP term, and reports Spearman's
rank correlation between each instrument's ordering and the leaderboard's.

WHAT THE MASK DOES, AND WHY IT IS THE PLATFORM'S RULE
-----------------------------------------------------
Forum 11516 (DrivenData staff, fetched 2026-09-25): known USGS/INGENIOUS faults are "masked /
excluded from evaluation, so they do not count towards penalty terms". So the FP sum may not
charge a prediction for sitting on a catalogue pixel, and the truth of a proxy population is the
pixels that population contributes. Emulated here as

    truth  = the population's pixels
    FP region = valid(template) & ~labels        (labels = pixels the platform masks)

The FP cost of a pixel that IS truth is zero by construction (k(d=0) = 1), so excluding label
pixels from the FP region never hides a real penalty - it removes exactly the charge the forum says
the platform removes.

READING THE OUTPUT
------------------
`rho` is Spearman's rho over the ranked files: +1 = the instrument reproduces the board's order
exactly, 0 = the instrument carries no information about it. With five files, |rho| >= 0.9 has a
two-sided p of about 0.037 under the null of no association, so a high rho is meaningful and a
rho near zero is not "proof of nothing" - it is an instrument this size cannot afford to trust.
The script prints the numbers and refuses to call any instrument validated.

USAGE
    python scripts/rank_instruments.py --out data/evidence/rank_instruments.json
    python scripts/rank_instruments.py --pred path/to/candidate.tif --label my-candidate \
        --out /tmp/rank.json          # score one extra file against the same instruments

Every figure printed is measured here; nothing in this file is typed from memory.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import GtContext, score_within_mask                          # noqa: E402

ALPHA, BETA, R_PX = 0.2, 0.8, 3

# The five leaderboard files. `score` is the number the platform returned; `sha8` is the file's
# identity, verified by scripts/leaderboard_anchor.py. Owner-reported account/notes, 2026-09-25.
SCORED = [
    dict(sha8="7f00890a", score=0.1563, account="extradr19", label="CNN ensemble 11-fold",
         path="data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"),
    dict(sha8="f68e590f", score=0.1560, account="smashi34", label="GEMSDOE2 dual-family union",
         path="data/evidence/leaderboard_anchor/gemsdoe2-dual-family-union-f68e590f.tif"),
    dict(sha8="f347b70daa", score=0.1193, account="smrtdoog5", label="Pindrop nodes",
         path="data/evidence/leaderboard_anchor/pindrop-v4-nodes-f347b70daa.tif"),
    dict(sha8="4e03fc9705", score=0.1152, account="SDCF9", label="Pindrop dense ridge",
         path="data/evidence/leaderboard_anchor/pindrop-v4-ridge-4e03fc9705.tif"),
    dict(sha8="37f9d5b855", score=0.0830, account="wbg1", label="Pindrop catalogue-gap target",
         path="data/evidence/leaderboard_anchor/pindrop-v4-discovery-37f9d5b855.tif"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_mask(path: Path) -> np.ndarray:
    """True where the raster is finite (the scored footprint)."""
    with rasterio.open(path) as src:
        return np.isfinite(src.read(1))


def read_bool(path: Path, thresh: float = 0.5) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1)
    return np.nan_to_num(a, nan=0.0) > thresh


def read_pred(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
    return np.nan_to_num(a, nan=0.0).clip(0.0, 1.0)


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman's rho with average ranks for ties; None when a vector is constant."""
    if len(x) != len(y) or len(x) < 2:
        return None

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def instrument(name: str, truth: np.ndarray, mask: np.ndarray, note: str) -> dict:
    ctx = GtContext(truth, R_pixels=R_PX)
    out = dict(name=name, note=note, n_truth=int(ctx.n_gt), fp_region_px=int(mask.sum()))
    return dict(ctx=ctx, mask=mask, meta=out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--proxy", default="data/evidence/proxy/proxy_catalogue.tif")
    ap.add_argument("--template", default="data/sample_submission.tif",
                    help="defines the scored footprint (finite pixels)")
    ap.add_argument("--pred", action="append", default=[],
                    help="extra candidate: path or path=label (scored, not ranked against the board)")
    ap.add_argument("--out", default="data/evidence/rank_instruments.json")
    a = ap.parse_args()

    valid = read_mask(Path(a.template))
    labels = read_bool(Path(a.labels))
    fp_region = valid & ~labels                       # the platform masks catalogue pixels (F2)
    proxy_arr = None
    proxy_path = Path(a.proxy)
    if proxy_path.exists():
        with rasterio.open(proxy_path) as src:
            proxy_arr = src.read(1)
        proxy_only = (proxy_arr == 2) & valid
    else:
        proxy_only = None

    # The mask restricts BOTH which truth pixels are counted and which prediction pixels may be
    # charged, so it must contain the truth of the instrument and exclude the catalogue pixels the
    # platform masks (their own FP charge is zero anyway: k(d=0) = 1).
    instruments = [instrument("labels_in_domain", labels & valid, valid,
                              "the Quaternary catalogue the supervised files were TRAINED on: an "
                              "in-domain monitor, not a generalisation test")]
    if proxy_only is not None:
        instruments.append(instrument("proxy_sgmc_gap", proxy_only & valid, fp_region,
                                      "SGMC faults absent from the labels (Horton et al. 2017, "
                                      "DOI 10.3133/ds1052): out-of-domain for every file here"))

    files = [dict(s) for s in SCORED if Path(ROOT / s["path"]).exists()]
    missing = [s["sha8"] for s in SCORED if not Path(ROOT / s["path"]).exists()]
    for spec in a.pred:
        if "=" in spec:
            p, lab = spec.split("=", 1)
        else:
            p, lab = spec, Path(spec).stem
        files.append(dict(sha8=None, score=None, account=None, label=lab, path=p))

    rows: list[dict] = []
    for f in files:
        path = Path(f["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            rows.append(dict(f, status="MISSING"))
            continue
        pred = read_pred(path)
        row = dict(f, status="OK", sha256=sha256(path), emitted_px=int((pred > 0).sum()),
                   emitted_mass=round(float(pred.sum()), 1), instruments={})
        for inst in instruments:
            res = score_within_mask(pred, inst["ctx"], inst["mask"], alpha=ALPHA, beta=BETA)
            row["instruments"][inst["meta"]["name"]] = {
                "dti": None if res["dti"] is None else round(float(res["dti"]), 6),
                "TP_w": round(float(res["TP_w"]), 3),
                "FP_w": round(float(res["FP_w"]), 3),
                "FN_w": round(float(res["FN_w"]), 3),
                "coverage": (round(float(res["TP_w"]) / float(res["n_gt"]), 6)
                             if res["n_gt"] else None),
            }
        rows.append(row)

    # rank agreement: only over files that carry a leaderboard score
    scored_rows = [r for r in rows if r.get("score") is not None and r["status"] == "OK"]
    board = [r["score"] for r in scored_rows]
    verdict = {}
    for inst in instruments:
        name = inst["meta"]["name"]
        vals = [r["instruments"].get(name, {}).get("dti") for r in scored_rows]
        rho = spearman(board, [v if v is not None else float("nan") for v in vals]) \
            if all(v is not None for v in vals) else None
        order = sorted(zip(board, [r["sha8"] for r in scored_rows],
                           [r["instruments"][name]["dti"] for r in scored_rows]),
                       reverse=True)
        verdict[name] = {
            "spearman_rho_vs_leaderboard": None if rho is None else round(rho, 4),
            "n_files": len(scored_rows),
            "board_order": [s for _, s, _ in order],
            "instrument_order": [s for _, s in sorted(
                zip([r["instruments"][name]["dti"] for r in scored_rows],
                    [r["sha8"] for r in scored_rows]), reverse=True)],
            "instrument_values": {r["sha8"]: r["instruments"][name]["dti"] for r in scored_rows},
            "verdict": ("reproduces the board's order" if rho is not None and rho >= 0.9 else
                        "does NOT reproduce the board's order - do not select on it"
                        if rho is not None else "not computable"),
        }

    report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/rank_instruments.py",
        "purpose": ("measure whether a local stand-in for the hidden truth orders the five scored "
                    "files the way the public leaderboard does"),
        "metric": dict(R_px=R_PX, alpha=ALPHA, beta=BETA,
                       fp_region="valid(template) & ~labels",
                       source="problem description page 967, forum 11516"),
        "inputs": {
            "template": dict(path=a.template, valid_px=int(valid.sum())),
            "labels": dict(path=a.labels, px=int((labels & valid).sum())),
            "proxy": (None if proxy_only is None else
                      dict(path=a.proxy, px=int(proxy_only.sum()))),
        },
        "instruments": [i["meta"] for i in instruments],
        "files": rows,
        "missing_scored_files": missing,
        "rank_agreement": verdict,
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1) + "\n")

    print(f"{'file':<44}{'board':>8}", end="")
    for i in instruments:
        print(f"{i['meta']['name'][:18]:>20}", end="")
    print()
    for r in rows:
        if r["status"] != "OK":
            print(f"{r.get('sha8') or r['label']:<44}{'MISSING':>8}")
            continue
        print(f"{(r['sha8'] or r['label']) + ' ' + (r.get('account') or ''):<44}"
              f"{(r['score'] if r['score'] is not None else float('nan')):>8.4f}", end="")
        for i in instruments:
            v = r["instruments"][i["meta"]["name"]]["dti"]
            print(f"{v:>20.4f}" if v is not None else f"{'—':>20}", end="")
        print()
    print()
    for name, v in verdict.items():
        print(f"{name}: Spearman rho vs leaderboard = {v['spearman_rho_vs_leaderboard']} "
              f"over {v['n_files']} files -> {v['verdict']}")
        print(f"    board order      {v['board_order']}")
        print(f"    instrument order {v['instrument_order']}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
