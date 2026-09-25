#!/usr/bin/env python3
"""Set the emission budget from the metric's own algebra (STRATEGY.md H4), not from a proxy sweep.

WHY THIS SCRIPT EXISTS
----------------------
Two things were measured in session 2 of this fork and both point here:

1. ``scripts/rank_instruments.py`` showed that the SGMC catalogue-gap proxy - the population every
   emission-width decision in this project family was argued on - **anti-ranks** the public
   leaderboard (Spearman rho = -0.8), while the catalogue scored in-domain orders it at rho = +0.9.
   A floor/thinning sweep on a disqualified instrument cannot select a submission.
2. ``scripts/fit_hidden_prior.py`` showed the leaderboard itself is **unidentified**: five scores do
   not pin the hidden truth density, the smallest tolerance at which any density fits all five is
   0.028 DTI, and the point fit predicts 0.81 for the platform's own example file - so the board
   cannot certify a candidate either.

What is left is the part that needs no instrument at all: the metric's algebra.  With

    DTI = T / (0.2*(T + F) + 0.8*G)          T = TP_w, F = FP_w, G = |G| (hidden truth mass)

adding one binary pixel with true-positive gain dT and false-positive cost dF raises the score iff

    dT > 0.2 * DTI * dF / (1 - 0.2 * DTI)                                        [R2]

so at an operating point DTI the budget is simply "emit in decreasing expected-hit-rate order until
the marginal rate crosses b = 0.2*DTI/(1-0.2*DTI)".  Break-even marginal hit-rate: 0.020 at
DTI 0.10, 0.032 at 0.156 (our best file), 0.042 at 0.20, 0.065 at 0.305 (the leader).  This script
implements that rule exactly, with dT and dF measured by the repository's own metric
(``src/metrics.GtContext``) rather than approximated, and it reports the whole curve so the
turn-around point is visible instead of asserted.

WHAT IT IS NOT
--------------
It is not a detector and it does not know where the hidden faults are.  The *ranking* of candidate
pixels comes from an input favourability field; the *budget* comes from the algebra.  Both are
reported, and the report states which instrument the calibration used, because an in-domain
instrument is degenerate for any candidate that exploits the mask (its truth IS the mask).

USAGE
    # price an existing candidate: what marginal hit rate must its added pixels earn?
    python scripts/emit_by_marginal_rule.py --price docs/downloads/candidate_s5_catalogue_hedge.tif

    # budget a favourability field and write the submission raster
    python scripts/emit_by_marginal_rule.py --field data/derived/structural_targets.tif \
        --dti 0.1563 --out data/derived/marginal_emission.tif
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_R_PIXELS, GtContext  # noqa: E402
from src.submission_io import conform_to_template, sha256_file  # noqa: E402

SOURCES = [
    "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric",
    "https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516",
]


def breakeven_marginal_hit_rate(dti: float, alpha: float = DEFAULT_ALPHA) -> float:
    """b = alpha*DTI / (1 - alpha*DTI): the marginal hit rate an added pixel must beat."""
    if not 0.0 <= dti < 1.0:
        raise ValueError(f"DTI must be in [0, 1), got {dti}")
    return alpha * dti / (1.0 - alpha * dti)


def dti_from_terms(T: float, F: float, G: float, alpha: float = DEFAULT_ALPHA,
                   beta: float = DEFAULT_BETA) -> float:
    return T / (alpha * (T + F) + beta * G)


def _cone(R: int) -> np.ndarray:
    """Triangular kernel weights k(dy, dx) for |offset| <= R (the problem page's k(d) = 1 - d/R)."""
    r = int(R)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return np.clip(1.0 - np.hypot(yy, xx) / float(R), 0.0, 1.0)


class MarginalBudget:
    """Exact application of the marginal-inclusion rule over a ranked candidate list.

    ``ctx`` is a ``GtContext`` built on the calibration truth (the catalogue, in-domain).
    ``candidates`` is a boolean mask of pixels that may be emitted; ``order`` is the ranking
    (higher = emit first).  dT and dF are the metric's own TP_w / FP_w increments, computed
    incrementally against a running per-truth-pixel best-credit map, so the rule is exact rather
    than a linearisation.

    A consequence worth stating, because it decides what this tool can and cannot be used for:
    **a pixel further than R from every calibration-truth pixel has dT = 0 and dF = 1**, so the
    rule can never admit it.  Under a catalogue-calibrated reading the rule therefore only ever
    emits inside 300 m of the catalogue.  That is not a limitation of the implementation, it is
    the degeneracy STRATEGY.md H7 records: an instrument whose truth is the mask prices
    discovery at zero by construction.  ``--all-candidates`` keeps such pixels in the walk so the
    effect is measured rather than assumed.
    """

    def __init__(self, ctx: GtContext, candidates: np.ndarray, order: np.ndarray,
                 R: int = DEFAULT_R_PIXELS, initial: np.ndarray | None = None):
        self.ctx = ctx
        self.R = int(R)
        self.candidates = np.asarray(candidates, dtype=bool)
        self.order = np.asarray(order, dtype=np.float64)
        self.cone = _cone(self.R)
        self.fp_weight = ctx.fp_weight()
        # index of each ground-truth pixel in the context's (gy, gx) vectors, or -1
        self.idx_of = np.full(ctx.shape, -1, dtype=np.int32)
        if ctx.n_gt:
            self.idx_of[ctx.gy, ctx.gx] = np.arange(ctx.n_gt, dtype=np.int32)
        self.best = np.zeros(ctx.n_gt, dtype=np.float64)      # credit already earned per truth px
        self.T = 0.0
        self.F = 0.0
        if initial is not None:
            # start from what is already emitted: the rule then prices ADDITIONS, which is the
            # only question a new upload asks (STRATEGY.md D2: a union of existing fields is
            # break-even, so an addition must earn its own marginal rate)
            base = np.asarray(initial, dtype=bool) & np.ones(ctx.shape, dtype=bool)
            self.best = ctx.credit_vector(np.where(base, 1.0, 0.0).astype(np.float32)).astype(np.float64)
            pos = base
            self.F = float((1.0 - ctx.k_to_gt[pos]).sum()) if ctx.n_gt else float(pos.sum())
            self.T = float(self.best.sum())

    def _delta(self, y: int, x: int) -> tuple[float, float]:
        """(dT, dF) of emitting one binary pixel at (y, x), given what is already emitted."""
        r = self.R
        y0, y1 = max(0, y - r), min(self.ctx.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(self.ctx.shape[1], x + r + 1)
        box = self.idx_of[y0:y1, x0:x1]
        dT = 0.0
        oy, ox = y - y0, x - x0          # box origin relative to (y, x)
        for dy in range(y0 - y, y1 - y):
            row = box[dy + oy]
            crow = self.cone[dy + r]
            for dx in range(x0 - x, x1 - x):
                i = row[dx + ox]
                if i < 0:
                    continue
                k = crow[dx + r]
                if k <= 0.0:
                    continue
                gain = k - self.best[i]
                if gain > 0.0:
                    dT += gain
                    self.best[i] = k
        dF = float(self.fp_weight[y, x])
        return dT, dF

    def walk(self, b: float, mode: str = "rule", max_pixels: int | None = None,
             curve_every: int = 200):
        """Apply the marginal-inclusion rule in decreasing favourability order.

        ``mode="rule"`` is STRATEGY.md R2 read literally - a pixel is emitted iff adding it raises
        the score, so pixels that clear the bar later in the order are still admitted.
        ``mode="budget"`` is the stricter reading - stop at the first pixel that fails, which is
        the right reading when the field is a calibrated probability and the budget is scarce.
        Both are reported with the number of pixels skipped, because the difference between them
        is itself a fact about the field.
        """
        if mode not in ("rule", "budget"):
            raise ValueError(f"mode must be 'rule' or 'budget', got {mode!r}")
        H, W = self.candidates.shape
        ys, xs = np.nonzero(self.candidates)
        if ys.size == 0:
            return np.zeros((H, W), dtype=bool), []
        vals = self.order[ys, xs]
        finite = np.isfinite(vals)
        ys, xs, vals = ys[finite], xs[finite], vals[finite]
        # deterministic order: value desc, then row, then column
        ord_idx = np.lexsort((xs, ys, -vals))
        emitted = np.zeros((H, W), dtype=bool)
        curve = []
        n = 0
        skipped = 0
        last_rate = None
        stopped = None
        for i in ord_idx:
            y, x = int(ys[i]), int(xs[i])
            dT, dF = self._delta(y, x)
            rate = (dT / dF) if dF > 1e-12 else (np.inf if dT > 0 else 0.0)
            if rate < b:
                skipped += 1
                last_rate = rate
                if mode == "budget":
                    # the maximal PREFIX of the ranked list whose pixels all clear the bar
                    stopped = "budget: marginal rate fell below break-even"
                    break
                continue
            self.T += dT
            self.F += dF
            emitted[y, x] = True
            n += 1
            if max_pixels is not None and n >= max_pixels:
                stopped = f"max_pixels={max_pixels} reached"
                break
            if n % curve_every == 0 or n == 1:
                curve.append(dict(n=n, T=round(self.T, 3), F=round(self.F, 3),
                                  dti=round(dti_from_terms(self.T, self.F, self.ctx.n_gt), 6),
                                  marginal_hit_rate=(None if not np.isfinite(rate)
                                                     else round(float(rate), 5))))
        if stopped is None:
            stopped = "candidate list exhausted"
        curve.append(dict(n=n, T=round(self.T, 3), F=round(self.F, 3),
                          dti=round(dti_from_terms(self.T, self.F, self.ctx.n_gt), 6),
                          marginal_hit_rate=(None if last_rate is None or not np.isfinite(last_rate)
                                             else round(float(last_rate), 5)),
                          skipped_px=skipped, stopped_because=stopped))
        return emitted, curve


def price_candidate(path: Path, template: np.ndarray, labels: np.ndarray, valid: np.ndarray,
                    dti_points=(0.10, 0.1563, 0.20, 0.3049)) -> dict:
    """What a shipped file costs, and what its added pixels must earn to be worth uploading."""
    with rasterio.open(path) as src:
        field = src.read(1)
    emitted = np.isfinite(field) & (field > 0) & valid
    chargeable = emitted & ~labels
    return dict(
        path=str(path), sha256=sha256_file(path), bytes=path.stat().st_size,
        emitted_px=int(emitted.sum()),
        on_masked_catalogue_px=int((emitted & labels).sum()),
        chargeable_px=int(chargeable.sum()),
        chargeable_mass=float(chargeable.sum()),
        required_marginal_hit_rate={f"{d:.4f}": round(breakeven_marginal_hit_rate(d), 5)
                                    for d in dti_points},
        note=("chargeable_px is the false-positive mass a score of 0 would pay for; the rule says "
              "those pixels are worth emitting only where their expected hit rate exceeds the "
              "required_marginal_hit_rate at the operating point"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--labels", default="data/labels.tif",
                    help="the masked catalogue: the in-domain calibration truth")
    ap.add_argument("--field", default=None, help="favourability raster on the competition grid")
    ap.add_argument("--base", default=None,
                    help="an already-shipped field: the rule then prices ADDITIONS to it, "
                         "starting from the credit it already earns")
    ap.add_argument("--dti", type=float, default=0.1563,
                    help="operating point for the budget (our best public score)")
    ap.add_argument("--out", default="data/derived/marginal_emission.tif")
    ap.add_argument("--report", default="data/evidence/emission/marginal_rule.json")
    ap.add_argument("--price", action="append", default=[],
                    help="price an existing candidate .tif instead of budgeting a field")
    ap.add_argument("--walk", choices=("rule", "budget"), default="rule",
                    help="rule = emit every pixel that raises the score (R2 read literally); "
                         "budget = stop at the first pixel that does not")
    ap.add_argument("--all-candidates", action="store_true",
                    help="keep pixels further than R from the calibration truth in the walk "
                         "(they can never clear the bar; measuring that is the point)")
    ap.add_argument("--hedge-catalogue", action="store_true", default=True,
                    help="also emit the masked catalogue pixels (charge-free, forum 11516)")
    ap.add_argument("--no-hedge-catalogue", dest="hedge_catalogue", action="store_false")
    ap.add_argument("--max-pixels", type=int, default=None)
    a = ap.parse_args()

    with rasterio.open(a.template) as src:
        tmpl = src.read(1)
        profile = src.profile.copy()
    valid = np.isfinite(tmpl)
    with rasterio.open(a.labels) as src:
        labels = np.nan_to_num(src.read(1), nan=0.0) > 0.5
    labels &= valid
    ctx = GtContext(labels.astype(np.float32), DEFAULT_R_PIXELS)

    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/emit_by_marginal_rule.py",
        "metric": dict(R_px=DEFAULT_R_PIXELS, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
                       formula="DTI = T / (alpha*(T+F) + beta*G); emit binary (R1); "
                               "marginal rule dT > alpha*DTI*dF/(1-alpha*DTI) (R2)"),
        "sources": SOURCES,
        "calibration_truth": dict(path=a.labels, px=int(ctx.n_gt),
                                  note=("the catalogue the supervised files trained on; an "
                                        "in-domain monitor, not a generalisation test, and "
                                        "degenerate for candidates that exploit the mask")),
        "footprint": dict(valid_px=int(valid.sum()), catalogue_px=int(labels.sum()),
                          chargeable_px=int((valid & ~labels).sum())),
    }

    if a.price:
        out["priced"] = [price_candidate(Path(p), tmpl, labels, valid) for p in a.price]
        print(f"{'file':<44}{'emitted':>10}{'chargeable':>12}{'required @0.1563':>18}")
        for p in out["priced"]:
            print(f"{Path(p['path']).name:<44}{p['emitted_px']:>10,}{p['chargeable_px']:>12,}"
                  f"{p['required_marginal_hit_rate']['0.1563']:>18.5f}")
        rep = Path(a.report)
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {rep}")
        return 0

    if not a.field:
        ap.error("give --field to budget, or --price to price an existing candidate")
    with rasterio.open(a.field) as src:
        raw = src.read(1)
        field_tags = {src.descriptions[i] for i in range(src.count)}
        uint8_field = src.dtypes[0] == "uint8"
    field = raw.astype(np.float64)
    if uint8_field:
        # probability x 255, as written by scripts/train_context_detector.py --dtype uint8
        field = field / 255.0
    field = np.where(np.isfinite(field), field, 0.0)

    b = breakeven_marginal_hit_rate(a.dti)
    near_truth = binary_dilation(labels, structure=np.ones((2 * DEFAULT_R_PIXELS + 1,) * 2, bool))
    base = None
    if a.base:
        with rasterio.open(a.base) as src:
            base_field = src.read(1)
        base = np.isfinite(base_field) & (base_field > 0) & valid
    candidates = valid & ~labels
    if base is not None:
        candidates &= ~base
    if not a.all_candidates:
        candidates &= near_truth
    budget = MarginalBudget(ctx, candidates=candidates, order=field, initial=base)
    emitted, curve = budget.walk(b, mode=a.walk, max_pixels=a.max_pixels)
    if base is not None:
        emitted = emitted | base
    if a.hedge_catalogue:
        emitted = emitted | labels

    field_out = np.where(emitted, np.float32(1.0), np.float32(0.0))
    field_out, conformance = conform_to_template(field_out, tmpl)
    profile.update(dtype="float32", count=1, nodata=float("nan"), compress="lzw",
                   tiled=True, blockxsize=256, blockysize=256)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(field_out, 1)
        dst.set_band_description(
            1, f"5GEMSDOE marginal-rule emission (b={b:.5f} at DTI {a.dti}) from "
               f"{Path(a.field).name}")

    dti_cal, (T, F, FN) = ctx.score(field_out, return_components=True)
    out.update(
        field=dict(path=a.field, descriptions=sorted(t for t in field_tags if t),
                   dtype="uint8 (probability x 255)" if uint8_field else "float32"),
        base=(None if base is None else dict(path=a.base, px=int(base.sum()),
                                             note="the walk prices additions to this field, "
                                                  "starting from the credit it already earns")),
        budget=dict(operating_dti=a.dti, breakeven_marginal_hit_rate=round(b, 6),
                    walk=a.walk,
                    candidate_px=int(candidates.sum()),
                    candidate_rule=("valid & ~catalogue & within R of the catalogue: a pixel "
                                    "further than R from every truth pixel has dT = 0 and can "
                                    "never clear the bar"
                                    if not a.all_candidates else "valid & ~catalogue (all)"),
                    walk_rule=("emit in decreasing favourability; a pixel is emitted iff adding it "
                               "raises the score, i.e. iff dT > alpha*DTI*dF/(1-alpha*DTI)")),
        result=dict(emitted_px=int(emitted.sum()),
                    chargeable_px=int((emitted & ~labels).sum()),
                    on_masked_catalogue_px=int((emitted & labels).sum()),
                    T=round(T, 3), F=round(F, 3), FN_w=round(FN, 3),
                    dti_on_calibration_truth=round(dti_cal, 6)),
        curve=curve,
        conformance=conformance,
        output=dict(path=str(out_path), sha256=sha256_file(out_path),
                    bytes=out_path.stat().st_size),
        caveat=("the budget is exact; the ranking is only as good as the input field, and the "
                "calibration truth is the masked catalogue, which cannot price discovery"),
    )
    rep = Path(a.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(out, indent=1) + "\n")
    print(f"break-even marginal hit rate at DTI {a.dti}: {b:.5f}")
    print(f"emitted {int(emitted.sum()):,} px ({int((emitted & ~labels).sum()):,} chargeable, "
          f"{int((emitted & labels).sum()):,} on the masked catalogue)")
    print(f"on the calibration truth: T={T:,.1f} F={F:,.1f} DTI={dti_cal:.4f}")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} B) and {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
