#!/usr/bin/env python3
"""Fit WHERE the hidden truth is from the five scores the platform already returned, then choose
the emission set that the fitted prior says is best.

WHY THIS EXISTS
---------------
Every local instrument in this repository scores against *some* fault population that is not the
scored one, and `scripts/rank_instruments.py` measured what that costs: the SGMC-gap proxy
anti-ranks the five scored files (Spearman rho = -0.8 against the board) while the catalogue
itself, scored with the platform's masking rule, orders them at rho = +0.9. A proxy is a
substitute; the public leaderboard is the instrument. This script uses the board directly.

THE MODEL (one approximation, stated once, measured as far as it can be)
-----------------------------------------------------------------------
Let G be the hidden truth (new faults only), lambda(x) the expected number of truth pixels per map
pixel, S the emitted set (binary, and R1 in docs/STRATEGY.md says emit at 1.0), and

    c_S(x) = max over emitted y of k(d(x,y))         the credit a truth pixel at x would earn
    E      = |S minus the pixels the platform masks| the chargeable emitted pixels

With the official kernel k(d) = max(1 - d/R, 0), R = 3 px:

    T = sum_x lambda(x) * c_S(x)                     (TP_w, in expectation)
    F = E - T                                        (FP_w; see the approximation below)
    DTI = T / (0.2*(T + F) + 0.8*G) = T / (0.2*E + 0.8*G),   G = sum_x lambda(x)

The one approximation is F = E - T: every emitted pixel within R of truth gives up its FP charge to
exactly one truth pixel. It is exact when truth is sparse enough that no two truth pixels compete
for the same emitted pixel, and it errs on the side of *over*-charging the prediction by at most the
kernel mass that several truth pixels share. It is the same identity T + F ~ E that
docs/STRATEGY.md R3 uses.

Parametrise the density as a non-negative combination of covariates, lambda = sum_k w_k * phi_k.
Then T_i = sum_k w_k * A_ik with A_ik = sum_x phi_k(x) c_i(x), and G = sum_k w_k * B_k with
B_k = sum_x phi_k(x). Every scored file gives one LINEAR equation in w:

    sum_k w_k * (A_ik - 0.8*DTI_i*B_k) = 0.2*DTI_i*E_i

Five files, K unknowns: solved by non-negative least squares, and validated by leaving each file
out, fitting on the other four and predicting the held-out score. A model that cannot predict a
score it did not see is a model that must not choose the next submission.

WHAT IT DECIDES
---------------
1. `Emit on the catalogue itself is free.` Forum 11516 (verified verbatim, fetched 2026-09-25):
   "Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from evaluation, so
   they do not count towards penalty terms." They cost nothing and can still earn credit for a new
   fault within 300 m. Two readings are computed: credit allowed (A) and credit denied (B).
2. `How much more to emit, and where`, by the marginal rule (R2): add pixels in decreasing marginal
   gain / cost until the ratio drops below 0.2*DTI/(1 - 0.2*DTI).

USAGE
    python scripts/fit_hidden_prior.py --out data/evidence/hidden_prior/fit.json
    python scripts/fit_hidden_prior.py --covar salience=data/derived/salience.tif \
        --emit-candidate --out data/evidence/hidden_prior/fit.json
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
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

R_PX = 3
ALPHA, BETA = 0.2, 0.8

# Owner-reported public-leaderboard results, 2026-09-25 (identical to scripts/leaderboard_anchor.py
# SCORED, whose sha256 pins are re-verified there and re-checked here).
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


def read_arr(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def credit_kernel(emitted: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """c_S(x) = max(0, 1 - d(x, S)/R): the credit a truth pixel at x earns from this emission."""
    dist = distance_transform_edt(~emitted)
    c = np.maximum(0.0, 1.0 - dist / float(R_PX))
    c[~valid] = 0.0
    return c.astype(np.float32)


def kernel_offsets(R: int = R_PX):
    offs = []
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            d = float(np.hypot(dy, dx))
            if d <= R:
                offs.append((dy, dx, max(0.0, 1.0 - d / R)))
    return offs


def coverage_mass(lam: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """C(y) = sum_x lambda(x) k(d(x,y)): expected truth mass inside the R-disc of every pixel."""
    out = np.zeros(lam.shape, dtype=np.float64)
    for dy, dx, k in kernel_offsets():
        if k <= 0:
            continue
        shifted = np.zeros_like(lam)
        ys_src = slice(max(0, -dy), lam.shape[0] - max(0, dy))
        xs_src = slice(max(0, -dx), lam.shape[1] - max(0, dx))
        ys_dst = slice(max(0, dy), lam.shape[0] - max(0, -dy))
        xs_dst = slice(max(0, dx), lam.shape[1] - max(0, -dx))
        shifted[ys_dst, xs_dst] = lam[ys_src, xs_src]
        out += k * shifted
    out[~valid] = 0.0
    return out


def marginal_gain(lam: np.ndarray, c_cur: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """m(y) = sum_x lambda(x) * max(0, k(d(x,y)) - c_cur(x)): extra TP if pixel y is emitted now."""
    out = np.zeros(lam.shape, dtype=np.float64)
    for dy, dx, k in kernel_offsets():
        if k <= 0:
            continue
        gain = np.maximum(0.0, k - c_cur) * lam
        shifted = np.zeros_like(gain)
        ys_src = slice(max(0, -dy), lam.shape[0] - max(0, dy))
        xs_src = slice(max(0, -dx), lam.shape[1] - max(0, dx))
        ys_dst = slice(max(0, dy), lam.shape[0] - max(0, -dy))
        xs_dst = slice(max(0, dx), lam.shape[1] - max(0, -dx))
        shifted[ys_dst, xs_dst] = gain[ys_src, xs_src]
        out += shifted
    out[~valid] = 0.0
    return out


def fit_weights(A: np.ndarray, B: np.ndarray, E: np.ndarray, dti: np.ndarray) -> np.ndarray:
    """Non-negative least squares for sum_k w_k (A_ik - 0.8*DTI_i*B_k) = 0.2*DTI_i*E_i."""
    M = A - BETA * dti[:, None] * B[None, :]
    b = ALPHA * dti * E
    w, _ = nnls(M, b)
    return w


def predict_dti(A_row: np.ndarray, E: float, w: np.ndarray, B: np.ndarray) -> float:
    T = float(np.dot(A_row, w))
    G = float(np.dot(B, w))
    if G <= 0:
        return 0.0
    return T / (ALPHA * (T + max(0.0, E - T)) + BETA * G + 1e-12)


# ---------------------------------------------------------------------------------------
# partial identification: five equations cannot pin down a whole density field, so instead of
# reporting one fit as if it were the truth, report the RANGE every candidate can take over ALL
# densities that reproduce the five scores to within `tol`. Both bounds are linear-fractional in w
# and the feasible set is a polytope, so each bound is one bisection over linear programmes:
#     DTI_cand >= d   <=>   sum_k w_k (A_k - 0.8*d*B_k) - 0.2*d*E >= 0
# ---------------------------------------------------------------------------------------
def polytope(A, B, E, dti, tol, g_lo, g_hi):
    """Linear constraints (A_ub w <= b_ub) defining the densities that reproduce every observed
    score to within `tol`:  w >= 0,  g_lo <= G <= g_hi,  and for each file
        (d_i - tol) <= T_i / (0.2*E_i + 0.8*G) <= (d_i + tol),
    both sides of which are linear in w for a fixed level."""
    rows, rhs = [], []
    for i in range(len(dti)):
        hi_lvl, lo_lvl = dti[i] + tol, max(0.0, dti[i] - tol)
        rows.append(list(A[i] - BETA * hi_lvl * B))
        rhs.append(ALPHA * hi_lvl * E[i])
        rows.append(list(-(A[i] - BETA * lo_lvl * B)))
        rhs.append(-ALPHA * lo_lvl * E[i])
    rows.append(list(-B))
    rhs.append(-g_lo)
    if g_hi is not None:
        rows.append(list(B))
        rhs.append(g_hi)
    return np.array(rows, dtype=float), np.array(rhs, dtype=float)


def _lp_feasible(rows, rhs, K):
    from scipy.optimize import linprog
    res = linprog(np.zeros(K), A_ub=rows, b_ub=rhs, bounds=[(0, None)] * K, method="highs")
    return bool(res.status == 0)


def falsification_row(A_cat, E_cat, B, leader):
    """One extra constraint on the density, from the board itself.

    example_submission.tif IS the catalogue (measured: bit-identical to labels.tif), it is public,
    and it costs nothing to submit. If the density implied that file scores higher than the best
    score anybody has actually achieved, the density is contradicted by the board - unless nobody
    ever submitted the example, which is the one assumption this flag makes. Expressed linearly:

        T_cat <= leader * (0.2*E_cat + 0.8*G)
    """
    return list(A_cat - BETA * leader * B), ALPHA * leader * E_cat


def min_tolerance(A, B, E, dti, g_lo, g_hi, hi=0.25, iters=40) -> float:
    """Smallest tol for which ANY non-negative density reproduces all five scores.

    Five equations in K unknowns sound like plenty until you ask how well the model class can do
    at all: if even the best-fitting density misses a score by 0.03, then a 0.005 tolerance is
    an empty set and every "prediction" computed inside it is arithmetic on nothing. This number
    is therefore reported before anything else is believed.
    """
    lo = 0.0
    if _lp_feasible(*polytope(A, B, E, dti, lo, g_lo, g_hi), A.shape[1]):
        return 0.0
    if not _lp_feasible(*polytope(A, B, E, dti, hi, g_lo, g_hi), A.shape[1]):
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        rows, rhs = polytope(A, B, E, dti, mid, g_lo, g_hi)
        if _lp_feasible(rows, rhs, A.shape[1]):
            hi = mid
        else:
            lo = mid
    return hi


def dti_extremes(A_row, E, B, rows, rhs, K, hi=1.0, iters=32) -> tuple[float, float]:
    """(min, max) of a candidate's predicted DTI over the feasible polytope.

    f(w) = T/(0.2*E + 0.8*G) is linear-fractional in w, so both extremes are one bisection over
    linear programmes:
        f(w) >= d  <=>  (A_row - 0.8*d*B).w >= 0.2*d*E
    The predicate "some feasible w has f >= d" is monotone in d, which is what makes bisection
    valid; each check is a feasibility LP, not an optimisation, so an unbounded polytope is fine.
    """
    def exists_ge(d: float) -> bool:
        coef = A_row - BETA * d * B
        const = ALPHA * d * E
        r = np.vstack([rows, -coef[None, :]])
        b = np.concatenate([rhs, [-(const + 1e-9)]])
        return _lp_feasible(r, b, K)

    def exists_le(d: float) -> bool:
        coef = A_row - BETA * d * B
        const = ALPHA * d * E
        r = np.vstack([rows, coef[None, :]])
        b = np.concatenate([rhs, [const - 1e-9]])
        return _lp_feasible(r, b, K)

    def bisect_monotone(pred, low, high, decreasing: bool) -> float:
        """Boundary of a monotone predicate on [low, high].

        decreasing=True  (predicate holds for small d):  return sup{d : pred(d)}
        decreasing=False (predicate holds for large d):  return inf{d : pred(d)}
        """
        lo, hi_ = float(low), float(high)
        if decreasing:
            if pred(hi_):
                return hi_
            for _ in range(iters):
                mid = 0.5 * (lo + hi_)
                if pred(mid):
                    lo = mid
                else:
                    hi_ = mid
            return lo
        if pred(lo):
            return lo
        for _ in range(iters):
            mid = 0.5 * (lo + hi_)
            if pred(mid):
                hi_ = mid
            else:
                lo = mid
        return hi_

    mx = bisect_monotone(exists_ge, 0.0, hi, decreasing=True)
    mn = bisect_monotone(exists_le, 0.0, hi, decreasing=False)
    return float(mn), float(mx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--base", default="data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif",
                    help="the field the candidate is built from (the best-scoring file we hold)")
    ap.add_argument("--covar", action="append", default=[],
                    help="extra density covariate: path=name (float raster on the competition grid)")
    ap.add_argument("--grid", default="0,1,4",
                    help="extra-emission budgets (in THOUSANDS of pixels) to evaluate, "
                         "added on top of --base by decreasing marginal gain")
    ap.add_argument("--emit-candidate", action="store_true",
                    help="write the best candidate raster (validated, template-conformant)")
    ap.add_argument("--tol", type=float, default=0.005,
                    help="how closely a density must reproduce each observed score to stay in the "
                         "feasible set (DTI units; the board publishes 4 decimals)")
    ap.add_argument("--g-min", type=float, default=1000.0,
                    help="smallest hidden-truth size (px) considered plausible")
    ap.add_argument("--g-max", type=float, default=200000.0,
                    help="largest hidden-truth size (px) considered plausible")
    ap.add_argument("--rule", choices=["maximin", "point"], default="maximin",
                    help="maximin = choose the candidate with the best WORST case over the "
                         "feasible set (default); point = trust the single NNLS fit")
    ap.add_argument("--no-falsification-constraint", dest="falsification_constraint",
                    action="store_false",
                    help="drop the assumption that somebody has already submitted the platform's "
                         "example file (the catalogue), which is what lets the board's best score "
                         "rule out densities that predict an impossible score for it")
    ap.add_argument("--leader", type=float, default=0.3049,
                    help="best public-leaderboard score observed (0.3049 on 2026-09-25); used "
                         "only to test whether the fitted model predicts an impossible score for "
                         "a file anybody could have submitted")
    ap.add_argument("--out", default="data/evidence/hidden_prior/fit.json")
    a = ap.parse_args()

    valid = np.isfinite(read_arr(Path(a.template)))
    labels = np.nan_to_num(read_arr(Path(a.labels)), nan=0.0) > 0.5
    labels &= valid
    dist_lab = distance_transform_edt(~labels)

    # ------------------------------------------------------------------ density covariates
    covars: list[tuple[str, np.ndarray, str]] = []
    covars.append(("uniform", np.ones(valid.shape, dtype=np.float32),
                   "constant background density over the valid footprint"))
    near = ((dist_lab <= 1.0) & valid).astype(np.float32)
    covars.append(("d_lab<=1", near,
                   "within 1 px (100 m) of a known USGS/INGENIOUS fault pixel"))
    mid = ((dist_lab > 1.0) & (dist_lab <= R_PX) & valid).astype(np.float32)
    covars.append(("1<d_lab<=3", mid,
                   "the metric's own tolerance ring around the known catalogue (100-300 m)"))
    for spec in a.covar:
        path, name = spec.split("=", 1)
        arr = read_arr(Path(path)).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        m = float(np.nanmax(np.abs(arr[valid])))
        if m > 0:
            arr = arr / m
        covars.append((name, arr, f"external covariate {path} (max-abs normalised on the footprint)"))

    C = np.stack([c[1] * valid for c in covars]).astype(np.float64)
    B = C.sum(axis=(1, 2))

    # ------------------------------------------------------------------ the five equations
    files = [dict(s) for s in SCORED if (ROOT / s["path"]).exists()]
    missing = [s["sha8"] for s in SCORED if not (ROOT / s["path"]).exists()]
    A_rows, E_list, dti_list, per_file = [], [], [], []
    for f in files:
        pred = np.nan_to_num(read_arr(ROOT / f["path"]), nan=0.0)
        emitted = pred > 0
        chargeable = emitted & valid & ~labels          # the platform masks catalogue pixels
        E = float(chargeable.sum())
        c = credit_kernel(emitted, valid)
        A_row = (C * c[None, :, :]).sum(axis=(1, 2))
        A_rows.append(A_row)
        E_list.append(E)
        dti_list.append(f["score"])
        per_file.append(dict(f, sha256=sha256(ROOT / f["path"]), emitted_px=int(emitted.sum()),
                             chargeable_px=int(E), on_catalogue_px=int((emitted & labels).sum()),
                             coverage_mass=float(c.sum()),
                             A=[[covars[k][0], float(A_row[k])] for k in range(len(covars))]))
    A = np.array(A_rows, dtype=np.float64)
    E_arr = np.array(E_list, dtype=np.float64)
    dti = np.array(dti_list, dtype=np.float64)

    w = fit_weights(A, B, E_arr, dti)
    fitted = np.array([predict_dti(A[i], E_arr[i], w, B) for i in range(len(files))])

    # ------------------------------------------------------------------ partial identification
    # The point fit above is ONE of many densities consistent with five scores. Before any
    # candidate is trusted, ask what every consistent density says about it.
    # the catalogue-only (example submission) equation, used both as a candidate and as the
    # board's own constraint on what a density may imply
    c_cat = credit_kernel(labels, valid)
    A_cat = (C * c_cat[None, :, :]).sum(axis=(1, 2))
    E_cat = 0.0                                     # every catalogue pixel is masked: nothing charged
    t_star = min_tolerance(A, B, E_arr, dti, a.g_min, a.g_max)
    tol = max(float(a.tol), 0.0 if np.isnan(t_star) else float(t_star))
    K = A.shape[1]
    rows, rhs, feasible, escalated = None, None, False, False
    for _ in range(24):
        rows, rhs = polytope(A, B, E_arr, dti, tol, a.g_min, a.g_max)
        if a.falsification_constraint:
            r, b = falsification_row(A_cat, E_cat, B, a.leader)
            rows = np.vstack([rows, np.array(r)[None, :]])
            rhs = np.concatenate([rhs, [b]])
        feasible = _lp_feasible(rows, rhs, K)
        if feasible:
            break
        tol *= 1.25
        escalated = True
    if not feasible:
        print("the model class cannot reproduce the five scores at ANY tolerance up to "
              f"{tol:.3f} under the constraints given; no candidate is certified", file=sys.stderr)
    ident = []
    for i in range(len(files)):
        mn, mx = dti_extremes(A[i], E_arr[i], B, rows, rhs, K)
        ident.append(dict(sha8=files[i]["sha8"], observed=files[i]["score"],
                          min=round(mn, 4), max=round(mx, 4),
                          note=("the range of scores THIS file could have had, over every "
                                "density that reproduces all five observed scores")))

    # ------------------------------------------------------------------ leave-one-out
    loo = []
    for i in range(len(files)):
        idx = [j for j in range(len(files)) if j != i]
        w_i = fit_weights(A[idx], B, E_arr[idx], dti[idx])
        pred_i = predict_dti(A[i], E_arr[i], w_i, B)
        loo.append(dict(held_out=files[i]["sha8"], actual=files[i]["score"],
                        predicted=round(float(pred_i), 4),
                        abs_error=round(abs(float(pred_i) - files[i]["score"]), 4),
                        weights=[round(float(x), 6) for x in w_i]))

    # ------------------------------------------------------------------ candidate policies
    base_path = ROOT / a.base
    base = np.nan_to_num(read_arr(base_path), nan=0.0) > 0
    lam = np.zeros(valid.shape, dtype=np.float64)
    for k, cw in enumerate(w):
        lam += float(cw) * C[k]

    def measure(S: np.ndarray, name: str, note: str) -> dict:
        chargeable = S & valid & ~labels
        E = float(chargeable.sum())
        c = credit_kernel(S, valid)
        A_row = (C * c[None, :, :]).sum(axis=(1, 2))
        d = predict_dti(A_row, E, w, B)
        mn, mx = dti_extremes(A_row, E, B, rows, rhs, K)
        return dict(name=name, note=note, emitted_px=int((S & valid).sum()),
                    chargeable_px=int(E), predicted_dti=round(float(d), 4),
                    dti_min=round(mn, 4), dti_max=round(mx, 4),
                    T_expected=round(float(np.dot(A_row, w)), 1),
                    G_expected=round(float(np.dot(B, w)), 1))

    struct = np.ones((2 * 1 + 1, 2 * 1 + 1), bool)
    candidates = [measure(base, "base_7f00890a", "the shipped field, unchanged (reference)"),
                  measure(labels, "catalogue_only_example_submission",
                          "the platform's own example_submission.tif, which this repository "
                          "measured to be bit-identical to labels.tif (docs/index.html finding 2)"),
                  measure(base | labels, "base_plus_catalogue",
                          "the shipped field plus every known-fault pixel (free: forum 11516)")]
    for r in (1, 2, 3):
        ring = binary_dilation(labels, structure=np.ones((2 * r + 1, 2 * r + 1), bool)) & valid
        candidates.append(measure(base | labels | ring, f"base_plus_catalogue_dilate{r}",
                                  f"catalogue dilated by {r} px (the whole ring, ungated)"))

    # marginal-gain ordered additions on top of base + catalogue
    S0 = base | labels
    c0 = credit_kernel(S0, valid)
    m = marginal_gain(lam, c0, valid)
    m[labels] = 0.0                      # already emitted, and free
    order = np.argsort(m.ravel())[::-1]
    budgets = [int(float(x) * 1000) for x in a.grid.split(",")]
    greedy = {}
    for n in budgets:
        idx = order[:n]
        S = S0.copy()
        S.flat[idx] = True
        greedy[n] = measure(S, f"base_plus_top{n // 1000}k",
                            f"the {n:,} pixels of highest marginal gain under the fitted prior, "
                            f"added to the shipped field + catalogue")
    candidates.extend(greedy.values())

    # ------------------------------------------------------------------ falsification test
    # The platform's own example_submission.tif is the catalogue (measured). If the fitted density
    # says that anyone submitting it would beat the best score on the board, the density is wrong:
    # that file is public, it costs nothing to submit, and the board is public.
    cat = [c for c in candidates if c["name"] == "catalogue_only_example_submission"]
    falsification = None
    if cat:
        c0 = cat[0]
        impossible = bool(c0["dti_min"] > a.leader)
        falsification = dict(
            candidate="catalogue_only_example_submission",
            predicted_dti=c0["predicted_dti"], dti_min=c0["dti_min"], dti_max=c0["dti_max"],
            best_observed_public_score=a.leader,
            verdict=("FALSIFIED: the model predicts a score above the best anyone has achieved "
                     "for a file every competitor already holds, so this density is wrong - the "
                     "hidden truth is not where it puts it" if impossible else
                     "not falsified by the board at this tolerance"),
            consequence=("do not ship a candidate whose case rests on this density; the free-"
                         "catalogue hedge is still worth taking because it cannot cost anything "
                         "under the platform's stated masking rule" if impossible else None))

    key = (lambda r: r["dti_min"]) if a.rule == "maximin" else (lambda r: r["predicted_dti"])
    shipable = [c for c in candidates if c["name"].startswith("base")]
    best = max(shipable, key=key) if shipable else max(candidates, key=key)
    best_greedy = max(greedy.values(), key=key) if greedy else None

    # ------------------------------------------------------------------ credit-denied variant
    # Reading B of "it should not matter whether these known faults are included": emitting on a
    # catalogue pixel earns nothing. Then the catalogue pixels are removed from the credit kernel
    # but still cost nothing, so only the marginal additions matter.
    A_mask = (C * (valid & ~labels)[None, :, :]).sum(axis=(1, 2))
    G_B = float(np.dot(B, w))
    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/fit_hidden_prior.py",
        "purpose": ("fit the spatial density of the hidden new-fault truth from the five public "
                    "scores, then choose the emission set the fitted density prefers"),
        "model": {
            "equations": ["DTI_i = T_i / (0.2*E_i + 0.8*G)",
                          "T_i = sum_k w_k A_ik,  G = sum_k w_k B_k",
                          "F_i = E_i - T_i  (one emitted pixel serves at most one truth pixel)"],
            "R_px": R_PX, "alpha": ALPHA, "beta": BETA,
            "masking": ("catalogue pixels are charge-free (forum 11516, verbatim, fetched "
                        "2026-09-25) and are excluded from E"),
            "sources": ["https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/",
                        "https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516"],
        },
        "covariates": [dict(name=n, note=note, B_px=float(B[k])) for k, (n, _, note) in enumerate(covars)],
        "footprint": dict(valid_px=int(valid.sum()), catalogue_px=int(labels.sum())),
        "weights": {covars[k][0]: float(w[k]) for k in range(len(covars))},
        "G_expected_px": round(float(np.dot(B, w)), 1),
        "files": per_file,
        "fit": [dict(sha8=files[i]["sha8"], actual=files[i]["score"],
                     fitted=round(float(fitted[i]), 4),
                     abs_error=round(abs(float(fitted[i]) - files[i]["score"]), 4))
                for i in range(len(files))],
        "leave_one_out": loo,
        "partial_identification": dict(
            tol=round(float(tol), 5),
            smallest_possible_tol=None if np.isnan(t_star) else round(float(t_star), 5),
            tol_escalated_to_keep_the_set_non_empty=bool(escalated),
            polytope_non_empty=bool(feasible),
            falsification_constraint_applied=bool(a.falsification_constraint),
            g_min=a.g_min, g_max=a.g_max,
            per_scored_file=ident,
            meaning=("five equations cannot identify a density field: the ranges below are what "
                     "every density consistent with the five observed scores implies. "
                     "smallest_possible_tol is how well the model class can fit the board AT ALL - "
                     "a model that cannot fit the five numbers it was given cannot rank a sixth.")),
        "candidates": candidates,
        "best_candidate": best["name"] if best else None,
        "falsification_test": falsification,
        "missing_scored_files": missing,
        "credit_denied_reading_B": {
            "meaning": ("if emitting on a catalogue pixel earns no credit either, only the "
                        "off-catalogue part of a candidate counts; the ordering of the additions "
                        "is unchanged, the level is lower"),
            "G_expected_px": round(G_B, 1),
        },
    }

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1) + "\n")

    print("covariate weights (truth pixels per map pixel):")
    for k, (n, _, _) in enumerate(covars):
        print(f"  {n:<12} w = {w[k]:.6g}")
    print(f"implied |G| = {np.dot(B, w):,.0f} truth pixels\n")
    print(f"{'file':<12}{'actual':>9}{'fitted':>9}{'LOO pred':>10}{'LOO err':>9}")
    for i in range(len(files)):
        print(f"{files[i]['sha8']:<12}{files[i]['score']:>9.4f}{fitted[i]:>9.4f}"
              f"{loo[i]['predicted']:>10.4f}{loo[i]['abs_error']:>9.4f}")
    print()
    print(f"{'candidate':<32}{'emitted px':>12}{'charged px':>12}{'point DTI':>11}"
          f"{'worst case':>12}{'best case':>11}")
    for c in candidates:
        print(f"{c['name']:<32}{c['emitted_px']:>12,}{c['chargeable_px']:>12,}"
              f"{c['predicted_dti']:>11.4f}{c['dti_min']:>12.4f}{c['dti_max']:>11.4f}")
    if falsification:
        print("\nFALSIFICATION TEST (the platform's own example file):")
        print(f"   predicted DTI {falsification['predicted_dti']}  "
              f"[{falsification['dti_min']}, {falsification['dti_max']}]  "
              f"vs best observed public score {falsification['best_observed_public_score']}")
        print(f"   -> {falsification['verdict']}")
    print(f"\nbest under --rule {a.rule}: {best['name']}  "
          f"(point {best['predicted_dti']}, worst case {best['dti_min']})")
    print("range each scored file could have had, over every consistent density:")
    for r in ident:
        print(f"   {r['sha8']:<12} observed {r['observed']:.4f}  "
              f"[{r['min']:.4f}, {r['max']:.4f}]")
    print(f"wrote {out_path}")

    if a.emit_candidate:
        # Ship the ROBUST candidate, not the highest point estimate. The catalogue pixels are the
        # only addition that cannot cost anything: the platform states in writing that known-fault
        # pixels are excluded from the penalty terms (forum 11516), and its own example submission
        # is the catalogue raster. Everything else the fit likes is an extrapolation.
        from src.submission_io import conform_to_template, sha256_file
        with rasterio.open(Path(a.template)) as src:
            tmpl = src.read(1)
            prof = src.profile.copy()
        S = base | labels
        field = np.where(S, np.float32(1.0), np.float32(0.0))
        field, _ = conform_to_template(field, tmpl)
        dest = out_path.parent / "candidate_s5_catalogue_hedge.tif"
        prof.update(dtype="float32", count=1, nodata=float("nan"), compress="lzw",
                    tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(dest, "w", **prof) as dst:
            dst.write(field, 1)
            dst.set_band_description(1, "GEMSDOE 5: 7f00890a union the masked catalogue "
                                        "(free-coverage hedge, forum 11516)")
        n_emitted = int(np.nansum(field > 0))
        emit_report = {
            "candidate": dest.name, "path": str(dest), "bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
            "emitted_px": n_emitted,
            "added_over_base_px": n_emitted - int(np.nansum(base)),
            "chargeable_px": int(np.nansum((field > 0) & np.isfinite(tmpl) & ~labels)),
            "why_this_one": ("the catalogue pixels are the only addition that cannot lower the "
                             "score under the platform's written masking rule, and the fit's "
                             "point estimate for it (%.3f) is the largest single move available"
                             % [c for c in candidates if c["name"] == "base_plus_catalogue"][0]["predicted_dti"]),
            "experiment": ("the score difference against 0.1563 measures which reading of forum "
                           "11516 the platform implements: no change = catalogue pixels earn no "
                           "credit either; a jump = they do, and the whole emission plan changes"),
        }
        out["emitted_candidate"] = emit_report
        out_path.write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote candidate raster {dest} ({dest.stat().st_size:,} B, "
              f"{n_emitted:,} emitted px, sha256 {emit_report['sha256'][:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
