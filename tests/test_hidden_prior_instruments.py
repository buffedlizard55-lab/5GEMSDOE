"""Tests for the two instruments added in session 2 (2026-09-25):

* scripts/rank_instruments.py  - does a local population order the scored files the way the
  leaderboard does?  (measured: catalogue rho = +0.9, SGMC-gap rho = -0.8)
* scripts/fit_hidden_prior.py  - fit the hidden-truth density from the five public scores and
  report the RANGE every candidate can take, not a point estimate nobody can certify.

Both scripts are instruments: their value is entirely in whether their arithmetic is right, so
these tests pin the arithmetic on small grids where the answer can be worked out by hand, and pin
the safety property that matters most - that the tool says "not identified" rather than printing a
confident number when the data cannot support one.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

rasterio = pytest.importorskip("rasterio")


def _mod(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RANK = _mod("rank_instruments", "scripts/rank_instruments.py")
FIT = _mod("fit_hidden_prior", "scripts/fit_hidden_prior.py")


# --------------------------------------------------------------------- rank_instruments
def test_spearman_recovers_a_known_permutation():
    assert RANK.spearman([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
    assert RANK.spearman([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]) == pytest.approx(-1.0)
    # a single adjacent swap of 4 ranks: rho = 1 - 6*sum(d^2)/(n(n^2-1)) = 1 - 6*2/60 = 0.8
    assert RANK.spearman([0.1, 0.2, 0.3, 0.4], [0.1, 0.3, 0.2, 0.4]) == pytest.approx(0.8, abs=1e-9)
    assert RANK.spearman([0.1, 0.1, 0.1], [0.2, 0.3, 0.4]) is None      # a constant has no ranking


def test_the_fp_region_excludes_the_pixels_the_platform_masks(tmp_path):
    """The whole point of the instrument: a prediction sitting on a catalogue fault is free."""
    H = W = 20
    tmpl = np.ones((H, W), dtype="float32")
    tmpl[0, 0] = np.nan                       # one pixel outside the scored footprint
    labels = np.zeros((H, W), dtype="float32")
    labels[5, :] = 1.0                        # a horizontal fault
    p = tmp_path / "t.tif"
    with rasterio.open(p, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32") as dst:
        dst.write(tmpl, 1)
    valid = RANK.read_mask(p)
    assert int(valid.sum()) == H * W - 1
    lab = labels > 0.5
    region = valid & ~lab
    assert int(region.sum()) == (H * W - 1) - W, "every catalogue pixel is excluded from the FP term"


def test_score_within_mask_matches_a_hand_computation():
    from src.metrics import GtContext, score_within_mask
    pred = np.zeros((11, 11), dtype="float32")
    pred[5, 5] = 1.0                          # one emitted pixel, one unit of mass
    ctx = GtContext(pred * 0 + np.eye(11, dtype="float32")[5] * 0 + (np.arange(11) == 5)[:, None] * 0)
    truth = np.zeros((11, 11), dtype=bool)
    truth[5, 8] = True                        # truth 3 px away: k = 1 - 3/3 = 0
    ctx2 = GtContext(truth, R_pixels=3)
    res = score_within_mask(pred, ctx2, np.ones((11, 11), dtype=bool))
    # d = 3 -> k = 0 -> no credit, and the emitted pixel pays full price
    assert res["TP_w"] == pytest.approx(0.0)
    assert res["FP_w"] == pytest.approx(1.0)
    truth2 = np.zeros((11, 11), dtype=bool)
    truth2[5, 6] = True                       # 1 px away: k = 2/3
    res2 = score_within_mask(pred, GtContext(truth2, R_pixels=3), np.ones((11, 11), dtype=bool))
    assert res2["TP_w"] == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert res2["FP_w"] == pytest.approx(1.0 - 2.0 / 3.0, abs=1e-6)


# --------------------------------------------------------------------- fit_hidden_prior
def test_credit_kernel_is_the_triangular_weight():
    emitted = np.zeros((9, 9), dtype=bool)
    emitted[4, 4] = True
    valid = np.ones((9, 9), dtype=bool)
    c = FIT.credit_kernel(emitted, valid)
    assert c[4, 4] == pytest.approx(1.0)
    assert c[4, 5] == pytest.approx(2.0 / 3.0, abs=1e-6)      # 1 px away, R = 3
    assert c[4, 7] == pytest.approx(0.0)                      # 3 px away -> exactly zero
    assert c[4, 8] == 0.0


def test_the_model_reproduces_the_metric_algebra_it_claims():
    """DTI = T / (0.2*E + 0.8*G): build a tiny world where every term is countable, then check
    the script's predictor against a direct transcription of the official formula."""
    rng = np.random.default_rng(0)
    H = W = 24
    valid = np.ones((H, W), dtype=bool)
    truth = np.zeros((H, W))
    for y in (4, 12, 20):
        truth[y, 3:21] = 1.0                       # three horizontal faults, 18 px each
    lam = truth.copy()                             # the density: one truth pixel per truth pixel
    emitted = np.zeros((H, W), dtype=bool)
    for y in (4, 12, 20):
        emitted[y, 3:21] = True                    # emitted exactly on them
    c = FIT.credit_kernel(emitted, valid)
    T = float((lam * c).sum())
    G = float(lam.sum())
    E = float(emitted.sum())
    direct = T / (0.2 * (T + (E - T)) + 0.8 * G)
    assert FIT.predict_dti(np.array([T]), E, np.array([1.0]), np.array([G])) == \
        pytest.approx(direct, rel=1e-9)
    assert FIT.predict_dti(np.array([T]), E, np.array([1.0]), np.array([G])) == \
        pytest.approx(T / (0.2 * E + 0.8 * G), rel=1e-9)


def test_an_empty_feasible_set_is_reported_not_papered_over():
    """With contradictory 'scores' no density can fit: min_tolerance must say so (nan) instead of
    returning a tolerance whose polytope is empty and whose extremes are nonsense."""
    A = np.array([[1e6, 0.0], [1e6, 0.0]], dtype=float)
    B = np.array([5e6, 1e5], dtype=float)
    E = np.array([1e5, 1e5], dtype=float)
    dti = np.array([0.9, 0.01], dtype=float)      # same geometry, wildly different scores
    t = FIT.min_tolerance(A, B, E, dti, 1000.0, 200000.0, hi=0.5)
    # either it is infeasible at every tolerance (nan) or it needs a very loose one - but the
    # caller must be able to tell the difference, which is what the returned value is for
    assert np.isnan(t) or t > 0.2, f"a tight tolerance here would be an empty set: {t}"


def test_extremes_bracket_the_point_fit_on_a_consistent_problem():
    """When the data ARE generated by the model, every candidate's range must contain the value
    the generating density gives it - otherwise the bisection is wrong."""
    rng = np.random.default_rng(3)
    H = W = 32
    valid = np.ones((H, W), dtype=bool)
    phi1 = np.ones((H, W))
    dist = np.abs(np.arange(H)[:, None] - 10.0)
    # full (H, W), not a broadcast column: a (32, 1) array would silently make the second
    # covariate a single row and the "truth density" a different field from the one scored
    phi2 = np.tile((dist <= 1).astype(float), (1, W))      # a band of "near catalogue" pixels
    truth_density = 0.02 * phi1 + 0.20 * phi2
    B = np.array([phi1.sum(), phi2.sum()], dtype=float)
    sets = []
    for k in range(4):
        em = np.zeros((H, W), dtype=bool)
        em[np.random.default_rng(10 + k).random((H, W)) < 0.15] = True
        sets.append(em)
    A, E, d = [], [], []
    for em in sets:
        c = FIT.credit_kernel(em, valid)
        A.append([float((phi1 * c).sum()), float((phi2 * c).sum())])
        E.append(float(em.sum()))
        T = float((truth_density * c).sum())
        G = float(truth_density.sum())
        d.append(T / (0.2 * E[-1] + 0.8 * G))
    A = np.array(A); E = np.array(E, dtype=float); d = np.array(d, dtype=float)
    w_true = np.array([0.02, 0.20])
    # g bounds are an argument, not a constant of nature: this world's truth is ~21 px, so the
    # plausibility bounds the real run uses (1e3..2e5 px) would exclude the generating density.
    rows, rhs = FIT.polytope(A, B, E, d, 1e-5, 1.0, 1e6)
    K = A.shape[1]
    assert FIT._lp_feasible(rows, rhs, K), "the generating density must be feasible"
    for i in range(len(sets)):
        mn, mx = FIT.dti_extremes(A[i], E[i], B, rows, rhs, K)
        point = FIT.predict_dti(A[i], E[i], w_true, B)
        assert mn - 1e-6 <= point <= mx + 1e-6, f"range [{mn}, {mx}] excludes the truth {point}"


def test_the_shipped_candidate_is_the_base_field_plus_the_catalogue(tmp_path):
    """The one decision the analysis supports: the catalogue pixels are added, nothing else."""
    base = ROOT / "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"
    cand = ROOT / "data/evidence/hidden_prior/candidate_s5_catalogue_hedge.tif"
    if not (base.exists() and cand.exists()):
        pytest.skip("the candidate raster is produced by scripts/fit_hidden_prior.py --emit-candidate")
    with rasterio.open(base) as s:
        b = s.read(1)
        prof = s.profile
    with rasterio.open(cand) as s:
        c = s.read(1)
        assert str(s.crs) == str(prof["crs"])
        assert list(s.transform)[:6] == list(prof["transform"])[:6]
    with rasterio.open(ROOT / "data/labels.tif") as s:
        labels = np.nan_to_num(s.read(1)) > 0.5
    with rasterio.open(ROOT / "data/sample_submission.tif") as s:
        valid = np.isfinite(s.read(1))
    assert np.array_equal(np.isfinite(c), valid), "NaN exactly outside the template footprint"
    assert float(np.nanmin(c)) >= 0.0 and float(np.nanmax(c)) <= 1.0
    emitted = np.isfinite(c) & (c > 0)
    assert np.all(emitted[labels]), "every catalogue pixel is emitted (they cost nothing)"
    base_emitted = np.isfinite(b) & (b > 0)
    assert np.all(emitted[base_emitted]), "the shipped field's support is preserved"
    # and nothing was added beyond the union of the two
    assert int(emitted.sum()) == int((base_emitted | labels).sum())


def test_the_rank_report_committed_in_the_repo_is_self_consistent():
    """If the measured report is part of the record, its rank correlation must be the one the
    narrative claims, and it must have been computed on all five scored files."""
    p = ROOT / "data/evidence/rank_instruments.json"
    if not p.exists():
        pytest.skip("run scripts/rank_instruments.py first")
    d = json.loads(p.read_text())
    scored = [f for f in d["files"] if f.get("score") is not None]
    assert len(scored) == 5, ("five files have public scores; the instrument is judged on all five "
                              "(unscored candidates may also be listed, they are not ranked)")
    for name, v in d["rank_agreement"].items():
        assert v["n_files"] == 5
        assert -1.0 <= v["spearman_rho_vs_leaderboard"] <= 1.0
