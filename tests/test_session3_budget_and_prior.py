"""Session 5GEMSDOE-3 (2026-09-25): the tools that set the emission budget, the literature prior
intersected with a detector, and the independent second judge.

WHY THIS FILE EXISTS
--------------------
Three new load-bearing pieces of machinery landed this session and each of them can be wrong in a
way that only a test catches:

* ``scripts/emit_by_marginal_rule.py`` - the metric's marginal-inclusion rule, applied exactly.
  If its dT/dF increments are wrong, every "this pixel is worth emitting" statement in the
  repository is wrong, and nothing else in the repo would notice.
* ``scripts/build_dilational_annulus.py`` - the Faulds & Hinz prior intersected with a detector.
  If the geometry it inherits changes, the prior silently changes with it.
* ``scripts/verify_candidates_independently.py`` - a second, independent judge.  If the two
  implementations of the metric disagree, the repository has been reporting a number only one
  implementation produced.

All tests build tiny synthetic rasters in ``tmp_path``; none of them needs the 419 MB feature
stack, so they run in the same CI job as everything else.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TR = Affine(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


emr = _load("emit_by_marginal_rule")
ann = _load("build_dilational_annulus")
bst = _load("build_structural_targets")


def _grid(path: Path, arr: np.ndarray, dtype="float32", nodata=float("nan")):
    prof = dict(driver="GTiff", dtype=dtype, count=1, width=arr.shape[1], height=arr.shape[0],
                crs="EPSG:32611", transform=TR, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


# --------------------------------------------------------------------------------------
# the algebra
# --------------------------------------------------------------------------------------
def test_breakeven_matches_the_published_numbers():
    """0.020 at DTI 0.10, 0.032 at 0.156, 0.065 at 0.305 - STRATEGY.md R2's table.

    b = alpha*DTI/(1 - alpha*DTI) with alpha = 0.2, so each value is checked against the closed
    form rather than against a rounded table entry.
    """
    for dti, expected in ((0.10, 0.02041), (0.1563, 0.03227), (0.20, 0.04167), (0.3049, 0.06494)):
        assert emr.breakeven_marginal_hit_rate(dti) == pytest.approx(expected, abs=1e-4), dti
    with pytest.raises(ValueError):
        emr.breakeven_marginal_hit_rate(1.0)


def test_dti_from_terms_agrees_with_src_metrics():
    from src.metrics import GtContext
    gt = np.zeros((40, 40), np.float32)
    gt[20, 5:35] = 1.0
    pred = np.zeros((40, 40), np.float32)
    pred[18:22, 8:32] = 1.0
    dti, (T, F, FN) = GtContext(gt, 3).score(pred, return_components=True)
    assert emr.dti_from_terms(T, F, gt.sum()) == pytest.approx(dti, abs=1e-9)


# --------------------------------------------------------------------------------------
# the marginal walk
# --------------------------------------------------------------------------------------
def _tiny_context():
    gt = np.zeros((41, 41), np.float32)
    gt[20, 10:31] = 1.0                      # a 21-px horizontal truth line
    from src.metrics import GtContext
    return GtContext(gt, 3), gt


def test_a_pixel_further_than_R_from_truth_is_never_admitted():
    """The rule cannot buy discovery from a catalogue-calibrated instrument - by construction."""
    ctx, gt = _tiny_context()
    far = np.zeros((41, 41), bool)
    far[2:6, 2:6] = True                     # nowhere near the truth line
    budget = emr.MarginalBudget(ctx, candidates=far, order=np.ones((41, 41)))
    emitted, curve = budget.walk(0.03227)
    assert emitted.sum() == 0
    assert curve[-1]["skipped_px"] == 16


def test_the_walk_emits_near_truth_and_reports_what_it_skipped():
    ctx, gt = _tiny_context()
    cand = np.zeros((41, 41), bool)
    cand[19:22, 8:34] = True                 # a 3-px band straddling the truth
    order = np.zeros((41, 41))
    order[20, :] = 1.0                       # the exact line first
    budget = emr.MarginalBudget(ctx, candidates=cand, order=order)
    emitted, curve = budget.walk(0.03227)
    # the line itself has dT/dF >> break-even, so it is admitted
    assert emitted[20, 10:31].all()
    assert budget.T > 0
    # the curve's final n is exactly what was emitted
    assert curve[-1]["n"] == int(emitted.sum())
    assert curve[-1]["n"] + curve[-1]["skipped_px"] == int(cand.sum())


def test_budget_mode_takes_the_maximal_clearing_prefix():
    """Prefix semantics: the first ranked pixel that fails ends the emission.

    Measured here rather than assumed - the band starts 4 px from the truth, so its very first
    candidate has dT = 0 and budget mode emits nothing at all, while rule mode keeps looking and
    finds the part of the band that sits on the truth.  This is why the script's default is
    ``rule`` and why budget mode is the natural reading only for a CALIBRATED probability field,
    where the ranking is the probability and the prefix degenerates to a threshold.
    """
    ctx, _ = _tiny_context()
    cand = np.zeros((41, 41), bool)
    cand[19:22, 6:36] = True
    cand[33:36, 6:36] = True                 # a far band, also a candidate
    order = np.zeros((41, 41))
    order[20, :] = 1.0                       # the good line ranks FIRST
    order[33:36, 6:36] = 0.5                 # the far band ranks second
    e_bud, curve_bud = emr.MarginalBudget(ctx, cand, order).walk(0.03227, mode="budget")
    e_rule, curve_rule = emr.MarginalBudget(ctx, cand, order).walk(0.03227, mode="rule")
    assert curve_bud[-1]["stopped_because"].startswith("budget:")
    assert e_bud.sum() == 0, "the first ranked candidate is 4 px from the truth: dT = 0"
    assert e_rule[20, 10:31].all(), "rule mode still emits the part that clears the bar"
    assert curve_rule[-1]["stopped_because"] == "candidate list exhausted"


def test_budget_mode_is_degenerate_when_the_best_pixel_is_far_from_the_truth():
    """The finding, not a bug: a catalogue-calibrated budget cannot buy off-catalogue emission."""
    ctx, _ = _tiny_context()
    cand = np.zeros((41, 41), bool)
    cand[19:22, 6:36] = True
    cand[33:36, 6:36] = True
    order = np.zeros((41, 41))
    order[33:36, 6:36] = 1.0                 # the far band ranks FIRST
    order[20, :] = 0.5
    e_bud, _ = emr.MarginalBudget(ctx, cand, order).walk(0.03227, mode="budget")
    e_rule, _ = emr.MarginalBudget(ctx, cand, order).walk(0.03227, mode="rule")
    assert e_bud.sum() == 0, "budget mode stops before it emits anything"
    assert e_rule[20, 10:31].all(), "rule mode keeps looking and finds the good line"


def test_starting_from_a_base_field_prices_additions_only():
    """The 0.1563 field already earns its credit; an addition must earn its own marginal rate."""
    ctx, gt = _tiny_context()
    base = np.zeros((41, 41), bool)
    base[20, 10:31] = True
    cand = np.zeros((41, 41), bool)
    cand[19:22, 8:34] = True
    cand &= ~base                            # the base is already emitted: it is not an addition
    budget = emr.MarginalBudget(ctx, candidates=cand, order=np.ones((41, 41)), initial=base)
    emitted, _ = budget.walk(0.03227)
    # the base's credit is already banked, so T starts at the base's TP, not at zero
    assert budget.T >= ctx.credit_vector(base.astype(np.float32)).sum() - 1e-6


# --------------------------------------------------------------------------------------
# the structural geometry the prior inherits
# --------------------------------------------------------------------------------------
def test_structural_marks_finds_the_settings_it_claims():
    lab = np.zeros((60, 60), bool)
    lab[30, 5:25] = True                     # a line: two terminations
    lab[10:30, 15] = True                    # a crossing line through it: one intersection
    valid = np.ones((60, 60), bool)
    mk = bst.structural_marks(lab, valid)
    assert mk["counts"]["terminations"] >= 2
    assert mk["counts"]["intersections"] >= 1
    assert mk["counts"]["step_over_pairs"] >= 0


def test_structural_marks_is_deterministic():
    lab = np.zeros((60, 60), bool)
    lab[30, 5:25] = True
    lab[10:30, 40] = True
    valid = np.ones((60, 60), bool)
    a = bst.structural_marks(lab, valid)["counts"]
    b = bst.structural_marks(lab, valid)["counts"]
    assert a == b


# --------------------------------------------------------------------------------------
# the annulus builder, end to end on a synthetic grid
# --------------------------------------------------------------------------------------
def _tiny_inputs(tmp_path: Path):
    lab = np.zeros((60, 60), np.float32)
    lab[30, 5:55] = 1.0
    valid = np.ones((60, 60), bool)
    tmpl = np.where(valid, 0.0, np.nan).astype(np.float32)
    base = np.zeros((60, 60), np.float32)
    base[30, 20:30] = 1.0
    prob = np.zeros((60, 60), np.float32)
    prob[29:32, 5:55] = 0.8                  # a confident band on the same line
    return dict(
        labels=_grid(tmp_path / "labels.tif", lab, "float32", -1.0),
        template=_grid(tmp_path / "tmpl.tif", tmpl),
        base=_grid(tmp_path / "base.tif", base),
        detector=_grid(tmp_path / "prob.tif", prob),
    )


def test_annulus_builder_writes_a_conformant_superset(tmp_path):
    p = _tiny_inputs(tmp_path)
    out = tmp_path / "cand.tif"
    rep = tmp_path / "rep.json"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dilational_annulus.py"),
                           "--labels", str(p["labels"]), "--template", str(p["template"]),
                           "--base", str(p["base"]), "--detector", str(p["detector"]),
                           "--halo", "1", "--quantile", "0.5",
                           "--out", str(out), "--report", str(rep),
                           "--downloads", str(tmp_path / "dl")],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    r = json.loads(rep.read_text())
    with rasterio.open(out) as src:
        arr = src.read(1)
        assert src.count == 1 and src.dtypes[0] == "float32"
        assert src.crs.to_epsg() == 32611
    with rasterio.open(p["template"]) as src:
        tv = np.isfinite(src.read(1))
    assert np.isfinite(arr[tv]).all(), "finite inside the footprint"
    assert np.isnan(arr[~tv]).all(), "NaN outside the footprint"
    assert ((arr[tv] >= 0) & (arr[tv] <= 1)).all()
    # it is a superset of the base, and every base pixel survives
    with rasterio.open(p["base"]) as src:
        b = np.nan_to_num(src.read(1), nan=0.0) > 0
    assert (arr[b] > 0).all()
    assert r["chosen"]["chargeable_px"] >= r["chosen"]["kept_px"]
    assert r["chosen"]["required_marginal_hit_rate_at_0_1563"] == pytest.approx(0.03227, abs=1e-4)


# --------------------------------------------------------------------------------------
# the independent second judge
# --------------------------------------------------------------------------------------
def test_the_two_judges_agree_on_the_same_file(tmp_path):
    """Our validator and the vendored independent one must not disagree on a legal file."""
    vci = _load("verify_candidates_independently")
    valid = np.ones((60, 60), bool)
    tmpl = np.where(valid, 0.0, np.nan).astype(np.float32)
    tpath = _grid(tmp_path / "tmpl.tif", tmpl)
    pred = np.zeros((60, 60), np.float32)
    pred[29:32, 10:50] = 1.0
    ppath = _grid(tmp_path / "pred.tif", pred)
    ours = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_submission.py"),
                           "--pred", str(ppath), "--sample", str(tpath),
                           "--train", str(tpath)],
                          capture_output=True, text=True, cwd=ROOT)
    theirs = vci._vendor().validate_submission(str(ppath), str(tpath))
    assert ours.returncode == 0
    assert theirs["ok"] is True
    assert theirs["checks"]["finite_inside_polygon"] is True


def test_the_independent_judge_rejects_a_nan_inside_the_footprint(tmp_path):
    """The exact condition behind the platform's 'Predicted values must be in range [0, 1]'."""
    vci = _load("verify_candidates_independently")
    valid = np.ones((30, 30), bool)
    tmpl = np.where(valid, 0.0, np.nan).astype(np.float32)
    tpath = _grid(tmp_path / "tmpl.tif", tmpl)
    bad = np.where(valid, 0.5, np.nan).astype(np.float32)
    bad[10, 10] = np.nan
    bpath = _grid(tmp_path / "bad.tif", bad)
    ours = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_submission.py"),
                           "--pred", str(bpath), "--sample", str(tpath), "--train", str(tpath)],
                          capture_output=True, text=True, cwd=ROOT)
    theirs = vci._vendor().validate_submission(str(bpath), str(tpath))
    assert ours.returncode != 0, "our gate must fail on a NaN inside the scored footprint"
    assert theirs["ok"] is False
    assert theirs["checks"]["finite_inside_polygon"] is False


def test_uint8_probability_field_round_trips(tmp_path):
    """1/255 must not move anything the marginal rule can resolve."""
    rng = np.random.default_rng(0)
    p = rng.random((50, 50)).astype(np.float32)
    u8 = np.rint(np.clip(p, 0, 1) * 255.0).astype(np.uint8)
    path = _grid(tmp_path / "u8.tif", u8, "uint8", 0)
    with rasterio.open(path) as src:
        assert src.dtypes[0] == "uint8"
        back = src.read(1).astype(np.float32) / 255.0
    assert float(np.max(np.abs(back - p))) <= 1.0 / 255.0 + 1e-6


# ---------------------------------------------------------------------------
# GDR 355: the first real geothermal truth set, and the contradiction it exposes
# ---------------------------------------------------------------------------

def _gdr_analysis():
    ag = importlib.import_module("scripts.analyze_gdr355_inventory")
    return ag


def test_the_gdr355_workbook_is_read_from_its_own_bytes():
    """The catalogue is in the repository, so its frequencies are measured, not quoted."""
    xls = ROOT / "data/external/gdr/faulds_structural_inventory_great_basin.xls"
    if not xls.exists():
        pytest.skip("the runner has not fetched GDR 355 into this checkout")
    rep = _gdr_analysis().analyse(xls, None)
    assert rep["shape"]["rows"] == 426 and rep["shape"]["cols"] == 23
    b = rep["blind"]
    assert b["counts"]["yes"] == 165 and b["counts"]["no"] == 261
    # the irregularity is recorded in the evidence, not silently resolved
    assert "opposite" in b["irregularity"].lower() or "OPPOSITE" in b["irregularity"]
    # and the measured frequencies are all BELOW the published ones
    for name, row in rep["structural_settings"]["published_vs_measured"].items():
        assert row["measured_all_rows"] < row["published"], name


def test_the_blind_column_definition_is_quoted_verbatim_from_the_workbook():
    xls = ROOT / "data/external/gdr/faulds_structural_inventory_great_basin.xls"
    if not xls.exists():
        pytest.skip("the runner has not fetched GDR 355 into this checkout")
    rep = _gdr_analysis().analyse(xls, None)
    definition = rep["blind"]["definition_verbatim"]
    assert "surface manifestations" in definition
    assert "hot springs" in definition


def test_the_117_in_footprint_systems_are_found_only_after_transforming_coordinates():
    """Degrees compared against metres reports 0 inside; the transform is what makes S7 real."""
    xls = ROOT / "data/external/gdr/faulds_structural_inventory_great_basin.xls"
    raster = ROOT / "data/training_features.tif"
    if not xls.exists() or not raster.exists():
        pytest.skip("GDR 355 or the assembled feature stack is absent from this checkout")
    rep = _gdr_analysis().analyse(xls, raster)
    fp = rep["footprint"]
    assert fp["coordinate_columns"].startswith("X_Nad83")
    assert fp["systems_inside"] == 117, fp["systems_inside"]
    assert fp["systems_outside"] == 309
    assert "Beowawe" in fp["inside_names"] and "Steamboat" in fp["inside_names"]
    assert fp["median_distance_outside_km"] > 100.0
