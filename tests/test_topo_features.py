"""The scarp channel must see what a 100 m slope cannot: a sub-pixel scarp.

Synthetic terrain, two 100 m cells with the SAME mean 100 m gradient:
  * cell A: a smooth planar hillside (every 10 m sub-cell has the same gentle slope);
  * cell B: flat ground with one 4 m vertical step (a scarp) across it.
A 100 m slope treats them alike.  slope_p90 / slope_max / steep_frac / curvature must separate
them, and the whole file must land on the template grid with NaN exactly where the template is NaN.
"""
from __future__ import annotations

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
from scripts.build_topo_features import BANDS, features_from_fine_dem  # noqa: E402


def test_scarp_cell_separates_from_planar_cell_at_equal_coarse_slope():
    f = 10
    # cell A: plane rising 4 m across 100 m (2.3 deg everywhere)
    x = np.arange(f) * 10.0
    plane = np.tile((x / 90.0) * 4.0, (f, 1))
    # cell B: flat at 4 m (continuous with A's right edge, no seam) then a 4 m step -> same 100 m rise
    step = np.tile(np.where(np.arange(f) >= f // 2, 8.0, 4.0), (f, 1))
    dem = np.concatenate([plane, step], axis=1)             # (10, 20) fine px -> (1, 2) coarse
    out = features_from_fine_dem(dem, 10.0, f)
    assert out.shape == (len(BANDS), 1, 2)
    b = dict(zip(BANDS, out[:, 0, :]))
    # coarse relief identical
    assert abs(b["relief_local"][0] - b["relief_local"][1]) < 1e-3
    # but the scarp cell has a far steeper tail
    assert b["slope_max"][1] > 3 * b["slope_max"][0]
    assert b["slope_p90"][1] > b["slope_p90"][0]
    assert b["curv_prof_absmax"][1] > 3 * b["curv_prof_absmax"][0]
    assert b["steep_frac"][0] == 0.0 and b["steep_frac"][1] > 0.0
    # a one-sided break is directionally coherent
    assert b["aspect_coherence"][1] > 0.9


def test_all_nan_fine_block_gives_all_nan_coarse_block():
    dem = np.full((20, 20), np.nan, dtype=np.float32)
    out = features_from_fine_dem(dem, 10.0, 10)
    assert out.shape == (len(BANDS), 2, 2) and np.isnan(out).all()


def test_half_missing_cell_is_masked():
    f = 10
    dem = np.random.default_rng(0).normal(1000, 1, (f, f)).astype(np.float32)
    dem[:, : f // 2 + 1] = np.nan   # 60 % missing -> masked
    out = features_from_fine_dem(dem, 10.0, f)
    assert np.isnan(out).all()


def test_end_to_end_on_template_grid(tmp_path):
    """Run the CLI on a synthetic DEM and a small template; the output must match the template grid."""
    H, W, coarse, f = 8, 12, 100.0, 10
    tr = Affine(coarse, 0, 500000.0, 0, -coarse, 4400000.0)
    crs = "EPSG:32611"
    template = np.ones((H, W), np.float32)
    template[:, :3] = np.nan                              # left strip outside the footprint
    with rasterio.open(tmp_path / "template.tif", "w", driver="GTiff", dtype="float32", count=1, height=H, width=W,
                       crs=crs, transform=tr, nodata=np.nan) as ds:
        ds.write(template, 1)
    # fine DEM covering the same extent at 10 m, in the same CRS, with a N–S scarp at column 70
    fine_tr = Affine(coarse / f, 0, 500000.0, 0, -coarse / f, 4400000.0)
    yy, xx = np.mgrid[0:H * f, 0:W * f]
    dem = (1500 + 0.01 * xx + np.where(xx >= 70, 6.0, 0.0)).astype(np.float32)
    with rasterio.open(tmp_path / "dem.tif", "w", driver="GTiff", dtype="float32", count=1, height=H * f, width=W * f,
                       crs=crs, transform=fine_tr, nodata=-999999.0) as ds:
        ds.write(dem, 1)
    out = tmp_path / "topo.tif"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/build_topo_features.py"), "--dem", str(tmp_path / "dem.tif"),
                        "--template", str(tmp_path / "template.tif"), "--out", str(out), "--chunk", "5"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with rasterio.open(out) as ds:
        assert (ds.height, ds.width, ds.count) == (H, W, len(BANDS))
        assert tuple(ds.transform)[:6] == tuple(tr)[:6]
        assert ds.descriptions[0] == "slope_p90"
        a = ds.read()
    assert np.isnan(a[:, :, :3]).all(), "outside the template footprint must be NaN"
    assert np.isfinite(a[0, :, 3:]).all()
    # the scarp at fine column 70 sits in coarse column 7; slope_max must peak there
    col_peak = int(np.nanargmax(np.nanmean(a[1], axis=0)))
    assert col_peak == 7, np.nanmean(a[1], axis=0)
    rep = json.loads(out.with_suffix(".report.json").read_text())
    assert rep["chunks_done"] >= 1 and rep["coverage_of_footprint"] == pytest.approx(1.0)


def test_aux_feature_hook_appends_bands_and_rejects_grid_drift(tmp_path):
    from src.dataset import _maybe_add_aux_features
    H, W = 6, 7
    tr = Affine(100.0, 0, 0.0, 0, -100.0, 600.0)
    meta = {"transform": tr, "crs": rasterio.crs.CRS.from_epsg(32611)}
    X = np.zeros((H, W, 2), np.float32)
    good = tmp_path / "aux.tif"
    with rasterio.open(good, "w", driver="GTiff", dtype="float32", count=3, height=H, width=W, crs="EPSG:32611",
                       transform=tr, nodata=np.nan) as ds:
        ds.write(np.arange(3 * H * W, dtype=np.float32).reshape(3, H, W))
        for i, n in enumerate(("a", "b", "c"), 1):
            ds.set_band_description(i, n)
    X2, tags = _maybe_add_aux_features(X, meta, [good])
    assert X2.shape == (H, W, 5) and [t["band_name"] for t in tags] == ["a", "b", "c"]
    bad = tmp_path / "bad.tif"
    with rasterio.open(bad, "w", driver="GTiff", dtype="float32", count=1, height=H, width=W, crs="EPSG:32611",
                       transform=tr * Affine.translation(1, 0), nodata=np.nan) as ds:
        ds.write(np.zeros((1, H, W), np.float32))
    with pytest.raises(ValueError):
        _maybe_add_aux_features(X, meta, [bad])
    with pytest.raises(FileNotFoundError):
        _maybe_add_aux_features(X, meta, [tmp_path / "nope.tif"])
