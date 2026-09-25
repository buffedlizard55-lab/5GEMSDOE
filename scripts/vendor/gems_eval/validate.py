"""Submission checks against the published format: single-band float32 GeoTIFF, same CRS /
transform / shape as the features, values in [0, 1], NaN outside the study polygon."""
from __future__ import annotations
import numpy as np


def validate_submission(sub_path: str, template_path: str) -> dict:
    """Returns a dict of checks (True = pass) plus summary statistics. Needs rasterio."""
    import rasterio
    with rasterio.open(template_path) as t:
        t_shape, t_crs, t_tr = (t.height, t.width), t.crs, t.transform
        tmpl = t.read(1)
        inside = tmpl >= 0 if t.nodata is None or not np.isnan(t.nodata) else ~np.isnan(tmpl)
    with rasterio.open(sub_path) as s:
        arr = s.read(1); dtype = s.dtypes[0]; count = s.count
        checks = {
            "single_band": count == 1,
            "float32": dtype == "float32",
            "same_shape": (s.height, s.width) == t_shape,
            "same_crs": s.crs == t_crs,
            "same_transform": s.transform.almost_equals(t_tr),
        }
    finite = np.isfinite(arr)
    checks["values_in_unit_interval"] = bool((arr[finite] >= 0).all() and (arr[finite] <= 1).all())
    checks["nan_outside_polygon"] = bool((~finite[~inside]).all()) if (~inside).any() else True
    checks["finite_inside_polygon"] = bool(finite[inside].all())
    # a collapsed model (fault everywhere, or nowhere) passes every format check; catch it here
    frac = float(np.mean(arr[inside] >= 0.5)) if inside.any() else 0.0
    checks["plausible_coverage"] = bool(0.001 <= frac <= 0.30)
    stats = {
        "mean_inside": float(np.nanmean(arr[inside])) if inside.any() else float("nan"),
        "frac_ge_0.5_inside": float(np.mean(arr[inside] >= 0.5)) if inside.any() else float("nan"),
        "nan_fraction": float((~finite).mean()),
    }
    return {"checks": checks, "ok": all(checks.values()), **stats}
