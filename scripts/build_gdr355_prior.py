#!/usr/bin/env python3
"""Build a geothermal-system prior raster from the GDR 355 Great Basin structural inventory.

WHY THIS EXISTS
---------------
`labels.tif` is a *fault* catalogue and the hidden truth is *new faults*.  Every local instrument
this project family has had so far is a stand-in for fault pixels (the supplied catalogue, the
SGMC gap).  The GDR 355 inventory (Faulds, 2013, CC BY 4.0,
https://gdr.openei.org/submissions/355, DOI 10.15121/1148722) is something different: 426
geothermal systems with measured coordinates, maximum temperatures and the *structural setting*
each sits in.  117 of the 426 fall inside the scored footprint
(`scripts/analyze_gdr355_inventory.py`, `data/evidence/gdr/inventory_analysis.json`).  A geothermal
system marks a fluid-conducting structural corridor; the new faults the experts mapped are most
likely to lie in the same corridors - i.e. near known systems - which makes the systems a
geographically *different* truth proxy than any fault population we have held.

WHAT THIS SCRIPT MEASURES
-------------------------
1. Projects every system's NAD83 coordinates (EPSG:4269, the exact transform the analysis script
   already uses) into the competition grid (EPSG:32611) and keeps the systems inside the scored
   footprint.
2. For each radius R in {3, 10, 20, 30} px (300 m .. 3 km) writes a binary disk raster
   `data/derived/gdr355_prior_R{R}.tif` (uint8, LZW, 1 inside the union of the disks).
3. Records, per system: name, max temperature (MAX_MAXT, the workbook's max of MAX_TEMP/THERMOM),
   structural setting (Primary_S_Code decoded by the workbook's own Field Definitions table),
   Quaternary-faulting column, and the *measured* distance to the nearest supplied-catalogue pixel
   in `data/labels.tif` (the statistic that decides whether the prior reaches off-catalogue ground,
   where the hidden truth lives).
4. Records, per radius: disk pixel count, footprint fraction, the fraction of disk pixels that sit
   on masked known-fault pixels (their FP charge is zero by the platform's own rule, forum 11516),
   and the overlap between the prior and each scored file's emission support.

Nothing here is a prediction: the disks are a geometric prior to be *tested* as an instrument by
`scripts/rank_instruments.py --instrument-tif` (does it order the five already-scored files the way
the public leaderboard does?) before it may touch any emission decision.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
XLS = ROOT / "data/external/gdr/faulds_structural_inventory_great_basin.xls"
OUT_DIR = ROOT / "data/derived"
REPORT = ROOT / "data/evidence/gdr/prior_report.json"
RADII_PX = (3, 10, 20, 30)

SETTING_NAMES = {
    1: "termination of normal fault",
    2: "stepover",
    3: "accommodation zone",
    4: "major normal fault",
    5: "fault bend",
    6: "antithetic normal fault",
    7: "fault intersection",
    8: "displacement transfer zone",
    9: "pull apart",
    10: "undetermined",
    11: "volcanic",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_systems() -> pd.DataFrame:
    inv = pd.read_excel(XLS, sheet_name="StructureInventory")
    x = pd.to_numeric(inv["X_Nad83"], errors="coerce")
    y = pd.to_numeric(inv["Y_Nad83"], errors="coerce")
    ok = x.notna() & y.notna()
    out = inv.loc[ok].copy()
    ux, uy = warp_transform(CRS.from_epsg(4269), CRS.from_epsg(32611),
                            x[ok].tolist(), y[ok].tolist())
    out["utm_x"] = np.asarray(ux)
    out["utm_y"] = np.asarray(uy)
    out["temp_c"] = pd.to_numeric(out["MAX_MAXT"], errors="coerce")
    primary = pd.to_numeric(out["Primary_S_Code"], errors="coerce")
    out["setting"] = primary.map(lambda c: SETTING_NAMES.get(int(c), "uncoded")
                                 if pd.notna(c) else "uncoded")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", default=str(ROOT / "data/sample_submission.tif"))
    ap.add_argument("--labels", default=str(ROOT / "data/labels.tif"))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--report", default=str(REPORT))
    a = ap.parse_args(argv)

    with rasterio.open(a.sample) as ds:
        template = ds.read(1)
        transform = ds.transform
        crs = ds.crs
        height, width = template.shape
        west, south, east, north = ds.bounds
    valid = np.isfinite(template)
    n_valid = int(valid.sum())
    labels = np.nan_to_num(rasterio.open(a.labels).read(1)) > 0.5

    sys_ = load_systems()
    inside = (sys_["utm_x"].between(west, east) & sys_["utm_y"].between(south, north))
    sys_in = sys_[inside.values].copy()

    # pixel coordinates (grid is 100 m; transform.xy maps row/col -> utm)
    sys_in["col"] = (sys_in["utm_x"] - west) / transform.a
    sys_in["row"] = (north - sys_in["utm_y"]) / abs(transform.e)
    sys_in["col_i"] = sys_in["col"].round().astype(int)
    sys_in["row_i"] = sys_in["row"].round().astype(int)
    in_grid = sys_in["col_i"].between(0, width - 1) & sys_in["row_i"].between(0, height - 1)
    sys_in = sys_in[in_grid.values]

    # distance to the nearest supplied-catalogue pixel (the platform masks those)
    dist_known = distance_transform_edt(~(labels & valid))  # px, 0 on a label pixel
    def known_km(r: int, c: int) -> float:
        return round(float(dist_known[r, c]) * 0.1, 2)

    systems = []
    for _, r in sys_in.iterrows():
        systems.append({
            "name": str(r["NAME"]).strip(),
            "temp_c": (None if pd.isna(r["temp_c"]) else round(float(r["temp_c"]), 1)),
            "setting": r["setting"],
            "quaternary_faulting": (None if pd.isna(r["Quaternary faulting Bell"])
                                    else str(r["Quaternary faulting Bell"]).strip()),
            "workbook_distance_to_fault": (None if pd.isna(r["Distance to fault (km) Bell"])
                                           else str(r["Distance to fault (km) Bell"]).strip()),
            "row": int(r["row_i"]), "col": int(r["col_i"]),
            "distance_to_supplied_catalogue_km": known_km(int(r["row_i"]), int(r["col_i"])),
        })
    systems.sort(key=lambda s: (-(s["temp_c"] or 0.0), s["name"]))

    # per-radius disk rasters + statistics
    yy, xx = np.mgrid[0:height, 0:width]
    radii_rows = []
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for R in RADII_PX:
        disk = np.zeros((height, width), dtype=np.uint8)
        for s in systems:
            r0, c0 = s["row"], s["col"]
            r_a = max(r0 - R, 0)
            r_b = min(r0 + R, height - 1)
            c_a = max(c0 - R, 0)
            c_b = min(c0 + R, width - 1)
            # xx indexes columns, yy indexes rows (np.mgrid[0:h, 0:w] convention)
            sub_r, sub_c = yy[r_a:r_b + 1, c_a:c_b + 1].astype(np.float64), \
                xx[r_a:r_b + 1, c_a:c_b + 1].astype(np.float64)
            d = np.hypot(sub_r - r0, sub_c - c0)
            disk[r_a:r_b + 1, c_a:c_b + 1] = np.where(d <= R + 1e-9, 1,
                                                      disk[r_a:r_b + 1, c_a:c_b + 1])
        if not disk.any():
            raise RuntimeError(
                f"R={R} disk raster is empty: {len(systems)} systems produced no pixels - "
                "the row/col arithmetic is wrong, refusing to write an empty prior")
        path = out_dir / f"gdr355_prior_R{R}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1,
                           dtype="uint8", crs=crs, transform=transform,
                           compress="lzw") as dst:
            dst.write(disk.astype(np.uint8), 1)
        on_known = int((disk.astype(bool) & labels).sum())
        radii_rows.append({
            "radius_px": R,
            "radius_m": R * 100,
            "raster": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "disk_px": int(disk.sum()),
            "fraction_of_valid": round(float(disk[valid].sum() / n_valid), 6),
            "on_known_fault_px": on_known,
            "on_known_fraction": (round(on_known / int(disk.sum()), 4) if disk.sum() else None),
        })

    report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/build_gdr355_prior.py",
        "purpose": "geothermal-system prior (GDR 355) as a geometric truth proxy to be tested as a "
                   "RANKING instrument by scripts/rank_instruments.py before any use",
        "source": {
            "dataset": "Structural Inventory of Great Basin Geothermal Systems (Faulds 2013)",
            "landing_page": "https://gdr.openei.org/submissions/355",
            "doi": "10.15121/1148722",
            "license": "CC BY 4.0",
            "file": str(XLS.relative_to(ROOT)),
            "sha256": sha256(XLS),
            "coordinate_transform": "EPSG:4269 (NAD83 geographic) -> EPSG:32611, "
                                    "identical to scripts/analyze_gdr355_inventory.py",
        },
        "systems_inside_footprint": len(systems),
        "systems": systems,
        "temperature_c": {
            "mean": round(float(np.nanmean([s["temp_c"] for s in systems])), 1),
            "max": round(float(np.nanmax([s["temp_c"] for s in systems])), 1),
            "ge_150": sum(1 for s in systems if (s["temp_c"] or 0) >= 150),
        },
        "setting_counts_in_footprint": {k: int(v) for k, v in
                                        pd.Series([s["setting"] for s in systems]).value_counts().items()},
        "quaternary_faulting_counts": {k: int(v) for k, v in
                                       pd.Series([s["quaternary_faulting"] for s in systems]).value_counts(dropna=False).items()},
        "distance_to_supplied_catalogue_km": {
            "mean": round(float(np.mean([s["distance_to_supplied_catalogue_km"] for s in systems])), 2),
            "median": round(float(np.median([s["distance_to_supplied_catalogue_km"] for s in systems])), 2),
            "within_300m": sum(1 for s in systems if s["distance_to_supplied_catalogue_km"] <= 0.3),
            "within_1km": sum(1 for s in systems if s["distance_to_supplied_catalogue_km"] <= 1.0),
            "farther_than_1km": sum(1 for s in systems if s["distance_to_supplied_catalogue_km"] > 1.0),
        },
        "radii": radii_rows,
    }
    out_path = Path(a.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"systems inside footprint: {len(systems)}")
    print(f"distance to supplied catalogue: mean {report['distance_to_supplied_catalogue_km']['mean']} km, "
          f"median {report['distance_to_supplied_catalogue_km']['median']} km, "
          f">{1.0} km: {report['distance_to_supplied_catalogue_km']['farther_than_1km']}")
    for r in radii_rows:
        print(f"  R={r['radius_px']:>2d} px ({r['radius_m']:>4d} m): {r['disk_px']:>8,d} px "
              f"({r['fraction_of_valid']:.2%} of valid) on_known={r['on_known_fraction']:.1%} "
              f"-> {r['raster']}")
    print(f"wrote {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
