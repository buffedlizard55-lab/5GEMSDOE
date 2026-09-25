#!/usr/bin/env python3
"""Measure the Faulds & Hinz Great Basin structural inventory (GDR 355) from its own bytes.

Why this exists
---------------
`docs/GEOTHERMAL_SCIENCE.md` 3 has quoted, since session 1, the *published* structural-setting
frequencies of the Great Basin geothermal catalogue (step-overs ~32 %, terminations 25 %,
intersections 22 %, bends 2 %, range-front 1 %) and the claim that **39 % of the systems are
blind**. Both came from the GBCGE award page, not from the dataset. The dataset itself is now in
the repository - the runner fetched it with `scripts/fetch_gdr_inventory.py` (the push-path
trigger on `.github/triggers/fetch-gdr`) and committed it to
`data/external/gdr/faulds_structural_inventory_great_basin.xls`, sha256 `843feb7c93f0a232...`.

This script reads those bytes and reports what they actually say, so the science page can cite
measurements instead of a secondary summary.

What it measures
----------------
1. The `Blind` column's own distribution, quoted verbatim from the workbook's *Field Definitions*
   sheet, because the two sources disagree about what the column means (see the irregularity
   recorded below - this script does not resolve it, it makes it auditable).
2. The primary structural-setting frequencies, on both denominators a reader might want
   (all 426 rows, and the 362 rows that carry a code), with the code->name table read from the
   *Field Definitions* sheet rather than hard-coded.
3. Whether any of the 426 systems falls inside the competition raster's scored footprint - the
   question that decides whether this catalogue can be used as a truth set at all.

Outputs `data/evidence/gdr/inventory_analysis.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLS = ROOT / "data/external/gdr/faulds_structural_inventory_great_basin.xls"
SHEET = "StructureInventory"
DEFS = "Field Definitions"
OUT = ROOT / "data/evidence/gdr/inventory_analysis.json"

# The code->setting table is on the workbook's own Field Definitions sheet, in the two columns
# that follow the Primary_S_Code row. It is read, not hard-coded, so a re-issued dataset with an
# extra code does not silently change our weights.
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

# The frequencies the science page has quoted since session 1 (source: the GBCGE award page).
PUBLISHED = {
    "stepover": 0.32,
    "termination of normal fault": 0.25,
    "fault intersection": 0.22,
    "accommodation zone": 0.09,
    "displacement transfer zone": 0.05,
    "pull apart": 0.03,
    "fault bend": 0.02,
    "major normal fault": 0.01,
}


def _codes(series: pd.Series) -> pd.Series:
    """Primary_S_Code is numeric but arrives with '?' suffixes in the Cashman column; be strict."""
    return pd.to_numeric(series, errors="coerce")


def analyse(xls: Path, raster: Path | None) -> dict:
    inv = pd.read_excel(xls, sheet_name=SHEET)
    defs = pd.read_excel(xls, sheet_name=DEFS, header=None)

    # ---- 1. the Blind column, and its own definition -------------------------------
    blind_counts = {str(k): int(v) for k, v in inv["Blind"].value_counts(dropna=False).items()}
    blind_definition = ""
    for _, row in defs.iterrows():
        if str(row.iloc[0]).strip().lower() == "blind" and isinstance(row.iloc[1], str):
            blind_definition = row.iloc[1].strip()
            break

    # ---- 2. structural settings on both denominators -------------------------------
    primary = _codes(inv["Primary_S_Code"])
    coded = primary.notna()
    setting_counts = {SETTING_NAMES[int(c)]: int(n) for c, n in primary[coded].value_counts().items()}
    total = int(len(inv))
    n_coded = int(coded.sum())
    freq_all = {k: v / total for k, v in setting_counts.items()}
    freq_coded = {k: v / n_coded for k, v in setting_counts.items()}
    compared = {}
    for name, published in PUBLISHED.items():
        compared[name] = {
            "published": published,
            "measured_all_rows": freq_all.get(name),
            "measured_coded_rows": freq_coded.get(name),
        }

    # ---- 3. does any system sit inside the scored footprint? -----------------------
    # X_Nad83 / Y_Nad83 are NAD83 geographic degrees; the raster is EPSG:32611 metres. Comparing
    # them directly reports 0 inside for a catalogue that plainly overlaps the region, so the
    # coordinates are transformed first.
    footprint = {}
    if raster is not None and raster.exists():
        import rasterio
        from rasterio.crs import CRS
        from rasterio.warp import transform as warp_transform

        with rasterio.open(raster) as ds:
            west, south, east, north = ds.bounds
            x = pd.to_numeric(inv["X_Nad83"], errors="coerce")
            y = pd.to_numeric(inv["Y_Nad83"], errors="coerce")
            ok = x.notna() & y.notna()
            utm_x, utm_y = warp_transform(
                CRS.from_epsg(4269), CRS.from_epsg(32611),
                x[ok].tolist(), y[ok].tolist())
            utm_x = pd.Series(utm_x, index=x[ok].index)
            utm_y = pd.Series(utm_y, index=y[ok].index)
            inside = utm_x.between(west, east) & utm_y.between(south, north)
            dx = np.maximum.reduce([west - utm_x, utm_x - east, np.zeros(len(utm_x))])
            dy = np.maximum.reduce([south - utm_y, utm_y - north, np.zeros(len(utm_y))])
            d_km = np.sqrt(dx ** 2 + dy ** 2) / 1000.0
            footprint = {
                "raster": str(raster.relative_to(ROOT)),
                "crs": str(ds.crs),
                "bounds": [west, south, east, north],
                "coordinate_columns": "X_Nad83 / Y_Nad83 (NAD83 degrees) transformed to EPSG:32611",
                "systems_with_coordinates": int(ok.sum()),
                "systems_inside": int(inside.sum()),
                "systems_outside": int((~inside).sum()),
                "min_distance_outside_km": round(float(np.min(d_km[~inside])), 1) if (~inside).any() else 0.0,
                "median_distance_outside_km": round(float(np.median(d_km[~inside])), 1) if (~inside).any() else 0.0,
                "inside_names": inv.loc[inside[inside].index, "NAME"].astype(str).tolist()[:20],
            }

    temp = pd.to_numeric(inv["MAX_TEMP"], errors="coerce")
    quaternary = {str(k): int(v) for k, v in inv["Quaternary faulting Bell"].value_counts(dropna=False).items()}

    return {
        "generated_by": "scripts/analyze_gdr355_inventory.py",
        "source": {
            "dataset": "Structural Inventory of Great Basin Geothermal Systems and Definition of "
                       "Favorable Structural Settings",
            "landing_page": "https://gdr.openei.org/submissions/355",
            "doi": "10.15121/1148722",
            "license": "CC BY 4.0",
            "author": "Faulds, James E. (University of Nevada, 2013)",
            "file": str(XLS.relative_to(ROOT)),
            "sha256": _sha(XLS),
        },
        "shape": {"rows": total, "cols": int(inv.shape[1])},
        "blind": {
            "counts": blind_counts,
            "fraction_yes": round(blind_counts.get("yes", 0) / total, 4),
            "definition_verbatim": blind_definition,
            "irregularity": (
                "The GBCGE award page states that 39 % of the 426 catalogued systems are BLIND, "
                "meaning no surface hot springs or fumaroles. The workbook's own Field Definitions "
                "sheet defines Blind = 'Yes if the system is associated with active surface "
                "manifestations including natural hot springs (not open flowing wells), warm "
                "springs (>20 C), or fumaroles; otherwise No' - i.e. the OPPOSITE. 165/426 = 38.7 % "
                "carry 'yes', which matches the page's 39 %, so the page appears to have counted "
                "the column's 'yes' values as blind. This script records the disagreement and does "
                "not resolve it: until the dataset owner clarifies, the 39 % figure must not be "
                "cited as fact in this repository."
            ),
        },
        "structural_settings": {
            "counts": setting_counts,
            "rows_total": total,
            "rows_with_a_primary_code": n_coded,
            "rows_without_a_primary_code": total - n_coded,
            "frequency_all_rows": {k: round(v, 4) for k, v in freq_all.items()},
            "frequency_coded_rows": {k: round(v, 4) for k, v in freq_coded.items()},
            "published_vs_measured": compared,
            "note": (
                "The published 32/25/22 % frequencies do not reproduce on either denominator. They "
                "are most plausibly quoted over the *favourable* settings only, or over the "
                "secondary (Cashman) codes; both are reported in inventory.json. "
                "scripts/build_structural_targets.py weights the settings it can recover from a "
                "trace raster, so its weights should be re-derived from frequency_all_rows rather "
                "than from the page."
            ),
        },
        "quaternary_faulting": quaternary,
        "max_temp_c": {
            "count": int(temp.count()),
            "min": float(temp.min()),
            "max": float(temp.max()),
            "mean": round(float(temp.mean()), 1),
            "mean_blind_yes": round(float(temp[inv["Blind"] == "yes"].mean()), 1),
            "mean_blind_no": round(float(temp[inv["Blind"] == "no"].mean()), 1),
            "note": "The temperature contrast is the wrong way round for the usual 'blind systems are hotter' heuristic, which is a second reason not to lean on the Blind column.",
        },
        "footprint": footprint,
    }


def _sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xls", default=str(XLS))
    ap.add_argument("--raster", default=str(ROOT / "data/training_features.tif"),
                    help="the competition grid, used only to test whether any system falls inside it")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    if not Path(a.xls).exists():
        print(f"missing {a.xls} - run scripts/fetch_gdr_inventory.py on a runner first")
        return 2
    try:
        import pandas  # noqa: F401
    except ImportError:
        print("pandas missing")
        return 2

    rep = analyse(Path(a.xls), Path(a.raster) if a.raster else None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")

    b = rep["blind"]
    s = rep["structural_settings"]
    print(f"rows {rep['shape']['rows']} x {rep['shape']['cols']}")
    print(f"Blind: {b['counts']} -> {b['fraction_yes']:.1%} 'yes'  (definition contradicts the 39% claim)")
    print(f"settings, all rows: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(
        s["frequency_all_rows"].items(), key=lambda kv: -kv[1])))
    print(f"published vs measured (all rows): " + ", ".join(
        f"{k} {v['published']:.0%}/{v['measured_all_rows']:.0%}" for k, v in s["published_vs_measured"].items()))
    fp = rep["footprint"]
    if fp:
        print(f"footprint: {fp['systems_inside']} inside / {fp['systems_outside']} outside the "
              f"raster bounds (coordinates transformed to EPSG:32611); "
              f"median outside distance {fp['median_distance_outside_km']} km")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
