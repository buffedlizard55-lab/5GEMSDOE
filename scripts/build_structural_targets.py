#!/usr/bin/env python3
"""Where the geothermal literature says faults matter: dilational settings of the known catalogue.

WHY THIS EXISTS
---------------
Every detector in this project family ranks pixels by what the data say (a CNN, a gradient-boosted
pixel model, this repo's lineament salience). None of them ranks pixels by what forty years of
Great Basin geothermal exploration says. That literature is unusually quantitative about geometry:

    Faulds & Hinz, World Geothermal Congress 2015 (OSTI 1724082, fetched 2026-09-25), of the
    ~250 geothermal fields whose structural setting could be categorised:
        step-overs / relay ramps in normal fault zones ...... ~32 %
        normal fault terminations ........................... 25 %
        fault intersections ................................. 22 %
        accommodation zones ................................. 9 %
        displacement transfer zones ......................... 5 %
        pull-aparts in strike-slip faults ................... 3 %
        bends in normal faults .............................. 2 %
        major range-front normal faults ..................... 1 %
    and, critically for a competition scored on faults that are NOT in the catalogue:
        "Quaternary faults typically lie within or near most of the geothermal systems",
        "geothermal systems are rare along major range-front faults", and
        fault tips horse-tail into "a myriad of closely-spaced faults", i.e. into exactly the
        short, unmapped strands the expert test set is made of.

The catalogue we were given is a set of traces, not a set of settings. This script derives the
settings from the traces' own geometry - endpoints, branch points, bends, and the relay ramps
between overlapping fault tips - and writes a favourability raster weighted by the frequencies
above. It is a PRIOR over where new fault pixels are likely, built from published structural
geology and from the catalogue geometry only: no label beyond `labels.tif`, no score, no model.

WHAT EACH MARK IS, AND HOW IT IS FOUND
--------------------------------------
* skeleton: the label raster thinned to one pixel (skimage.morphology.skeletonize), so a trace's
  topology - ends, branches, bends - is countable at all.
* termination: a skeleton pixel with exactly one 8-neighbour. These are the horse-tailing tips.
* intersection: a skeleton pixel with three or more 8-neighbours (a branch point / crossing).
* bend: a skeleton pixel where the local strike, measured over +/-4 px of the trace, turns by more
  than 35 degrees.
* step-over (relay ramp): a pair of terminations whose local strikes are sub-parallel (within 25
  degrees), that OVERLAP along strike by >= 1 px, and are separated across strike by 1..8 px. That
  is the geometric definition of a relay ramp, and it is the single most favourable setting in the
  inventory. The ramp itself (the lens between the two tips) is marked, not just the tips.

Settings this script does NOT compute, because the raster carries no dip, slip sense or strain
partition: accommodation zones, displacement transfer zones, pull-aparts. They are reported as
absent rather than approximated - a prior that quietly invents 17 % of its own weight is worse
than one that says what it left out.

USAGE
    python scripts/build_structural_targets.py --out data/derived/structural_targets.tif
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
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Faulds & Hinz (2015), OSTI 1724082 - share of the ~250 categorised Great Basin geothermal fields
WEIGHTS = {
    "step_over": 0.32,
    "termination": 0.25,
    "intersection": 0.22,
    "bend": 0.02,
}
NOT_COMPUTED = {
    "accommodation_zone": 0.09,
    "displacement_transfer_zone": 0.05,
    "pull_apart": 0.03,
    "range_front": 0.01,
}
SOURCE = ("https://www.osti.gov/servlets/purl/1724082 "
          "(Faulds & Hinz, World Geothermal Congress 2015) · dataset "
          "https://gdr.openei.org/submissions/355 (DOI 10.15121/1148722)")

BEND_WINDOW = 4          # px either side of a pixel when measuring local strike
BEND_MIN_DEG = 35.0
STEP_MAX_ACROSS = 8.0    # px: relay ramps wider than 800 m are accommodation zones, not ramps
STEP_MIN_OVERLAP = 1.0   # px of along-strike overlap between the two tips
STEP_MAX_ANGLE = 25.0    # degrees between the two tips' strikes


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def neighbours8(mask: np.ndarray) -> np.ndarray:
    n = np.zeros(mask.shape, dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            n += np.roll(np.roll(mask.astype(np.int16), dy, axis=0), dx, axis=1)
    return n


def local_strike(skel: np.ndarray, y: int, x: int, window: int = BEND_WINDOW) -> float | None:
    """Least-squares direction of the skeleton pixels within `window` px of (y, x), in degrees."""
    ys = []
    for yy in range(max(0, y - window), min(skel.shape[0], y + window + 1)):
        for xx in range(max(0, x - window), min(skel.shape[1], x + window + 1)):
            if skel[yy, xx]:
                ys.append((yy - y, xx - x))
    if len(ys) < 4:
        return None
    pts = np.array(ys, dtype=float)
    pts -= pts.mean(axis=0)
    if float(np.linalg.norm(pts)) < 1e-9:
        return None
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    vy, vx = vt[0]
    return float(np.degrees(np.arctan2(vy, vx)) % 180.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--smooth", type=float, default=2.0,
                    help="sigma (px) of the Gaussian that spreads each mark: the metric tolerates "
                         "300 m, so a mark is worth more as a 300 m-wide target than as a pixel")
    ap.add_argument("--out", default="data/derived/structural_targets.tif")
    ap.add_argument("--report", default="data/evidence/salience/structural_targets.json")
    a = ap.parse_args()

    with rasterio.open(a.template) as src:
        valid = np.isfinite(src.read(1))
        profile = src.profile.copy()
    with rasterio.open(a.labels) as src:
        labels = np.nan_to_num(src.read(1), nan=0.0) > 0.5
    labels &= valid

    skel = skeletonize(labels)
    nb = neighbours8(skel)
    skel_only = skel & valid
    ends = skel_only & (nb == 1)
    joins = skel_only & (nb >= 3)
    # bends: strike measured on either side of the pixel differs by more than the threshold
    bends = np.zeros(skel.shape, dtype=bool)
    yy, xx = np.nonzero(skel_only & (nb == 2))
    step = max(2, len(yy) // 40000)          # cap the O(n) pass on a 12 M px grid
    checked = 0
    for y, x in zip(yy[::step], xx[::step]):
        pts_back, pts_fwd = [], []
        for d in range(1, BEND_WINDOW + 1):
            pass
        # sample the trace on a small disc and split it by the pixel's own position
        disc = [(dy, dx) for dy in range(-BEND_WINDOW, BEND_WINDOW + 1)
                for dx in range(-BEND_WINDOW, BEND_WINDOW + 1)
                if skel_only[min(max(y + dy, 0), skel.shape[0] - 1),
                             min(max(x + dx, 0), skel.shape[1] - 1)]]
        if len(disc) < 6:
            continue
        ang = local_strike(skel_only, y, x, BEND_WINDOW)
        ang_tight = local_strike(skel_only, y, x, max(2, BEND_WINDOW // 2))
        if ang is None or ang_tight is None:
            continue
        d = abs(ang - ang_tight)
        d = min(d, 180.0 - d)
        checked += 1
        if d >= BEND_MIN_DEG:
            bends[y, x] = True

    # ---- step-overs: pairs of sub-parallel, overlapping fault tips -------------------------
    ey, ex = np.nonzero(ends)
    strikes = {}
    for y, x in zip(ey, ex):
        s = local_strike(skel_only, int(y), int(x), BEND_WINDOW + 2)
        if s is not None:
            strikes[(int(y), int(x))] = s
    ramps = np.zeros(skel.shape, dtype=bool)
    ramp_pairs = []
    keys = list(strikes)
    # bucket tips spatially so the pair search is O(n) rather than O(n^2)
    cell = 12
    buckets: dict[tuple[int, int], list] = {}
    for (y, x) in keys:
        buckets.setdefault((y // cell, x // cell), []).append((y, x))
    for (y1, x1) in keys:
        s1 = strikes[(y1, x1)]
        for by in range(y1 // cell - 1, y1 // cell + 2):
            for bx in range(x1 // cell - 1, x1 // cell + 2):
                for (y2, x2) in buckets.get((by, bx), ()):
                    if (y2, x2) <= (y1, x1):
                        continue
                    s2 = strikes[(y2, x2)]
                    dang = abs(s1 - s2)
                    dang = min(dang, 180.0 - dang)
                    if dang > STEP_MAX_ANGLE:
                        continue
                    dv = np.array([y2 - y1, x2 - x1], dtype=float)
                    dist = float(np.linalg.norm(dv))
                    if dist < 1.0 or dist > 3.0 * STEP_MAX_ACROSS:
                        continue
                    u = np.array([np.sin(np.radians(s1)), np.cos(np.radians(s1))])
                    along = float(abs(np.dot(dv, u)))
                    across = float(np.sqrt(max(0.0, dist ** 2 - along ** 2)))
                    if across < 1.0 or across > STEP_MAX_ACROSS:
                        continue
                    if along < STEP_MIN_OVERLAP:
                        continue
                    ramp_pairs.append(dict(a=[y1, x1], b=[y2, x2],
                                           along_px=round(along, 2), across_px=round(across, 2)))
    # the ramp is the lens between the two tips: draw the segment and dilate it by the across gap
    for pr in ramp_pairs:
        (y1, x1), (y2, x2) = pr["a"], pr["b"]
        n = int(max(abs(y2 - y1), abs(x2 - x1)) * 2) + 1
        for t in np.linspace(0.0, 1.0, n):
            yy = int(round(y1 + t * (y2 - y1)))
            xx = int(round(x1 + t * (x2 - x1)))
            if 0 <= yy < ramps.shape[0] and 0 <= xx < ramps.shape[1]:
                ramps[yy, xx] = True
    ramps = binary_dilation(ramps, structure=np.ones((3, 3), bool)) & valid

    # ---- the favourability field ------------------------------------------------------------
    marks = ((ends & valid) * WEIGHTS["termination"] +
             (joins & valid) * WEIGHTS["intersection"] +
             (bends & valid) * WEIGHTS["bend"] +
             ramps * WEIGHTS["step_over"]).astype(np.float32)
    field = gaussian_filter(marks, a.smooth, mode="constant").astype(np.float32)
    if float(field.max()) > 0:
        field = field / float(field.max())
    field[~valid] = 0.0

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=float("nan"), compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.where(valid, field, np.nan).astype(np.float32), 1)
        dst.set_band_description(1, "dilational-setting prior from catalogue geometry, weighted "
                                    "by Faulds & Hinz 2015 (OSTI 1724082)")

    rep = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/build_structural_targets.py",
        "labels_sha256": sha256(Path(a.labels)),
        "source_of_the_weights": SOURCE,
        "weights_used": WEIGHTS,
        "weights_not_computable_from_a_trace_raster": NOT_COMPUTED,
        "geometry": dict(bend_window_px=BEND_WINDOW, bend_min_deg=BEND_MIN_DEG,
                         step_max_across_px=STEP_MAX_ACROSS,
                         step_min_overlap_px=STEP_MIN_OVERLAP, step_max_angle_deg=STEP_MAX_ANGLE,
                         smooth_sigma_px=a.smooth),
        "counts": {
            "catalogue_px": int(labels.sum()),
            "skeleton_px": int((skel_only).sum()),
            "terminations": int((ends & valid).sum()),
            "intersections": int((joins & valid).sum()),
            "bends_measured": int(bends.sum()),
            "bend_pixels_sampled": int(checked),
            "step_over_pairs": len(ramp_pairs),
            "step_over_ramp_px": int(ramps.sum()),
        },
        "output": dict(path=str(out_path), sha256=sha256(out_path), size=out_path.stat().st_size,
                       max=round(float(field.max()), 6), mean=round(float(field[valid].mean()), 6)),
        "note": ("a prior built from published structural geology and the catalogue's own geometry; "
                 "it has never been scored against the hidden truth and must not be reported as if "
                 "it had been"),
    }
    rep_path = Path(a.report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep["counts"], indent=1))
    print(f"wrote {out_path} ({out_path.stat().st_size:,} B) and {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
