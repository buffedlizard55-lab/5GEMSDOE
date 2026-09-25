#!/usr/bin/env python3
"""Fetch the Great Basin structural inventory of geothermal systems (GDR 355) and summarise it.

WHY
---
Every truth population this project has ever scored against is a *fault* catalogue. The prize is
about geothermal discovery, and there is a public catalogue of the actual geothermal systems in
this region, with the structural setting each one occupies:

    Faulds, J.E. (2013). Structural Inventory of Great Basin Geothermal Systems and Definition of
    Favorable Structural Settings. University of Nevada. Geothermal Data Repository.
    https://gdr.openei.org/submissions/355  ·  DOI 10.15121/1148722  ·  CC BY 4.0

It holds 426 systems (structural setting, primary fault orientation, presence/absence of Quaternary
faulting, reservoir lithology, geothermometry, recent magmatism, blind vs surface expression) for
western/central/NW/NE Nevada, eastern California, southern Oregon and western Utah — the same
region as the GeoDAWN footprint. The published frequencies that weight
`scripts/build_structural_targets.py` come from it (via Faulds & Hinz, WGC 2015, OSTI 1724082).

This script fetches the spreadsheet, records what it actually is (size, hash, magic bytes — an HTML
error page saved as .xls is a classic silent failure), and prints the setting counts so the next
session can use it as an independent target/validation population.

It is written to run on a GitHub-hosted runner: gdr.openei.org is NOT reachable from the
development sandbox (curl exit 35, verified 2026-09-25).

USAGE
    python scripts/fetch_gdr_inventory.py --out-dir data/external/gdr
    python scripts/fetch_gdr_inventory.py --url <xls-url> --out-dir data/external/gdr
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

DEFAULT_URL = ("https://gdr.openei.org/files/355/"
               "Faulds_Structural_Inventory_Great_Basin_DE-EE0002748%20(1).xls")

# Published frequencies (Faulds & Hinz 2015, OSTI 1724082) - what the fetched counts are checked
# against. A mismatch is a finding about the file, not a reason to abandon it.
PUBLISHED = {
    "step_over_relay_ramp": 0.32,
    "termination": 0.25,
    "intersection": 0.22,
    "accommodation_zone": 0.09,
    "displacement_transfer_zone": 0.05,
    "pull_apart": 0.03,
    "bend": 0.02,
    "range_front": 0.01,
}

UA = "Mozilla/5.0 (GEMS Prize research; contact via the repository)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, timeout: int = 300) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        status = getattr(r, "status", 200)
        ctype = r.headers.get("Content-Type", "")
    dest.write_bytes(data)
    head = data[:8]
    # Legacy .xls (OLE2) starts D0 CF 11 E0; a zip-based xlsx starts PK\x03\x04; an HTML error
    # page starts '<'. Anything else is not a spreadsheet and must not be trusted downstream.
    kind = ("ole2-xls" if head[:4] == b"\xd0\xcf\x11\xe0" else
            "zip-xlsx" if head[:4] == b"PK\x03\x04" else
            "html-or-text" if data[:1:1] == b"<" or head[:1] in (b"<", b"{") else
            "unknown")
    return dict(url=url, http_status=status, content_type=ctype, bytes=len(data),
                sha256=sha256(dest), magic=kind, path=str(dest))


def summarise(path: Path) -> dict:
    """Read the sheet if a reader is available; otherwise report that it was not read."""
    out: dict = {"read": False}
    try:
        import pandas as pd  # type: ignore
    except Exception:
        out["note"] = "pandas not installed: the file is committed unread; install pandas+xlrd to summarise"
        return out
    try:
        frames = pd.read_excel(path, sheet_name=None)
    except Exception as exc:                       # xlrd missing for .xls, or a password, or HTML
        out["note"] = f"could not parse the spreadsheet: {type(exc).__name__}: {exc}"
        return out
    sheets = {}
    for name, df in frames.items():
        sheets[name] = dict(rows=int(df.shape[0]), cols=int(df.shape[1]),
                            columns=[str(c) for c in df.columns][:40])
        guess = [c for c in df.columns if "structural" in str(c).lower() or "setting" in str(c).lower()]
        for col in guess[:1]:
            counts = Counter(str(v).strip() for v in df[col].tolist() if str(v).strip())
            sheets[name]["setting_counts"] = dict(counts.most_common(30))
    out.update(read=True, sheets=sheets)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out-dir", default="data/external/gdr")
    ap.add_argument("--report", default="data/evidence/gdr/inventory.json")
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    dest = out_dir / "faulds_structural_inventory_great_basin.xls"
    try:
        fetched = fetch(a.url, dest, a.timeout)
    except Exception as exc:
        print(f"FAILED to fetch {a.url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    ok = fetched["bytes"] > 50_000 and fetched["magic"] in ("ole2-xls", "zip-xlsx")
    summary = summarise(dest) if ok else {"read": False, "note": "not parsed: the download is not a spreadsheet"}
    rep = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"dataset": "Structural Inventory of Great Basin Geothermal Systems and Definition "
                              "of Favorable Structural Settings",
                   "landing_page": "https://gdr.openei.org/submissions/355",
                   "doi": "10.15121/1148722", "license": "CC BY 4.0",
                   "author": "Faulds, James E. (University of Nevada, 2013)"},
        "fetch": fetched,
        "looks_like_a_spreadsheet": bool(ok),
        "published_frequencies_for_comparison": PUBLISHED,
        "summary": summary,
    }
    rep_path = Path(a.report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"bytes": fetched["bytes"], "magic": fetched["magic"],
                      "sha256": fetched["sha256"][:16], "ok": ok,
                      "sheets": list((summary.get("sheets") or {}).keys())}, indent=1))
    print(f"wrote {dest} and {rep_path}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
