#!/usr/bin/env python3
"""Parse a saved public-leaderboard capture into auditable evidence JSON.

The sandbox shell cannot reach www.drivendata.org (egress allow-list), so a fresh board reading
is captured with the Arena fetch_page tool and saved verbatim as markdown under
`data/evidence/leaderboard/leaderboard_<date>.md` with a provenance header (url, fetch date).
This script parses that capture into `leaderboard_snapshot_<date>.json` and runs self-checks
that pin the parse to the truth this project family already knows:

  * ranks are 1..N with no gaps
  * scores are non-increasing
  * the five accounts owned by this project family are present with EXACTLY the scores the
    committed bytes earned (sha256-pinned in scripts/leaderboard_anchor.py) - a capture in
    which one of them moved without a new upload is a stale or corrupt capture

If a self-check fails the script exits 1 and writes nothing.  The raw capture stays the
primary record; the JSON is derived from it, never the other way round.

Usage:
    python scripts/fetch_leaderboard_snapshot.py data/evidence/leaderboard/leaderboard_2026-09-26.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/evidence/leaderboard"

# This project family's accounts and the exact public score each uploaded byte earned.
# (account, pinned score, file identity; the files themselves are sha256-pinned in
# scripts/leaderboard_anchor.py and committed under data/evidence/leaderboard_anchor/.)
OURS = {
    "extradr19": {"score": 0.1563, "sha8": "7f00890a", "site": "GEMSDOE"},
    "smashi34": {"score": 0.1560, "sha8": "f68e590f", "site": "GEMSDOE2"},
    "smrtdoog5": {"score": 0.1193, "sha8": "f347b70daa", "site": "GEMSDOE3"},
    "SDCF9": {"score": 0.1152, "sha8": "4e03fc9705", "site": "GEMSDOE3"},
    "wbg1": {"score": 0.0830, "sha8": "37f9d5b855", "site": "GEMSDOE3"},
}

ROW_RE = re.compile(r"^\|\s*#(\d+)\s*\|(.+)\|\s*$", re.M)


def parse_row(line: str) -> dict | None:
    m = ROW_RE.match(line.strip())
    if not m:
        return None
    rank = int(m.group(1))
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    # cells: [rank, members, participant, score, shared]
    if len(cells) < 5:
        return None
    members = re.findall(r"\[([^\]]+)\]\(https://www\.drivendata\.org/users/[^/]+/", cells[1])
    part = cells[2]
    sm = re.search(r"⸱\s*<br>\s*(\d+)\s+submissions?", part)
    lm = re.search(r"<br>\s*([0-9]+[a-z]*(?:\s+[0-9]+[a-z]*)*)\s*ago<br>", part)
    nm = re.match(r"^(?:\[([^\]]+)\]\([^)]*\)|([^<\[]+))", part)
    name = (nm.group(1) or nm.group(2) or "").strip().replace("\\_", "_")
    try:
        score = float(cells[3].strip())
    except ValueError:
        return None
    return {
        "rank": rank,
        "members": members,
        "display_name": name,
        "last_submission": lm.group(1).strip() + " ago" if lm else None,
        "submissions": int(sm.group(1)) if sm else None,
        "score": score,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capture", help="path to the raw markdown capture")
    a = ap.parse_args(argv)
    cap = Path(a.capture).resolve()
    if not cap.exists():
        print(f"FAIL: capture not found: {cap}")
        return 1
    text = cap.read_text()

    rows = []
    for line in text.splitlines():
        r = parse_row(line)
        if r is not None:
            rows.append(r)
    rows.sort(key=lambda r: r["rank"])

    errors = []
    ranks = [r["rank"] for r in rows]
    if ranks != list(range(1, len(rows) + 1)):
        errors.append(f"ranks are not contiguous 1..{len(rows)}: {ranks}")
    scores = [r["score"] for r in rows]
    if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
        errors.append("scores are not non-increasing with rank")

    ours_rows, improvements = [], []
    for acct, pin in OURS.items():
        hit = [r for r in rows if acct in r["members"]]
        if len(hit) != 1:
            errors.append(f"account {acct!r}: expected exactly 1 row, found {len(hit)}")
            continue
        r = hit[0]
        # The public board shows the BEST score per account, so it can only rise.  A capture
        # below the score the committed bytes earned is stale/corrupt; a capture above it
        # means a new file was uploaded and scored - report it, do not reject it.
        if r["score"] < pin["score"] - 1e-9:
            errors.append(
                f"account {acct!r}: capture shows {r['score']} but the committed bytes of "
                f"{pin['sha8']} earned exactly {pin['score']} - stale or corrupt capture")
        elif r["score"] > pin["score"] + 1e-9:
            improvements.append(
                f"account {acct!r}: board now shows {r['score']}, above the {pin['score']} the "
                f"committed bytes of {pin['sha8']} earned - a new file was uploaded and scored")
        ours_rows.append({
            "account": acct, "rank": r["rank"], "score": r["score"],
            "pinned_score": pin["score"], "submissions": r["submissions"],
            "last_submission": r["last_submission"], "file_sha8": pin["sha8"],
            "site": pin["site"],
        })

    if errors:
        print("CAPTURE SELF-CHECK FAILED:")
        for e in errors:
            print("  -", e)
        return 1

    date = cap.stem.split("_")[-1]
    out = OUT_DIR / f"leaderboard_snapshot_{date}.json"
    leader = rows[0]
    top5_cutoff = rows[4]["score"] if len(rows) >= 5 else None
    report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/fetch_leaderboard_snapshot.py",
        "url": "https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/",
        "capture": str(cap.relative_to(ROOT)),
        "observed_date": date,
        "participants_on_page": len(rows),
        "leader": {
            "account": leader["display_name"] or (leader["members"][0] if leader["members"] else None),
            "score": leader["score"],
            "submissions": leader["submissions"],
            "last_submission": leader["last_submission"],
        },
        "top5_cutoff_public_dti": top5_cutoff,
        "our_accounts": sorted(ours_rows, key=lambda r: r["rank"]),
        "rows": rows,
        "self_checks": [
            "ranks contiguous 1..N",
            "scores non-increasing with rank",
            "each family account's board score is >= the score its committed bytes earned "
            "(sha256 pins in scripts/leaderboard_anchor.py); a strict improvement is reported "
            "below, a shortfall fails the capture",
        ],
        "improvements_over_committed_bytes": improvements,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    for imp in improvements:
        print("  NEW:", imp)
    print(f"OK: {len(rows)} rows, leader {report['leader']['account']} {report['leader']['score']}, "
          f"top-5 cutoff {top5_cutoff}")
    for r in ours_rows:
        gap5 = round(top5_cutoff - r["score"], 4) if top5_cutoff else None
        print(f"  #{r['rank']:<3d} {r['account']:<11s} {r['score']:.4f} ({r['submissions']} submission(s), "
              f"last {r['last_submission']})  gap-to-top5={gap5}")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
