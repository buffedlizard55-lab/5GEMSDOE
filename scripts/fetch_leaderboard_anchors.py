#!/usr/bin/env python3
"""Restore the sha256-pinned scored-submission bytes for scripts/leaderboard_anchor.py.

The anchor script measures five files that were uploaded to the DrivenData public
leaderboard and two arms that were never uploaded.  Their bytes live in the sibling
repositories' published folders (GEMSDOE / GEMSDOE2 / GEMSDOE3 under this org); this
script brings every one of them into data/evidence/leaderboard_anchor/ and verifies
each byte against the sha256 pinned in scripts/leaderboard_anchor.py.

Provenance of the bytes (each row in SCORED carries its own "origin"):
  * GEMSDOE files        -> this repository carries a full copy of GEMSDOE, so the
                            origin path usually exists here already (local copy, no
                            network); the sibling repo is the fallback.
  * GEMSDOE2 / GEMSDOE3  -> sibling repos under buffedlizard55-lab.

Network path (sandbox egress: api.github.com is reachable; raw.githubusercontent.com
is not, so the fetch goes through the contents endpoint with
"Accept: application/vnd.github.raw+json", which streams the file bytes from
api.github.com itself).

Every action is logged to data/evidence/leaderboard_anchor/fetch_log.json:
  status        verified_local | copied_from_origin | fetched_api | FAILED_*
  sha256        measured after the copy/fetch, compared with the pin
  bytes         measured

Exit codes: 0 = every file present and verified; 1 = at least one file missing or
mismatched.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_DIR = ROOT / "data/evidence/leaderboard_anchor"
LOG = ANCHOR_DIR / "fetch_log.json"
ORG = "buffedlizard55-lab"
API = "https://api.github.com/repos/{repo}/contents/{path}"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_specs():
    sys.path.insert(0, str(ROOT / "scripts"))
    import leaderboard_anchor as la
    return la.SCORED


def parse_origin(origin: str) -> tuple[str, str]:
    """'GEMSDOE2 docs/foo.tif' -> ('buffedlizard55-lab/GEMSDOE2', 'docs/foo.tif')."""
    repo_name, _, path = origin.partition(" ")
    return f"{ORG}/{repo_name}", path


def fetch_api(repo: str, path: str, dest: Path) -> tuple[int, int]:
    """Fetch raw bytes through api.github.com (raw.githubusercontent.com is blocked here)."""
    url = API.format(repo=repo, path=urllib.parse.quote(path))
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.raw+json",
                                               "User-Agent": "5gemsdoe-anchor-restore"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    dest.write_bytes(data)
    return r.status, len(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=str(ANCHOR_DIR))
    ap.add_argument("--force", action="store_true",
                    help="re-fetch even when the local bytes already verify")
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = load_specs()
    log_rows, failures = [], []

    for spec in specs:
        dest = out_dir / spec["file"]
        row = {"file": spec["file"], "pin_sha256": spec["sha256"], "origin": spec["origin"]}
        settled = False

        if dest.exists() and not a.force:
            got = sha256(dest)
            row["bytes"] = dest.stat().st_size
            if got == spec["sha256"]:
                row.update({"sha256": got, "status": "verified_local"})
                settled = True
            else:
                row["sha256_local_mismatch"] = got
                dest.unlink()  # do not trust mismatched bytes; restore below

        if not settled:
            repo, path = parse_origin(spec["origin"])
            # 1) local copy under this repository (GEMSDOE's tree is copied here)
            local = ROOT / path if repo.endswith("/GEMSDOE") else None
            if local and local.exists():
                got = sha256(local)
                if got == spec["sha256"]:
                    dest.write_bytes(local.read_bytes())
                    row.update({"sha256": got, "bytes": local.stat().st_size,
                                "status": "copied_from_origin"})
                    log_rows.append(row)
                    continue
                row["local_origin_sha256"] = got
            # 2) sibling repo via api.github.com
            try:
                code, n = fetch_api(repo, path, dest)
                got = sha256(dest)
                row["http_status"] = code
                row["bytes"] = n
                if got == spec["sha256"]:
                    row.update({"sha256": got, "status": "fetched_api"})
                else:
                    row["sha256"] = got
                    dest.unlink()
                    failures.append(spec["file"])
                    row["status"] = "FAILED_sha256_fetched"
                log_rows.append(row)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                failures.append(spec["file"])
                row["status"] = f"FAILED_fetch ({e.__class__.__name__})"
                log_rows.append(row)
        if settled:
            log_rows.append(row)

    LOG.write_text(json.dumps({
        "fetched_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetched_by": "scripts/fetch_leaderboard_anchors.py",
        "provenance": "bytes published by the sibling repositories under buffedlizard55-lab; "
                      "each sha256 pinned in scripts/leaderboard_anchor.py SCORED",
        "rows": log_rows,
    }, indent=1))
    for r in log_rows:
        print(f"  {r['status']:22s} {r['file']}")
    if failures:
        print(f"FAILURES: {len(failures)} file(s) missing or mismatched: {failures}")
        return 1
    print(f"OK: {len(log_rows)}/{len(specs)} files present and sha256-verified "
          f"(log: {LOG.relative_to(ROOT)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
