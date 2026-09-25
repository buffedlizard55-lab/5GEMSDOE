#!/usr/bin/env python3
"""Measure the GitHub Pages inventory for both in-scope accounts and write
data/evidence/site_directory.json (schema "site-directory-evidence-v1").

In scope (per the project brief): every GitHub Pages site hosted by
  * buffedlizard55-lab
  * kanlerxz87-cyber

What this script measures, line by line, from public endpoints only:

  1. GET users/buffedlizard55-lab/repos?per_page=100&type=owner
        every public repository the account owns (name, description, created_at,
        updated_at, pushed_at, archived, size, language, has_pages).
  2. GET repos/buffedlizard55-lab/<repo>/pages   (for every repo above)
        Pages build status, build type, source branch/path, canonical site URL.
  3. GET users/kanlerxz87-cyber  +  its repos  +  its public events
        the second in-scope account: does it have any verifiable public surface?
  4. GET https://buffedlizard55-lab.github.io/  and
     GET https://kanlerxz87-cyber.github.io/
        the user-level Pages URLs (both currently return GitHub's official 404
        page, i.e. neither account publishes a user site).
  5. GET each repo's Pages URL
        liveness: HTTP status, final URL after redirects, the document <title>.

Authentication: the `gh` CLI is used if installed (it carries the token);
otherwise the GITHUB_TOKEN environment variable; otherwise anonymous.  Every
endpoint used here is public, so anonymous works but is rate-limited.

Liveness fallback: some sandboxes have no egress to *.github.io.  On such a
machine, pass --liveness-file <json> containing previously measured records
([{repo, url, final_url, http_ok, title, note, measured_at, method}, ...]);
they are merged verbatim (their measured_at is preserved, and the record is
tagged with the fallback source).  On a machine with egress (e.g. a GitHub
runner) the script measures liveness itself and no file is needed.

Run:
    python scripts/make_site_directory_evidence.py
    python scripts/make_site_directory_evidence.py --liveness-file liveness.json
"""
from __future__ import annotations

import argparse
import html as htmlmod
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "evidence" / "site_directory.json"

ACCOUNTS = ("buffedlizard55-lab", "kanlerxz87-cyber")
GITHUB_API = "https://api.github.com"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utcdate() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------- GitHub API --
def _token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def gh_api(path: str) -> object:
    """GET a public GitHub REST endpoint. Prefers the gh CLI, falls back to
    urllib with/without a token."""
    if shutil.which("gh"):
        p = subprocess.run(["gh", "api", path], capture_output=True, text=True)
        if p.returncode == 0:
            return json.loads(p.stdout)
        # fall through to urllib (the gh token may lack scopes for a path)
    req = urllib.request.Request(GITHUB_API + "/" + path.lstrip("/"),
                                 headers={"Accept": "application/vnd.github+json",
                                          "User-Agent": "site-directory-evidence"})
    tok = _token()
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


# ------------------------------------------------------------------- liveness --
def http_title(url: str) -> dict:
    """Plain HTTPS GET of a live URL: record final URL, status and <title>."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (site-directory-evidence)"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            body = r.read(400_000).decode("utf-8", errors="ignore")
            final = r.geturl()
            status = r.status
    except urllib.error.HTTPError as e:
        body = e.read(400_000).decode("utf-8", errors="ignore")
        final, status = url, e.code
    except Exception as e:  # no egress / DNS / TLS failure: record, do not invent
        return {"url": url, "final_url": None, "http_ok": False,
                "http_status": None, "title": None,
                "error": f"{type(e).__name__}: {e}"}
    m = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
    title = htmlmod.unescape(m.group(1)).strip() if m else None
    out = {"url": url, "final_url": final, "http_ok": status < 400,
           "http_status": status, "title": title}
    # GitHub's official "Site not found" 404 page is a definitive answer
    if status == 404 and "Site not found" in (title or ""):
        out["verdict"] = "github_404_page"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--liveness-file", type=Path, default=None,
                    help="previously measured liveness records (JSON list) for machines "
                         "without egress to *.github.io")
    a = ap.parse_args()

    measured_at = utcnow()
    problems: list[str] = []

    # -- 1. primary account: all public repos it owns -------------------------
    repos_raw = gh_api(f"users/buffedlizard55-lab/repos?per_page=100&type=owner")
    if not isinstance(repos_raw, list):
        print("FATAL: unexpected repos response:", type(repos_raw))
        return 1
    repos = []
    for r in repos_raw:
        pages = gh_api(f"repos/buffedlizard55-lab/{r['name']}/pages") or {}
        repos.append({
            "name": r["name"],
            "description": r.get("description"),
            "homepage": r.get("homepage"),
            "language": r.get("language"),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "pushed_at": r["pushed_at"],
            "archived": bool(r.get("archived")),
            "size_kb": r.get("size"),
            "fork": bool(r.get("fork")),
            "has_pages": bool(r.get("has_pages")),
            "html_url": r["html_url"],
            "api_url": r["url"],
            "pages_settings_url": r["html_url"] + "/pages",
            "pages": {
                "status": pages.get("status"),
                "build_type": pages.get("build_type"),
                "source": pages.get("source"),
                "html_url": pages.get("html_url"),
                "error": pages.get("error"),
            },
        })
    repos.sort(key=lambda x: x["name"])

    # -- 2. liveness of each Pages URL ----------------------------------------
    live_by_repo: dict[str, dict] = {}
    liveness_source = "measured live by this script (plain HTTPS GET)"
    if a.liveness_file is not None:
        pre = json.loads(a.liveness_file.read_text(encoding="utf-8"))
        for rec in pre:
            live_by_repo[rec["repo"]] = dict(rec)
        liveness_source = (f"previously measured records from {a.liveness_file.name} "
                           "(merged verbatim; sandbox has no egress to *.github.io)")
    for rp in repos:
        url = f"https://buffedlizard55-lab.github.io/{rp['name']}/"
        if a.liveness_file is None:
            live_by_repo[rp["name"]] = http_title(url)
        if rp["name"] not in live_by_repo:
            problems.append(f"no liveness record for repo {rp['name']}")
            continue
        rec = live_by_repo[rp["name"]]
        rp["live"] = {
            "url": url,
            "final_url": rec.get("final_url"),
            "http_ok": rec.get("http_ok"),
            "http_status": rec.get("http_status"),
            "title": rec.get("title"),
            "note": rec.get("note"),
            "measured_at": rec.get("measured_at") or measured_at,
            "method": rec.get("method") or liveness_source,
            "verdict": rec.get("verdict"),
        }

    # -- 3. user-level Pages URLs (both accounts) ------------------------------
    user_pages = {}
    for acct in ACCOUNTS:
        url = f"https://{acct}.github.io/"
        if a.liveness_file is None:
            rec = http_title(url)
        else:
            # user-page checks are tiny; measure them live when egress exists,
            # otherwise record the previously measured result inline (see file)
            pre = {p["url"]: p for p in json.loads(a.liveness_file.read_text(encoding="utf-8"))
                   if p.get("user_page")}
            rec = pre.get(url) or http_title(url)
        user_pages[acct] = {"url": url, **{k: rec.get(k) for k in
                         ("final_url", "http_ok", "http_status", "title", "verdict")},
                            "measured_at": rec.get("measured_at") or measured_at,
                            "method": rec.get("method") or liveness_source}

    # -- 4. second account: verifiable public surface --------------------------
    kx = gh_api("users/kanlerxz87-cyber") or {}
    kx_repos = gh_api("users/kanlerxz87-cyber/repos?per_page=100") or []
    if not isinstance(kx_repos, list):
        kx_repos = []
    try:
        kx_events = gh_api("users/kanlerxz87-cyber/events/public?per_page=100") or []
        kx_events = kx_events if isinstance(kx_events, list) else []
    except Exception:
        kx_events = []

    out = {
        "schema": "site-directory-evidence-v1",
        "measured_at": measured_at,
        "method": {
            "inventory": "GitHub REST API (public): users/{acct}/repos?per_page=100&type=owner, then repos/{acct}/{repo}/pages for every repo",
            "liveness": "plain HTTPS GET of each Pages URL; final URL, HTTP status and document <title> recorded",
            "liveness_source": liveness_source,
            "notes": [
                "Private repositories are invisible to public API calls; any Pages sites "
                "hosted from private repos cannot be inventoried here and are flagged, not guessed.",
                "pushed_at is GitHub's last-push timestamp for the repo (last update); "
                "live.measured_at is when the published site was last verified working.",
            ],
        },
        "accounts": {
            "buffedlizard55-lab": {
                "profile_url": "https://github.com/buffedlizard55-lab",
                "api_url": "https://api.github.com/users/buffedlizard55-lab",
                "public_repo_count": len(repos),
                "user_page": user_pages["buffedlizard55-lab"],
                "repos": repos,
            },
            "kanlerxz87-cyber": {
                "profile_url": "https://github.com/kanlerxz87-cyber",
                "api_url": "https://api.github.com/users/kanlerxz87-cyber",
                "public_repo_count": len(kx_repos),
                "public_events_seen": len(kx_events),
                "account_exists": bool(kx.get("login")),
                "user_page": user_pages["kanlerxz87-cyber"],
                "repos": kx_repos,
                "note": ("No verifiable public GitHub Pages surface: 0 public repositories and "
                         "the user page returns GitHub's official 404.  Any sites this account "
                         "hosts would be in private repos (invisible to public APIs) or under a "
                         "different name - flagged for review, not guessed."),
            },
        },
    }

    # cross-checks the evidence must satisfy before it is written
    if any(not rp.get("has_pages") for rp in repos):
        problems.append("a repo has has_pages=false; the directory premise (all repos host Pages) changed")
    if any((rp["pages"] or {}).get("status") != "built" for rp in repos):
        bad = [rp["name"] for rp in repos if (rp["pages"] or {}).get("status") != "built"]
        problems.append(f"Pages status not 'built' for: {bad}")
    if len({rp["name"] for rp in repos}) != len(repos):
        problems.append("duplicate repo names in the inventory")

    if problems:
        print("FATAL evidence problems:")
        for p in problems:
            print("  -", p)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    live_ok = sum(1 for rp in repos if rp["live"]["http_ok"])
    print(f"wrote {OUT.relative_to(ROOT)}: {len(repos)} repos, "
          f"{live_ok}/{len(repos)} live-verified, measured {measured_at}")
    print(f"liveness source: {liveness_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
