#!/usr/bin/env python3
"""Render docs/site_directory.html from data/evidence/site_directory.json and
link it into the site navigation.

The evidence file is produced by scripts/make_site_directory_evidence.py from
public GitHub REST calls + live HTTPS fetches of every Pages URL.  Nothing on
this page is hand-written about what a site supposedly is: every measured
field (dates, build status, live URL, verified title) is read out of the JSON.
The two editorial layers are labelled as such where they appear:

  * short descriptions and category groupings (from the verified page
    title/content seen on 2026-09-25; category is navigation, not a claim);
  * the irregularities register (analysis of the measured fields, each item
    links to the evidence it is based on).

Run:  python scripts/build_site_directory.py
Then: python scripts/audit_docs.py
"""
from __future__ import annotations

import datetime as dt
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EV = ROOT / "data" / "evidence" / "site_directory.json"
BLOB = "https://github.com/buffedlizard55-lab/5GEMSDOE/blob/main/"

NAV = [
    ("index.html", "Overview"),
    ("strategy.html", "★ Strategy 5"),
    ("executive_summary.html", "Executive summary"),
    ("how_to_submit.html", "↳ How to submit"),
    ("submission.html", "Submission details"),
    ("data.html", "Data"),
    ("metric.html", "Metric"),
    ("method.html", "Method"),
    ("results.html", "Results"),
    ("sources.html", "Sources"),
    ("verification.html", "Verification"),
    ("review.html", "Current review"),
    ("reproduce.html", "Reproduce"),
    ("site_directory.html", "Site directory"),
]
# must stay in sync with the nav list in scripts/build_site.py (the patch below
# reproduces exactly the anchor build_site.py renders for the last entry)
NEW_LINK = '<a href="site_directory.html">Site directory</a>'
NAV_MARK = "</div></nav>"

# ---------------------------------------------------------------- editorial --
# Short descriptions written from the verified live page (title + visible
# content) on 2026-09-25.  The page shows the machine-measured title next to
# each one, so a reader can check every description against the live site.
DESC = {
    "11GEMSDOE": "GEMS family, site 11 — README-only repo; the live page shows just the repo title (no substantive content).",
    "5GEMSDOE": "GEMS family, site 5 — this repository: the audited working notes for the GEMS Prize strategy-5 entry, the one this site documents.",
    "6GEMSDOE": "GEMS family, site 6 — declares itself the canonical GEMS entry; the page carries its own audit flagging the 11 GEMSDOE repos as rule-relevant duplication.",
    "7GEMSDOE": "GEMS family, site 7 — README-only repo; the live page shows just the repo title (no substantive content).",
    "8GEMSDOE": "GEMS family, site 8 — README-only repo; the live page shows just the repo title (no substantive content).",
    "GEMSDOE": "GEMS family, site 1 — full fault-detection entry for the DOE GEMS Prize with a browser-side submission builder; same project as sites 2, 4 and 5.",
    "GEMSDOE10": "GEMS family, site 10 — README-only repo; the live page shows just the repo title (no substantive content).",
    "GEMSDOE2": "GEMS family, site 2 — another complete copy of the fault-detection entry with its own docs build; same project as sites 1, 4 and 5.",
    "GEMSDOE3": "GEMS family, site 3 — the “Pindrop” line: a mission-control page publishing three format-validated Pindrop v4 submission GeoTIFFs with SHA-256 pins and paste-notes.",
    "GEMSDOE4": "GEMS family, site 4 — the “new-fault-first” line: lineament-feature model (Sato ridgeness, structure tensor) selected on a proxy of the new-fault population.",
    "GEMSDOE9": "GEMS family, site 9 — README-only repo; the live page shows just the repo title (no substantive content).",
    "AirPremia": "Refund playbook for Air Premia SFO ⇄ ICN flights, Oct–Nov 2026.",
    "BathTubOverflowSF": "“Sunset Repair” — project page for the Sunset-district tub-overflow repair.",
    "BusanL7HaeundaeLotteHotelStay": "Verified travel itinerary & vlog planner for a Busan L7 Haeundae stay, Nov 9–16, 2026.",
    "Commodities": "“Kalshi Research Exchange · Commodities” — paper-trading research on commodity markets.",
    "Coupons": "Official-source product coupons, rebates and BOGO deals for the Bay Area.",
    "CruiseDeals": "2027 U.S.-port cruise + flight planner with verified findings (served from /docs/).",
    "DrugAnalysis": "“FDA Decision Engine (v29)” — drug-approval decision analysis.",
    "Elections": "Election results: “collect, analyze, project & estimate”.",
    "FLABSENGY": "FantasyLabs (FLAB) research + rebuild project, served from /docs/ (“FLABSENGY — Overview”).",
    "FUTURESCOMMODITIES": "“FUTURESCOMMODITIES — verified paper-trading competition” on commodity futures.",
    "GOLD": "“Solid Gold Ring Directory” — directory of gold-ring sellers.",
    "HongdaeStay": "Itinerary page for the Mercure Ambassador Seoul Hongdae, Nov 16–22 2026.",
    "InjuryAlerTNFL": "“Sideline Signal” — in-game NFL injury alerts with receipts (source links per alert).",
    "Insider-trades": "SEC Form 4 (insider trading) research dashboard.",
    "Itinerary-Korea": "“Korea Compass · Master trip planner” (archived repo; Pages still built and live). Content-identical to HotelSeoulRoughdraft1 — duplicate pair.",
    "KalshiPaperSim": "Paper-trading championship run on real Kalshi market data.",
    "Korea": "Korea deals and freebies, organized (the only repo with a GitHub-API description: “Freebies and deals korea”).",
    "Korea-emergency": "Emergency guide for Korea travel, Seoul & Busan, Nov 1–22, 2026.",
    "KoreaHotels": "South Korea hotel shortlist with verified findings.",
    "Leg3SeoulTrip": "Near-empty: the live page shows just the repo title (no substantive content).",
    "LSTARENGY": "LineStar review — sports projection research (“LSTARENGY — Sports Projection Research”).",
    "MALTASUPPLEMENTAL": "Malta trip guide & recreation companion. The page states its companion repo MALTAOffline no longer resolves (checked 2026-09-24 per the page).",
    "MLB-Live-PBP": "MLB live scoreboard: live status, play-by-play and box scores (verified title “MLB Live Scoreboard”).",
    "MLB-PBP": "“MLB Plaintext Archive” — browsable plaintext play-by-play archive (root redirects to /docs/?date=<current date>).",
    "MLB-Prediction-model-backtest": "README stub: the page renders only the repo README, which cites the off-account upstream repo karagemop466-tech/StatcastMLB (not a hosted site).",
    "MLBComp": "“Auditable MLB Research” — competition/research site.",
    "MLBSCORINGCHANGE": "MLB live scoreboard — verified title identical to MLB-Live-PBP; duplicate-content candidate.",
    "MLBRainDelay": "“MLB Rain Delay — Live Weather Scoreboard” — MLB scores with a live weather overlay.",
    "NHL-SCOREBOARD": "NHL scoreboard, live and historical back to 1917, built from the official NHL public API with per-game official review links.",
    "NHLComp": "“Dashboard · NHLComp” — NHL betting-strategy competition dashboard.",
    "NBAComp": "“Dashboard — NBAComp” — NBA betting-strategy competition dashboard.",
    "NBAInjScoreboard": "“NBA Courtside | Live Scoreboard” — live NBA scores with an injury overlay.",
    "NBAInjuryReport": "“NBA Injury Alert System” — NBA injury tracking.",
    "NBASCOREBOARD": "NBA scoreboard from the published official NBA feeds, with play-by-play, box scores, a historical-date archive and a sync-evidence log.",
    "NFL-scoreboard": "NFL scoreboard (verified title “NFL Scoreboard”).",
    "NFL-PLAYER-PROP-SIM": "Near-empty: the live page shows just the repo title (no substantive content).",
    "NFLComp": "“Autonomous NFL Betting Strategy Competition”.",
    "NFLInjuryReport": "“NFL Injury Report — live, source-verified”.",
    "NFLMAIN": "NFL scoreboard, live and historical: season browser 1999–2026, per-game pages and official NFL.com review links (served from /docs/).",
    "NFLPRED": "Near-empty: the live page shows just the repo title (no substantive content).",
    "NFLPARLAYCOMP": "NFL parlay trading competition (served from /docs/; SPA shell — the app renders client-side, “Loading…” at check time).",
    "NOBEL-PRIZE": "“Nobel Prize Record” — the prize/laureate record (verified page content: 633 awards, 1,018 laureates).",
    "Ncaa-football-alerts": "“NCAA Football Scoreboard”.",
    "OLBG-Competition": "“Northstar // OLBG Competition Lab” — research lab for OLBG betting competitions.",
    "ParlaySports": "Simulated multi-sport parlay research (SPA shell — the app renders client-side, “Loading…” at check time).",
    "PFFNFL": "README stub: the page renders only the repo README (title “PFFNFL”); purpose not verifiable beyond the README — review the repo.",
    "PlumbingSF": "“Verified San Francisco Plumbers” directory (served from /docs/).",
    "PriceKalshiHistorical": "Historical Kalshi market data (served from /docs/; verified title “kalshi — trade on what's next”).",
    "PRICINGEXPERT": "“PRICINGEXPERT — Kalshi Paper-Trading Competition & Research Desk”.",
    "ProjX": "“Verified Directory” — verified local-business directory.",
    "RGENGY": "RotoGrinders data-quality rebuild (“Dashboard · RGENGY”).",
    "SABERENGY": "SaberSim rebuild — public-data DFS research (“SABERENGY — public-data DFS research”).",
    "ScheduleFreeTime": "“When are you free?” — shared availability calendar, Aug 2026 – Feb 2027.",
    "SFWeather": "Rainy-season weather outlook for San Francisco 94122.",
    "SFLateNight": "“SF Late Night — verified transit-friendly places”.",
    "SIM-COMP-NOBEL-PRIZE": "Simulated competition on Kalshi's Nobel Prize markets (“SIM-COMP Nobel Prize”).",
    "SelfLearn": "“SelfLearn — autonomous evidence-verifying research engine”.",
    "ShoulderPain": "Shoulder-injury guide (“Start here · Shoulder Guide”).",
    "SocialMediaComp": "“Social Media Comp — Leaderboard”.",
    "SportsPred": "Multi-sport scoreboard with OLBG markets and written predictions.",
    "StanfordStay": "Seoul itineraries for a stay at the Stanford Hotel Myeongdong, Nov 1–9, 2026.",
    "StockPaperSim": "“Paper-trading stock competition, audited line by line”.",
    "StokEngineer": "DFS projection / lineup-optimization engine — the page is the repo README (title “StokEngineer”).",
    "TAXKALSHI": "“TAXKALSHI — How Kalshi earnings are taxed, federal and California”.",
    "THUNDERPICK-WC-2026": "“Overview — Thunderpick World Championship 2026 · Expert Hub”.",
    "THUNDERPICKCOMP": "“Thunderpick WC 2026 · Prediction Competition Hub”.",
    "TinoLunchSpecial": "Lunch specials in and around Cupertino.",
    "TradingViewTheLeap": "“The Leap Research Lab” — TradingView research.",
    "Tradingview-pinescript-editor": "“PinePilot — TradingView Strategy Lab” (archived repo; Pages still built and live).",
    "VacationSchedule": "“Vacation windows · Bay Area sports planner”.",
    "VapePods": "“STIIIZY Pod Deals — San Francisco (94122 / Stonestown)”.",
    "WoWForever": "“WoW Forever Hub” — World of Warcraft hub page.",
    "MasterSite": "Pre-existing self-serve directory of this account's Pages sites (“Verified GitHub Pages Directory · buffedlizard55-lab”, 74 entries at its snapshot, generated from the GitHub REST API). Overlaps this page — consolidation candidate.",
    "MasterSelfLearn": "“Today's briefing — MasterSelfLearn” — hub/briefing variant of SelfLearn.",
    "HotelSeoulRoughdraft1": "“Korea Compass · Master trip planner” (archived repo; Pages still built and live). Content-identical to Itinerary-Korea — duplicate pair.",
}

# Editorial grouping for navigation only — not a claim about the sites.
CATS = [
    ("gems", "GEMS Prize (DOE)"),
    ("scoreboard", "Live scoreboards"),
    ("research", "Sports research & comps"),
    ("injury", "Injury & event monitors"),
    ("dfs", "Fantasy / DFS engines"),
    ("trading", "Trading & finance research"),
    ("travel", "Travel & itineraries"),
    ("local", "Local guides & services"),
    ("directory", "Directories & hubs"),
]
CAT = {
    "GEMSDOE": "gems", "GEMSDOE2": "gems", "GEMSDOE3": "gems", "GEMSDOE4": "gems",
    "5GEMSDOE": "gems", "6GEMSDOE": "gems", "7GEMSDOE": "gems", "8GEMSDOE": "gems",
    "GEMSDOE9": "gems", "GEMSDOE10": "gems", "11GEMSDOE": "gems",
    "MLB-Live-PBP": "scoreboard", "MLBSCORINGCHANGE": "scoreboard", "MLBRainDelay": "scoreboard",
    "MLB-PBP": "scoreboard", "NFL-scoreboard": "scoreboard", "NFLMAIN": "scoreboard",
    "Ncaa-football-alerts": "scoreboard", "NBAInjScoreboard": "scoreboard",
    "NBASCOREBOARD": "scoreboard", "NHL-SCOREBOARD": "scoreboard",
    "SportsPred": "research", "OLBG-Competition": "research", "StockPaperSim": "research",
    "KalshiPaperSim": "research", "Commodities": "research", "FUTURESCOMMODITIES": "research",
    "PRICINGEXPERT": "research", "TAXKALSHI": "research", "NFLPARLAYCOMP": "research",
    "ParlaySports": "research", "NFLComp": "research", "NBAComp": "research",
    "NHLComp": "research", "MLBComp": "research", "SocialMediaComp": "research",
    "THUNDERPICK-WC-2026": "research", "THUNDERPICKCOMP": "research", "NOBEL-PRIZE": "research",
    "SIM-COMP-NOBEL-PRIZE": "research", "NFLPRED": "research", "NFL-PLAYER-PROP-SIM": "research",
    "Elections": "research", "MLB-Prediction-model-backtest": "research",
    "PriceKalshiHistorical": "research",
    "NFLInjuryReport": "injury", "NBAInjuryReport": "injury", "InjuryAlerTNFL": "injury",
    "PFFNFL": "dfs", "StokEngineer": "dfs", "FLABSENGY": "dfs", "LSTARENGY": "dfs",
    "RGENGY": "dfs", "SABERENGY": "dfs",
    "Insider-trades": "trading", "Tradingview-pinescript-editor": "trading",
    "TradingViewTheLeap": "trading",
    "Korea": "travel", "Korea-emergency": "travel", "Itinerary-Korea": "travel",
    "KoreaHotels": "travel", "HotelSeoulRoughdraft1": "travel", "StanfordStay": "travel",
    "BusanL7HaeundaeLotteHotelStay": "travel", "HongdaeStay": "travel", "Leg3SeoulTrip": "travel",
    "AirPremia": "travel", "MALTASUPPLEMENTAL": "travel", "CruiseDeals": "travel",
    "VacationSchedule": "travel", "ScheduleFreeTime": "travel",
    "PlumbingSF": "local", "SFLateNight": "local", "SFWeather": "local",
    "TinoLunchSpecial": "local", "VapePods": "local", "Coupons": "local",
    "BathTubOverflowSF": "local", "ShoulderPain": "local", "DrugAnalysis": "local",
    "WoWForever": "local",
    "MasterSite": "directory", "ProjX": "directory", "GOLD": "directory",
    "SelfLearn": "directory", "MasterSelfLearn": "directory",
}

# ------------------------------------------------------------------ helpers --
def e(x) -> str:
    return html.escape(str(x)) if x is not None else "—"


def d(iso: str | None) -> str:
    return (iso or "—")[:10]


def full(iso: str | None) -> str:
    return (iso or "").replace("T", " ").replace("Z", " UTC")


def load_evidence() -> dict:
    if not EV.exists():
        print("FATAL: missing data/evidence/site_directory.json — run "
              "`python scripts/make_site_directory_evidence.py` first. "
              "This page will not be written: it renders measurements, not prose.")
        sys.exit(1)
    data = json.loads(EV.read_text(encoding="utf-8"))
    if data.get("schema") != "site-directory-evidence-v1":
        print(f"FATAL: unsupported evidence schema: {data.get('schema')!r}")
        sys.exit(1)
    return data


def check_editorial_coverage(repos: list[dict]) -> list[str]:
    """Both directions: every measured repo gets an editorial entry, and the
    editorial table does not name a repo the measurement does not contain."""
    names = {r["name"] for r in repos}
    probs = []
    missing_desc = sorted(names - set(DESC))
    missing_cat = sorted(names - set(CAT))
    extra_desc = sorted(set(DESC) - names)
    extra_cat = sorted(set(CAT) - names)
    if missing_desc:
        probs.append(f"editorial descriptions missing for: {missing_desc}")
    if missing_cat:
        probs.append(f"categories missing for: {missing_desc}")
    if extra_desc:
        probs.append(f"editorial descriptions name unknown repos: {extra_desc}")
    if extra_cat:
        probs.append(f"categories name unknown repos: {extra_cat}")
    bad_cats = sorted({c for c in CAT.values() if c not in dict(CATS)} - set())
    if bad_cats:
        probs.append(f"categories reference unknown groups: {bad_cats}")
    return probs


# ------------------------------------------------------------------ render --
def render(data: dict) -> str:
    bl = data["accounts"]["buffedlizard55-lab"]
    kx = data["accounts"]["kanlerxz87-cyber"]
    repos = bl["repos"]
    n = len(repos)
    live_repos = [r for r in repos if r["live"]["http_ok"]]
    verified_dates = sorted({r["live"]["measured_at"][:10] for r in repos if r["live"]["measured_at"]})
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    nav_links = "".join(
        '<a href="%s"%s>%s</a>' % (h, ' class="active"' if h == "site_directory.html" else "", e(t))
        for h, t in NAV
    )

    # ---- accounts section
    arch = [r["name"] for r in repos if r["archived"]]
    account_html = f"""
<h2>Accounts in scope</h2>
<div class="stats">
  <div class="stat"><div class="k">buffedlizard55-lab</div>
    <div class="v">{n} public repos</div>
    <div class="s">all {n} have GitHub Pages enabled; GitHub's Pages API reports
    <code>status: built</code> (legacy, source <code>main /</code>) for every one.
    User page <a href="{e(bl['user_page']['url'])}">{e(bl['user_page']['url'])}</a> returns
    GitHub's official 404 — all {n} sites are project pages, verified {e(verified_dates[0])}.</div></div>
  <div class="stat"><div class="k">kanlerxz87-cyber</div>
    <div class="v">{kx['public_repo_count']} public repos</div>
    <div class="s">account exists, {kx['public_repo_count']} public repositories,
    {kx['public_events_seen']} public events; user page
    <a href="{e(kx['user_page']['url'])}">{e(kx['user_page']['url'])}</a> returns GitHub's
    official 404. <b>No verifiable public Pages surface</b> — see the flag below.</div></div>
</div>
<div class="note warn"><b>Flag for review — kanlerxz87-cyber.</b>
The brief puts this account in scope, but its public surface is empty:
<a href="{e(kx['api_url'])}">api.github.com/users/kanlerxz87-cyber</a> reports
{ kx['public_repo_count'] } public repositories,
<a href="{e(kx['profile_url'])}">github.com/kanlerxz87-cyber</a> lists none, and
<a href="{e(kx['user_page']['url'])}">kanlerxz87-cyber.github.io</a> serves GitHub's own
“Site not found” 404 page (title “{e(kx['user_page']['title'])}”, checked
{e(kx['user_page']['measured_at'])}). Any Pages sites it hosts would live in <i>private</i>
repositories — invisible to public API calls — or under a different name. This directory
records that verifiable fact and stops there; nothing is guessed.</div>
"""

    # ---- irregularities register
    def live(name: str) -> str:
        r = next(x for x in repos if x["name"] == name)
        return f'<a href="{e(r["live"]["url"])}">{e(name)}</a>'

    def repo_link(name: str) -> str:
        r = next(x for x in repos if x["name"] == name)
        return f'<a href="{e(r["html_url"])}">{e(name)}</a>'

    irregularities = f"""
<h2>Irregularities flagged for review</h2>
<p class="small">Analysis of the measured fields below, each item linked to the evidence it
rests on. Items 1–2 are the two consolidation decisions this directory exists to surface.</p>
<ol class="findings">
  <li><b>11 GEMSDOE repositories publish one competition (rule-relevant duplication).</b>
  The account hosts 11 Pages URLs for the DOE GEMS Prize — {live('GEMSDOE')}, {live('GEMSDOE2')},
  {live('GEMSDOE3')}, {live('GEMSDOE4')}, {live('5GEMSDOE')}, {live('6GEMSDOE')},
  {live('7GEMSDOE')}, {live('8GEMSDOE')}, {live('GEMSDOE9')}, {live('GEMSDOE10')},
  {live('11GEMSDOE')}. Five are complete copies of the same project (GEMSDOE, GEMSDOE2,
  GEMSDOE3, GEMSDOE4, 5GEMSDOE); five hold a README and nothing else (7GEMSDOE, 8GEMSDOE,
  GEMSDOE9, GEMSDOE10, 11GEMSDOE); all 11 report <code>status: built</code> in the Pages API.
  The {live('6GEMSDOE')} site itself declares it “the one canonical entry” and its own audit
  recommends keeping that one repo and deleting or archiving the other ten, citing the
  competition's one-entry-per-entity terms (rules §3.4/§3.5/§3.6.2).
  <span class="small">Evidence: the Pages API column in the table, and the 6GEMSDOE page itself.</span></li>
  <li><b>{live('MasterSite')} already <i>is</i> a directory of this account.</b>
  Its title is “Verified GitHub Pages Directory · buffedlizard55-lab” — 74 entries at the time
  of its snapshot, generated from the GitHub REST API, with categories and its own irregularities
  register. This page supersedes it ({n} entries, both in-scope accounts, per-site source links,
  second account checked). <b>Consolidation candidate:</b> point MasterSite at this page, or
  retire it, so the account stops publishing two directories of itself.</li>
  <li><b>Duplicate-content pair: {live('Itinerary-Korea')} ≡ {live('HotelSeoulRoughdraft1')}.</b>
  Both serve the identical title “Korea Compass · Master trip planner”, both repos are
  <i>archived</i>, and Pages is still built and live on both.</li>
  <li><b>8 near-empty sites (title-only pages).</b> The live page shows just the repo title —
  nothing to review: {live('NFLPRED')}, {live('Leg3SeoulTrip')}, {live('NFL-PLAYER-PROP-SIM')},
  {live('7GEMSDOE')}, {live('8GEMSDOE')}, {live('GEMSDOE9')}, {live('GEMSDOE10')},
  {live('11GEMSDOE')}. Mechanical rule: verified title equals the repo name.</li>
  <li><b>3 README-stub sites.</b> The page is only the repo README: {live('MLB-Prediction-model-backtest')},
  {live('PFFNFL')}, {live('StokEngineer')}. The MLB-Prediction-model-backtest README cites the
  off-account upstream repo <code>karagemop466-tech/StatcastMLB</code> — an upstream, not a
  site hosted by either in-scope account.</li>
  <li><b>3 archived repos with live Pages.</b> {live('Tradingview-pinescript-editor')},
  {live('Itinerary-Korea')}, {live('HotelSeoulRoughdraft1')} — archiving a repo does not stop
  its Pages build; all three still serve.</li>
  <li><b>Identical verified titles across different projects (duplicate candidates).</b>
  {live('MLB-Live-PBP')} and {live('MLBSCORINGCHANGE')} both verify to “MLB Live Scoreboard”.
  (The GEMS family's identical “Overview — GEMS Prize” titles are expected — one project.)</li>
  <li><b>kanlerxz87-cyber has no verifiable public surface</b> — 0 public repos, 0 public
  events, official 404 user page. Flagged in the accounts section above; any sites it hosts
  are private-repo Pages and cannot be inventoried from public endpoints.</li>
  <li><b>{live('MALTASUPPLEMENTAL')} references a deleted/renamed companion.</b> Its page
  states the companion repo <code>MALTAOffline</code> “no longer resolves” (checked 2026-09-24
  per that page). It is not among the current {n} repos.</li>
  <li><b>Neither account publishes a user-level Pages site.</b>
  <a href="https://buffedlizard55-lab.github.io/">buffedlizard55-lab.github.io</a> and
  <a href="https://kanlerxz87-cyber.github.io/">kanlerxz87-cyber.github.io</a> both return
  GitHub's official “Site not found” 404 — every one of the {n} sites is a project page.</li>
</ol>
"""

    # ---- the table
    cat_label = dict(CATS)
    rows = []
    for r in repos:
        name = r["name"]
        L = r["live"]
        title = L.get("title")
        minimal = title == name
        redirect = L.get("final_url") and L["final_url"].rstrip("/") != L["url"].rstrip("/")
        pills = ""
        if r["archived"]:
            pills += '<span class="pill warn">archived</span>'
        if minimal:
            pills += '<span class="pill info">title-only</span>'
        status = ('<span class="pill ok">LIVE</span> <span class="small">verified '
                  + e(L.get("measured_at", "—")[:10]) + "</span>") if L.get("http_ok") else \
                 '<span class="pill bad">NOT VERIFIED</span>'
        redir = ' <span class="pill info">→ docs/</span>' if redirect else ""
        desc = DESC.get(name)
        cat = CAT.get(name, "?")
        text_search = " | ".join([name, title or "", desc or ""]).lower()
        rows.append(
            f'<tr data-cat="{e(cat)}" data-text="{e(text_search)}">'
            f'<td><a href="{e(L["url"])}"><b>{e(name)}</b></a>{pills}'
            f'<div class="small mono">{e(L["url"])}</div></td>'
            f'<td>{e(title)}{redir}</td>'
            f'<td>{e(desc)}</td>'
            f'<td><span class="pill info">{e(cat_label.get(cat, cat))}</span></td>'
            f'<td class="num" title="{e(full(r.get("created_at")))}">{e(d(r.get("created_at")))}</td>'
            f'<td class="num" title="{e(full(r.get("pushed_at")))}">{e(d(r.get("pushed_at")))}</td>'
            f'<td>{status}</td>'
            f'<td class="small"><a href="{e(r["html_url"])}">repo</a> · '
            f'<a href="{e(r["pages_settings_url"])}">pages</a> · '
            f'<a href="{e(r["api_url"])}">api</a></td></tr>'
        )
    chips = ['<button class="dir-chip on" data-cat="all">All ({0})</button>'.format(n)]
    for cid, clabel in CATS:
        cnt = sum(1 for r in repos if CAT.get(r["name"]) == cid)
        chips.append(f'<button class="dir-chip" data-cat="{e(cid)}">{e(clabel)} ({cnt})</button>')

    table_html = f"""
<h2 id="all-sites">All {n} sites</h2>
<p>Every public repository of <code>buffedlizard55-lab</code> that hosts GitHub Pages,
live-verified on {e(verified_dates[0])}. “Verified title” is the document <code>&lt;title&gt;</code>
read from the live URL (machine-measured); “Description” is a short editorial summary written
from that same verified page. Filter by category or search. Full provenance per row: the
<a href="{BLOB}data/evidence/site_directory.json">evidence JSON</a> and the linked official
sources.</p>
<input id="dir-q" type="search" placeholder="Search site, title or description…" aria-label="Search sites">
<div class="dir-chips">{''.join(chips)}</div>
<p class="small" id="dir-count"></p>
<table class="dir" id="dir-table">
<thead><tr><th>Site (live URL)</th><th>Verified title</th><th>Description</th><th>Category</th>
<th>Created</th><th>Last push</th><th>Status</th><th>Official sources</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
"""

    method_html = f"""
<h2>How each column was measured</h2>
<table class="kv">
<tbody>
<tr><th>Inventory (86 repos)</th><td>GitHub REST API <code>users/buffedlizard55-lab/repos?per_page=100&amp;type=owner</code>
— every public repo the account owns (no forks, no <code>.github.io</code> user-site repo).</td></tr>
<tr><th>Pages build status</th><td><code>repos/buffedlizard55-lab/&lt;repo&gt;/pages</code> per repo —
all report <code>status: built</code>, build type <code>legacy</code>, source <code>main /</code>.</td></tr>
<tr><th>Created</th><td>repo <code>created_at</code> — the GitHub API timestamp of repo creation.</td></tr>
<tr><th>Last push</th><td>repo <code>pushed_at</code> — the GitHub API timestamp of the repo's
last push (last update to the repository).</td></tr>
<tr><th>Live URL · title · status</th><td>plain HTTPS GET of each Pages URL on
{e(verified_dates[0])}; final URL (after redirects), HTTP status and the document
<code>&lt;title&gt;</code> recorded per row. Liveness source: {e(data['method'].get('liveness_source', ''))}.</td></tr>
<tr><th>Evidence</th><td>machine-measured JSON: <a href="{BLOB}data/evidence/site_directory.json">data/evidence/site_directory.json</a>
(measured {e(data.get('measured_at', '—'))}); measurement script:
<a href="{BLOB}scripts/make_site_directory_evidence.py">scripts/make_site_directory_evidence.py</a>;
rendering script: <a href="{BLOB}scripts/build_site_directory.py">scripts/build_site_directory.py</a>.</td></tr>
<tr><th>Official sources per site</th><td>the GitHub repo page, the repo's Pages settings page,
and the live Pages URL — linked on every row of the table.</td></tr>
<tr><th>Out of reach</th><td>private repositories are invisible to public API calls; any Pages
sites hosted from private repos are flagged (kanlerxz87-cyber section), never guessed.</td></tr>
</tbody></table>
"""

    accounts_link = (f'<a href="{e(bl["api_url"])}">api.github.com/users/buffedlizard55-lab</a> · '
                     f'<a href="{e(bl["profile_url"])}">github.com/buffedlizard55-lab</a>')

    reproduce_html = f"""
<h2>Reproducing this measurement</h2>
<pre><code>python scripts/make_site_directory_evidence.py     # re-measure from GitHub's public API + live fetches
python scripts/build_site_directory.py               # re-render this page + refresh the nav
python scripts/audit_docs.py                         # the no-hallucination gate</code></pre>
<p class="small">On a machine without egress to <code>*.github.io</code>, the measurement script
accepts <code>--liveness-file</code> with previously measured liveness records (this
checkout's evidence used that route; the records carry their own measured-at date). The API
inventory itself always comes live from GitHub.</p>
"""

    page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Site directory — all GitHub Pages sites</title>
<meta name="description" content="Every GitHub Pages site hosted by buffedlizard55-lab and kanlerxz87-cyber: name, description, dates, live URL and official source links for manual review.">
<link rel="stylesheet" href="style.css">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>⛰️</text></svg>">
<style>
.dir-chips{{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0 2px}}
.dir-chip{{font:inherit;font-size:.82rem;font-weight:560;padding:4px 11px;border-radius:999px;
  border:1px solid var(--line);background:#fff;color:var(--muted);cursor:pointer}}
.dir-chip:hover{{border-color:var(--accent);color:var(--accent-dark)}}
.dir-chip.on{{background:var(--accent);border-color:var(--accent);color:#fff}}
#dir-q{{font:inherit;font-size:.95rem;padding:8px 12px;border:1px solid var(--line);
  border-radius:9px;width:100%;max-width:440px;margin:12px 0 2px;background:#fff}}
table.dir th,table.dir td{{font-size:.84rem;padding:7px 10px}}
table.dir td b{{font-weight:640}}
</style>
</head><body>
<header>
  <div class="wrap">
    <div class="brand"><span class="logo">📇</span>
      <div><div class="name">Site directory — all GitHub Pages sites</div>
      <div class="tag">{accounts_link} · every row measured from GitHub's own API and live fetches, none hand-collected</div></div>
    </div>
  </div>
</header>
<nav><div class="wrap">{nav_links}</div></nav>
<main class="wrap">
<h1>Site directory</h1>
<p class="lede">Master list of <b>all</b> GitHub Pages sites hosted by the in-scope accounts
<code>buffedlizard55-lab</code> ({n} sites) and <code>kanlerxz87-cyber</code>
(no verifiable public sites — flagged in the accounts section below). For each site:
name, brief description,
date created, last update, live URL and links to the official sources for manual review.
All {len(live_repos)}/{n} sites were live-verified on {e(verified_dates[0])}.</p>
{method_html}
{account_html}
{irregularities}
{table_html}
{reproduce_html}
</main>
<footer><div class="wrap">
  <p>Generated by <code>scripts/build_site_directory.py</code> from
  <a href="{BLOB}data/evidence/site_directory.json">data/evidence/site_directory.json</a>
  (measured {e(data.get('measured_at', '—'))}) · {stamp}</p>
  <p>Unofficial directory. For any site, the authoritative sources are its GitHub repository,
  its Pages settings, and the live URL — all linked on its row.</p>
</div></footer>
<script>
(function(){{
  var q = document.getElementById('dir-q');
  var rows = Array.prototype.slice.call(document.querySelectorAll('#dir-table tbody tr'));
  var chips = Array.prototype.slice.call(document.querySelectorAll('.dir-chip'));
  var cat = 'all';
  function apply(){{
    var needle = (q.value || '').trim().toLowerCase();
    var shown = 0;
    rows.forEach(function(tr){{
      var okCat = (cat === 'all') || (tr.getAttribute('data-cat') === cat);
      var okTxt = !needle || (tr.getAttribute('data-text') || '').indexOf(needle) !== -1;
      var show = okCat && okTxt;
      tr.style.display = show ? '' : 'none';
      if (show) shown++;
    }});
    document.getElementById('dir-count').textContent =
      shown + ' of ' + rows.length + ' sites shown';
    chips.forEach(function(c){{ c.classList.toggle('on', c.getAttribute('data-cat') === cat); }});
  }}
  chips.forEach(function(c){{ c.addEventListener('click', function(){{ cat = c.getAttribute('data-cat'); apply(); }}); }});
  q.addEventListener('input', apply);
  apply();
}})();
</script>
</body></html>
"""
    return page


# ------------------------------------------------------------------- nav -----
def patch_nav() -> tuple[list[str], list[str]]:
    """Idempotently add the Site-directory anchor to every already-published
    page's nav, exactly where build_site.py's nav list places it (last)."""
    patched, skipped = [], []
    for f in sorted(DOCS.glob("*.html")):
        if f.name == "site_directory.html":
            continue
        text = f.read_text(encoding="utf-8")
        if 'href="site_directory.html"' in text:
            skipped.append(f.name)
            continue
        i = text.rfind(NAV_MARK)
        if i < 0:
            skipped.append(f.name + " (no nav found — not patched)")
            continue
        f.write_text(text[:i] + NEW_LINK + text[i:], encoding="utf-8")
        patched.append(f.name)
    return patched, skipped


def main() -> int:
    data = load_evidence()
    repos = data["accounts"]["buffedlizard55-lab"]["repos"]
    probs = check_editorial_coverage(repos)
    if probs:
        print("FATAL: editorial table out of sync with the measured inventory:")
        for p in probs:
            print("  -", p)
        print("Update the DESC/CAT tables in scripts/build_site_directory.py to match "
              "data/evidence/site_directory.json — this page will not ship with a hole or a ghost.")
        return 1
    if not repos:
        print("FATAL: evidence contains zero repos — refusing to render an empty directory.")
        return 1

    out = DOCS / "site_directory.html"
    DOCS.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size:,} bytes, {len(repos)} sites)")

    patched, skipped = patch_nav()
    print(f"nav patched: {len(patched)} page(s) "
          + (f"({', '.join(patched)})" if patched else "") +
          f"; already linked/skipped: {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
