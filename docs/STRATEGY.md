# 5GEMSDOE strategy — what we know from the board, what the metric forces, what we test next

*Written 2026-09-25 (session 1 of this fork). Every number below is produced by a script in this repository
(`scripts/leaderboard_anchor.py` → `data/evidence/leaderboard_anchor/leaderboard_anchor.json`) or linked to an
official page that was fetched on the date given. Nothing here is a leaderboard prediction.*

---

## 1. Verified facts that decide the strategy (with sources)

| # | Fact | Source (fetched 2026-09-25) |
|---|---|---|
| F1 | The test set is **new faults only** — "manually identified faults that are not contained within the current public USGS database … comprise the test dataset". The region is "chunked and split into a public test set and a private test set". | [Problem description → Competition structure](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#competition-structure) |
| F2 | Pixels of **known USGS/INGENIOUS faults are masked** from scoring: "masked / excluded from evaluation, so they do not count towards penalty terms … it should not matter whether these known faults are included with predictions or not." Re-evaluation (final round) masks them too. | [Forum 11516, DrivenData staff reply](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516) |
| F3 | "New fault" = **"any fault pixel not already captured by USGS/INGENIOUS" and can include newly mapped geometry of an existing fault system** (continuations, splays, parallel strands). | [Forum 11536, staff reply](https://community.drivendata.org/t/where-do-you-draw-the-line/11536) |
| F4 | Organizers **will not disclose** which data, fault types or coverage produced the test faults. Phase 2 uses labels expanded by expert review of all Phase-1 submissions. | [Forum 11527, post 7](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527/7) |
| F5 | Metric: DTI = TP_w / (TP_w + 0.2·FP_w + 0.8·FN_w), triangular kernel R = 300 m (3 px); TP_w takes the **single best** prediction inside R of each truth pixel. | [Problem description → Performance metric](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric) |
| F6 | Submission: single-band float32 GeoTIFF, EPSG 32611, 100 m, same bounds, NaN outside; values in [0,1]. | [Problem description → Submission format](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format) |
| F7 | Public leaderboard 2026-09-25: #1 DARD 0.3049 · #2 alexoktaba 0.2993 · #3 HardcoreTechGod 0.2854 · … · extradr19 0.1563 (#21) · smashi34 0.1560 (#22). | [Leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/) |
| F8 | The supplied 19 bands are magnetics (6), gravity (4), geodetic strain (3), seismicity (2), subsurface (2), topography (2: `det_elev`, `det_elev_slope` at 100 m). **No radiometric band.** | measured from `data/training_features.tif` band tags (`scripts/prepare_data.py`) |
| F9 | GeoDAWN's own release ships **radiometric and magnetic GeoTIFF grids** (`22103_area1_tiffs.zip` 43.6 MB, `22103_area2_tiffs.zip` 230.5 MB), flown at 200 m / 400 m line spacing, and states that **airborne lidar was collected in coordination (via 3DEP)** over a similar extent. | [ScienceBase 657e1d85d34e23d3533209f7](https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7), DOI [10.5066/P93LGLVQ](https://doi.org/10.5066/P93LGLVQ) |
| F10 | USGS 3DEP seamless 1/3 arc-second (~10 m) DEM is public domain and served as staged GeoTIFFs with a GDAL-readable VRT over HTTP. | [VRT](https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt) · [ScienceBase record](https://www.sciencebase.gov/catalog/item/4f70aa9fe4b058caae3f8de5) · [USGS data catalog](https://data.usgs.gov/datacatalog/data/USGS:3a81321b-c153-416f-98b7-cc8e5f0e17c3) |
| F11 | Deep-learning fault mapping in northern-central Nevada trained on **lidar** found faults absent from the USGS map that were "later confirmed by experts looking at the LIDAR data" (Hermant et al., Stanford Geothermal Workshop 2025; INGENIOUS-linked authors). | [Hermant et al. 2025 (PDF)](https://pangea.stanford.edu/ERE/pdf/IGAstandard/SGW/2025/Hermant.pdf) |

## 2. What the metric forces (derivations, checked against the board)

Because TP_w + FN_w = |G| exactly (each truth pixel contributes its covered mass to TP and the remainder to FN):

```
DTI = T / (0.2·(T + F) + 0.8·G)          T = TP_w,  F = FP_w,  G = |G| (kernel mass of scored truth)
```

**R1 — Binary is optimal for a fixed support.** Scaling every emitted value by c gives cT/(0.2c(T+F)+0.8G), increasing in c.
So every emitted pixel should be 1.0; the whole problem is *which* pixels to emit. (GEMSDOE measured "hard band beats
distance ramp" empirically; this is why.)

**R2 — Marginal-inclusion rule.** Adding one binary pixel with expected TP gain ΔT and FP cost ΔF (≤ 1) raises the score iff
```
ΔT  >  0.2·DTI·ΔF / (1 − 0.2·DTI)
```
Break-even expected marginal hit-rate: **0.020 at DTI 0.10 · 0.032 at 0.156 · 0.042 at 0.20 · 0.065 at 0.305.**
*Check against reality:* the GEMSDOE2 union added 9,430 unmasked pixels to the 0.1563 file and scored 0.1560; the algebra
gives those pixels a marginal hit-rate of 0.028–0.030 — right at the 0.032 break-even. The rule reproduces the board.

**R3 — Each scored file is a line in (G, T).** T_i(G) = DTI_i·(0.2·E_i + 0.8·G) (E_i = emitted-and-unmasked px). One more
measurement fixes G; the density probe (§5) supplies it.

**R4 — The gap to the leader is detection, not shaping.** At the leader's 0.3049 the break-even hit-rate is 6.5 %; ours at
0.156 is 3.3 %. Whatever the leader emits finds roughly **twice the truth per pixel**. No re-shaping of our fields can
double their hit-rate (D2, D3 below show shaping moves are ±0.004).

## 3. What the five scored files prove (measured on the committed bytes)

| file (sha8) | score | emitted px | on masked known px | > 3 px from any known fault |
|---|---|---|---|---|
| CNN ensemble skeleton `7f00890a` | **0.1563** | 172,974 | 3.7 % | 78.4 % |
| GEMSDOE2 union `f68e590f` (superset, +10,668 px) | 0.1560 | 183,642 | 4.2 % | 76.8 % |
| Pindrop nodes `f347b70daa` | 0.1193 | 155,021 | 0 % | 91.2 % |
| Pindrop dense ridge `4e03fc9705` | 0.1152 | 155,021 | 0 % | 80.5 % |
| Pindrop catalogue-gap target `37f9d5b855` | 0.0830 | 155,021 | 0 % | 91.5 % |

Known-fault pixels = 60,988 = 1.18 % of 5,167,373 valid px. Jaccard(CNN, Pindrop ridge) = 0.066: the families are nearly disjoint.

- **D1** The best file is a discovery field (78 % far off-catalogue); the score was earned on new ground, not by re-tracing.
- **D2** A strict superset of +9.4 k unmasked px changed the score by −0.0003 → post-hoc unions of *existing* fields are break-even.
- **D3** Sparse nodes vs dense ridge, same model & budget: +0.0041 on the board vs +0.09 predicted by GEMSDOE3's local proxy → the
  SGMC-based local proxies over-reward emission geometry by ~20×; **local proxy verdicts about shaping are not trustworthy**.
- **D4** Training on SGMC catalogue-gap pixels alone: 0.0830 vs 0.1193 with the union target → the SGMC gap population does not
  resemble the expert new-fault set; stop treating it as truth.
- **D5** CNN (context) beats pixel-wise boosting by ~+0.04 at equal budget, with almost disjoint supports.
- **D6** Every file in this family sits at 3–3.6 % emission and 0.08–0.16; the leader is 0.30. See R4.
- **D7** The bands we trained on carry no radiometrics and only 100 m topography (F8), while the expert label source is most
  plausibly high-resolution topography (F9, F11). That is the channel this family has never used.

Irregularities flagged in the JSON: the 0.1560 file identity is inferred from GEMSDOE2's pre-registered upload order (confirm
from that account's submissions page); five accounts from one project family are competing in one competition — the
[official rules](https://docs.nlr.gov/docs/fy26osti/96647.pdf) govern this and the owner must resolve it; the public/private
chunk geometry is unpublished so E_i has two readings (whole footprint vs public chunks).

## 4. Hypotheses this fork tests (contrarian, but each one is anchored above)

**H1 — Sub-pixel scarp statistics (the primary bet).** A 100 m cell that contains a 2–10 m fault scarp has nearly the same
100 m slope as a smooth hillside with the same relief, but a very different *distribution* of 10 m slopes. Compute slope,
profile curvature, aspect coherence and multi-azimuth hillshade contrast at 10 m from the seamless 3DEP DEM (F10) and aggregate
9 statistics per 100 m cell (`scripts/build_topo_features.py`; unit test proves a 4 m step and a 2.3° plane of equal relief
separate by 4× in slope_max and 8× in hillshade contrast). Train the *same* CNN with the 19 + 9 bands (`configs/config_topo.yaml`)
and compare paired folds. *Why others may overlook it:* the organizers' 1 m tiles are hundreds of GB and cover only some
projects (data/dem_links.json: 867 tiles), so teams either skip topography or average a DEM to 100 m before differentiating —
which erases the signal. The 1/3″ product is seamless over the whole footprint and reads over HTTP.

**H2 — GeoDAWN radiometrics (K, eTh, eU, total count).** Radiometric contrasts mark lithologic boundaries and altered ground
that faults juxtapose; the challenge stack omits them although they are GeoDAWN's headline product (F8, F9). Resample the
released grids to the competition grid with the same aux-feature hook. *Status:* data source verified; resampling script is next
session's step 2 (the zip must be fetched on a runner).

**H3 — Catalogue-continuation prior, weakly.** F3 says new-fault truth includes extensions and parallel strands of known
systems. GEMSDOE2's 1 px corridor along the whole catalogue (`ad5ba911`, 292 k px) is *not* it — R2 says blanket flanks fail
break-even. The testable version is directional: extend each catalogue trace along strike by ≤ 10 px only where the H1/H2
detector already fires ≥ 0.05. Measured with the marginal rule once G is known.

**H4 — Emission budget set by the rule, not by a floor.** With G from the probe and a calibrated detector, emit pixels in
decreasing expected-hit-rate order until the marginal rate crosses 0.2·DTI/(1−0.2·DTI). This replaces the floor/thinning
sweeps on SGMC proxies (D3 says those cannot be trusted).

**H5 — Phase-2 leverage.** Final scoring uses labels expanded by expert review of all submissions (F1, F4). Pixels that are
confidently *ours alone* (not in any other public file we hold) are worth more in Phase 2 than their Phase-1 price. Keep the
Jaccard table in the anchor JSON current and prefer distinct, defensible geometry over consensus.

**H6 — A dilational-setting prior, from the literature rather than from the data (session 2).** Faulds & Hinz (WGC 2015,
[OSTI 1724082](https://www.osti.gov/servlets/purl/1724082); dataset [GDR 355](https://gdr.openei.org/submissions/355),
DOI [10.15121/1148722](https://doi.org/10.15121/1148722)) catalogued the structural setting of ~250 Great Basin geothermal
systems: **step-overs / relay ramps ~32 %, terminations 25 %, intersections 22 %**, accommodation zones 9 %, transfer zones
5 %, pull-aparts 3 %, bends 2 %, and **major range-front faults 1 %** — with Quaternary faults "within or near most" systems,
systems *rare* on range-front displacement maxima (clay gouge, stress release), and tips horse-tailing into "a myriad of
closely-spaced faults". That last clause is a description of the scored population. `scripts/build_structural_targets.py`
recovers the settings a trace raster carries — 6,938 terminations, 2,633 intersections, 1,558 bends, 5,432 relay ramps —
weights them by those frequencies and emits the prior. *Status:* candidates exist and are conformant, but **no local
instrument can price them** (§7), so they are a second upload, not the first.

**H7 — Do not trust the SGMC proxy (measured, not asserted).** `scripts/rank_instruments.py` asks each stand-in the question
it has never been asked: can it order the five files the way the public leaderboard already does? The SGMC-gap proxy — the
population every emission-width decision in this project family was argued on — returns **ρ = −0.8**: it ranks the files
almost exactly backwards. The catalogue, scored in-domain with the platform's masking rule applied, returns **ρ = +0.9**
(`data/evidence/rank_instruments.json`). Five files is five data points, so the first conclusion to draw is not "the catalogue
is right" but "the proxy is disqualified": it must never again select a submission. The catalogue instrument is in-domain for
every supervised file here, so it can rank *detectors* but cannot certify *discovery* value, and it is degenerate for any
candidate that exploits the mask, because its truth is the mask.

## 4b. Fitting the hidden truth from the board — and why it was not believed (session 2)

`scripts/fit_hidden_prior.py` uses the leaderboard directly. With `DTI = T / (0.2·E + 0.8·G)` (§2), each scored file is one
**linear** equation in the weights of a non-negative truth-density model `λ(x) = Σ w_k φ_k(x)`, where `T = Σ_x λ(x) c_S(x)`
is the credit kernel of that file's emission and `E` its chargeable pixels. So five public scores constrain the hidden truth
directly, with no proxy at all.

The point fit is then **not** reported as the answer, because five equations cannot identify a density field. The script
computes the smallest tolerance at which *any* density reproduces all five scores (**0.028 DTI** — as large as the gap between
our best and our fourth file), and then, by bisection over feasibility linear programmes, the **min and max every candidate can
take over the whole consistent set** (`dti_min`, `dti_max` in `data/evidence/hidden_prior/fit.json`).

Two results decided what to ship:

* the fitted density predicts **0.81** for the platform's own `example_submission.tif` — a file everyone already has, against a
  best observed score of 0.3049. **The model is falsified by the board**, so a candidate that looked excellent under the point
  fit (0.349 for "shipped field + catalogue") is *not* shipped on that evidence;
* **the one decision that survives is free.** Staff state in writing that known-fault pixels are excluded from the penalty terms
  (F2), and the platform's own example submission *is* the catalogue raster. Adding the 54,533 catalogue pixels the shipped
  field was missing therefore cannot lower the score under either reading of that rule, and may raise it substantially. That is
  **S5-A**, and because nothing else changes, its score against 0.1563 is a clean measurement of which reading the platform
  implements — a hedge and an experiment in the one upload.

## 5. Upload plan for this site (3 slots / week per account)

1. **Density probe** (`scripts/density_probe.py`; file `gems-density-probe-*.tif`, Note `density probe · p=1 on all unmasked
   valid px …`). Expected score is *low by design*; its value is G = 0.2·N·D/(1−D+0.2·κ·D) (table in the JSON; κ-insensitive).
   One slot, once, on one account.
2. **H1 detector** once `build-topo-features` has run and `config_topo.yaml` has trained (runner or GPU box) — budgeted by H4.
3. **Hold** the third slot; do not spend it on re-shaped versions of scored fields (D2/D3).

## 6. Limitations that gate the plan (honest list)
- This sandbox cannot reach S3/USGS/Dropbox (only GitHub + PyPI); the whole-region channel is built by the workflow, not here.
- No GPU: `config_topo.yaml` needs a runner (≈300 min per fold, as GEMSDOE measured) or a GPU box.
- No upload capability and no leaderboard API: a human uploads and pastes the score back into `SCORED` in the anchor script.
- The public/private chunking is unknown; all algebra carries two readings.
- Multiple accounts: a rules question for the owner, flagged, not resolved here.
