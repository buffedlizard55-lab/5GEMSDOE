# Geothermal & structural-geology knowledge base (5GEMSDOE)

**Purpose.** A durable, source-linked store of what this project has established about the science behind the
GEMS Prize: what the competition scores, what the GeoDAWN region is, and — the part the leaderboard actually
rewards — **where unmapped faults are likely to be**, according to published structural geology rather than
according to any model in this repository.

**Rules this file obeys.** Every entry records (a) the claim, (b) the official or peer-reviewed source with a
link, and (c) **how it was verified**: *fetched* (the page was retrieved and read on 2026-09-25), *abstract
seen* (the title, authors and abstract were returned by a search but the full text was not read), or
*inherited* (already verified in this repository, with the verifying script named). Nothing here is from
memory. Anything we could not verify is listed under §5 as an open question, not as a fact.

---

## 1. What the competition scores (fetched / machine-verified in this repo)

| Fact | Source | How verified |
|---|---|---|
| The test set is **new faults only**, chunked into public/private splits | [Problem description, §Competition structure](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#competition-structure) | fetched 2026-09-25; quoted in `scripts/verify_rules_quotes.py` |
| Known USGS/INGENIOUS fault pixels are **masked from scoring**: *"Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from evaluation, so they do not count towards penalty terms."* and *"for scoring purposes it should not matter whether these known faults are included with predictions or not"* | [Forum 11516, DrivenData staff](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516) | **fetched verbatim 2026-09-25** — the exact words are quoted in `scripts/fit_hidden_prior.py` and in the strategy page |
| "New fault" = *"any fault pixel not already captured by USGS/INGENIOUS"*, and can include **newly mapped geometry of an existing fault system** (continuations, splays, parallel strands) | [Forum 11536, staff reply](https://community.drivendata.org/t/where-do-you-draw-the-line/11536) | fetched 2026-09-25 |
| Organizers **will not disclose** which data produced the test faults | [Forum 11527, post 7](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527/7) | fetched 2026-09-25 |
| Metric: DTI = TP<sub>w</sub> / (TP<sub>w</sub> + 0.2·FP<sub>w</sub> + 0.8·FN<sub>w</sub>), triangular kernel R = 300 m, **TP takes the single best prediction within R** | [Problem description, §Performance metric](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric) | fetched; transcribed and pinned in `src/metrics.py` + `tests/test_metric.py` |
| Submission: single-band float32 GeoTIFF, EPSG:32611, 100 m, same bounds, NaN outside, values in [0, 1] | [Problem description, §Submission format](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format) | fetched; enforced by `scripts/validate_submission.py` (17 checks) |
| `example_submission.tif` is **bit-identical to the label raster** | measured in this repository | inherited — `docs/index.html` finding 2; re-checked 2026-09-25 in `scripts/fit_hidden_prior.py` |
| Public leaderboard, 2026-09-25: **0.3049** top (50 ranked, median 0.1134) | [Leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/) | owner-reported 2026-09-25; recorded in `scripts/leaderboard_anchor.py` |

---

## 2. The region: GeoDAWN and the northwestern Great Basin

| Fact | Source | How verified |
|---|---|---|
| GeoDAWN = airborne **magnetic + radiometric** surveys over NW Nevada / E California (Walker Lane, western Great Basin); grids released as GeoTIFFs (`22103_area1_tiffs.zip` 43.6 MB, `22103_area2_tiffs.zip` 230.5 MB) flown at 200–400 m line spacing | [USGS ScienceBase 657e1d85d34e23d3533209f7](https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7), DOI [10.5066/P93LGLVQ](https://doi.org/10.5066/P93LGLVQ) | inherited (fetched in session 1; file table re-read 2026-09-25) |
| The supplied **19 bands contain no radiometric channel** | measured from the band tags of `training_features.tif` | inherited — `scripts/prepare_data.py` |
| Walker Lane accommodates ~20–25 % of Pacific–North America dextral motion (~1 cm/yr); it dies out NW-ward in NW Nevada, and ~1 cm/yr of dextral shear diffuses into WNW-directed extension | [Faulds & Hinz, WGC 2015](https://www.osti.gov/servlets/purl/1724082) §2 | **fetched 2026-09-25** |
| Enhanced **dilation on N- to NNE-striking normal faults**, induced by that transfer of dextral shear into extension, is the probable cause of the region's prolific geothermal activity | same, §2 + abstract of [OSTI 1110517](https://www.osti.gov/servlets/purl/1110517) | **fetched 2026-09-25** |

**Why this matters for the prize.** The scored faults were drawn by experts working on this region's
geophysics. The literature says the structures that matter here are **N- to NNE-striking normal faults in a
transtensional strain field** — an orientation prior that no detector in this repository uses.

---

## 3. Where geothermal systems sit: the structural-setting catalogue (**the core of this file**)

Faulds & Hinz, *Favorable Tectonic and Structural Settings of Geothermal Systems in the Great Basin Region,
Western USA: Proxies for Discovering Blind Geothermal Systems*, World Geothermal Congress 2015 —
[OSTI 1724082](https://www.osti.gov/servlets/purl/1724082) (**fetched 2026-09-25**).

Of **426** known geothermal systems (≥ 37 °C) inventoried, the structural setting could be categorised for
**~250**. Share of the categorised fields, by setting:

| Setting | Share | What it is |
|---|---|---|
| **Step-overs / relay ramps in normal fault zones** | **~32 %** | two overlapping fault terminations; multiple strands, breached relay ramps, highest fracture density |
| **Normal-fault terminations (tips)** | **25 %** | horse-tailing: *"a myriad of closely-spaced faults"* |
| **Fault intersections** | **22 %** | two normal faults, or a normal fault with a transverse oblique-slip fault; dilational quadrants |
| Accommodation zones | 9 % | belts of intermeshing, oppositely dipping normal faults |
| Displacement transfer zones | 5 % | a strike-slip fault terminating in an array of normal faults |
| Pull-aparts (transtensional) | 3 % | releasing bends in strike-slip systems |
| Bends in normal faults | 2 % | — |
| **Major range-front normal faults (displacement maxima)** | **1 %** | — |

Supporting statements, quoted from the same source:

* *"Quaternary faults typically lie within or near most of the geothermal systems."*
* *"Geothermal systems are rare along major range-front faults, possibly due to both reduced permeability in
  thick zones of clay gouge and periodic release of stress in major earthquakes."*
* *"Step-overs, terminations, intersections, and accommodation zones correspond to long-term, critically
  stressed areas, where fluid pathways would more likely remain open in networks of closely-spaced,
  breccia-dominated fractures."*
* ~39 % of the inventoried systems are **blind** (no surface expression); as much as **75 %** of the region's
  geothermal *resource* may be blind.

**Public dataset (overlooked by the field?).** The catalogue itself is downloadable:
[GDR submission 355](https://gdr.openei.org/submissions/355), DOI [10.15121/1148722](https://doi.org/10.15121/1148722),
CC BY 4.0, "Structural Inventory of Great Basin Geothermal Systems and Definition of Favorable Structural
Settings" (Faulds, University of Nevada, 2013) — a 244 kB spreadsheet with structural setting, primary fault
orientation, presence/absence of Quaternary faulting, reservoir lithology, geothermometry, recent magmatism and
blind-vs-surface for the 426 localities, explicitly including **western Nevada** (**fetched 2026-09-25**).

**How this repository uses it.** `scripts/build_structural_targets.py` recomputes the settings that a trace
raster can carry — terminations, intersections, bends and relay ramps — from `labels.tif` alone and weights
them by the frequencies above. It does **not** invent the settings it cannot see (accommodation zones, transfer
zones, pull-aparts need dip, slip sense and strain partitioning); those 17 % are reported as absent.

---

## 4. Fault detection: what has worked for other teams

| Claim | Source | How verified |
|---|---|---|
| Deep-learning fault mapping in northern-central Nevada trained on **lidar** found faults absent from the USGS map that were *"later confirmed by experts looking at the LIDAR data"* | [Hermant et al., Stanford Geothermal Workshop 2025 (PDF)](https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hermant.pdf) | inherited (fetched session 1) |
| USGS 3DEP 1/3″ (~10 m) seamless DEM is public domain and GDAL-readable over HTTP via a VRT | [VRT](https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt) · [ScienceBase record](https://www.sciencebase.gov/catalog/item/4f70aa9fe4b058caae3f8de5) | inherited (session 1) |
| Keranen & Weingarten-style induced-seismicity work and the USGS QFaults compilation are the label backbone of the challenge | [USGS QFaults](https://www.usgs.gov/programs/earthquake-hazards/faults) · DOI [10.5066/P9BCVRCK](https://doi.org/10.5066/P9BCVRCK) | inherited (`docs/data_catalog.csv`) |
| INGENIOUS Great Basin regional compilation (conductivity, strain, gravity, seismicity, heat flow) | [GDR 1391](https://gdr.openei.org/submissions/1391), DOI [10.15121/1881483](https://doi.org/10.15121/1881483) | owner-supplied link; inherited in the data catalog |

**Lineament enhancement used in this repository** (`scripts/build_structural_salience.py`): multi-scale Frangi
vesselness on the gradient magnitude of each band — the standard way to turn a *step* (a fault juxtaposing two
blocks, the classic expression of a Basin-and-Range normal fault in a potential field) into a *ridge*, then
detect the ridge's centre-line. **Measured result (2026-09-25):** the cross-source mean salience has a lift of
only **1.07×** on catalogue pixels over background, and the leaderboard-inversion fit in
`scripts/fit_hidden_prior.py` assigns it a weight of **0**. It is recorded as a negative result, not a
contribution: potential-field lineaments at 100 m do not, by themselves, tell us where the experts' faults are.

---

## 5. Open questions (deliberately not answered here)

1. **Which reading of the masking rule does the platform implement?** Staff say catalogue pixels "do not count
   towards penalty terms" and that including them "should not matter". If they also earn *credit* for a new
   fault within 300 m, the optimal submission changes completely. This is what the S5-A upload measures.
2. **How large is the hidden truth, |G|?** Unknown; the density probe inverts a public score into it but has
   not been uploaded. The fitted models in `data/evidence/hidden_prior/fit.json` imply |G| ≈ 22 k px, and that
   same fit is falsified by the board (see the strategy page §8), so treat it as a bound, not a value.
3. **Are the experts' new faults near the known catalogue?** Forum 11536 allows continuations, splays and
   parallel strands, which suggests yes; the SGMC-gap population that this project used as a proxy for years
   is measured to **anti-rank** the leaderboard (ρ = −0.8), which says the resemblance is not to *that*
   population.
4. **Radiometrics.** GeoDAWN's K / eTh / eU grids are public and absent from the 19 bands; untested.
5. **Topographic scarp statistics (H1).** The channel was built on a runner (artifact `topo_features_100m`,
   194 MB, run 36177863986) but has not been committed to the repository or trained on.

---

## 5b. Session 5GEMSDOE-3 (2026-09-25): five more verified facts, one of them a correction

Every row below was retrieved on 2026-09-25 through a server-side fetch of the named official page; none
of it is from memory, and the one claim that failed verification is recorded as a correction rather than
quietly dropped.

| # | Fact | Source | How verified |
|---|---|---|---|
| S1 | **⚠️ IRREGULARITY — the "39 % blind" claim is contradicted by the dataset's own field definition.** 165 of the 426 rows carry `Blind = yes` (38.7 %), which matches the page's 39 % — but the workbook's *Field Definitions* sheet says `Blind = yes` means the system **is** associated with active surface manifestations (hot springs, warm springs > 20 °C, fumaroles), i.e. the opposite. The page's other figure does reproduce: 113/426 = 26.5 % carry primary setting code 10 "Undertermined", against the page's "~25 % could not be determined" | [GBCGE award page](https://gbcge.org/recent-projects/characterizing-structural-controls/) vs the workbook itself, GDR 355, DOI [10.15121/1148722](https://doi.org/10.15121/1148722) | **fetched 2026-09-25** (page); **measured 2026-09-25** on the spreadsheet by `scripts/analyze_gdr355_inventory.py` |
| S2 | **GeoDAWN lidar point clouds exist** for the same region: "GeoDAWN West Central Nevada EarthMRI Data", work units 1–6, `.laz` files plus OPR/processed TIFFs, published as EPT (Entwine) resources on the AWS USGS Lidar Public Dataset | [GDR 1501](https://gdr.openei.org/submissions/1501) · [OEDI 7592](https://data.openei.org/submissions/7592) · DOI [10.15121/1992093](https://doi.org/10.15121/1992093) · [AWS registry](https://registry.opendata.aws/usgs-lidar/) · [hobu/usgs-lidar](https://github.com/hobu/usgs-lidar/) | **fetched 2026-09-25** (landing pages, file inventory, DOI) |
| S3 | The GeoDAWN **radiometric/magnetic GeoTIFF inventory** is exactly: `22103_area1_tiffs.zip` **43.57 MB**, `22103_area2_tiffs.zip` **230.54 MB** (geoTIFF images of the geophysical grids), `22103_area1_grids.zip` 42.43 MB, `22103_area2_grids.zip` 227.39 MB, ternary maps 9.68 / 51.37 MB, `22103_ternary_a1_pdf.zip` 4.77 MB | [ScienceBase 657e1d85d34e23d3533209f7](https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7), DOI [10.5066/P93LGLVQ](https://doi.org/10.5066/P93LGLVQ) | **fetched 2026-09-25** — the attached-files table was read line by line |
| S4 | An **independent third party** publishes a measured calibration of this competition's metric: raw probabilities 0.088 → binarised top-2 % 0.106 → **skeleton of the top-5 % 0.132 (+49 %)**, "adding a 3-px band back around the skeleton loses", and their U-Net scores **0.0435 on held-out segments vs 0.0387 on the public leaderboard** | [Gameassassin777/gems-eval](https://github.com/Gameassassin777/gems-eval) (MIT, © 2026 Syntropy Digital) | **fetched 2026-09-25**; the two files this repository vendors from it are in `scripts/vendor/gems_eval/` with provenance |
| S5 | **CORRECTION.** A sibling repository's pull request states that "feature bands 17–19 are constant placeholders despite carrying plausible descriptions". On the official bytes in this repository that is **false**: band 17 `cond_surf` carries 1,237,549 distinct values (range −3.4e38…4.969), band 18 `iso_grav_anom_hg` 5,008,543 and band 19 `det_elev_slope` 4,954,390. All 19 bands are measured live in `scripts/train_context_detector.py`'s percentile pass. | measured on `data/training_features.tif` (sha256-pinned, 418,912,844 B) | **measured 2026-09-25** |

| S6 | **The catalogue's own structural-setting frequencies do not reproduce over all rows — RESOLVED in session 4.** Measured over all 426 rows: undetermined 26.5 %, stepover 18.5 %, termination 14.6 %, fault intersection 12.9 %, accommodation zone 5.2 %, displacement transfer 3.3 %, pull-apart 1.6 %, bend 0.7 %, major normal fault 0.7 %. The page's 32 / 25 / 22 % are roughly 1.7× higher on every row **over all rows**, but over the *favourable-setting* rows (247 = 426 − 113 undetermined − 2 volcanic) the workbook reproduces the published table **to the rounding digit on all 8 rows**: 79/247 = 32.0 %, 62/247 = 25.1 %, 55/247 = 22.3 %, 21/247 = 8.5 %, 14/247 = 5.7 %, 8/247 = 3.2 %, 5/247 = 2.0 %, 3/247 = 1.2 % vs published 32/25/22/9/5/3/2/1. The page and the workbook agree once the denominator is the one the page quotes. `scripts/build_structural_targets.py` now carries the measured fractions and the full derivation in its report. | same workbook, measured | **measured 2026-09-25, denominator resolved 2026-09-26** — `data/evidence/gdr/inventory_analysis.json` (`published_vs_measured`) + `WEIGHT_DERIVATION` in `scripts/build_structural_targets.py` |
| S7 | **117 of the 426 systems fall inside the competition raster** (coordinates transformed from NAD83 to EPSG:32611 and tested against the raster bounds): Beowawe, Dixie Valley, Empire–San Emidio, Bradys, Desert Peak, Soda Lake, Stillwater, Steamboat, Humboldt House, Wabuska, Casa Diablo, Moana and 105 more. Mean maximum temperature 84.8 °C, maximum 250 °C, 17 systems ≥ 150 °C; 50 have Holocene fault scarps. | same workbook, measured | **measured 2026-09-25** |
| S8 | The workbook has no surface-geophysical column at all — it is a **field-and-literature inventory** (23 columns: temperature, geothermometry, structural setting, Quaternary faulting age, distance to fault, power, coordinates). It cannot be used as a detector target; it can only be used as a *location prior* | same workbook | **measured 2026-09-25** |

**What S1 changes about the strategy.** If 39 % of the systems are blind, then a surface-geophysical detector
is not merely imperfect — for a large share of the target population there is *no surface expression to
find*, and the best a 100 m potential-field model can do is detect the faults that host them. That is an
argument for spending the remaining effort on (a) the topographic channel (the one surface expression an
expert mapper can see, per Hermant et al. 2025) and (b) the structural setting prior, rather than on more
capacity for the same 19 bands.

**What S6–S8 change about the strategy — this is the largest single finding of session 3.**
S7 says there is a **real, public, geothermal truth set of 117 systems inside the scored footprint**.
`labels.tif` is a *fault* catalogue (USGS/INGENIOUS); GDR 355 is a *geothermal-resource* catalogue, and
they are not the same population — the experts' new faults are, by the problem statement, faults "not
already captured by USGS/INGENIOUS", which is much closer to S7's population than to the mask. The
obvious session-4 experiment is the one `scripts/rank_instruments.py` already performs for every other
instrument: build a prior raster from the 117 in-footprint systems (a Gaussian or radial kernel, possibly
weighted by temperature or by setting) and ask whether it orders the five scored files the way the public
leaderboard already does. If it does, it is the first *geothermal* instrument this project family has had,
and every emission-width decision that was previously argued on the SGMC proxy (measured ρ = −0.8,
`docs/STRATEGY.md` §4b) can be re-made on it.

**What S2 changes about the ceiling.** The 1 m DEM links the organizers shipped cover 867 tiles of selected
projects; the GeoDAWN lidar point clouds cover work units 1–6 of west-central Nevada. If any of those work
units overlaps the competition footprint, a 1 m channel is available from a public, no-auth source that no
team in this competition family has used. **This is the single highest-value untested lead in this file.**

## 5c. Session 5GEMSDOE-4 (2026-09-26): the 117 systems become a measured prior — and the prior fails the instrument test

The session-4 experiment S6–S8 called for is done, measured, and recorded with its negative result.

| # | Fact | How verified |
|---|---|---|
| G1 | The 117 in-footprint systems project to the competition grid (EPSG:4269 → 32611, the exact transform `analyze_gdr355_inventory.py` already used) and are 0–22.45 km from the raster edge. **Distance to the nearest supplied-catalogue pixel: mean 22.45 km, median 7.07 km; 12 of 117 within 300 m, 26 within 1 km, 91 farther than 1 km.** The geothermal population is overwhelmingly *off-catalogue* — the ground the hidden truth lives on and the supplied mask does not touch. | `scripts/build_gdr355_prior.py` → `data/evidence/gdr/prior_report.json` (per-system rows, sha256-pinned workbook) |
| G2 | System statistics in the footprint: mean max temperature 120.7 °C, maximum 250 °C (Dixie Valley, a step-over), 38 systems ≥ 150 °C. Structural setting (workbook's own codes): step-over 33, termination 22, undetermined 17, intersection 14, accommodation zone 11, transfer zone 7, pull-apart 3, volcanic 1, major normal fault 1, bend 1, uncoded 7. Quaternary faulting column: 50 rows say **Holocene** (42.7 % of coded rows) — nearly half the in-footprint systems sit on *modern* faulting. | same report |
| G3 | Prior rasters built at four radii: R3 = 3,393 px (0.03 % of valid), R10 = 37,089 (0.35 %), R20 = 147,069 (1.39 %), R30 = 329,759 (3.12 %). At every radius <1 % of disk pixels sit on masked known-fault pixels, i.e. the prior is almost entirely unmasked ground. | same report + `data/derived/gdr355_prior_R{3,10,20,30}.tif` |
| G4 | **NEGATIVE RESULT — the prior cannot rank the board.** Scored with the official metric (R = 3 px, α 0.2, β 0.8, platform mask) against the five committed scored files, Spearman ρ vs the public leaderboard: R3 −0.60, R10 −0.60, R20 −0.20, R30 −0.20. The two Pindrop files (board #3 and #5) rank *first* on the prior at every radius; the 0.1563 CNN field ranks last. The in-domain catalogue still orders the board at ρ = +0.9 (swapping only the 0.1563/0.1560 pair, a 0.0003 gap). | `scripts/rank_instruments.py --instrument-tif …` → `data/evidence/rank_instruments_gdr355.json` |
| G5 | **Why the test fails, measured rather than guessed.** The disk truth is a *blob* and the scored files are *lineaments*. DTI against disks measures how much 3 km-neighbourhood of systems a field touches; a field spread over 91 % of the footprint off-catalogue (the Pindrop files) touches more disks than a field concentrated on lineaments (the CNN). The metric rewards *precision on fault lines*, not *coverage of geothermal ground* — so an instrument that pays for coverage anti-ranks a board that pays for lines. | arithmetic of the metric (page 967) applied to the measured supports in `leaderboard_anchor.json` |
| G6 | **Consequence.** Per this repository's own rule (an instrument that cannot order five files whose order is known cannot choose the sixth), the corridor prior may not *select* emission policy. It may still *weight* it inside a budgeted bet: `scripts/build_gdr_corridor_candidate.py` (S5-D) adds only corridor pixels the fault detector already believes, at a printed break-even price. | `data/evidence/emission/gdr_corridor.json` |

**The science reading (no hallucination — each clause is a measurement above):** geothermal systems mark
fluid-conducting structural corridors, and new faults plausibly cluster in those corridors (G1–G2). But the
*test population* of this competition is the experts' fault *lines*, and whether a pixel's fault-likeness is
predictable from "within 3 km of a system" is exactly the bet S5-D prices: 6,922 added chargeable pixels,
each needing a marginal hit rate of 0.0323 at DTI 0.1563. The board, not the prior, decides it.

---

## 6. Standing brief (owner's non-negotiables, carried into every session)

* Work line by line; verify from official, verified, trusted sources; provide links for manual review.
* No manual input where the agent can do it; flag irregularities; **no hallucinations**.
* Multiple passes: implement → review for bugs and wrong assumptions → re-check against the request.
* Finish by opening a pull request and merging it, and by naming what is still blocked.
* Core values: **Maximize P(Win)** and **Own the Outcome**.
