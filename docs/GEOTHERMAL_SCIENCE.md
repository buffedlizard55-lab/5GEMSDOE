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

## 6. Standing brief (owner's non-negotiables, carried into every session)

* Work line by line; verify from official, verified, trusted sources; provide links for manual review.
* No manual input where the agent can do it; flag irregularities; **no hallucinations**.
* Multiple passes: implement → review for bugs and wrong assumptions → re-check against the request.
* Finish by opening a pull request and merging it, and by naming what is still blocked.
* Core values: **Maximize P(Win)** and **Own the Outcome**.
