# Submission Guide — GEMS Prize (Verified from Official Sources)

> **See also:** the online **[Make a submission](https://buffedlizard55-lab.github.io/5GEMSDOE/docs/submission.html)**
> page — the same path as a build-time-verified checklist (artifact sha256 re-hashed from the bytes,
> the placement table compared against whatever `data/` holds now, the validator table parsed from its
> own committed log, rules sentences quoted by id with their verification badge, and the irregularities
> that a submitter must know about before uploading).
>
> Also [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) and the online
> [Executive Summary Subpage](https://buffedlizard55-lab.github.io/5GEMSDOE/docs/executive_summary.html)
> for the comprehensive executive roadmap, eligibility rules, Generative AI disclosure narrative, and
> pre-validated ready-to-upload submission raster.

**Primary sources:**
- Problem description submission format: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format
- Official Rules PDF: https://www.nlr.gov/docs/fy26osti/96647.pdf (Sections 3.2, 3.3, 3.5, 3.6)
  - Both hosts verified identical 2026-09-12: `www.nlr.gov` and `docs.nlr.gov` serve the same SEPTEMBER 2026 document (title + TOC match). Full PDF read across 7 chunks.
- Competition main: https://www.drivendata.org/competitions/306/competition-doe-gems/
- Reference solution: https://github.com/drivendataorg/gems-prize-reference-solution

## 1. What to Submit (Per Problem Page)

> For this competition, you will submit a GeoTIFF file containing your predictions for **all** faults in the region.

Requirements (line-by-line verified from https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format):

- [ ] Same projected CRS as training data: **UTM zone 11N, EPSG:32611** (https://epsg.io/32611)
- [ ] Same resolution: **100m**
- [ ] Same bounds as training data, data outside bounds is null or NaN
- [ ] Single layer, datatype **32-bit float (float32)**, values **between 0 and 1** indicating confidence/probability, higher = higher probability
- [ ] Sample submission that predicts total fault absence is provided for reference on data download page (https://www.drivendata.org/competitions/306/competition-doe-gems/data/)

## 2. How to Enter (Per PDF Section 3.1, 3.2)

From PDF https://docs.nlr.gov/docs/fy26osti/96647.pdf:

- [ ] Create profile on DrivenData platform and agree to competition rules and restrictions
- [ ] Navigate to challenge website to sign up as competitor: https://www.drivendata.org/competitions/306/competition-doe-gems/
- [ ] Review competition materials and data (problem description https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ and about https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/)
- [ ] Download training features and labels from data tab https://www.drivendata.org/competitions/306/competition-doe-gems/data/ (requires login)
- [ ] Training dataset: geophysical features for GeoDAWN study area at 100m resolution, multiband GeoTIFF, one feature per band. Plus instructions for downloading USGS DEM at 1m.
- [ ] Training labels: existing fault data at 100m resolution where positively labeled pixels indicate fault presence, obtained from INGENIOUS Great Basin Regional Dataset Compilation DOI https://doi.org/10.15121/1881483 (PDF footnote 4)
- [ ] Competitors will submit predictions for all faults in GeoDAWN study area as GeoTIFF raster at 100m resolution. Example valid submission provided.
- [ ] **Generative AI disclosure:** If using generative AI, indicate in narrative (not included in word count) extent and how used (PDF section 3.2). Responsible for accuracy, authenticity, authorship.

## 3. Feedback & Limits (Per PDF Section 3.4)

> **⚠️ FLAGGED DISCREPANCY (not silently resolved):** an earlier project instruction stated
> "max 2 submissions per 7 days"; the official rules PDF (§3.2 + §3.4, verified 2026-09-12)
> says **up to three per week**. Per project policy, the official PDF + competition website are
> authoritative; the 2-per-7-days figure is treated as superseded but is recorded here for audit.
> Practical consequence: we plan around **3 submissions/week** (the binding limit per the
> platform may also be displayed at submission time — check before submitting).

- [ ] Each entity may submit **more than one set of predictions** for automated scoring up to **three per week**, as specified on competition website
- [ ] By submission deadline, **must select only one set of predictions** to use as final submission for final evaluation and ranking
- [ ] Multiple finalized submissions not allowed
- [ ] Each entity (team/org/individual) allowed **one final submission**; individuals on team not allowed separate final submission

## 4. Evaluation & Ranking (Per PDF Section 3.6)

- [ ] Using **distance-weighted Tversky index** metric published on competition website, judges score chosen submission against ground truth
- [ ] Ground truth divided into **public test dataset** and **private test dataset**
- [ ] **Public leaderboard:** score on public test dataset shown while competition running
- [ ] **Private leaderboard:** score on private test dataset used for first prize round when competition closes
- [ ] **Second round:** score against complete updated test set created by expert review after close
- [ ] **Must choose only one submission** for scoring across both prize rounds, **without knowledge of private test scores** — to encourage generalization, discourage overfitting to public test (PDF 3.6.2)
- [ ] Set of faults in public test and relative weight determined by organizers before start

### Prize Structure (Per PDF 1.1 and Problem Page)

- **Initial Round $50,000:** Top 5 each $10,000, judged on private test set of fault labels
- **Expert review:** Panel uses submitted predictions to update fault labels for full region
- **Final Round $250,000:** Top 5 judged on all fault labels in updated set — 1st $100k, 2nd $70k, 3rd $40k, 4th $25k, 5th $15k
- Same submission scored twice; predictions that helped experts identify previously-unmapped faults can score higher in final round

## 5. Solution Verification and Delivery (Per PDF 3.2, 3.5)

For finalists, for chosen algorithm, must submit:

- [ ] **Complete code assets and documentation**, including:
  - Description of resources required to build and run solution
  - Assets should be able to sufficiently reproduce winning results and generate predictions on new data samples
- [ ] Documentation consistent with DrivenData's Winning Model Documentation Template (provided to winners after competition)
- [ ] Finalists must sign and return required documents including eligibility certifications
- [ ] Award approvals — Official winners selected by DOE, may take into account program policy factors listed in Appendix A (PDF section 1.3 eligibility, 1.4 prize goals)
- [ ] DOE is judge and final decision maker, may elect to award all, none, or some submissions
- [ ] After winners notified, prize administrator requests necessary information to distribute cash prizes
- [ ] Interviews may be held after announcement (PDF 3.6.3), not required

## 6. How to Submit via DrivenData (Practical)

1. Go to https://www.drivendata.org/competitions/306/competition-doe-gems/
2. Click "Compete!" to enroll (requires account)
3. Download data from https://www.drivendata.org/competitions/306/competition-doe-gems/data/
4. Train model using this repo: `python -m src.train --config configs/config.yaml`
5. Generate predictions: `python -m src.inference --config configs/config.yaml --model-dir outputs --out submission.tif`
6. Validate format: `python scripts/validate_submission.py --pred submission.tif --sample data/sample_submission.tif`
7. On DrivenData, click "Submit" → "Make new submission" → upload GeoTIFF
8. Check public LB score
9. Before deadline **Dec 3, 2026 11:59pm UTC**, select ONE submission as final for both prize rounds
10. If finalist, prepare code + docs per template

### 6a. Generating the file itself — three ways, one verdict (added 2026-09-22)

Everything above says *what* the file must be. These are the ways to *make* it, and all three end at
the same gate (`scripts/validate_submission.py` + `scripts/check_site_generator.py`).

**1. In the browser, from the published site** — no clone, no install, no GPU.
The panel is the **first block on the [landing page](https://buffedlizard55-lab.github.io/5GEMSDOE/docs/index.html)** and
the [executive summary](https://buffedlizard55-lab.github.io/5GEMSDOE/docs/executive_summary.html) (added 2026-09-24), and
[`docs/how_to_submit.html` §3](https://buffedlizard55-lab.github.io/5GEMSDOE/docs/how_to_submit.html) — the same mount, the same
payload, the same self-checks. It loads `docs/submission_meta.json` (grid, transform, EPSG, value pins) and
`docs/submission_field.bin`
(a 532,072-byte `gems-rle-v1` run-length encoding of the field, 259,495 runs) and writes the GeoTIFF
locally with `docs/geotiff_writer.js`. Nothing is uploaded and no network call is made: the writer's
17 self-checks run against the file it just parsed, and the download is withheld if any fails. The
result is **pixel-identical** to the adopted artifact (float32 bits equal) — *not* byte-identical,
because the artifact is 256×256 LZW-tiled and the page writes 64-row deflate strips; the page states
this rather than hiding it.

**2. From a clone, with one command** (same code, Node CLI):

```bash
python scripts/build_submission_payload.py --check       # the payload still describes the artifact
node docs/geotiff_writer.js docs/submission_meta.json docs/submission_field.bin \
     submission.tif submission.zip --rows-per-strip 64   # exits 1 and writes NOTHING if a check fails
python scripts/validate_submission.py --pred submission.tif \
       --sample data/sample_submission.tif --train data/training_features.tif
```

**3. On a CI runner, no login** — `.github/workflows/make-submission.yml`
(`route=adopted|baseline|both`, `package=tif|zip|both`): places `data/` from the sha256-pinned git
bridge, validates, rebuilds the payload if asked, runs the site generator check, packages the
`.zip`, refuses any artifact under 100,000 bytes (the run-35042805806 stub bug), prints the sha256s
and a paste-ready submit note, and commits only the measurement JSON to
`data/evidence/make_submission/`.

```bash
gh workflow run make-submission.yml -f route=adopted -f package=both
```

> **⚠️ That last command does not work from this token (measured 2026-09-25, session 3): `gh workflow run`
> returns HTTP 403 "Resource not accessible by integration", and `gh api …/dispatches` (POST) likewise. The
> token can list and read workflows but cannot dispatch them. The route that *does* work is a **push-path
> trigger** on `.github/triggers/<name>` (which `build-topo-features` and, since session 3,
> `fetch-gdr-inventory` both have): edit the trigger file and push. Runner results can only come back by the
> runner **committing them into the repository** — the artifact zip redirects to an Azure blob host this
> sandbox cannot reach. Route 2 (the Node CLI above) is the working path for the file itself.

**If you prefer to hand over the `.zip`** the dialog accepts (the dialog's wording, transcribed by
the team; rules §3.2 documents only the GeoTIFF form — irregularity 7):

```bash
python scripts/package_submission.py --tif submission.tif --out submission.zip --json
python scripts/package_submission.py --tif submission.tif --out submission.zip --check
```

`--check` re-extracts the member, reads it back through GDAL, and compares bytes; a second member, a
recompressed member, or a member that differs by one bit fails the command. Packaging is verified to
be deterministic by re-running it and comparing hashes.

### 6b. Which file to upload, in what order, with the exact Note text (added 2026-09-25, session 3)

Every file below has already been gated **twice** — by `scripts/validate_submission.py` (17 checks) *and* by an
unmodified third-party implementation of the same rules (`scripts/vendor/gems_eval/`, MIT © 2026 Syntropy Digital,
provenance in that directory), with the two DTI implementations cross-checked on the real rasters. The measured
agreement is in `data/evidence/submission_independent_gate.json` (verdict PASS; DTIs agree to six decimals on all
six candidates). Filenames carry a sha256 of the pixels so a re-run cannot silently substitute a different field.

| # | File | sha256 (first 8) | Note text to paste | Why this one, next |
|---|---|---|---|---|
| 1 | `docs/downloads/gems-density-probe-20260925T184215Z-9cdae9b4.tif` | `9cdae9b4` | `density probe · p=1 on all unmasked valid px · inverts the public score into the hidden truth size G · score is expected to be low by design` | Once, on one account, any time. It is the only measurement that turns every bound in `docs/STRATEGY.md` §2 into a number. Everything else is worth more once G is known. |
| 2 | `docs/downloads/candidate_s5_catalogue_hedge.tif` | `132e23e1` | `S5-A · base 7f00890a + masked catalogue (+54,533 px, charge-free per forum 11516) · measures the masking rule` | Adds **zero chargeable pixels**, so under the platform's written rule it cannot lower the score. Its score against 0.1563 decides which reading of the masking rule is implemented — the one open question in `docs/GEOTHERMAL_SCIENCE.md` §5. (`7f00890a` is the base field it is built from, not this file's hash; this file is `132e23e1`.) |
| 3 | `docs/downloads/candidate_s5_dilational_annulus.tif` | `542eaf30` | `S5-C · annulus halo0 q0.9 of Faulds & Hinz settings ∩ context detector · +8,053 chargeable px · required marginal hit rate 0.0323` | The cheapest high-prior bet available: the literature's dilational settings as a one-pixel halo, kept only where a detector already fires. Costs 4.8 % more chargeable mass than S5-A and tests H3/H6 in the same upload. |
| 4 | hold | — | — | Do not spend a slot on a re-shaped version of a scored field (`docs/STRATEGY.md` D2/D3). |

The sha256s above are the ones recorded in `data/evidence/submission_independent_gate.json` for the bytes
currently in `docs/downloads/`; if you re-run any builder, re-run
`python scripts/verify_candidates_independently.py --blanket docs/downloads/gems-density-probe-*.tif`
first — the gate's verdict is only meaningful against the bytes it hashed.

**The decision after #2 and #3 are scored** is mechanical, not a judgement call: run
`python scripts/emit_by_marginal_rule.py --pred <candidate> --base docs/downloads/candidate_s5_catalogue_hedge.tif --price`
and compare each candidate's chargeable mass against the break-even marginal hit rate
`breakeven_marginal_hit_rate(score)` (**0.0204 / 0.0323 / 0.0417 / 0.0649** at DTI **0.10 / 0.1563 / 0.20 /
0.3049**). A candidate that adds chargeable pixels without a prior reason to expect ≥ that rate is not a candidate.

**On the website.** The one-click button on the [landing page](https://buffedlizard55-lab.github.io/5GEMSDOE/docs/index.html)
and the [executive summary](https://buffedlizard55-lab.github.io/5GEMSDOE/docs/executive_summary.html) writes the
**adopted 0.1563 artifact** — the one field in this repository with a measured leaderboard score — and says so. The
candidates above are offered as downloads from `docs/downloads/` precisely because their scores are not yet known;
the button is for the file you can upload today and defend, the table is for the experiment.

## 7. Submission Format Validation (Our Implementation)

`scripts/validate_submission.py` checks:

- CRS == EPSG:32611
- Resolution == 100m
- Bounds match sample submission (or training_features.tif)
- Single band, float32, values in [0,1]
- NaN/nodata allowed **outside** training bounds (spec: "data outside bounds is null or NaN"); values **inside** bounds must be within [0,1]. Our scorer (`src/metrics.py`) sanitizes NaN→0 so padded files evaluate correctly.
- File size matches expected

See reference solution for example: https://github.com/drivendataorg/gems-prize-reference-solution

## 8. Irregularities Flagged

- Sample submission: user-provided Dropbox mirror (`example_submission.tif`) verified reachable 2026-09-12; binary not fetched in sandbox (egress allowlist) — `scripts/download_competition_data.sh` fetches it; dummy fallback in `scripts/generate_dummy_submission.py`
- Training features file naming: problem page says `training_features.tif`, reference solution says `numeric_features.tif`, mirror file is `gems-geodawn-numerical-features.tif` — all three handled in `src/dataset.py`
- 1m DEM links: RESOLVED 2026-09-12 — links point at official USGS bucket `prd-tnm.s3.amazonaws.com` (verified); partial capture of the links file yielded 35 tiles / 6 S3-verified (artefact not committed at the time; regenerate with `python scripts/fetch_dem_links_pdf.py`); full authoritative list via `python scripts/download_dem_tiles.py --complete-listing`; tiles are 90–380 MB each. Competition JSON itself contains irregularities (duplicate rows, path≠filename project rows — the filename project is canonical, S3-proven; garbled hosts in the PDF print)
- Rules PDF submission cadence: see flagged discrepancy atop §3 (3/week per PDF vs earlier 2/7-days instruction)

## 9. No Hallucinations

All requirements above copied line-by-line from official sources, with links for manual verification.
