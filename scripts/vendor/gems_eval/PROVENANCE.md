# Vendored third-party code: `gems_eval` — an INDEPENDENT implementation of this
# competition's metric and submission rules.

## What it is and why it is here

`dti.py` and `validate.py` are copied **unmodified** from

    https://github.com/Gameassassin777/gems-eval      (MIT License, (c) 2026 Syntropy Digital)

fetched 2026-09-25 through `https://codeload.github.com/Gameassassin777/gems-eval/tar.gz/refs/heads/main`
(the only egress this sandbox has besides PyPI).  `LICENSE` and `README.upstream.md` are the
upstream files, unmodified.

The repository's own metric is `src/metrics.py`.  An implementation that shares an author with
the thing it checks cannot catch a misreading of the problem page, and the whole point of this
project is that every number is either measured here or linked to an official source.  So the
submission gate and the metric parity check are run against BOTH implementations:

* `scripts/validate_submission.py` (ours, 17 checks) and
* `gems_eval.validate.validate_submission` (theirs: single band, float32, same shape/CRS/transform,
  values in [0,1], NaN outside the study polygon, **finite inside it**, and a plausible-coverage
  band that rejects an all-zero or all-one file).

Their validator's `finite_inside_polygon` / `nan_outside_polygon` pair is an independent
statement of exactly the rule the DrivenData upload form enforces as
`Predicted values must be in range [0, 1]` — which is the rejection this repository root-caused
on 2026-09-24 (`src/submission_io.conform_to_template`).

## What the upstream README independently reports (fetched 2026-09-25, not measured here)

* TP saturates per truth pixel: a thick band earns nothing beyond its best pixel per truth pixel
  and pays for every off-line pixel.
* **Thin lines win**: raw probabilities 0.088 -> binarised top-2% 0.106 -> skeleton of the top-5%
  0.132 (+49 %); adding a 3 px band back around the skeleton loses.  This is an independent
  confirmation of this repository's own thinning measurements (`scripts/decide_emission_width.py`).
* The score is dominated by the false-positive term when the scored faults are sparse.
* **Calibration data point**: their U-Net scores 0.0435 on their held-out segments and 0.0387 on
  the public leaderboard - i.e. a spatially held-out local score tracks the board to ~12 %
  relative.  That is the one external evidence that a local instrument CAN rank submissions,
  which the SGMC catalogue-gap proxy in this repository demonstrably cannot (Spearman rho = -0.8,
  `data/evidence/rank_instruments.json`).

None of those numbers is reproduced here; they are cited as an external measurement with its
source, which is the only honest way to use them.
