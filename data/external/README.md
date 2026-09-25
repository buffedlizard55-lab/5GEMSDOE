# data/external — auxiliary feature rasters on the competition grid

Files here are **built, never committed** (hundreds of MB). Each must sit on the EXACT competition grid
(3292×3730, EPSG:32611, 100 m, transform 243350/4508550) — `src/dataset.py:_maybe_add_aux_features` refuses anything else.

| file | producer | where it runs | how it is used |
|---|---|---|---|
| `topo_features_100m.tif` (9 bands) | `scripts/build_topo_features.py` from the public 3DEP 1/3″ DEM VRT | `.github/workflows/build-topo-features.yml` (artifact `topo_features_100m`) or any machine that reaches S3 | `configs/config_topo.yaml` → `data.aux_feature_paths` |
| `topo_features_100m.report.json` | same | same; copied to `data/evidence/topo/` by the workflow | rendered on `docs/strategy.html` §6 |

Download the artifact from the Actions run and place it here before `python -m src.train --config configs/config_topo.yaml`.
