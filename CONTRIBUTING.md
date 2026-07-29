# Contributing to Trazo

Issues and pull requests are welcome: https://github.com/wri/trazo/issues

For questions about the technical note, the training data, or applying Trazo to a
new region, contact **Tristan Grupp — tristan.grupp@wri.org**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip wheel setuptools
pip install -e ".[pt1,pt2,dev]"  # add pt3/pt4/pt5 if you touch those steps
```

Python 3.10, 3.11 or 3.12. Steps 3, 4 and 5 need 3.11 or 3.12 (`ftw-tools`
requires `>=3.11,<3.13`).

## Before you open a PR

```bash
trazo-smoke               # offline end-to-end check, under a minute
pytest -m "not heavy"     # core suite
pytest -m heavy           # only if you have the pt3/pt4 extras installed
```

CI runs the core suite on 3.10, 3.11 and 3.12, plus a job that installs the
training stack and imports every module. That second job exists because a package
rename once left Step 4 unimportable for months and nothing caught it — if you add
a module, add it to `tests/test_imports.py`.

## Conventions

- **CLI entry points** take `argv` (`def main(argv=None)`) so they can be called
  in-process from tests and notebooks, not only from a shell.
- **Every flag a CLI accepts must reach the thing it configures.** A flag that is
  parsed and dropped is worse than no flag at all; there are tests asserting this
  for the Step 3 and Step 4 CLIs.
- **Column names stay at 10 characters or fewer.** The ESRI Shapefile driver
  truncates longer ones silently, and the documented schema has to match what
  users actually get.
- **Loud failure over silent fallback.** Loading a checkpoint that matches no
  parameters, or injecting LoRA that wraps no layers, raises instead of quietly
  training from scratch.
- Docstrings explain *why* a non-obvious choice was made, not what the next line
  does.

## Repository layout

```
src/trazo/
  pt1_createdata/   Step 1: gridding, planting/harvest chips, active sampling
  pt2_dataprep/     Step 2: stacks, masks, parquet, hkl/zarr export
  pt3_finetune/     Step 3: fine-tuning strategies, LoRA, UPGD, merging
  pt4_train/        Step 4: training, models, losses, metrics
  pt5_inference/    Step 5: tile pair selection and inference
  smoke.py          Offline end-to-end install check
tests/              pytest suite
scratch/            Exploratory notebooks from the technical note (not packaged)
validation/         Annotation-quality analyses (not packaged)
```

## License

Contributions are accepted under [CC BY 4.0](LICENSE), the license this project
is released under.
