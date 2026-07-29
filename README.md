# Trazo

<p align="center">
  <img src="https://github.com/wri/trazo/raw/main/assets/fieldscropped2.gif" alt="Field Boundaries" />
</p>

<p></p>

This is the documentation of the code of the WRI technical note that focuses on how to create training data, sample training data, fine tune models, and train models. This documentation of the technical note also has functionality that can serve researchers and others who want to apply their own field boundaries to create custom models. We hope these tools can help those in the agricultural and food sectors make better land use and sourcing decisions.

<p></p>

Trazo includes end to end utilities for creating training data, preparing chips and masks, and running inference for field boundary modeling. This repository builds on the Fields of the World repo: https://github.com/fieldsoftheworld for documenting efforts to scale FTW to new geographies.
This package also has several inference scripts for making model testing easier, such as comparing models on multiple Sentinel-2 tile sites and pulling Sentinel-2 imagery for a user's study area.

<p></p>

This front page gives you a practical map of all five steps. Step 1 will be built out further with additional training data sampling strategies.

Trazo was created as a joint effort between World Resources Institute and the Kerner Lab at Arizona State University, funded by the Walmart Foundation. Significant support was provided by Land and Carbon Lab at WRI.
<p></p>

<b><u>Trazo</u></b> is the Spanish word for brushstroke, from the verb trazar; to trace, to draw, to plot.
By tracing fields by hand in satellite imagery, we can choose the most powerful samples for creating robust, generalizable field boundary detection models. Each sketched field in our dataset adds knowledge about the wide diversity of agricultural systems in South America. These sketches teach models the culture of agriculture and how landscapes differ.

Trazo echoes the English word "trace": to follow a path, uncover origins and make hidden connections legible. Trazo is both about creating powerful, diverse training data and the aim of tracing commodities through the supply chain, so that agriculture can be monitored for deforestation.

**License:** [CC BY 4.0](LICENSE) — use it, adapt it, build on it commercially, just credit the source. See [NOTICE.md](NOTICE.md) for the citation and third-party components.

**New here?** `trazo_walkthrough.ipynb` in the repo root runs the pipeline end to end in a notebook. The five steps below are the reference for each stage.

---

## Install

Python 3.10, 3.11 or 3.12. Steps 1 and 2 work on all three; **Steps 3, 4 and 5 need 3.11 or 3.12**, because `ftw-tools` requires `>=3.11,<3.13`.

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip wheel setuptools

# core install (Step 1.1 gridding + Step 2 data prep)
pip install -e .
```

Then add the extras for the steps you actually run:

| Extra | Step | Adds |
|---|---|---|
| `pt1` | 1.2 planting/harvest imagery | pystac-client, odc-stac, planetary-computer, rasterio, rioxarray |
| `pt2` | 2 data prep | rasterio, rtree, pandas, zarr, hickle |
| `pt3` | 3 fine-tuning and merging | torch, lightning, torchgeo, smp, kornia, ftw-tools |
| `pt4` | 4 training | everything in `pt3` plus ultralytics, torchvision, hickle |
| `pt5` | 5 inference | ftw-tools (provides the `ftw` CLI) |
| `active` | 3.2 active-learning sampling | ftw-tools, torch, rasterio |
| `dev` | tests | pytest |
| `all` | everything above | |

```bash
pip install -e ".[pt1]"           # Step 1.2
pip install -e ".[pt2]"           # Step 2
pip install -e ".[pt3]"           # Step 3
pip install -e ".[pt4]"           # Step 4
pip install -e ".[pt5]"           # Step 5
pip install -e ".[all]"           # everything
```

Dependency versions are pinned to ranges rather than left open, so a run today and a run in two years resolve to compatible stacks.

### Verify the install

Before pulling a single Sentinel-2 scene, check that the install is wired up:

```bash
trazo-smoke
# or: python -m trazo.smoke
```

This builds synthetic fields and synthetic chips, runs the real Step 1.1 gridding and the full Step 2 chain over them, and asserts every expected output exists. It needs the `pt2` extra, touches no network, and finishes in well under a minute. Add `--work-dir ./smoke` to keep the outputs and look at them.

### Console scripts

Installing the package registers:

| Script | Module form | What it runs |
|---|---|---|
| `trazo-pt1-create grid` | `python -m trazo.pt1_createdata.gridding` | Step 1.1 gridding |
| `trazo-pt1-create chips` | `python -m trazo.pt1_createdata.plantingharvest` | Step 1.2 planting/harvest chips |
| `trazo-pt2-dataprep <sub>` | `python -m trazo.pt2_dataprep <sub>` | Step 2 subcommands |
| `trazo-pt3-finetune <sub>` | `python -m trazo.pt3_finetune.cli <sub>` | Step 3 fine-tuning and merging |
| `trazo-pt4-train fit\|test` | `python -m trazo.pt4_train.cli fit\|test` | Step 4 training |
| `trazo-pt5-infer <sub>` | `python -m trazo.pt5_inference.cli <sub>` | Step 5 selection and inference |
| `trazo-active-sample` | `python -m trazo.pt1_createdata.active_sample` | Step 3.2 active-learning sampling |
| `trazo-smoke` | `python -m trazo.smoke` | Offline install check |

Every command accepts `--help`.

---

## Folder conventions

A typical project layout after Step 1 and Step 2, following FTW structure:

```
trazo/
  seasontifs/                         # SOS/EOS GeoTIFFs that ship with the repo
  spatial/sentinel_2_index_shapefile.geojson

  data/
    conab2020/
      conab2020_grid.shp              # user grid (chip_id column)
      window_a/                       # 4 band chips from Step 1 (planting), FTW style
      window_b/                       # 4 band chips from Step 1 (harvest), FTW style
      sized256/                       # optional normalized 8 band chips
      s2_images/window_a/             # split out of 8 band chips, FTW style
      s2_images/window_b/
      label_masks/instance/
      label_masks/semantic_2class/
      label_masks/semantic_3class/
      hkl                             #  hickle version of window_a, window_b, label_masks combined
      region_boundingbox256.geojson
      chips_region.parquet
      S2_best_pairs_summary.csv       # Step 5 selection summary
```

---

# Step 1: Create grids and data windows from grids

### > 1.1 Creating grids from your field boundaries

`gridding.py` — build a standardized FTW-style AOI grid from a field boundary dataset.

### What it does
- Reads an input field boundary file (shp, geojson, gpkg, or parquet).
- Dissolves the input and builds a fishnet grid of square cells over its extent.
- Reprojects to a projected CRS (Brazil Albers by default; override or disable as needed).
- Writes a `chip_id` per cell plus coverage statistics, then saves the grid.

### Outputs
- A spatial file (`<name>_grid.<ext>`) with columns `chip_id` (int), `chip_area`, `cov_area`, `cov_pct`, and `geometry` in the chosen projected CRS.
- A dissolved copy of the input (`<name>_dissolved.shp`).

Column names are all 10 characters or shorter on purpose: the ESRI Shapefile driver silently truncates longer names, so the columns you get from `--output-format shp` are the same ones you get from geojson, gpkg and parquet.

### Run

```bash
# Build a 2.56 km grid (256 px at 10 m) over a field boundary file
trazo-pt1-create grid \
  --input "/path/to/field_boundaries.shp" \
  --output-dir "/path/to/grids" \
  --cell-size-meters 2560 \
  --output-format geojson

# equivalent module form
python -m trazo.pt1_createdata.gridding --input ... --output-dir ...
```

Useful options:

```bash
--target-projected-epsg 32721    # force a specific projected CRS instead of Brazil Albers
--no-brazil-albers               # do not default to Brazil Albers
--input-epsg-if-missing 4326     # assume this EPSG if the input has no CRS
--output-format {shp,geojson,gpkg,parquet}
```

Run `trazo-pt1-create grid --help` for the full list.

After you have these grids, open the grids and your field boundaries in your preferred geometry editing software; ArcPro, QGIS, Collect Earth Online, etc. Fill in ALL fields within each chip. If you do not fill in all fields, you must use presence-only labels, which weight the background (non-fields) and the unlabeled fields with the value `3`. These values are excluded when calculating loss during fine-tuning/training. Presence-only masking will be added to Step 2 later. You can use the output of Step 1.2 as the imagery to label. Be sure that whatever reference year you choose matches when the rest of your field boundaries were produced.

### > 1.2 Creating harvest/planting images for every grid

`plantingharvest.py` — produce two 4 band chips per AOI: a planting window and a harvest window. Selection uses SOS and EOS rasters to target month ranges and prioritizes low cloud cover at the chip level.

- `window_a/<chip>.tif`  B04, B03, B02, B08 at planting
- `window_b/<chip>.tif`  B04, B03, B02, B08 at harvest
- `<chip>__stack8.tif`   optional 8 band stack written during the sweep
- `missing_windows.txt` and `errors.txt` audit logs

### Key features

- Queries the Microsoft Planetary Computer STAC (requires the `pt1` extras)
- Chip-level cloud fraction using the SCL mask
- Target windows around SOS or EOS with fallback expansion
- Uses SOS/EOS rasters from `--season-tifs-dir`, falling back to the bundled `seasontifs/`

### Run

```bash
trazo-pt1-create chips \
  --input "/path/to/your/grid_or_folder" \
  --year-constant 2020 \
  --planting-span-months 2 \
  --harvest-span-months 2 \
  --scene-cloud-threshold-pct 90 \
  --patch-cloud-threshold-frac 0.08 \
  --target-cloud-max-frac 0.01 \
  --preferred-cloud-max-frac 0.02 \
  --chip-size 256 \
  --batch-size 10
```

Notes on inputs:

- Pass `--year-constant <YEAR>` for a single year, or `--year-field <column>` to read the year per feature from the grid.
- Chip IDs come from `--chipid-field`, which defaults to `chip_id` — the column Step 1.1 writes, so the two steps line up with no extra flags. Grids written by older versions carrying `cell_id` or `cellid` are detected automatically and reported in the log; `--fallback-id-field` (default `aoi_id`) is the last resort.
- Bands of interest are B04, B03, B02, B08.
- SOS/EOS rasters are read from `--season-tifs-dir`; if not provided, the bundled `seasontifs/` copies are used.

Run `trazo-pt1-create chips --help` for all options and defaults.

---

# Step 2: Data preparation

These tools convert the raw outputs of Step 1 into standardized chips, masks, and a one-row-per-chip GeoParquet.

### CLI

```bash
trazo-pt2-dataprep <subcommand> [args...]
# or, without the installed script:
python -m trazo.pt2_dataprep <subcommand> [args...]
```

Subcommands:

| Subcommand      | What it does                                       | Main flag |
|-----------------|----------------------------------------------------|-----------|
| `pair-stacks`   | Pair `window_a` and `window_b` into 8 band stacks  | `--window-a-dir`, `--window-b-dir`, `--out-dir` |
| `resize-256`    | Normalize stacks to 256x256 into `sized256`        | `--base-folder` |
| `chips-bboxes`  | Create WGS84 bbox GeoJSON for each 256 chip        | `--folder` |
| `make-masks`    | Make instance and semantic masks, split A and B    | `--base-folder`, `--fields-shp` |
| `chips-parquet` | Build per chip GeoParquet metadata                 | `--base-folder`, `--fields-shp` |
| `scale-u16`     | Convert chips to uint16 in [0, 10000]              | `--base-folder` |
| `export-hkl`    | Export FTW-style `.hkl` files from windows + masks | `--root` |
| `export-zarr`   | Export FTW-style Zarr files from windows + masks   | `--root` |

`--base-folder` and `--folder` can be repeated to process several regions in one call; `--root` takes a single FTW-style root containing country folders. Run `python -m trazo.pt2_dataprep <subcommand> --help` for each subcommand's full arguments.

### Recommended order

1. `pair-stacks` — pair A and B into 8 band stacks
2. `resize-256` — normalize to 256x256 (optional but recommended)
3. `chips-bboxes` — create chip bounding boxes
4. `make-masks` — make masks and split back out to `s2_images/window_a` and `s2_images/window_b`
5. `chips-parquet` — build the per chip GeoParquet
6. `scale-u16` — scale to uint16 (optional, can be run before or after masks)
7. `export-hkl` or `export-zarr` — optional, only if you want FTW-style packed archives instead of loose GeoTIFFs. Step 4 reads either.

### Quick start

Every command below is exactly what `trazo-smoke` runs, so if the smoke check passes these work on your machine.

```bash
# 1. Pair
python -m trazo.pt2_dataprep pair-stacks \
  --window-a-dir /data/region/window_a \
  --window-b-dir /data/region/window_b \
  --out-dir /data/region --overwrite

# 2. Normalize to 256
python -m trazo.pt2_dataprep resize-256 --base-folder /data/region --overwrite

# 3. BBoxes
python -m trazo.pt2_dataprep chips-bboxes --folder /data/region

# 4. Masks and split
python -m trazo.pt2_dataprep make-masks \
  --base-folder /data/region --fields-shp /data/region/fields.shp --boundary-px 1

# 5. Parquet
python -m trazo.pt2_dataprep chips-parquet \
  --base-folder /data/region --fields-shp /data/region/fields.shp \
  --split-train 0.85 --split-val 0.15 --split-test 0.0

# 6. Scale to uint16
python -m trazo.pt2_dataprep scale-u16 --base-folder /data/region

# 7. Optional: pack into FTW-style archives
python -m trazo.pt2_dataprep export-hkl  --root /data --mask-type semantic_3class
python -m trazo.pt2_dataprep export-zarr --root /data --mask-type semantic_3class
```

Notes

- `make-masks` selects features that intersect each chip without clipping, which avoids fake edges along chip borders
- 3 class masks write the boundary as a thin ribbon that overwrites the interior where they overlap
- `chips-parquet` writes exactly one row per chip and embeds a MultiPolygon of intersecting fields for the chip extent
- If `--fields-shp` is omitted, `make-masks` and `chips-parquet` look for a single `.shp` in the base folder

---

# Step 3: Fine-tuning and Active-Learning Sampling

Step 3 covers two related workflows: adapting a pretrained checkpoint to a new region, and intelligently selecting the most informative chips for annotation.

Install with `pip install -e ".[pt3]"` (add `".[active]"` for 3.2).

### > 3.1 Fine-tuning a checkpoint

Fine-tuning updates a pretrained model with region-specific data. It buys target-region accuracy and, done naively, pays for it in **catastrophic forgetting** — the model gets good at your region and bad everywhere else. Trazo ships four fine-tuning strategies and a merging step that recovers most of what full fine-tuning forgets.

```bash
trazo-pt3-finetune run \
  --strategy full \
  --checkpoint /models/ftw_3class.ckpt \
  --data-dir   /data/region \
  --output-dir /models/ft_full
```

With no `--config`, each strategy uses its bundled starter config. `trazo-pt3-finetune configs` prints where they live so you can copy and edit one:

```bash
trazo-pt3-finetune configs
```

**Strategies.** Results are from the technical note's Chiquitania experiment: 76 chips (36 train / 20 val / 20 test), 300 epochs with early stopping at patience 100, U-Net with an EfficientNet-b3 backbone. The FTW baseline scored 42.5% pixel IoU and 6.51% object recall on that test set.

| `--strategy` | What it updates | Pixel IoU | Object recall | When to use it |
|---|---|---|---|---|
| `full` | every weight | **84.4%** | 34.16% | Best on the target region. Forgets the most. |
| `lastlayer` | segmentation head only | 62.7% | 25.68% | Cheapest. Underfits. |
| `lora` | rank-`r` adapters, base frozen | 64.0% | 28.33% | Small trainable footprint, low memory. |
| `upgd` | every weight, utility-gated optimizer | 65.6% | 27.57% | Full-model updates with some forgetting protection. |

Common overrides:

```bash
trazo-pt3-finetune run --strategy lora \
  --config my_config.yaml \
  --checkpoint /models/ftw_3class.ckpt \
  --data-dir /data/region --output-dir /models/ft_lora \
  --max-epochs 100 --lr 1e-4 \
  --data.init_args.batch_size=16     # any extra LightningCLI override passes through
```

Every flag this CLI accepts becomes a config override — nothing you type is ignored.

### > 3.1b Merging to limit catastrophic forgetting

Fine-tuning on 36 Chiquitania chips took India's pixel IoU from 96.0% to 0.1%. Merging the fine-tuned weights back toward the base model recovers most of it. **MagMax** keeps, for each weight, the larger-magnitude change across the checkpoints being merged, then applies that to the base:

```bash
trazo-pt3-finetune merge \
  --base      /models/ftw_3class.ckpt \
  --finetuned /models/ft_full/lightning_logs/version_0/checkpoints/best.ckpt \
  --output    /models/merged.ckpt \
  --method    magmax \
  --scaling-coef 1.0
```

In the note's cross-region evaluation MagMax scored the best pixel IoU on Kenya (65.9%) and India (43.4%), recovered Brazil to 77.2%, and still held 79.2% on the target region — against 84.4% for unmerged full fine-tuning. `--method sum` does plain task-vector addition instead, which is the tool to reach for when combining several regional models.

### > 3.1c The Colab notebook

`src/trazo/pt3_finetune/Cerrado_Fine_Tuning.ipynb` is a Google Colab reference for the experiment side: loss function comparison (Focal, Tversky, Dice, Jaccard, asymmetric α/β), class weight tuning, learning rate sweeps (5e-6, 1e-5, 5e-5), batch size testing (8, 16, 32, 64), patience tuning, and side-by-side prediction visualizations. It uses `ftw model fit` from `ftw-tools`. To adapt it for a new region, update the data paths and the YAML config blocks.

### > 3.2 Active-learning sampling

`trazo-active-sample` scores every chip in a directory using a model checkpoint and selects the most informative ones for re-labeling using four complementary strategies.

Install with `pip install -e ".[active]"`.

**Four sampling strategies:**

| Strategy | Description |
|----------|-------------|
| `lc_discrepancy_all` | Chips with largest absolute disagreement between the LC raster and model predictions (all non-background pixels) |
| `lc_discrepancy_interiors` | Same discrepancy, but counting only field-interior predictions |
| `low_confidence` | Chips where the model's mean softmax max-probability is lowest |
| `low_confidence_ag20pct` | Same as above, restricted to chips that are ≥20 % agricultural land |

**Run:**

```bash
trazo-active-sample \
  --chips-dir   /data/grids_tiff \
  --checkpoint  /models/best.ckpt \
  --ag-raster   /data/ag_binary.tif \
  --output-dir  /data/active_learning \
  --n-per-strategy 100
```

**Output layout:**

```
<output-dir>/
  scores.json                       # per-chip score table
  lc_discrepancy_all.json           # strategy A selection metadata
  lc_discrepancy_all/               # chip TIFFs for strategy A
  lc_discrepancy_interiors.json
  lc_discrepancy_interiors/
  low_confidence.json
  low_confidence/
  low_confidence_ag20pct.json
  low_confidence_ag20pct/
  inference/preds/                  # (optional) prediction TIFFs
  inference/conf/                   # (optional) confidence TIFFs
```

Run `trazo-active-sample --help` for all options.

---

# Step 4: Model training

Step 4 contains the training infrastructure for field boundary segmentation models built on PyTorch Lightning and TorchGeo. Train custom models from scratch or fine-tune pretrained checkpoints on your own field boundaries.

Install with `pip install -e ".[pt4]"`.

### Quick start

Training is driven by a YAML config through the LightningCLI. A working example config ships with the package at `src/trazo/pt4_train/configs/example_3class.yaml` — copy it and edit the paths, class weights and loss:

```bash
# Fit a model
trazo-pt4-train fit \
  --config src/trazo/pt4_train/configs/example_3class.yaml \
  --data-dir /data/region \
  --output-dir /models/checkpoints
```

`--data-dir` overrides `data.init_args.root` and `--output-dir` overrides `trainer.default_root_dir`, so the command line wins over the config.

```bash
# Test a checkpoint on one or more countries/regions
trazo-pt4-train test \
  --model /models/checkpoints/best.ckpt \
  --countries region \
  --dir /data/ftw
```

Architecture, loss, channels, class weights and augmentation toggles are all config keys. See `src/trazo/pt4_train/settings.py`, `models.py`, `losses.py` and `trainers.py` for the full set.

### Architecture support

The training system supports multiple semantic segmentation architectures:

- **U-Net** and **U-Net (reduced)** with configurable encoder backbones
- **UperNet** for hierarchical feature extraction
- **FCN** (Fully Convolutional Network)
- **DeepLabV3+** for atrous spatial pyramid pooling
- **FCSiamDiff**, **FCSiamConc**, **FCSiamAvg** for bi-temporal change detection

All models support ImageNet weights.

### Loss functions

| Loss                | Description                                                    |
|---------------------|----------------------------------------------------------------|
| `ce`                | Cross-entropy with optional class weights                      |
| `pixel_weighted_ce` | Gaussian-weighted CE emphasizing boundary neighborhoods        |
| `jaccard`           | Jaccard/IoU loss for overlap maximization                      |
| `focal`             | Focal loss for hard example mining                             |
| `tversky`           | Tversky loss with tunable precision/recall tradeoff            |
| `dice`              | Dice coefficient loss                                          |
| `ce+dice`           | Combined cross-entropy and Dice                                |
| `logcoshdice`       | Log-cosh Dice for smooth gradients                             |
| `logcoshdice+ce`    | Log-cosh Dice combined with CE                                 |
| `ftnmt`             | Fields of The World FTnMT loss                                 |
| `ce+ftnmt`          | Combined cross-entropy and FTnMT                               |
| `tversky_ce`        | Tversky-Focal combined with cross-entropy                      |
| `localtversky`      | Locally weighted (spatially adaptive) Tversky-Focal            |
| `customtversky`     | Configurable Tversky with optional per-class alpha/beta        |

Most losses support `ignore_index` to exclude unlabeled or uncertain pixels (e.g., class 3 for presence-only labels).

### Training features

- **Multi-class segmentation**: Background, field interior, field boundary (+ optional unknown class)
- **Temporal fusion**: Stack two Sentinel-2 dates (planting + harvest)
- **Data augmentation**: Rotation, flipping, brightness, sharpness, random crops, channel shuffling
- **Adaptive normalization**: Random divisor augmentation for radiometric robustness
- **Metrics**: Per-class IoU, precision, recall, object-level F1
- **Learning rate scheduling**: Cosine annealing with configurable patience
- **Checkpoint management**: Best model selection, early stopping, resume from checkpoint
- **Distributed training**: Multi-GPU support via PyTorch Lightning

### Data structure expected

Step 4 expects data prepared by Step 2:

```
data/region/
  s2_images/
    window_a/
      <chip_id>__stack8.tif  # 4 bands from planting window
    window_b/
      <chip_id>__stack8.tif  # 4 bands from harvest window
  label_masks/
    instance/
      <chip_id>__instance.tif
    semantic_2class/
      <chip_id>__semantic_2class.tif  # background + field
    semantic_3class/
      <chip_id>__semantic_3class.tif  # background + interior + boundary
  chips_region.parquet  # Metadata with train/val/test splits
```

### Monitoring training

All metrics are logged to TensorBoard and console:

```bash
tensorboard --logdir /models/checkpoints/lightning_logs
```

### Model checkpoints

```python
from trazo.pt4_train.trainers import CustomSemanticSegmentationTask

model = CustomSemanticSegmentationTask.load_from_checkpoint(
    "/models/checkpoints/best.ckpt"
)
```

### Tips for best results

1. **Start with pretrained weights**: Use ImageNet backbones for faster convergence
2. **Class weighting**: Upweight rare classes (boundaries) to improve detection. The note used 0.05 / 0.20 / 0.75 for background / interior / boundary
3. **Loss selection**: `tversky_ce` works well for imbalanced boundary detection
4. **Augmentation**: Enable preprocessing augmentation for radiometric robustness across sensors
5. **Temporal stacking**: Use both windows (`stacked`) for maximum information
6. **Batch size**: Larger batches (32-64) improve stability with boundary-focused losses
7. **Learning rate**: The note used 0.003 with a 300-epoch budget and early stopping
8. **Validation**: Use a different region for validation to test generalization

---

# Step 5: Select per tile pairs and run inference

Step 5 contains utilities to choose the best two Sentinel-2 scenes per tile, write 8 band stacks for those pairs, and run inference with one or many checkpoints.

### CLI

```bash
trazo-pt5-infer <subcommand> [args...]
# or, without the installed script:
python -m trazo.pt5_inference.cli <subcommand> [args...]
```

Subcommands:

| Subcommand             | What it does                                                           | Module called                       |
|------------------------|------------------------------------------------------------------------|-------------------------------------|
| `tilepairs`            | AOI based selector and 8 band stack writer                             | `tilepairs.main()`                  |
| `tilepairs-tilelist`   | Tile list selector using SOS/EOS windows and progressive cloud caps    | `tilepairs_tilelist.main()`         |
| `tilepairs-advanced`   | AOI based selector with extra tunables for power users                 | `tilepairs_advanced.main()`         |
| `multi-infer`          | Run every checkpoint in a folder over every stack in a folder          | `multimodelinference.main()`        |
| `batch-infer`          | Run a single checkpoint over a folder of stacks                        | `batchinference.main()`             |

Use `multi-infer` when comparing checkpoints against each other, and `batch-infer` when you have already picked one.

### > 5.1 AOI based tile pairs

Finds the best two dates per tile over your AOI and year. Groups items by `s2:mgrs_tile`, tries cloud thresholds in order, and chooses the pair that minimizes total cloud cover, then prefers longer gaps, then earlier first date. Writes an 8 band stack per pair and a summary CSV.

```bash
python -m trazo.pt5_inference.cli tilepairs \
  --aoi-shp /data/aois/merged.shp \
  --year 2022 \
  --output-dir /data/stacks \
  --min-month-gap 4 \
  --cloud-thresholds 5 7 9 10 \
  --bands B04 B03 B02 B08 \
  --full-tile \
  --write-dtype uint16
```

### > 5.2 Tile list with SOS/EOS windows

If you have a list of MGRS tiles, you can drive selection by SOS and EOS windows per tile. The script can also build that tile list by intersecting your AOI with the built-in index at `spatial/sentinel_2_index_shapefile.geojson`.

```bash
python -m trazo.pt5_inference.cli tilepairs-tilelist \
  --tiles-shp /path/to/sentinel_2_tile_index.shp \
  --tile-id-col Name \
  --year 2024 \
  --output-csv /data/S2_tile_pairs.csv \
  --min-month-gap 4 \
  --cloud-thresholds 0 1 2 3 5 7 10 \
  --aoi-shp /data/aois/merged.shp \
  --build-tiles-from-aoi   # build the tile list by intersecting the repo index
```

Outputs a CSV of selected per-tile date pairs.

### > 5.3 Multi checkpoint inference

Runs the FTW model inference on every stack in a folder for each checkpoint in a folder. Creates an output subfolder per checkpoint.

```bash
python -m trazo.pt5_inference.cli multi-infer \
  --input-dir /data/stacks \
  --checkpoint-dir /models/ftw_checkpoints \
  --output-dir /data/inference/inf_output \
  --gpu-id 0 \
  --overwrite
```

Requires the `ftw` CLI (from `ftw-tools`) in your environment.

---

## End to end at a glance

If you start with a grid of AOIs and a field polygon layer:

1. **Step 1**: `trazo-pt1-create grid` to build the grid, then `trazo-pt1-create chips` to produce `window_a` and `window_b`
2. **Step 2**:
   - `pair-stacks` to create 8 band stacks
   - `resize-256` if you want normalized chips
   - `chips-bboxes` to write 256 bounding boxes
   - `make-masks` to split and build masks
   - `chips-parquet` to produce training metadata
   - `scale-u16` to standardize radiometry
3. **Step 3**: `trazo-pt3-finetune run` to adapt an existing checkpoint, then `trazo-pt3-finetune merge` if you need the base model's global performance back
4. **Step 4**: `trazo-pt4-train fit` to train from your own data
5. **Step 5**:
   - `tilepairs` or `tilepairs-tilelist` to select per tile pairs and write stacks
   - `multi-infer` or `batch-infer` to run checkpoints over those stacks

---

## Notes

- The repo includes SOS and EOS GeoTIFFs under `seasontifs/` as a bundled fallback for Step 1.2.
- The repo includes a Sentinel-2 tile index at `spatial/sentinel_2_index_shapefile.geojson` for building tile lists from an AOI.
- All scripts accept `--help` to see the full argument list and defaults.
- `scratch/` and `validation/` hold exploratory notebooks and annotation-quality analyses from the technical note. They are kept for provenance and are not part of the installable package.

## Tests

```bash
pip install -e ".[pt2,dev]"
pytest -m "not heavy"       # core + Step 2 + the offline smoke path
pytest -m heavy             # needs the pt3/pt4 extras
```

CI runs the core suite on Python 3.10, 3.11 and 3.12 and the heavy suite on 3.12.

## Contributing and contact

Issues and pull requests are welcome at https://github.com/wri/trazo/issues.

For questions about the technical note, the training data, or applying Trazo to a
new region, contact **Tristan Grupp — tristan.grupp@wri.org**.

## Acknowledgements and Funding

This project was funded by a Walmart Foundation grant with support from Land and Carbon Lab, Taylor Geospatial Engine, and WRI's Data Lab. We would like to thank data partners and collaborators in South America working to advance sustainable agriculture who gave time, expertise, and feedback.
