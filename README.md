# Trazo

<p align="center">
  <img src="assets/fieldscropped2.gif" alt="Field Boundaries" />
</p>
<b><u>Trazo</u></b> is the Spanish word for brushstroke, from the verb trazar; to trace, to draw, to plot. 
By tracing fields by hand in satellite imagery, we can choose the most powerful samples for creating robust, generalizable field boundary detection models. Each sketched field in our dataset adds knowledge about the wide diversity of agricultural systems in South America. These sketches teach models the culture of agriculture and how landscapes differ.

Trazo echos the English word "trace": to follow path, uncover origins and make hidden connections legible. Trazo is both about creating powerful, diverse training data and the aim of of tracing commodities through the supply chain, so that agriculture can be monitored for deforestation.

<p></p>

This is the documentation of a WRI technical note that focuses on how to create training data, sample training data, fine tune models, and train models. This documentation of the technical note also has functionality that can serve researchers and others who want to apply their own field boundaries to create custom models. We hope these tools can help those in the agricultural and food sectors make better land use and sourcing decisions.

<p></p>

Trazo includes end to end utilities for creating training data, preparing chips and masks, and running inference for field boundary modeling. This repository builds on the Fields of the World repo: https://github.com/fieldsoftheworld for documenting efforts to scale FTW to new geographies.
This package also has a several inference scripts for making model test testing easier, such as comparing models on multiple Sentinel-2 tile sites and pulling Sentinel-2 imagery for a user's study area.

<p></p>

This front page gives you a practical map of Step 1, Step 2, and Step 5 with quick starts and command examples. Step 3 and 4 to come. Step 1 will be built out more with training data sampling strategies.

Trazo was created a joint effort between World Resources Institute and the Kerner Lab at Arizona State University, funded by the Walmart Foundation. Significant support was provided by Land and Carbon Lab at WRI.
<p></p>

---

## Install

Use a clean Python 3.10 to 3.12 environment.

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip wheel setuptools
pip install -e .
```

Core dependencies used across steps include geopandas, rasterio, rioxarray, odc.stac, pystac-client, planetary-computer, shapely, numpy, and tqdm.

---

## Folder conventions

A typical project layout after Step 1 and Step 2, following FTW structure:

```
toolkit-for-traceability/
  seasontifs/                         # SOS/EOS GeoTIFFs that ship with the repo
  spatial/sentinel_2_index_shapefile.geojson

  data/
    conab2020/
      conab2020_grid.shp              # user grid
      window_a/                       # 4 band chips from Step 1 (planting), FTW style
      window_b/                       # 4 band chips from Step 1 (harvest), FTW style
      sized256/                       # optional normalized 8 band chips
      s2_images/window_a/             # split out of 8 band chips, FTW style
      s2_images/window_b/
      label_masks/instance/
      label_masks/semantic_2class/
      label_masks/semantic_3class/
      hkl                             #  hickle version of window_a, window_b, label_masks: these combined
      region_boundingbox256.geojson
      chips_region.parquet
      S2_best_pairs_summary.csv       # Step 5 selection summary
```

---

# Step 1: Create grids and data windows from grids

### > 1.1 Creating Grids from your field boundaries

gridding.py — build or standardize the AOI grid

Generates a grid of square AOIs or standardizes an existing grid so it can be used by Step 1B.

### What it does
- Creates a fishnet grid over an input geometry or bbox, or reads an existing grid.
- Ensures a unique ID column aoi_id exists.
- Writes a year column using a constant or a field you specify.
- Saves the grid to your chosen output path.

### Outputs
- A spatial file with an aoi_id string, year int, geometry with chosen CRS

### Run

```bash
# Example: build a 2.56 km grid (256 px at 10 m) over an AOI and stamp year 2020
python -m tkt.pt1_createdata.gridding \
  --aoi "/path/to/aoi_boundary.shp" \
  --out "/path/to/grids/conab2020_grid.shp" \
  --grid-size-m 2560 \
  --crs "EPSG:32721" \
  --year-constant 2020

# Example: standardize an existing grid in place
python -m tkt.pt1_createdata.gridding \
  --grid "/path/to/existing_grid.shp" \
  --write-inplace \
  --id-field cellid \
  --year-field crop_year
```

After you have these grids, open the grids and your field boundaries in your preferred geometry editing software; ArcPro, QGIS, Collect Earth Online, etc. Fill in ALL fields within each chip. If you do not fill in all fields, you must use presence-only labels, which weights the background (non-fields) and the unlabeled fields with the value '3'. These values are excluded when calculating loss during fine-tuning/training. Presence-only masking will be added to step-2 later. You can use the output of step 1B as the imagery to label. Be sure that whatever reference year you are choosing matches when the rest of your field boundaries were produced.

### > 1.2 Creating harvest/planting images for every grid

Produce two 4 band chips per AOI: a planting window and a harvest window. Selection uses SOS and EOS rasters to target month ranges and prioritizes low cloud cover at the chip level.

- `window_a/<chip>.tif`  B04, B03, B02, B08 at planting
- `window_b/<chip>.tif`  B04, B03, B02, B08 at harvest
- `<chip>__stack8.tif`   optional 8 band stack written during the sweep
- `missing_windows.txt` and `errors.txt` audit logs

### Key features

- Queries the Microsoft Planetary Computer STAC
- Chip level cloud fraction using SCL mask
- Target windows around SOS or EOS with fallback search
- Auto download of SOS and EOS rasters if they are not present
  - Tries the upstream research_products repo
  - Falls back to the repo copy under `seasontifs/`
  - Stores into your input folder under `eossos tifs/`

### Run

Use the CLI wrapper or call the module directly.

```bash
# CLI form
python -m tkt.pt1_createdata.plantingharvest   --input-shps  "/path/to/your/grid_or_folder"   --year        2020   --span-months-planting 2   --span-months-harvest  2   --scene-cloud-threshold 90   --patch-cloud-threshold 0.08   --target-cloud-max      0.01   --preferred-cloud-max   0.02   --chip-size 256   --batch-size 10   --overwrite
```

If your build registers an entrypoint for Step 1, you can also run:

```bash
trazo-pt1-create plantingharvest ...  # if present in your install
```

### Important inputs

- AOI grid shapefile or a folder of shapefiles
- Year column or a constant year
- Bands of interest are B04, B03, B02, B08
- SOS/EOS rasters
  - If missing, the script will download them into your input folder
  - Primary source: research_products repo
  - Fallback: `seasontifs/` inside this repo

---

# Step 2: Data preparation

These tools convert the raw outputs of Step 1 into standardized chips, masks, and a one row per chip GeoParquet.

### CLI

```
trazo-pt2-dataprep <subcommand> [args...]
```

Subcommands:

| Subcommand      | What it does                                      |
|-----------------|----------------------------------------------------|
| `pair-stacks`   | Pair `window_a` and `window_b` into 8 band stacks  |
| `resize-256`    | Normalize stacks to 256x256 into `sized256`        |
| `chips-bboxes`  | Create WGS84 bbox GeoJSON for each 256 chip        |
| `make-masks`    | Make instance and semantic masks, split A and B    |
| `chips-parquet` | Build per chip GeoParquet metadata                 |
| `scale-u16`     | Convert chips to uint16 in [0, 10000]              |

### Recommended order

1. Pair A and B into 8 band stacks
2. Resize to 256x256 (optional but recommended)
3. Create chip bounding boxes
4. Make masks and split back out to `s2_images/window_a` and `s2_images/window_b`
5. Build the per chip GeoParquet
6. Scale to uint16 (optional, can be run before or after masks)

### Quick start

```bash
# 1. Pair
trazo-pt2-dataprep pair-stacks   --window-a /data/region/window_a   --window-b /data/region/window_b   --out-dir  /data/region   --overwrite

# 2. Normalize to 256
trazo-pt2-dataprep resize-256   --folders /data/region

# 3. BBoxes
trazo-pt2-dataprep chips-bboxes   --folders /data/region

# 4. Masks and split
trazo-pt2-dataprep make-masks   --folders /data/region   --boundary-px 1

# 5. Parquet
trazo-pt2-dataprep chips-parquet   --folders /data/region   --train-ratio 0.85 --val-ratio 0.15 --test-ratio 0.0

# 6. Scale to uint16
trazo-pt2-dataprep scale-u16   --folders /data/region
```

Notes

- `make-masks` selects features that intersect each chip without clipping, which avoids fake edges along chip borders
- 3 class masks write boundary as a thin ribbon that overwrites interior where they overlap
- `chips-parquet` writes exactly one row per chip and embeds a MultiPolygon of intersecting fields for the chip extent

---

# Step 3: Fine Tuning and Data sampling (Coming soon)

Step 3 provides utilities fine-tuning and for intelligent sampling and subset creation to maximize model performance while minimizing time spent tracing fields. 

### Planned features

- **Active learning** subset selection based on model uncertainty and land cover discrepency
- **Temporal diversity** sampling across seasons
- **Window diversity** mosaic and chipped versions
- **Geographic balancing** to ensure representation across study areas

---

# Step 4: Model training

Step 4 contains the training infrastructure for field boundary segmentation models built on PyTorch Lightning and TorchGeo. Train custom models from scratch or fine-tune pretrained checkpoints on your own field boundaries.

### Architecture support

The training system supports multiple semantic segmentation architectures:

- **U-Net** and **U-Net (reduced)** with configurable encoder backbones
- **UperNet** for hierarchical feature extraction
- **FCN** (Fully Convolutional Network)
- **DeepLabV3+** for atrous spatial pyramid pooling
- **FCSiamDiff**, **FCSiamConc**, **FCSiamAvg** for bi-temporal change detection

All models support pretrained ImageNet weights for transfer learning.

### Loss functions

Step 4 includes a comprehensive suite of loss functions optimized for field boundary detection:

| Loss                              | Description                                                    |
|-----------------------------------|----------------------------------------------------------------|
| `ce`                              | Cross-entropy with optional class weights                      |
| `pixel_weighted_ce`               | Gaussian-weighted CE emphasizing boundary neighborhoods        |
| `jaccard`                         | Jaccard/IoU loss for overlap maximization                      |
| `focal`                           | Focal loss for hard example mining                             |
| `tversky`                         | Tversky loss with tunable precision/recall tradeoff            |
| `dice`                            | Dice coefficient loss                                          |
| `ce+dice`                         | Combined cross-entropy and Dice                                |
| `logcosh_dice`                    | Log-cosh Dice for smooth gradients                             |
| `logcosh_dice_ce`                 | Log-cosh Dice combined with CE                                 |
| `tversky_focal_ce`                | Three-way combination for robust training                      |
| `ftnmt`                           | Fields of The World FTnMT loss                                 |
| `locally_weighted_tversky_focal`  | Spatially adaptive Tversky-Focal                               |

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

### Quick start

```bash
# Example: Fine-tune a U-Net with ResNet-50 backbone on your custom data
python -m trazo.pt4_train.train \
  --data-root /data/region \
  --train-countries region \
  --val-countries region \
  --model unet \
  --backbone resnet50 \
  --in-channels 8 \
  --num-classes 4 \
  --loss tversky_focal_ce \
  --ignore-index 3 \
  --batch-size 16 \
  --lr 1e-4 \
  --max-epochs 50 \
  --gpus 1 \
  --output-dir /models/checkpoints
```

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

### Temporal options

Control how the two temporal windows are used:

| Option           | Description                                      | Input channels |
|------------------|--------------------------------------------------|----------------|
| `stacked`        | Concatenate window_a and window_b (default)      | 8              |
| `windowA`        | Use only planting window                         | 4              |
| `windowB`        | Use only harvest window                          | 4              |
| `median`         | Median across both windows                       | 4              |
| `random_window`  | Randomly select A or B each epoch                | 4              |

### Advanced configuration

Fine-tune model behavior with additional arguments:

```bash
# Class weights for imbalanced datasets
--class-weights 0.1 1.0 5.0 0.0  # [background, interior, boundary, ignore]

# Boundary pixel emphasis
--pixel-weight-scale 3.0  # Upweight boundary neighborhoods

# Per-class Tversky parameters
--per-class-tversky \
--alphas 0.3 0.5 0.7 \  # False negative penalty per class
--betas 0.7 0.5 0.3     # False positive penalty per class

# Freeze components for transfer learning
--freeze-backbone       # Fine-tune decoder only
--freeze-decoder        # Linear probe segmentation head

# Augmentation toggles
--random-shuffle        # Randomly swap temporal windows
--brightness-aug        # Random brightness adjustment
--preprocess-aug        # Random normalization divisor
--resize-aug            # Random crops and resizing
--resize-factor 1.5     # Upsample to 384x384

# Training efficiency
--num-workers 8
--pin-memory
--prefetch-factor 2
```

### Monitoring training

All metrics are logged to TensorBoard and console:

- **Training**: Loss, IoU (macro/micro), per-class IoU, precision, recall
- **Validation**: Same metrics plus object-level precision, recall, F1
- **Learning rate**: Tracked per epoch

View training progress:

```bash
tensorboard --logdir /models/checkpoints/lightning_logs
```

### Model checkpoints

Best models are saved based on validation loss. Load for inference or resume training:

```python
from trazo.pt4_train.trainers import CustomSemanticSegmentationTask

# Load from checkpoint
model = CustomSemanticSegmentationTask.load_from_checkpoint(
    "/models/checkpoints/best.ckpt"
)
```

### Tips for best results

1. **Start with pretrained weights**: Use ImageNet backbones for faster convergence
2. **Class weighting**: Upweight rare classes (boundaries) to improve detection
3. **Loss selection**: `tversky_focal_ce` works well for imbalanced boundary detection
4. **Augmentation**: Enable `preprocess_aug` for radiometric robustness across sensors
5. **Temporal stacking**: Use both windows (`stacked`) for maximum information
6. **Batch size**: Larger batches (32-64) improve stability with boundary-focused losses
7. **Learning rate**: Start with 1e-4, reduce if loss plateaus
8. **Validation**: Use a different region for validation to test generalization

---

# Step 5: Select per tile pairs and run inference

Step 5 contains utilities to choose the best two Sentinel 2 scenes per tile, write 8 band stacks for those pairs, and run inference with one or many checkpoints.

### CLI

```
trazo-pt5-infer <subcommand> [args...]
```

Subcommands:

| Subcommand             | What it does                                                           | Module called                       |
|------------------------|------------------------------------------------------------------------|-------------------------------------|
| `tilepairs`            | AOI based selector and 8 band stack writer                             | `tilepairs.main()`                  |
| `tilepairs-tilelist`   | Tile list selector using SOS/EOS windows and progressive cloud caps    | `tilepairs_tilelist.main()`         |
| `tilepairs-advanced`   | AOI based selector with extra tunables for power users                 | `tilepairs_advanced.main()`         |
| `multi-infer`          | Run multiple checkpoints across a folder of TIFF stacks                | `multimodelinference.main()`        |
| `batch-infer`          | General batch inference utility                                        | `batchinference.main()`             |

### > 5.1 AOI based tile pairs

Finds the best two dates per tile over your AOI and year. Groups items by `s2:mgrs_tile`, tries cloud thresholds in order, and chooses the pair that minimizes total cloud cover, then prefers longer gaps, then earlier first date. Writes an 8 band stack per pair and a summary CSV.

```bash
trazo-pt5-infer tilepairs   --aoi-shp /data/aois/merged.shp   --year 2022   --output-dir /data/stacks   --min-month-gap 4   --cloud-thresholds 5 7 9 10   --bands B04 B03 B02 B08   --full-tile   --write-dtype uint16
```

### > 5.2 Tile list with SOS/EOS windows

If you have a list of MGRS tiles, you can drive selection by SOS and EOS windows per tile. The script can also build that tile list by intersecting your AOI with the built in index:

- `spatial/sentinel_2_index_shapefile.geojson`

```bash
trazo-pt5-infer tilepairs-tilelist   --tile-shp /path/to/sentinel_2_tile_index.shp   --year 2024   --out-csv /data/S2_tile_pairs.csv   --min-gap-months 4   --cloud-seq 0 1 2 3 5 7 10   --use-aoi "/data/aois/merged.shp"   --auto-from-index   # to build the tile list by intersecting the repo index
```

Outputs a CSV with columns: `Tile,Window A,Window B,Nodata % A,Nodata % B`.

### > 5.3 Multi checkpoint inference

Runs the FTW model inference on every stack in a folder for each checkpoint in a folder. Creates an output subfolder per checkpoint.

```bash
trazo-pt5-infer multi-infer   --tif-dir       /data/stacks   --checkpoint-dir /models/ftw_checkpoints   --output-dir    /data/inference/inf_output   --gpu 0   --overwrite
```

Requires the `ftw` CLI in your environment.

---

## End to end at a glance

If you start with a grid of AOIs and a field polygon layer:

1. **Step 1**: run `plantingharvest` to produce `window_a` and `window_b`
2. **Step 2**:
   - `pair-stacks` to create 8 band stacks
   - `resize-256` if you want normalized chips
   - `chips-bboxes` to write 256 bounding boxes
   - `make-masks` to split and build masks
   - `chips-parquet` to produce training metadata
   - `scale-u16` to standardize radiometry
3. **Step 5**:
   - `tilepairs` or `tilepairs-tilelist` to select per tile pairs and write stacks
   - `multi-infer` to run checkpoints over those stacks

---

## Notes

- The repo includes SOS and EOS GeoTIFFs under `seasontifs/` as a fallback. The Step 1 script tries to download the upstream set first and falls back to the bundled copies when needed.
- The repo includes a Sentinel 2 tile index at `spatial/sentinel_2_index_shapefile.geojson` for building tile lists from an AOI.
- All scripts accept `--help` to see the full argument list and defaults.

## Acknowledgements and Funding


