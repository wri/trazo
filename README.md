# Toolkit for Traceability Package

<video src="assets/fieldscropped.mp4?raw=1" width="300" controls>
</video>

End to end utilities for creating training data, preparing chips and masks, and running inference for field boundary modeling. This repository builds on the Fields of the World repo: https://github.com/fieldsoftheworld

This front page gives you a practical map of Step 1, Step 2, and Step 5 with quick starts and command examples. Step 3 and 4 to come. Step 1 will be built out more with training data sampling strategies.

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

A typical project layout after Step 1 and Step 2:

```
toolkit-for-traceability/
  seasontifs/                         # SOS/EOS GeoTIFFs that ship with the repo
  spatial/sentinel_2_index_shapefile.geojson

  data/
    conab2020/
      conab2020_grid.shp              # user grid
      window_a/                       # 4 band chips from Step 1 (planting)
      window_b/                       # 4 band chips from Step 1 (harvest)
      sized256/                       # optional normalized 8 band chips
      s2_images/window_a/             # split out of 8 band chips
      s2_images/window_b/
      label_masks/instance/
      label_masks/semantic_2class/
      label_masks/semantic_3class/
      region_boundingbox256.geojson
      chips_region.parquet
      S2_best_pairs_summary.csv       # Step 5 selection summary
```

---

# Step 1 — Create data windows from grids

Produce two 4 band chips per AOI: a planting window and a harvest window. Selection uses SOS and EOS rasters to target month ranges and prioritizes low cloud cover at the chip level.

### What you get

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
tkt-pt1-create plantingharvest ...  # if present in your install
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

# Step 2 — Data preparation

These tools convert the raw outputs of Step 1 into standardized chips, masks, and a one row per chip GeoParquet.

### CLI

```
tkt-pt2-dataprep <subcommand> [args...]
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
tkt-pt2-dataprep pair-stacks   --window-a /data/region/window_a   --window-b /data/region/window_b   --out-dir  /data/region   --overwrite

# 2. Normalize to 256
tkt-pt2-dataprep resize-256   --folders /data/region

# 3. BBoxes
tkt-pt2-dataprep chips-bboxes   --folders /data/region

# 4. Masks and split
tkt-pt2-dataprep make-masks   --folders /data/region   --boundary-px 1

# 5. Parquet
tkt-pt2-dataprep chips-parquet   --folders /data/region   --train-ratio 0.85 --val-ratio 0.15 --test-ratio 0.0

# 6. Scale to uint16
tkt-pt2-dataprep scale-u16   --folders /data/region
```

Notes

- `make-masks` selects features that intersect each chip without clipping, which avoids fake edges along chip borders
- 3 class masks write boundary as a thin ribbon that overwrites interior where they overlap
- `chips-parquet` writes exactly one row per chip and embeds a MultiPolygon of intersecting fields for the chip extent

---

# Step 5 — Select per tile pairs and run inference

Step 5 contains utilities to choose the best two Sentinel 2 scenes per tile, write 8 band stacks for those pairs, and run inference with one or many checkpoints.

### CLI

```
tkt-pt5-infer <subcommand> [args...]
```

Subcommands:

| Subcommand             | What it does                                                           | Module called                       |
|------------------------|------------------------------------------------------------------------|-------------------------------------|
| `tilepairs`            | AOI based selector and 8 band stack writer                             | `tilepairs.main()`                  |
| `tilepairs-tilelist`   | Tile list selector using SOS/EOS windows and progressive cloud caps    | `tilepairs_tilelist.main()`         |
| `tilepairs-advanced`   | AOI based selector with extra tunables for power users                 | `tilepairs_advanced.main()`         |
| `multi-infer`          | Run multiple checkpoints across a folder of TIFF stacks                | `multimodelinference.main()`        |
| `batch-infer`          | General batch inference utility                                        | `batchinference.main()`             |

### 5.1 AOI based tile pairs

Finds the best two dates per tile over your AOI and year. Groups items by `s2:mgrs_tile`, tries cloud thresholds in order, and chooses the pair that minimizes total cloud cover, then prefers longer gaps, then earlier first date. Writes an 8 band stack per pair and a summary CSV.

```bash
tkt-pt5-infer tilepairs   --aoi-shp /data/aois/merged.shp   --year 2022   --output-dir /data/stacks   --min-month-gap 4   --cloud-thresholds 5 7 9 10   --bands B04 B03 B02 B08   --full-tile   --write-dtype uint16
```

### 5.2 Tile list with SOS/EOS windows

If you have a list of MGRS tiles, you can drive selection by SOS and EOS windows per tile. The script can also build that tile list by intersecting your AOI with the built in index:

- `spatial/sentinel_2_index_shapefile.geojson`

```bash
tkt-pt5-infer tilepairs-tilelist   --tile-shp /path/to/sentinel_2_tile_index.shp   --year 2024   --out-csv /data/S2_tile_pairs.csv   --min-gap-months 4   --cloud-seq 0 1 2 3 5 7 10   --use-aoi "/data/aois/merged.shp"   --auto-from-index   # to build the tile list by intersecting the repo index
```

Outputs a CSV with columns: `Tile,Window A,Window B,Nodata % A,Nodata % B`.

### 5.3 Multi checkpoint inference

Runs the FTW model inference on every stack in a folder for each checkpoint in a folder. Creates an output subfolder per checkpoint.

```bash
tkt-pt5-infer multi-infer   --tif-dir       /data/stacks   --checkpoint-dir /models/ftw_checkpoints   --output-dir    /data/inference/inf_output   --gpu 0   --overwrite
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
