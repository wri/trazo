#!/usr/bin/env bash
# pt2_dataprep/run.sh
# Orchestrate Step 2 data prep: pair stacks, resize to 256, make bboxes, masks + windows,
# build per chip GeoParquet, and optional scaling to uint16 in [0,10000].

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PAIR_SCRIPT="$SCRIPT_DIR/pair_stacks.py"
RESIZE_SCRIPT="$SCRIPT_DIR/resize_chips_256.py"
BBOX_SCRIPT="$SCRIPT_DIR/chips_to_bboxes.py"
MASKS_SCRIPT="$SCRIPT_DIR/make_masks_and_windows.py"
PARQUET_SCRIPT="$SCRIPT_DIR/build_chips_parquet.py"
SCALE_SCRIPT="$SCRIPT_DIR/scale_uint16.py"

usage() {
  cat <<'USAGE'
Usage:
  run.sh --base BASE_FOLDER [--fields FIELDS_SHP_OR_GEOJSON]
         [--pair-stacks --win-a DIR --win-b DIR --stacks-out DIR]
         [--resize] [--no-resize]
         [--bboxes] [--no-bboxes]
         [--masks] [--no-masks] [--boundary-px N] [--overwrite-masks]
         [--parquet] [--no-parquet] [--overwrite-parquet]
         [--scale] [--no-scale] [--scale-mode suffix|inplace] [--scale-suffix _u16]
         [--overwrite-resize] [--resize-only256] [--crop-size 256]

Required:
  --base BASE_FOLDER
      Base folder that holds your chips. This script writes outputs next to it.

Optional inputs:
  --fields PATH
      Fields polygon file used to create label masks. Required if --masks is enabled.

Stack pairing step (only if you start with separate window_a and window_b folders):
  --pair-stacks
  --win-a DIR
  --win-b DIR
  --stacks-out DIR
      Create 8 band stacks A(1-4) then B(5-8) at --stacks-out.

Resize to 256 step:
  --resize            Enable resize to 256x256 into <base>/sized256. Default on.
  --no-resize         Disable resize step.
  --overwrite-resize  Overwrite existing outputs in sized256.
  --resize-only256    When an image is already 256x256, keep or fix invalids and write to sized256.
  --crop-size N       Target side length. Default 256.

Bounding boxes step:
  --bboxes            Enable writing <base>/<basename>_boundingbox256.geojson. Default on.
  --no-bboxes         Disable bbox step.

Masks and per window outputs step:
  --masks             Enable creation of label masks and window_a/window_b from sized256. Default on.
  --no-masks          Disable masks step.
  --boundary-px N     Boundary thickness in pixels for semantic_3class masks. Default 1.
  --overwrite-masks   Overwrite window splits and masks if present.

Per chip GeoParquet step:
  --parquet           Enable building <base>/chips_<region>.parquet. Default on.
  --no-parquet        Disable parquet step.
  --overwrite-parquet Overwrite parquet if present.

Scale to uint16 step:
  --scale             Enable scaling or casting chips to uint16 in [0,10000]. Default off.
  --no-scale          Disable scaling step.
  --scale-mode M      "suffix" (default) or "inplace".
  --scale-suffix S    Suffix for output files in suffix mode. Default "_u16".

General:
  -h, --help          Show this help.

Examples:
  # Typical run from 8 band stacks already created by pt1:
  ./run.sh --base /data/region --fields /data/region/fields.shp

  # If starting with separate window_a and window_b folders:
  ./run.sh --base /data/region \
    --pair-stacks --win-a /data/region/window_a --win-b /data/region/window_b --stacks-out /data/region/stacks8 \
    --fields /data/region/fields.shp

USAGE
}

# Defaults
BASE=""
FIELDS=""
DO_PAIR=false
WIN_A=""
WIN_B=""
STACKS_OUT=""

DO_RESIZE=true
OVERWRITE_RESIZE=false
RESIZE_ONLY256=false
CROP_SIZE=256

DO_BBOXES=true

DO_MASKS=true
BOUNDARY_PX=1
OVERWRITE_MASKS=false

DO_PARQUET=true
OVERWRITE_PARQUET=false

DO_SCALE=false
SCALE_MODE="suffix"
SCALE_SUFFIX="_u16"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2;;
    --fields) FIELDS="$2"; shift 2;;

    --pair-stacks) DO_PAIR=true; shift;;
    --win-a) WIN_A="$2"; shift 2;;
    --win-b) WIN_B="$2"; shift 2;;
    --stacks-out) STACKS_OUT="$2"; shift 2;;

    --resize) DO_RESIZE=true; shift;;
    --no-resize) DO_RESIZE=false; shift;;
    --overwrite-resize) OVERWRITE_RESIZE=true; shift;;
    --resize-only256) RESIZE_ONLY256=true; shift;;
    --crop-size) CROP_SIZE="$2"; shift 2;;

    --bboxes) DO_BBOXES=true; shift;;
    --no-bboxes) DO_BBOXES=false; shift;;

    --masks) DO_MASKS=true; shift;;
    --no-masks) DO_MASKS=false; shift;;
    --boundary-px) BOUNDARY_PX="$2"; shift 2;;
    --overwrite-masks) OVERWRITE_MASKS=true; shift;;

    --parquet) DO_PARQUET=true; shift;;
    --no-parquet) DO_PARQUET=false; shift;;
    --overwrite-parquet) OVERWRITE_PARQUET=true; shift;;

    --scale) DO_SCALE=true; shift;;
    --no-scale) DO_SCALE=false; shift;;
    --scale-mode) SCALE_MODE="$2"; shift 2;;
    --scale-suffix) SCALE_SUFFIX="$2"; shift 2;;

    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

# Validate base
if [[ -z "$BASE" ]]; then
  echo "Error: --base is required"
  usage
  exit 1
fi
BASE="$(python - <<PY
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
"$BASE")"

if [[ ! -d "$BASE" ]]; then
  echo "Error: base folder does not exist: $BASE"
  exit 1
fi

echo "=== pt2_dataprep pipeline ==="
echo "Base: $BASE"

# 1) Pair stacks (optional)
if $DO_PAIR; then
  if [[ -z "$WIN_A" || -z "$WIN_B" || -z "$STACKS_OUT" ]]; then
    echo "Error: --pair-stacks requires --win-a, --win-b and --stacks-out"
    exit 1
  fi
  echo ""
  echo "[1/6] Pair stacks"
  python "$PAIR_SCRIPT" \
    --window-a "$WIN_A" \
    --window-b "$WIN_B" \
    --out "$STACKS_OUT" \
    --overwrite
  # After pairing, set BASE to stacks-out for subsequent steps if user wants that flow
  BASE="$STACKS_OUT"
fi

# 2) Resize to 256 (optional, default on)
if $DO_RESIZE; then
  echo ""
  echo "[2/6] Resize chips to ${CROP_SIZE}x${CROP_SIZE} into sized256"
  RESIZE_ARGS=(--base-folder "$BASE" --crop-size "$CROP_SIZE")
  $OVERWRITE_RESIZE && RESIZE_ARGS+=(--overwrite)
  $RESIZE_ONLY256 && RESIZE_ARGS+=(--only-accept-256)
  python "$RESIZE_SCRIPT" "${RESIZE_ARGS[@]}"
else
  echo ""
  echo "[2/6] Resize step skipped"
fi

# 3) Make bboxes (default on)
if $DO_BBOXES; then
  echo ""
  echo "[3/6] Build bounding boxes GeoJSON from sized256"
  python "$BBOX_SCRIPT" --base-folder "$BASE"
else
  echo ""
  echo "[3/6] Bboxes step skipped"
fi

# 4) Masks and windows (default on)
if $DO_MASKS; then
  if [[ -z "$FIELDS" ]]; then
    echo "Error: --fields is required when --masks is enabled"
    exit 1
  fi
  echo ""
  echo "[4/6] Create masks and split windows"
  MASK_ARGS=(--base-folder "$BASE" --fields-shp "$FIELDS" --boundary-px "$BOUNDARY_PX")
  $OVERWRITE_MASKS && MASK_ARGS+=(--overwrite)
  python "$MASKS_SCRIPT" "${MASK_ARGS[@]}"
else
  echo ""
  echo "[4/6] Masks and windows step skipped"
fi

# 5) Build per chip GeoParquet (default on)
if $DO_PARQUET; then
  echo ""
  echo "[5/6] Build chips GeoParquet"
  PARQ_ARGS=(--base-folder "$BASE")
  $OVERWRITE_PARQUET && PARQ_ARGS+=(--overwrite)
  python "$PARQUET_SCRIPT" "${PARQ_ARGS[@]}"
else
  echo ""
  echo "[5/6] Parquet step skipped"
fi

# 6) Optional scaling to uint16 in [0,10000]
if $DO_SCALE; then
  echo ""
  echo "[6/6] Scale or cast chips to uint16 in [0,10000]"
  SCALE_ARGS=(--base-folder "$BASE" --write-mode "$SCALE_MODE" --out-suffix "$SCALE_SUFFIX")
  python "$SCALE_SCRIPT" "${SCALE_ARGS[@]}"
else
  echo ""
  echo "[6/6] Scale step skipped"
fi

echo ""
echo "Done."
