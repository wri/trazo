#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pair 4 band Sentinel-2 chips from two folders into 8 band stacks.

Order of bands in the output:
    Bands 1–4: window A bands 1–4
    Bands 5–8: window B bands 1–4

Validation:
    - Both inputs must have 4 bands
    - Same CRS
    - Same transform
    - Same width / height

Usage examples
--------------

As a module:

    from pathlib import Path
    from trazo.pt2_dataprep.pair_stacks import pair_stacks

    pair_stacks(
        window_a_dir=Path("/data/window_a"),
        window_b_dir=Path("/data/window_b"),
        out_dir=Path("/data/stacked"),
        overwrite=False,
        compress="deflate",
    )

From CLI (once wired into your entrypoints):

    python -m tkt.pt2_dataprep.pair_stacks \
        --window-a-dir /data/window_a \
        --window-b-dir /data/window_b \
        --out-dir /data/stacked \
        --overwrite \
        --compress deflate
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import rasterio
from rasterio.io import DatasetReader


def list_tifs(folder: Path) -> Dict[str, Path]:
    """
    Return a mapping {filename: full_path} for .tif/.tiff in the given folder (non recursive).
    """
    mapping: Dict[str, Path] = {}
    for p in folder.glob("*.tif"):
        mapping[p.name] = p
    for p in folder.glob("*.tiff"):
        mapping[p.name] = p
    return mapping


def validate_pair(a_ds: DatasetReader, b_ds: DatasetReader) -> Tuple[bool, str]:
    """
    Validate that the two datasets form a compatible pair.

    Checks:
      - 4 bands each
      - Same CRS
      - Same transform
      - Same width / height

    Returns
    -------
    (ok, reason_if_invalid)
    """
    if a_ds.count != 4:
        return False, f"Window A has {a_ds.count} bands (expected 4)."
    if b_ds.count != 4:
        return False, f"Window B has {b_ds.count} bands (expected 4)."
    if a_ds.crs != b_ds.crs:
        return False, "CRS mismatch."
    if a_ds.transform != b_ds.transform:
        return False, "Affine transform mismatch."
    if (a_ds.width != b_ds.width) or (a_ds.height != b_ds.height):
        return False, "Raster dimension mismatch."
    return True, ""


def common_dtype(dtype_a, dtype_b) -> np.dtype:
    """
    Compute a safe common dtype for stacking.

    Examples:
      - uint16 + uint16 -> uint16
      - uint16 + int16  -> int32
      - float32 + uint16 -> float32
    """
    return np.result_type(np.dtype(dtype_a), np.dtype(dtype_b))


def stack_and_write(
    a_path: Path,
    b_path: Path,
    out_path: Path,
    compress: Optional[str],
) -> None:
    """
    Stack bands from A (1..4) then B (1..4) into an 8 band GeoTIFF at out_path.

    Parameters
    ----------
    a_path : Path
        Path to window A TIFF with 4 bands.
    b_path : Path
        Path to window B TIFF with 4 bands.
    out_path : Path
        Output TIFF path (8 band stack).
    compress : Optional[str]
        Compression mode: None, "deflate", "lzw", etc.
    """
    with rasterio.open(a_path) as a_ds, rasterio.open(b_path) as b_ds:
        ok, reason = validate_pair(a_ds, b_ds)
        if not ok:
            print(f"[SKIP] {a_path.name}: {reason}")
            return

        out_dtype = common_dtype(a_ds.dtypes[0], b_ds.dtypes[0])

        a_nodata = a_ds.nodata
        b_nodata = b_ds.nodata
        out_nodata = a_nodata if a_nodata is not None else (b_nodata if b_nodata is not None else None)

        profile = a_ds.profile.copy()
        profile.update(
            driver="GTiff",
            count=8,
            dtype=str(out_dtype),
            nodata=out_nodata,
        )

        if compress is None:
            profile.pop("compress", None)
        else:
            profile["compress"] = compress

        with rasterio.open(out_path, "w", **profile) as dst:
            for i in range(1, 5):
                arr = a_ds.read(i)
                if arr.dtype != out_dtype:
                    arr = arr.astype(out_dtype, copy=False)
                dst.write(arr, i)

            for i in range(1, 5):
                arr = b_ds.read(i)
                if arr.dtype != out_dtype:
                    arr = arr.astype(out_dtype, copy=False)
                dst.write(arr, 4 + i)

            tags = a_ds.tags().copy()
            tags.update({f"A_{k}": v for k, v in a_ds.tags().items()})
            tags.update({f"B_{k}": v for k, v in b_ds.tags().items()})
            dst.update_tags(**tags)

    print(f"[OK]  Wrote 8 band stack: {out_path.name}")


def pair_stacks(
    window_a_dir: Path,
    window_b_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    compress: Optional[str] = None,
) -> None:
    """
    Pair match TIFFs across two folders and write 8 band stacks.

    Parameters
    ----------
    window_a_dir : Path
        Folder containing A chips (4 band TIFFs).
    window_b_dir : Path
        Folder containing B chips (4 band TIFFs).
    out_dir : Path
        Output folder for 8 band stacks.
    overwrite : bool
        If True, overwrite existing outputs. If False, skip existing.
    compress : Optional[str]
        Compression for output GeoTIFFs. None, "deflate", "lzw", etc.
    """
    if not window_a_dir.is_dir():
        print(f"[ERROR] Window A folder does not exist: {window_a_dir}")
        sys.exit(1)
    if not window_b_dir.is_dir():
        print(f"[ERROR] Window B folder does not exist: {window_b_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    a_map = list_tifs(window_a_dir)
    b_map = list_tifs(window_b_dir)

    if not a_map:
        print(f"[WARN] No TIFFs found in Window A folder: {window_a_dir}")
    if not b_map:
        print(f"[WARN] No TIFFs found in Window B folder: {window_b_dir}")

    common_names: List[str] = sorted(set(a_map.keys()) & set(b_map.keys()))
    print(f"[INFO] Found {len(common_names)} matching filename(s).")

    skipped_existing = 0
    processed = 0

    for name in common_names:
        a_path = a_map[name]
        b_path = b_map[name]
        out_path = out_dir / name

        if out_path.exists() and not overwrite:
            skipped_existing += 1
            print(f"[SKIP] Exists: {out_path.name} (overwrite=False)")
            continue

        try:
            stack_and_write(a_path, b_path, out_path, compress=compress)
            processed += 1
        except Exception as e:
            print(f"[ERR] {name}: {e}")

    print("-" * 60)
    print(f"Total matches:           {len(common_names)}")
    print(f"Processed (written):     {processed}")
    print(f"Skipped (already exist): {skipped_existing} (overwrite={overwrite})")
    print(f"Output folder:           {out_dir}")


def parse_args(argv: Optional[List[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Pair match 4 band TIFFs across two folders and write 8 band stacks."
    )

    parser.add_argument(
        "--window-a-dir",
        required=True,
        help="Path to folder containing window A TIFFs.",
    )
    parser.add_argument(
        "--window-b-dir",
        required=True,
        help="Path to folder containing window B TIFFs.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output folder for 8 band stacks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs instead of skipping.",
    )
    parser.add_argument(
        "--compress",
        choices=["deflate", "lzw", "none"],
        default="none",
        help="Compression for output GeoTIFFs (default: none).",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    window_a_dir = Path(args.window_a_dir).expanduser().resolve()
    window_b_dir = Path(args.window_b_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    compress: Optional[str]
    if args.compress == "none":
        compress = None
    else:
        compress = args.compress

    pair_stacks(
        window_a_dir=window_a_dir,
        window_b_dir=window_b_dir,
        out_dir=out_dir,
        overwrite=args.overwrite,
        compress=compress,
    )


if __name__ == "__main__":
    main()
