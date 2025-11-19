#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create GeoJSON bounding boxes for 256x256 chips.

Scans each input folder's direct child folder "sized256" for .tif/.tiff chips.
For every chip:
  - Reads georeferencing
  - Transforms its bounds to EPSG:4326 with optional densification for rotated grids
  - Writes a rectangle polygon for the chip extent to a GeoJSON placed one level above "sized256"

Output file name:
  <parent_folder_name>_boundingbox256.geojson

Rules
  - No recursion. Only files directly inside "<folder>/sized256" are scanned.
  - If REQUIRE_256 is set, chips must be exactly 256 x 256 pixels to be included.
  - Chips without CRS are skipped.

Example CLI:
  python -m tkt.pt2_dataprep.chips_to_bboxes \
      --folder /data/stacked \
      --folder /more/stacked \
      --require-256 \
      --densify-pts 21
"""

import argparse
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box


def _sized256_dir(base: Path) -> Path:
    return base / "sized256"


def _list_tifs_one_level(folder: Path) -> List[Path]:
    tifs = sorted(folder.glob("*.tif"))
    tiffs = sorted(folder.glob("*.tiff"))
    return [*tifs, *tiffs]


def generate_bboxes_for_sized256(
    folder: Path,
    require_256: bool = True,
    densify_pts: int = 21,
) -> Optional[Path]:
    """
    Build a GeoJSON of chip bounding boxes for chips in <folder>/sized256.

    Returns the path to the written GeoJSON, or None if nothing was written.
    """
    sized = _sized256_dir(folder)
    if not sized.exists() or not sized.is_dir():
        print(f"[WARN] Missing folder: {sized}")
        return None

    tif_files = _list_tifs_one_level(sized)
    if not tif_files:
        print(f"[INFO] No TIFF files found in {sized}")
        return None

    records = []
    for tif in tif_files:
        try:
            with rasterio.open(tif) as src:
                if require_256 and not (src.width == 256 and src.height == 256):
                    print(f"[WARN] Skip non 256 chip: {tif.name} ({src.width}x{src.height})")
                    continue
                if src.crs is None:
                    print(f"[WARN] Skip no CRS: {tif.name}")
                    continue

                b = src.bounds
                try:
                    # Densify to better approximate rotated geotransforms when reprojecting
                    minx, miny, maxx, maxy = transform_bounds(
                        src.crs,
                        "EPSG:4326",
                        b.left, b.bottom, b.right, b.top,
                        densify_pts=densify_pts,
                    )
                except Exception:
                    # Fallback without densification
                    minx, miny, maxx, maxy = transform_bounds(
                        src.crs,
                        "EPSG:4326",
                        b.left, b.bottom, b.right, b.top,
                    )

                geom = box(minx, miny, maxx, maxy)
                records.append({"filename": tif.name, "geometry": geom})
        except Exception as e:
            print(f"[ERROR] {tif}: {e}")

    if not records:
        print(f"[INFO] No valid chips to write for {folder}")
        return None

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    out_name = f"{folder.name}_boundingbox256.geojson"
    out_path = folder / out_name

    try:
        gdf.to_file(out_path, driver="GeoJSON")
    except Exception as e:
        print(f"[ERROR] Failed writing {out_path}: {e}")
        return None

    print(f"[OK] Wrote {len(gdf)} chip bounding boxes to {out_path}")
    return out_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create GeoJSON bounding boxes for 256x256 chips found in <folder>/sized256."
        )
    )
    p.add_argument(
        "--folder",
        action="append",
        required=True,
        help="Parent folder that contains a 'sized256' subfolder. Can be repeated.",
    )
    p.add_argument(
        "--require-256",
        action="store_true",
        help="Only include chips that are exactly 256x256 pixels.",
    )
    p.add_argument(
        "--densify-pts",
        type=int,
        default=21,
        help="Number of intermediate points per edge for transform_bounds densification.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    folders = [Path(f).expanduser().resolve() for f in args.folder]

    for base in folders:
        generate_bboxes_for_sized256(
            folder=base,
            require_256=args.require_256,
            densify_pts=args.densify_pts,
        )

    print("Done.")


if __name__ == "__main__":
    main()
