#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 5 utility: select best Sentinel-2 scene pairs per tile and write 8-band stacks.

This script does two things:
  1. Finds the best two Sentinel-2 scenes per tile over a given AOI and year.
  2. Builds 8-band stacks for those scene pairs, clipped or masked to the AOI.

It does NOT run any model inference.

The basic selection logic:
  * Query Sentinel-2 L2A (Microsoft Planetary Computer) for a given YEAR and AOI.
  * Group scenes by MGRS tile (s2:mgrs_tile).
  * For each tile, search for a pair of scenes separated by at least MIN_MONTH_GAP months.
  * Try CLOUD_THRESHOLDS in order (e.g. 5, 7, 9, 10 percent) and use the first threshold
    that yields a valid pair.
  * Within that threshold, choose the pair that:
      - Minimizes total cloud cover (cloud1 + cloud2).
      - If tied, prefers larger temporal gap in days.
      - If still tied, prefers earlier first date.

Each chosen pair is written as an 8-band GeoTIFF:
  Bands 1-4: B04, B03, B02, B08 from the first date.
  Bands 5-8: B04, B03, B02, B08 from the second date.

A summary CSV is written in the output directory listing:
  tile, date1, date2, cloud1, cloud2, cloud_sum, gap_days, threshold_used, stack_path

Example:

  python -m tkt.pt5_inference.select_and_stack_pairs \
      --aoi-shp /path/to/aois/merged.shp \
      --year 2022 \
      --output-dir /path/to/output/stacks \
      --min-month-gap 4 \
      --cloud-thresholds 5 7 9 10 \
      --bands B04 B03 B02 B08 \
      --full-tile \
      --write-dtype uint16 \
      --compress deflate \
      --nodata 0
"""

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from pystac_client import Client
import planetary_computer as pc


def read_geometry(aoi_shp: str, simplify_tol: float) -> gpd.GeoSeries:
    """
    Read AOI shapefile, dissolve to a single geometry, project to EPSG:4326,
    and optionally simplify.

    Returns a shapely geometry in EPSG:4326.
    """
    gdf = gpd.read_file(aoi_shp)
    if gdf.crs is None:
        raise ValueError("Input AOI shapefile has no CRS")

    gdf = gdf.to_crs(4326)

    # union_all is GeoPandas 0.10+, unary_union is older
    if hasattr(gdf, "union_all"):
        dissolved = gdf.union_all()
    else:
        dissolved = gdf.unary_union

    if simplify_tol and simplify_tol > 0:
        dissolved = dissolved.simplify(simplify_tol, preserve_topology=True)

    return dissolved


def stac_client() -> Client:
    return Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


def stac_search(items_kwargs: Dict) -> List:
    api = stac_client()
    search = api.search(**items_kwargs, limit=1000, max_items=None)
    return list(search.item_collection())


def all_items_for_geometry(
    geom,
    start: str,
    end: str,
    collection: str,
) -> List:
    """
    Get all STAC items for a given geometry and date range.
    Tries intersects first, then falls back to bbox + post-filter.
    """
    try:
        return stac_search(
            {
                "collections": [collection],
                "intersects": mapping(geom),
                "datetime": f"{start}/{end}",
            }
        )
    except Exception:
        minx, miny, maxx, maxy = geom.bounds
        items = stac_search(
            {
                "collections": [collection],
                "bbox": [minx, miny, maxx, maxy],
                "datetime": f"{start}/{end}",
            }
        )
        return [it for it in items if it.intersects(mapping(geom))]


def group_by_tile(items: Sequence) -> Dict[str, List]:
    """
    Group STAC items by s2:mgrs_tile and sort each tile list by datetime.
    """
    by_tile: Dict[str, List] = defaultdict(list)
    for it in items:
        tile = it.properties.get("s2:mgrs_tile")
        if tile:
            by_tile[tile].append(it)

    for tile in by_tile:
        by_tile[tile].sort(key=lambda it: it.datetime)

    return by_tile


def pick_best_pair_in_tile(
    items_for_tile: Sequence,
    min_month_gap: int,
    cloud_cap: Optional[float] = None,
) -> Optional[Dict]:
    """
    For one tile, pick the best pair of scenes under a cloud_cap.

    Returns a dict with keys:
      it1, it2, t1, t2, c1, c2, cloud_sum, gap_days
    or None if no valid pair is found.
    """
    if cloud_cap is not None:
        items_for_tile = [
            it
            for it in items_for_tile
            if float(it.properties.get("eo:cloud_cover", 100.0)) < cloud_cap
        ]

    if len(items_for_tile) < 2:
        return None

    best = None
    min_gap_days = int(round(30 * min_month_gap))

    for i, it1 in enumerate(items_for_tile[:-1]):
        t1 = it1.datetime
        c1 = float(it1.properties.get("eo:cloud_cover", 100.0))

        # Candidate second images that satisfy the min gap
        candidates = [
            it
            for it in items_for_tile[i + 1 :]
            if (it.datetime - t1).days >= min_gap_days
        ]
        if not candidates:
            continue

        # Among those, pick with minimum cloud cover
        it2 = min(
            candidates,
            key=lambda it: float(it.properties.get("eo:cloud_cover", 100.0)),
        )
        t2 = it2.datetime
        c2 = float(it2.properties.get("eo:cloud_cover", 100.0))
        cloud_sum = c1 + c2
        gap_days = (t2 - t1).days

        cand = {
            "it1": it1,
            "it2": it2,
            "t1": t1,
            "t2": t2,
            "c1": c1,
            "c2": c2,
            "cloud_sum": cloud_sum,
            "gap_days": gap_days,
        }

        if best is None:
            best = cand
        else:
            better = (
                (cloud_sum < best["cloud_sum"])
                or (
                    cloud_sum == best["cloud_sum"]
                    and gap_days > best["gap_days"]
                )
                or (
                    cloud_sum == best["cloud_sum"]
                    and gap_days == best["gap_days"]
                    and t1 < best["t1"]
                )
            )
            if better:
                best = cand

    return best


def sign_asset_href(item, band: str) -> str:
    """
    Get a signed asset URL for a band using Planetary Computer token signing.
    """
    return pc.sign(item).assets[band].href


def build_target_profile(
    ref_href: str,
    clip_geom_wgs84,
    full_tile: bool,
    write_dtype: str,
    compress: Optional[str],
    nodata: Optional[float],
) -> Dict:
    """
    Build a raster profile for the output stack, based on a reference asset.
    """
    import rasterio.warp  # local import to avoid confusion with typing

    with rasterio.open(ref_href) as src:
        dst_crs = src.crs
        clip_gdf = gpd.GeoSeries([clip_geom_wgs84], crs=4326).to_crs(dst_crs)
        geom_proj = clip_gdf.iloc[0]

        if full_tile:
            height, width = src.height, src.width
            transform = src.transform
        else:
            out_image, out_transform = mask(src, [mapping(geom_proj)], crop=True)
            height, width = out_image.shape[1], out_image.shape[2]
            transform = out_transform

        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "transform": transform,
                "count": 1,
                "dtype": write_dtype,
                "crs": dst_crs,
                "tiled": True,
                "BIGTIFF": "YES" if compress is None else "IF_SAFER",
            }
        )

        # Only set nodata for full tile case if a nodata value is provided
        if full_tile and nodata is not None:
            profile["nodata"] = nodata
        else:
            profile.pop("nodata", None)

        if compress:
            profile["compress"] = compress
        else:
            profile.pop("compress", None)

    return profile


def stack_pair_to_tiff(
    it1,
    it2,
    clip_geom_wgs84,
    out_path: Path,
    bands: Sequence[str],
    full_tile: bool,
    write_dtype: str,
    compress: Optional[str],
    nodata: Optional[float],
) -> None:
    """
    For a given pair of STAC items and AOI, write an 8-band stack:
      B04, B03, B02, B08 from it1
      B04, B03, B02, B08 from it2
    or whatever bands are requested (duplicated for both dates).
    """
    import rasterio.warp  # local import

    if len(bands) != 4:
        raise ValueError("This helper assumes exactly 4 bands for now.")

    ref_href = sign_asset_href(it1, bands[0])
    dst_profile = build_target_profile(
        ref_href=ref_href,
        clip_geom_wgs84=clip_geom_wgs84,
        full_tile=full_tile,
        write_dtype=write_dtype,
        compress=compress,
        nodata=nodata,
    )

    mask_arr = None
    if full_tile:
        with rasterio.open(ref_href) as src_ref:
            dst_crs = src_ref.crs
            clip_gdf = gpd.GeoSeries([clip_geom_wgs84], crs=4326).to_crs(dst_crs)
            geom_proj = clip_gdf.iloc[0]
            mask_arr = geometry_mask(
                [mapping(geom_proj)],
                transform=dst_profile["transform"],
                invert=True,
                out_shape=(dst_profile["height"], dst_profile["width"]),
            )

    def read_and_resample(asset_href: str) -> np.ndarray:
        with rasterio.open(asset_href) as src:
            dst_arr = np.empty(
                (dst_profile["height"], dst_profile["width"]), dtype=np.float32
            )
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=dst_arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_profile["transform"],
                dst_crs=dst_profile["crs"],
                resampling=Resampling.bilinear,
            )

        if full_tile and mask_arr is not None:
            dst_arr = np.where(mask_arr, dst_arr, 0.0)

        if write_dtype == "uint16":
            dst_arr = np.clip(np.rint(dst_arr), 0, 65535).astype(np.uint16)
        elif write_dtype == "float32":
            dst_arr = dst_arr.astype(np.float32)
        else:
            raise ValueError(f"Unsupported write_dtype: {write_dtype}")

        return dst_arr

    band_arrays: List[np.ndarray] = []
    band_names: List[str] = []

    # First date
    for b, suffix in zip(bands, ["_1_B1", "_1_B2", "_1_B3", "_1_B4"]):
        href = sign_asset_href(it1, b)
        band_arrays.append(read_and_resample(href))
        band_names.append(f"{b}{suffix}")

    # Second date
    for b, suffix in zip(bands, ["_2_B1", "_2_B2", "_2_B3", "_2_B4"]):
        href = sign_asset_href(it2, b)
        band_arrays.append(read_and_resample(href))
        band_names.append(f"{b}{suffix}")

    stack = np.stack(band_arrays, axis=0)
    profile = dst_profile.copy()
    profile.update({"count": stack.shape[0]})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stack)
        # Add band name tags
        dst.update_tags(
            **{f"band_{i + 1}": name for i, name in enumerate(band_names)}
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select best Sentinel-2 scene pairs per tile for an AOI and year, "
            "then write 8-band stacks for each pair."
        )
    )

    parser.add_argument(
        "--aoi-shp",
        required=True,
        help="Path to AOI shapefile (geometries will be dissolved and reprojected to EPSG:4326).",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Calendar year to search Sentinel-2 imagery (YYYY).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for stacks and the summary CSV.",
    )
    parser.add_argument(
        "--collection",
        default="sentinel-2-l2a",
        help="STAC collection name to query (default: sentinel-2-l2a).",
    )
    parser.add_argument(
        "--min-month-gap",
        type=int,
        default=4,
        help="Minimum temporal gap between images in months (default: 4).",
    )
    parser.add_argument(
        "--cloud-thresholds",
        nargs="+",
        type=float,
        default=[5.0, 7.0, 9.0, 10.0],
        help=(
            "List of cloud cover thresholds to try in order (percent). "
            "Example: --cloud-thresholds 5 7 9 10"
        ),
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["B04", "B03", "B02", "B08"],
        help="Bands to include in the stack (exactly 4 expected, default: B04 B03 B02 B08).",
    )
    parser.add_argument(
        "--simplify-tol",
        type=float,
        default=0.0005,
        help="Optional geometry simplify tolerance in degrees (default: 0.0005).",
    )
    parser.add_argument(
        "--full-tile",
        action="store_true",
        help=(
            "If set, write full tile rasters and mask out pixels outside the AOI. "
            "If not set, write only the cropped AOI."
        ),
    )
    parser.add_argument(
        "--write-dtype",
        choices=["uint16", "float32"],
        default="uint16",
        help="Output data type for stacks (default: uint16).",
    )
    parser.add_argument(
        "--compress",
        choices=["deflate", "lzw"],
        default=None,
        help="Compression for output stacks (default: None).",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=0.0,
        help=(
            "Nodata value for full-tile outputs (only used when --full-tile is set). "
            "Default: 0.0"
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default="S2_best_pairs_summary.csv",
        help="Name of the summary CSV file (default: S2_best_pairs_summary.csv).",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    aoi_shp = Path(args.aoi_shp).expanduser().resolve()
    if not aoi_shp.exists():
        raise FileNotFoundError(f"AOI shapefile not found: {aoi_shp}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(args.bands) != 4:
        raise ValueError("This script currently expects exactly 4 bands.")

    print(f"[INFO] AOI shapefile:    {aoi_shp}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Year:             {args.year}")
    print(f"[INFO] Collection:       {args.collection}")
    print(f"[INFO] Min month gap:    {args.min_month_gap}")
    print(f"[INFO] Cloud thresholds: {args.cloud_thresholds}")
    print(f"[INFO] Bands:            {args.bands}")
    print(f"[INFO] Full tile:        {args.full_tile}")
    print(f"[INFO] Write dtype:      {args.write_dtype}")
    print(f"[INFO] Compress:         {args.compress}")
    print(f"[INFO] Nodata:           {args.nodata}")
    print(f"[INFO] Simplify tol:     {args.simplify_tol}")
    print("")

    # 1. Read AOI geometry and compute date range
    geom = read_geometry(str(aoi_shp), args.simplify_tol)
    start = f"{args.year}-01-01"
    end = f"{args.year}-12-31"

    # 2. Query Sentinel-2 items intersecting AOI and group by tile
    print("[INFO] Querying Sentinel-2 items from Planetary Computer...")
    all_items = all_items_for_geometry(
        geom,
        start,
        end,
        collection=args.collection,
    )
    by_tile = group_by_tile(all_items)
    tiles = sorted(by_tile.keys())

    if not tiles:
        print("No Sentinel-2 tiles intersect the input geometry for the given year.")
        return

    print(f"[INFO] Tiles intersecting geometry in {args.year}: {len(tiles)}")

    # 3. Pick best pair per tile under the cloud thresholds
    results: List[Dict] = []
    for tile in tiles:
        items_for_tile = by_tile[tile]
        found = None
        used_th = None
        for th in args.cloud_thresholds:
            candidate = pick_best_pair_in_tile(
                items_for_tile,
                min_month_gap=args.min_month_gap,
                cloud_cap=th,
            )
            if candidate:
                found = candidate
                used_th = th
                break
        if found:
            found["tile"] = tile
            found["threshold_used"] = used_th
            results.append(found)

    if not results:
        print("No valid pairs found for any tile under the provided thresholds.")
        return

    # Sort tiles by cloud_sum, then gap_days descending, then first date
    results.sort(
        key=lambda d: (d["cloud_sum"], -d["gap_days"], d["t1"])
    )

    print(f"[INFO] Tiles with best pairs found: {len(results)}")

    # 4. Write stacks and summary CSV
    summary_path = output_dir / args.summary_csv
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tile",
                "date1",
                "date2",
                "cloud1",
                "cloud2",
                "cloud_sum",
                "gap_days",
                "threshold_used",
                "stack_path",
            ]
        )

        for r in results:
            tile = r["tile"]
            t1: datetime = r["t1"]
            t2: datetime = r["t2"]

            t1s_compact = t1.strftime("%Y%m%d")
            t2s_compact = t2.strftime("%Y%m%d")
            stack_name = f"S2_Stack_{args.year}_{tile}_{t1s_compact}_{t2s_compact}.tif"
            stack_path = output_dir / stack_name

            print(f"\n[WRITE STACK] {stack_path}")
            stack_pair_to_tiff(
                it1=r["it1"],
                it2=r["it2"],
                clip_geom_wgs84=geom,
                out_path=stack_path,
                bands=args.bands,
                full_tile=args.full_tile,
                write_dtype=args.write_dtype,
                compress=args.compress,
                nodata=args.nodata,
            )

            writer.writerow(
                [
                    tile,
                    t1.strftime("%Y-%m-%d"),
                    t2.strftime("%Y-%m-%d"),
                    f"{r['c1']:.2f}",
                    f"{r['c2']:.2f}",
                    f"{r['cloud_sum']:.2f}",
                    r["gap_days"],
                    r["threshold_used"],
                    str(stack_path),
                ]
            )

    print(f"\n[DONE] Stacks and summary written to {summary_path}")


if __name__ == "__main__":
    main()
