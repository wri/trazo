#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 5 utility: select Sentinel-2 scene pairs per tile (tile list variant).

This tool is designed to reimplement and extend the logic from
"Create List of Sentinel2 Image Pairs based on Tile List.ipynb"
inside the tkt package.

Key features:

  * Operates on a list of Sentinel-2 tiles, either:
      - from a user provided tile shapefile, or
      - derived by intersecting a user AOI with the repo tile index
        at spatial/sentinel_2_index_shapefile.geojson.

  * Uses SOS and EOS rasters to define the preferred date range
    per tile, using exact extraction over tile polygons.

  * Falls back to a wider date range (the full calendar year)
    and progressively relaxes cloud cover constraints if needed.

  * Outputs a CSV listing one best pair per tile, with both STAC
    item IDs and basic metadata.

It does not run any model inference or write 8 band stacks.
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from pystac_client import Client
import planetary_computer as pc


# Defaults relative to the repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TILE_INDEX = _REPO_ROOT / "spatial" / "sentinel_2_index_shapefile.geojson"
DEFAULT_SOS_RASTER = _REPO_ROOT / "seasontifs" / "S2_SOS_WGS84.tif"
DEFAULT_EOS_RASTER = _REPO_ROOT / "seasontifs" / "S2_EOS_WGS84.tif"

DEFAULT_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass
class TilePair:
    tile_id: str
    it1: object
    it2: object
    t1: datetime
    t2: datetime
    c1: float
    c2: float
    cloud_sum: float
    gap_days: int
    cloud_threshold_used: float
    used_sos_eos_window: bool
    window_start: datetime
    window_end: datetime


def doy_to_date(year: int, doy: int) -> datetime:
    d = max(1, min(366, int(doy)))
    return datetime(year, 1, 1) + timedelta(days=d - 1)


def extract_mode_doy(raster_path: Path, geom_epsg4326) -> Optional[int]:
    """
    Extract a representative DOY value for a geometry using the mode
    of non zero, non nodata pixels.
    """
    if not raster_path.exists():
        return None

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster {raster_path} has no CRS")

        # Reproject geometry to raster CRS
        gseries = gpd.GeoSeries([geom_epsg4326], crs=4326).to_crs(src.crs)
        geom_proj = gseries.iloc[0]

        try:
            out, _ = mask(
                src,
                [mapping(geom_proj)],
                crop=True,
                filled=True,
                nodata=src.nodata if src.nodata is not None else 0,
            )
        except Exception:
            return None

        arr = out[0]
        nodata = src.nodata
        if nodata is not None:
            mask_valid = (arr != nodata) & (arr != 0)
        else:
            mask_valid = arr != 0

        vals = arr[mask_valid]
        if vals.size == 0:
            return None

        vals_int = vals.astype(np.int32)
        uniq, counts = np.unique(vals_int, return_counts=True)
        return int(uniq[np.argmax(counts)])


def stac_client(stac_url: str = DEFAULT_STAC_URL) -> Client:
    return Client.open(stac_url, modifier=pc.sign_inplace)


def search_items_for_tile(
    api: Client,
    tile_id: str,
    start_date: datetime,
    end_date: datetime,
    collection: str,
    cloud_cap: Optional[float],
) -> List:
    """
    Search Sentinel-2 items for a specific MGRS tile in a date range.
    """
    dt_range = f"{start_date.date()}/{end_date.date()}"

    query: Dict[str, Dict] = {
        "s2:mgrs_tile": {"eq": tile_id},
        "s2:nodata_pixel_percentage": {"lt": 100},
    }
    if cloud_cap is not None:
        query["eo:cloud_cover"] = {"lte": float(cloud_cap)}

    search = api.search(
        collections=[collection],
        datetime=dt_range,
        query=query,
        limit=500,
    )
    items = list(search.items())
    items.sort(key=lambda it: it.datetime)
    return items


def group_by_tile(items: Sequence) -> Dict[str, List]:
    """
    Group STAC items by s2:mgrs_tile.
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
) -> Optional[Dict]:
    """
    Given a list of STAC items for a single tile, pick the best pair.

    Ranking rules:
      1. Minimize (cloud1 + cloud2).
      2. If tie, maximize temporal gap (in days).
      3. If tie again, prefer earlier first date.
    """
    if len(items_for_tile) < 2:
        return None

    best: Optional[Dict] = None
    min_gap_days = int(round(30 * min_month_gap))

    for i, it1 in enumerate(items_for_tile[:-1]):
        t1 = it1.datetime
        c1 = float(it1.properties.get("eo:cloud_cover", 100.0))

        candidates = [
            it for it in items_for_tile[i + 1 :]
            if (it.datetime - t1).days >= min_gap_days
        ]
        if not candidates:
            continue

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


def compute_window_from_sos_eos(
    year: int,
    sos_doy: Optional[int],
    eos_doy: Optional[int],
    sos_buffer_days: int,
    eos_buffer_days: int,
) -> Tuple[datetime, datetime, bool]:
    """
    Derive a date window for a tile from SOS and EOS plus buffers.

    If SOS or EOS is missing, falls back to the full calendar year.
    Returns (start, end, used_sos_eos_flag).
    """
    full_start = datetime(year, 1, 1)
    full_end = datetime(year, 12, 31)

    if sos_doy is None or eos_doy is None:
        return full_start, full_end, False

    sos_date = doy_to_date(year, sos_doy)
    eos_date = doy_to_date(year, eos_doy)

    start = max(full_start, sos_date - timedelta(days=sos_buffer_days))
    end = min(full_end, eos_date + timedelta(days=eos_buffer_days))

    if start >= end:
        return full_start, full_end, False

    return start, end, True


def build_tiles_from_aoi(
    aoi_path: Path,
    tile_index_path: Path,
    tile_id_col: str,
) -> gpd.GeoDataFrame:
    """
    Intersect an AOI with the Sentinel-2 tile index to get a tile list.
    """
    aoi_gdf = gpd.read_file(aoi_path)
    if aoi_gdf.crs is None:
        raise ValueError(f"AOI file has no CRS: {aoi_path}")

    tile_gdf = gpd.read_file(tile_index_path)
    if tile_gdf.crs is None:
        # Assume the index is in EPSG:4326 if missing
        tile_gdf.set_crs(4326, inplace=True)

    tile_gdf = tile_gdf.to_crs(aoi_gdf.crs)

    # Spatial join: tiles that intersect AOI
    joined = gpd.sjoin(tile_gdf, aoi_gdf, how="inner", predicate="intersects")
    if tile_id_col not in joined.columns:
        raise ValueError(
            f"Tile ID column '{tile_id_col}' not found in tile index. "
            f"Columns: {list(joined.columns)}"
        )

    # Keep unique tiles and their geometry
    tiles_unique = joined.drop_duplicates(subset=[tile_id_col])[[tile_id_col, "geometry"]]
    tiles_unique = tiles_unique.set_index(tile_id_col, drop=False)
    tiles_unique = tiles_unique.to_crs(4326)

    return tiles_unique


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select Sentinel-2 scene pairs per tile using SOS/EOS derived "
            "date windows and progressive cloud constraints."
        )
    )

    # Tile sources
    parser.add_argument(
        "--tiles-shp",
        help="Path to a tile shapefile or GeoJSON with a tile ID column.",
    )
    parser.add_argument(
        "--tile-id-col",
        default="Name",
        help="Name of the tile ID column (default: Name).",
    )
    parser.add_argument(
        "--aoi-shp",
        help="Optional AOI shapefile; used if --build-tiles-from-aoi is set.",
    )
    parser.add_argument(
        "--build-tiles-from-aoi",
        action="store_true",
        help=(
            "If set, build the tile list by intersecting the AOI with the tile index "
            f"(default index: {DEFAULT_TILE_INDEX.name})."
        ),
    )
    parser.add_argument(
        "--tile-index",
        default=str(DEFAULT_TILE_INDEX),
        help=(
            "Path to Sentinel-2 tile index (GeoJSON or shapefile) used when "
            "--build-tiles-from-aoi is enabled. "
            f"Default: {DEFAULT_TILE_INDEX}"
        ),
    )

    # Year and STAC
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Calendar year to search Sentinel-2 imagery (YYYY).",
    )
    parser.add_argument(
        "--stac-url",
        default=DEFAULT_STAC_URL,
        help=f"STAC API URL (default: {DEFAULT_STAC_URL}).",
    )
    parser.add_argument(
        "--collection",
        default="sentinel-2-l2a",
        help="STAC collection name (default: sentinel-2-l2a).",
    )

    # SOS / EOS
    parser.add_argument(
        "--sos-raster",
        default=str(DEFAULT_SOS_RASTER),
        help=f"Path to SOS raster (default: {DEFAULT_SOS_RASTER}).",
    )
    parser.add_argument(
        "--eos-raster",
        default=str(DEFAULT_EOS_RASTER),
        help=f"Path to EOS raster (default: {DEFAULT_EOS_RASTER}).",
    )
    parser.add_argument(
        "--sos-buffer-days",
        type=int,
        default=30,
        help="Days before SOS to include in the search window (default: 30).",
    )
    parser.add_argument(
        "--eos-buffer-days",
        type=int,
        default=30,
        help="Days after EOS to include in the search window (default: 30).",
    )

    # Pair selection logic
    parser.add_argument(
        "--min-month-gap",
        type=int,
        default=5,
        help="Minimum temporal gap between images in months (default: 5).",
    )
    parser.add_argument(
        "--cloud-thresholds",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        help=(
            "List of cloud cover thresholds to try in order (percent). "
            "Example: --cloud-thresholds 0 1 2 3 4 5"
        ),
    )

    # Output
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to CSV where tile level pairs will be written.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    tile_index_path = Path(args.tile_index).expanduser().resolve()
    sos_raster = Path(args.sos_raster).expanduser().resolve()
    eos_raster = Path(args.eos_raster).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if args.build_tiles_from_aoi:
        if not args.aoi_shp:
            raise ValueError("You must provide --aoi-shp when using --build-tiles-from-aoi.")
        aoi_path = Path(args.aoi_shp).expanduser().resolve()
        if not aoi_path.exists():
            raise FileNotFoundError(f"AOI file not found: {aoi_path}")
        if not tile_index_path.exists():
            raise FileNotFoundError(f"Tile index not found: {tile_index_path}")
        tiles_gdf = build_tiles_from_aoi(
            aoi_path=aoi_path,
            tile_index_path=tile_index_path,
            tile_id_col=args.tile_id_col,
        )
    else:
        if not args.tiles_shp:
            raise ValueError(
                "You must provide either --tiles-shp, or use --build-tiles-from-aoi."
            )
        tiles_path = Path(args.tiles_shp).expanduser().resolve()
        if not tiles_path.exists():
            raise FileNotFoundError(f"Tile shapefile not found: {tiles_path}")

        tiles_gdf = gpd.read_file(tiles_path)
        if tiles_gdf.crs is None:
            raise ValueError(f"Tile file has no CRS: {tiles_path}")
        if args.tile_id_col not in tiles_gdf.columns:
            raise ValueError(
                f"Tile ID column '{args.tile_id_col}' not found in tile file. "
                f"Columns: {list(tiles_gdf.columns)}"
            )
        # Ensure unique tile ids
        tiles_gdf = tiles_gdf.dropna(subset=[args.tile_id_col])
        tiles_gdf = tiles_gdf.drop_duplicates(subset=[args.tile_id_col])
        tiles_gdf = tiles_gdf.to_crs(4326)

    if tiles_gdf.empty:
        raise ValueError("No tiles available after preprocessing.")

    print(f"[INFO] Number of tiles: {len(tiles_gdf)}")
    print(f"[INFO] Year:            {args.year}")
    print(f"[INFO] STAC URL:        {args.stac_url}")
    print(f"[INFO] Collection:      {args.collection}")
    print(f"[INFO] SOS raster:      {sos_raster}")
    print(f"[INFO] EOS raster:      {eos_raster}")
    print(f"[INFO] Cloud thresholds:{args.cloud_thresholds}")
    print(f"[INFO] Min month gap:   {args.min_month_gap}")
    print(f"[INFO] Output CSV:      {output_csv}")
    print("")

    api = stac_client(args.stac_url)

    results: List[TilePair] = []
    year = args.year

    for idx, row in tiles_gdf.iterrows():
        tile_id = str(row[args.tile_id_col])
        geom_4326 = row.geometry
        if geom_4326 is None or geom_4326.is_empty:
            print(f"[WARN] Tile {tile_id}: empty geometry, skipping.")
            continue

        sos_doy = extract_mode_doy(sos_raster, geom_4326)
        eos_doy = extract_mode_doy(eos_raster, geom_4326)
        start_window, end_window, used_sos_eos = compute_window_from_sos_eos(
            year=year,
            sos_doy=sos_doy,
            eos_doy=eos_doy,
            sos_buffer_days=args.sos_buffer_days,
            eos_buffer_days=args.eos_buffer_days,
        )

        print(
            f"[INFO] Tile {tile_id}: SOS={sos_doy}, EOS={eos_doy}, "
            f"window={start_window.date()} to {end_window.date()}, "
            f"used_sos_eos={used_sos_eos}"
        )

        pair_found: Optional[TilePair] = None

        # First pass: SOS/EOS window
        for cloud_th in args.cloud_thresholds:
            items = search_items_for_tile(
                api=api,
                tile_id=tile_id,
                start_date=start_window,
                end_date=end_window,
                collection=args.collection,
                cloud_cap=cloud_th,
            )
            if not items:
                continue

            best = pick_best_pair_in_tile(
                items_for_tile=items,
                min_month_gap=args.min_month_gap,
            )
            if best:
                pair_found = TilePair(
                    tile_id=tile_id,
                    it1=best["it1"],
                    it2=best["it2"],
                    t1=best["t1"],
                    t2=best["t2"],
                    c1=best["c1"],
                    c2=best["c2"],
                    cloud_sum=best["cloud_sum"],
                    gap_days=best["gap_days"],
                    cloud_threshold_used=cloud_th,
                    used_sos_eos_window=used_sos_eos,
                    window_start=start_window,
                    window_end=end_window,
                )
                break

        # Second pass: full calendar year if nothing found
        if pair_found is None:
            full_start = datetime(year, 1, 1)
            full_end = datetime(year, 12, 31)
            print(
                f"[INFO] Tile {tile_id}: falling back to full year "
                f"{full_start.date()} to {full_end.date()}."
            )
            for cloud_th in args.cloud_thresholds:
                items = search_items_for_tile(
                    api=api,
                    tile_id=tile_id,
                    start_date=full_start,
                    end_date=full_end,
                    collection=args.collection,
                    cloud_cap=cloud_th,
                )
                if not items:
                    continue

                best = pick_best_pair_in_tile(
                    items_for_tile=items,
                    min_month_gap=args.min_month_gap,
                )
                if best:
                    pair_found = TilePair(
                        tile_id=tile_id,
                        it1=best["it1"],
                        it2=best["it2"],
                        t1=best["t1"],
                        t2=best["t2"],
                        c1=best["c1"],
                        c2=best["c2"],
                        cloud_sum=best["cloud_sum"],
                        gap_days=best["gap_days"],
                        cloud_threshold_used=cloud_th,
                        used_sos_eos_window=False,
                        window_start=full_start,
                        window_end=full_end,
                    )
                    break

        if pair_found is None:
            print(f"[WARN] Tile {tile_id}: no valid pair found under given constraints.")
            continue

        results.append(pair_found)

    if not results:
        print("[INFO] No valid pairs found for any tile. No CSV written.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tile",
                "win_a_id",
                "win_b_id",
                "nodata_a",
                "nodata_b",
                "cloud1",
                "cloud2",
                "cloud_sum",
                "gap_days",
                "cloud_threshold_used",
                "used_sos_eos_window",
                "window_start",
                "window_end",
            ]
        )

        for tp in results:
            it1 = tp.it1
            it2 = tp.it2
            nodata_a = it1.properties.get("s2:nodata_pixel_percentage", None)
            nodata_b = it2.properties.get("s2:nodata_pixel_percentage", None)

            writer.writerow(
                [
                    tp.tile_id,
                    it1.id,
                    it2.id,
                    nodata_a,
                    nodata_b,
                    f"{tp.c1:.2f}",
                    f"{tp.c2:.2f}",
                    f"{tp.cloud_sum:.2f}",
                    tp.gap_days,
                    tp.cloud_threshold_used,
                    "yes" if tp.used_sos_eos_window else "no",
                    tp.window_start.date().isoformat(),
                    tp.window_end.date().isoformat(),
                ]
            )

    print(f"\n[DONE] Wrote tile pair CSV to {output_csv}")


if __name__ == "__main__":
    main()
