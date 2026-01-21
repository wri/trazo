#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Sentinel-2 tilepair selector for Step 5.

Features:
1. Load tiles explicitly provided by user, OR intersect user AOI with the
   internal sentinel_2_index_shapefile.geojson included in the repo.

2. Select two best image pairs per tile using:
      - Year filter
      - Cloud threshold ladder
      - Minimum temporal separation (months)
      - Increasing search windows if needed

3. Output: CSV containing tile, window A, window B, nodata %
"""

import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd
from datetime import datetime
from pystac_client import Client
import planetary_computer


S2_INDEX_PATH = Path("/workspaces/toolkit-for-traceability/spatial/sentinel_2_index_shapefile.geojson")


def load_tiles_from_intersection(aoi_path: Path) -> list:
    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError("AOI file is empty.")

    s2idx = gpd.read_file(S2_INDEX_PATH)
    if s2idx.empty:
        raise ValueError("Sentinel-2 index file is empty.")

    aoi = aoi.to_crs(s2idx.crs)
    inter = gpd.overlay(s2idx, aoi, how="intersection")

    if inter.empty:
        raise ValueError("AOI intersects zero Sentinel-2 tiles.")

    return sorted(inter["Name"].unique().tolist())


def load_tiles_from_file(tile_file: Path) -> list:
    gdf = gpd.read_file(tile_file)
    if "Name" not in gdf.columns:
        raise ValueError("Tile shapefile must have a 'Name' column.")
    tiles = sorted(gdf["Name"].unique().tolist())
    return tiles


def pick_pairs_from_stac(
    stac: Client,
    tile_id: str,
    year: int,
    cloud_ladder,
    min_month_gap_ladder,
):
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)

    for cloud, gap in zip(cloud_ladder, min_month_gap_ladder):
        search = stac.search(
            collections=["sentinel-2-l2a"],
            datetime=f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
            query={
                "s2:mgrs_tile": {"eq": tile_id},
                "eo:cloud_cover": {"lte": cloud},
                "s2:nodata_pixel_percentage": {"lt": 100},
            },
            limit=200,
        )
        items = list(search.items())
        items.sort(key=lambda x: x.datetime)

        pairs = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                d1 = items[i].datetime
                d2 = items[j].datetime
                if abs((d2 - d1).days) >= gap * 30:
                    id1 = items[i].id
                    id2 = items[j].id
                    n1 = items[i].properties.get("s2:nodata_pixel_percentage", None)
                    n2 = items[j].properties.get("s2:nodata_pixel_percentage", None)
                    pairs.append((id1, id2, n1, n2))
                    if len(pairs) >= 2:
                        return pairs
    return []


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Advanced tilepair selection with optional AOI intersection."
    )
    parser.add_argument("--tile-file", type=str, default=None, help="User-provided tile shapefile/geojson.")
    parser.add_argument("--aoi", type=str, default=None, help="User AOI; intersects sentinel index.")
    parser.add_argument("--year", type=int, required=True, help="Year to search.")
    parser.add_argument("--output", type=str, required=True, help="Output CSV.")
    parser.add_argument("--min-cloud", type=int, default=0)
    parser.add_argument("--min-month-gap", type=int, default=5)
    parser.add_argument("--expand-cloud-steps", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--expand-gap-steps", nargs="*", type=int, default=[5, 3, 3, 3, 3])
    args = parser.parse_args(args)

    if not args.tile_file and not args.aoi:
        raise ValueError("You must provide either --tile-file or --aoi.")

    if args.tile_file:
        tiles = load_tiles_from_file(Path(args.tile_file))
    else:
        tiles = load_tiles_from_intersection(Path(args.aoi))

    stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)

    cloud_ladder = [args.min-cloud] + args.expand_cloud_steps
    gap_ladder = [args.min_month_gap] + args.expand_gap_steps

    results = []
    for tile in tiles:
        pairs = pick_pairs_from_stac(stac, tile, args.year, cloud_ladder, gap_ladder)
        for p in pairs:
            results.append([tile, p[0], p[1], p[2], p[3]])

    df = pd.DataFrame(results, columns=["Tile", "WindowA", "WindowB", "NodataA", "NodataB"])
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
