#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Offline end-to-end smoke test for the Trazo install.

Builds synthetic field polygons and synthetic Sentinel-2-shaped chips, then runs
the real Step 1.1 gridding code and the whole Step 2 chain over them. Nothing is
downloaded: no Planetary Computer, no STAC, no Sentinel-2. The point is to answer
"is my install wired up correctly?" in under a minute, instead of finding out an
hour into an imagery pull.

Run it with::

    trazo-smoke                     # uses a temp directory, cleans up after
    trazo-smoke --work-dir ./smoke  # keeps the outputs so you can look at them

Requires the ``pt2`` extra (``pip install -e ".[pt2]"``) for rasterio.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

# Synthetic AOI: a small block in Mato Grosso, Brazil. Chosen so the default
# Brazil Albers reprojection in Step 1.1 is exercised on realistic coordinates.
AOI_LON = -55.5
AOI_LAT = -12.5

CHIP_PX = 256
CHIP_RES_M = 10.0
N_CHIPS = 2


def _log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


def make_fields(path: Path, n: int = 12) -> None:
    """Write a small polygon layer of fake fields around the synthetic AOI."""
    import geopandas as gpd
    from shapely.geometry import box

    rng = np.random.default_rng(7)
    polys = []
    for _ in range(n):
        # Fields between roughly 200 m and 900 m on a side, in degrees.
        w = rng.uniform(0.002, 0.009)
        h = rng.uniform(0.002, 0.009)
        x0 = AOI_LON + rng.uniform(-0.02, 0.02)
        y0 = AOI_LAT + rng.uniform(-0.02, 0.02)
        polys.append(box(x0, y0, x0 + w, y0 + h))

    gdf = gpd.GeoDataFrame({"field_id": range(n)}, geometry=polys, crs="EPSG:4326")
    gdf.to_file(path)
    _log(f"wrote {n} synthetic fields -> {path.name}")


def run_gridding(fields_path: Path, out_dir: Path) -> Path:
    """Run the real Step 1.1 gridding and return the grid path."""
    from trazo.pt1_createdata.gridding import create_grid

    create_grid(
        input_path=str(fields_path),
        output_dir=str(out_dir),
        cell_size_meters=2560.0,
        input_epsg_if_missing=4326,
        target_projected_epsg=None,
        use_brazil_albers=True,
        output_format="shp",
    )
    grids = sorted(out_dir.glob("*_grid.shp"))
    if not grids:
        raise AssertionError(f"Step 1.1 produced no *_grid.shp in {out_dir}")
    return grids[0]


def check_grid_columns(grid_path: Path) -> None:
    import geopandas as gpd

    grid = gpd.read_file(grid_path)
    missing = {"chip_id", "chip_area", "cov_area", "cov_pct"} - set(grid.columns)
    if missing:
        raise AssertionError(f"grid is missing expected columns: {sorted(missing)}")
    _log(f"grid has {len(grid)} cells with chip_id column")


def write_synthetic_chips(base: Path, grid_path: Path) -> List[str]:
    """Write N_CHIPS fake 4-band uint16 chips into window_a and window_b.

    These stand in for the Step 1.2 Planetary Computer pull. Band order matches
    the real pipeline (B04, B03, B02, B08) and the geotransform is a real UTM
    grid so downstream reprojection is genuinely exercised.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin

    from trazo.pt1_createdata.gridding import pick_utm_epsg_from_lonlat

    grid = gpd.read_file(grid_path)
    epsg = pick_utm_epsg_from_lonlat(AOI_LON, AOI_LAT)

    win_a = base / "window_a"
    win_b = base / "window_b"
    win_a.mkdir(parents=True, exist_ok=True)
    win_b.mkdir(parents=True, exist_ok=True)

    # Centroids are computed in the grid's own projected CRS, then moved to the
    # UTM zone the synthetic chips are written in.
    utm = grid.geometry.centroid.to_crs(epsg=epsg)

    rng = np.random.default_rng(11)
    names: List[str] = []
    for i in range(min(N_CHIPS, len(grid))):
        chip_id = int(grid.iloc[i]["chip_id"])
        name = f"chip_{chip_id}.tif"
        cx, cy = float(utm.iloc[i].x), float(utm.iloc[i].y)
        half = CHIP_PX * CHIP_RES_M / 2.0
        transform = from_origin(cx - half, cy + half, CHIP_RES_M, CHIP_RES_M)

        for window_dir, offset in ((win_a, 0), (win_b, 400)):
            data = (
                rng.integers(200, 3000, size=(4, CHIP_PX, CHIP_PX), dtype=np.uint16)
                + np.uint16(offset)
            )
            profile = {
                "driver": "GTiff",
                "height": CHIP_PX,
                "width": CHIP_PX,
                "count": 4,
                "dtype": "uint16",
                "crs": f"EPSG:{epsg}",
                "transform": transform,
            }
            with rasterio.open(window_dir / name, "w", **profile) as dst:
                dst.write(data)
        names.append(name)

    _log(f"wrote {len(names)} synthetic chip pairs into window_a/ and window_b/")
    return names


def run_step2(base: Path, fields_path: Path) -> None:
    """Run the whole Step 2 chain in the documented order."""
    from trazo.pt2_dataprep import (
        build_chips_parquet,
        chips_to_bboxes,
        make_masks_and_windows,
        pair_stacks,
        resize_chips_256,
        scale_uint16,
    )

    _log("pair-stacks")
    pair_stacks.main([
        "--window-a-dir", str(base / "window_a"),
        "--window-b-dir", str(base / "window_b"),
        "--out-dir", str(base),
        "--overwrite",
    ])

    _log("resize-256")
    resize_chips_256.main(["--base-folder", str(base), "--overwrite"])

    _log("chips-bboxes")
    chips_to_bboxes.main(["--folder", str(base)])

    _log("make-masks")
    make_masks_and_windows.main([
        "--base-folder", str(base),
        "--fields-shp", str(fields_path),
        "--boundary-px", "1",
        "--overwrite",
    ])

    _log("chips-parquet")
    build_chips_parquet.main([
        "--base-folder", str(base),
        "--fields-shp", str(fields_path),
        "--split-train", "0.85",
        "--split-val", "0.15",
        "--split-test", "0.0",
        "--overwrite",
    ])

    _log("scale-u16")
    scale_uint16.main(["--base-folder", str(base), "--overwrite"])


def check_outputs(base: Path) -> None:
    expectations = {
        "8-band stacks": list(base.glob("*.tif")),
        "sized256 chips": list((base / "sized256").glob("*.tif")),
        "chip bbox GeoJSON": list(base.glob("*.geojson")),
        "window_a splits": list((base / "s2_images" / "window_a").glob("*.tif")),
        "window_b splits": list((base / "s2_images" / "window_b").glob("*.tif")),
        "instance masks": list((base / "label_masks" / "instance").glob("*.tif")),
        "3-class masks": list((base / "label_masks" / "semantic_3class").glob("*.tif")),
        "chip parquet": list(base.glob("*.parquet")),
    }
    failures = [name for name, hits in expectations.items() if not hits]
    for name, hits in expectations.items():
        mark = "ok  " if hits else "MISS"
        print(f"  [{mark}] {name}: {len(hits)}")
    if failures:
        raise AssertionError(f"missing Step 2 outputs: {failures}")


def run(work_dir: Optional[Path] = None) -> Path:
    """Run the full smoke path. Returns the working directory used."""
    created_temp = work_dir is None
    base = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="trazo-smoke-"))
    base.mkdir(parents=True, exist_ok=True)

    try:
        _log(f"working in {base}")
        fields_path = base / "fields.shp"
        make_fields(fields_path)

        grid_path = run_gridding(fields_path, base)
        check_grid_columns(grid_path)

        write_synthetic_chips(base, grid_path)
        run_step2(base, fields_path)
        check_outputs(base)

        _log("PASS - install is wired up end to end (Step 1.1 + Step 2)")
        return base
    finally:
        if created_temp and work_dir is None:
            shutil.rmtree(base, ignore_errors=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline smoke test: synthetic fields and chips through Step 1.1 "
            "gridding and the full Step 2 chain. No network access required."
        )
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory to work in. Default: a temp directory that is deleted afterwards.",
    )
    args = parser.parse_args(argv)

    try:
        run(Path(args.work_dir) if args.work_dir else None)
    except Exception as exc:  # noqa: BLE001 - smoke test reports, does not re-raise
        print(f"[smoke] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
