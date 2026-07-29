#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create windows and masks for 8-band chips.

For each <base> folder:
  - Scan <base>/sized256 for .tif/.tiff chips (no recursion)
  - Split each 8-band chip into:
        <base>/s2_images/window_a/<name>.tif  (bands 1..4)
        <base>/s2_images/window_b/<name>.tif  (bands 5..8)
  - Build masks in <base>/label_masks/:
        instance/         uint32 ids starting at 1
        semantic_2class/  0 background, 1 interior
        semantic_3class/  0 background, 1 interior, 2 thin boundary

Field selection:
  - Reads a field polygon layer for the base folder
  - Selects features that intersect the chip polygon, but does not clip
    to the chip, which prevents synthetic boundaries along chip edges

Optional filtering:
  - If a file named "<base.name>_boundingbox256.geojson" exists next to <base>,
    only chips whose stem matches the bbox "aoi_id" list are processed.
    If that GeoJSON lacks "aoi_id", the script tries to derive it from a
    filename-like column.

Example:
  python -m trazo.pt2_dataprep.make_masks_and_windows \
    --base-folder /data/regionA \
    --base-folder /data/regionB \
    --fields-shp-name-contains field \
    --boundary-px 1 \
    --overwrite
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box as shapely_box


# ----------------------------
# Helpers: discovery
# ----------------------------

def _sized256_dir(base_dir: Path) -> Path:
    return base_dir / "sized256"


def _bbox_geojson_path(base_dir: Path) -> Optional[Path]:
    cand = base_dir / f"{base_dir.name}_boundingbox256.geojson"
    return cand if cand.exists() else None


def _list_sized256_tifs(base_dir: Path, valid_exts: Sequence[str]) -> List[Path]:
    sized = _sized256_dir(base_dir)
    if not sized.exists() or not sized.is_dir():
        return []
    files: List[Path] = []
    for ext in valid_exts:
        files.extend(sorted(p for p in sized.glob(f"*{ext}") if p.is_file()))
    return files


def _ensure_structure(base_dir: Path) -> Dict[str, Path]:
    lm = base_dir / "label_masks"
    dirs = {
        "inst": lm / "instance",
        "sem2": lm / "semantic_2class",
        "sem3": lm / "semantic_3class",
        "win_a": base_dir / "s2_images" / "window_a",
        "win_b": base_dir / "s2_images" / "window_b",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ----------------------------
# Helpers: IO
# ----------------------------

def _write_window_subset(
    src: rasterio.io.DatasetReader,
    out_path: Path,
    bands: List[int],
    overwrite: bool
) -> None:
    meta = src.meta.copy()
    meta.update({
        "count": len(bands),
        "compress": "deflate",
        "tiled": True,
        "blockxsize": min(256, src.width),
        "blockysize": min(256, src.height),
        "nodata": src.nodata
    })
    if out_path.exists():
        if not overwrite:
            return
        out_path.unlink()
    data = src.read(bands)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data)


def _write_mask(out_path: Path, arr: np.ndarray, src_profile: dict, dtype: str, overwrite: bool, nodata: int = 0) -> None:
    prof = src_profile.copy()
    prof.update({
        "count": 1,
        "dtype": dtype,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": min(256, prof.get("width", 256)),
        "blockysize": min(256, prof.get("height", 256))
    })
    # Clean non-applicable keys for masks
    for k in ("photometric",):
        if k in prof:
            prof.pop(k)
    if out_path.exists():
        if not overwrite:
            return
        out_path.unlink()
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr, 1)


# ----------------------------
# Helpers: vector prep
# ----------------------------

def _read_fields(fields_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(fields_path)
    # drop typical shape columns if present
    lower = {c.lower(): c for c in gdf.columns}
    to_drop = [lower[k] for k in ("shape_area", "shape_length", "shape_leng") if k in lower]
    if to_drop:
        gdf = gdf.drop(columns=to_drop)
    # keep only polygonal
    if "geometry" in gdf:
        try:
            gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        except Exception:
            pass
    return gdf


def _pick_fields_path(
    base_dir: Path,
    fields_shp: Optional[Path],
    name_contains: Optional[str]
) -> Optional[Path]:
    if fields_shp:
        return fields_shp if Path(fields_shp).exists() else None
    # auto discover first .shp
    shps = sorted(base_dir.glob("*.shp"))
    if not shps:
        return None
    if name_contains:
        candidates = [p for p in shps if name_contains.lower() in p.stem.lower()]
        if candidates:
            return candidates[0]
    # fallback: prefer any with "field" in name
    field_like = [p for p in shps if "field" in p.stem.lower()]
    if field_like:
        return field_like[0]
    # otherwise first shapefile
    return shps[0]


def _read_bbox_ids(bbox_path: Path) -> Optional[set]:
    try:
        gdf = gpd.read_file(bbox_path)
        if "aoi_id" not in gdf.columns:
            def _stem(row):
                for col in row.index:
                    low = str(col).lower()
                    if low in ("filename","location","fn","file","name","id","grid_id","grid","stem"):
                        v = row[col]
                        if isinstance(v, str) and v:
                            from os.path import basename, splitext
                            return splitext(basename(v))[0]
                return None
            gdf["aoi_id"] = gdf.apply(_stem, axis=1)
        gdf = gdf.loc[gdf["aoi_id"].notna()].copy()
        return set(gdf["aoi_id"].astype(str).tolist())
    except Exception:
        return None


# ----------------------------
# Helpers: selection and masks
# ----------------------------

def _chip_polygon(src: rasterio.io.DatasetReader):
    left, bottom, right, top = src.bounds
    return shapely_box(left, bottom, right, top)


def _select_fields_for_chip(fields: gpd.GeoDataFrame, chip_poly, chip_crs) -> gpd.GeoDataFrame:
    gdf = fields
    if gdf.crs != chip_crs:
        try:
            gdf = gdf.to_crs(chip_crs)
        except Exception:
            pass
    inter = gdf[gdf.intersects(chip_poly)]
    if inter.empty:
        return inter
    inter = inter.copy()
    inter["geometry"] = inter.geometry.apply(lambda g: g if (g and g.is_valid) else (g.buffer(0) if g else g))
    return inter


def _write_instance_mask(out_path: Path, fields_sel: gpd.GeoDataFrame, h: int, w: int, transform, profile, overwrite: bool) -> None:
    if fields_sel.empty:
        arr = np.zeros((h, w), dtype=np.uint32)
        _write_mask(out_path, arr, profile, "uint32", overwrite, 0)
        return
    ids = np.arange(1, len(fields_sel) + 1, dtype=np.uint32)
    shapes = [(g, int(i)) for g, i in zip(fields_sel.geometry, ids) if g is not None and not g.is_empty]
    if not shapes:
        arr = np.zeros((h, w), dtype=np.uint32)
    else:
        arr = rasterize(
            shapes,
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype="uint32",
            all_touched=False
        )
    _write_mask(out_path, arr, profile, "uint32", overwrite, 0)


def _write_sem2_mask(out_path: Path, fields_sel: gpd.GeoDataFrame, h: int, w: int, transform, profile, overwrite: bool) -> None:
    if fields_sel.empty:
        arr = np.zeros((h, w), dtype=np.uint8)
        _write_mask(out_path, arr, profile, "uint8", overwrite, 0)
        return
    shapes = [(g, 1) for g in fields_sel.geometry if g is not None and not g.is_empty]
    if not shapes:
        arr = np.zeros((h, w), dtype=np.uint8)
    else:
        arr = rasterize(
            shapes,
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False
        )
    _write_mask(out_path, arr, profile, "uint8", overwrite, 0)


def _write_sem3_mask_vector(
    out_path: Path,
    fields_sel: gpd.GeoDataFrame,
    h: int,
    w: int,
    transform,
    profile,
    boundary_px: int,
    overwrite: bool
) -> None:
    if fields_sel.empty:
        _write_mask(out_path, np.zeros((h, w), dtype=np.uint8), profile, "uint8", overwrite, 0)
        return

    geoms = [g if (g and g.is_valid) else (g.buffer(0) if g else None) for g in fields_sel.geometry]
    geoms = [g for g in geoms if g and not g.is_empty]
    if not geoms:
        _write_mask(out_path, np.zeros((h, w), dtype=np.uint8), profile, "uint8", overwrite, 0)
        return

    # 1) Interior
    interior = rasterize(
        [(g, 1) for g in geoms],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False
    )

    # 2) Thin boundary via buffering of geometry boundaries in pixel units
    px = max(abs(transform.a), abs(transform.e))
    radius = max(0.0, float(boundary_px)) * px
    buffered_bounds = []
    if radius > 0.0:
        for g in geoms:
            b = g.boundary.buffer(radius, cap_style=1, join_style=2)
            if not b.is_empty:
                buffered_bounds.append((b, 2))
    boundary = np.zeros((h, w), dtype=np.uint8)
    if buffered_bounds:
        boundary = rasterize(
            buffered_bounds,
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False
        )

    # 3) Compose boundary overwriting interior
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[interior == 1] = 1
    mask[boundary == 2] = 2
    _write_mask(out_path, mask, profile, "uint8", overwrite, 0)


# ----------------------------
# Processing one base
# ----------------------------

def process_base(
    base_dir: Path,
    fields_shp: Optional[Path],
    fields_shp_name_contains: Optional[str],
    valid_exts: Sequence[str],
    boundary_px: int,
    overwrite: bool
) -> None:
    print(f"\n=== Make masks and windows in: {base_dir} ===")
    if not base_dir.exists():
        print("  [WARN] base missing; skip")
        return

    dirs = _ensure_structure(base_dir)
    sized_tifs = _list_sized256_tifs(base_dir, valid_exts)
    if not sized_tifs:
        print("  [WARN] no chips in sized256; skip")
        return

    bbox_ids = None
    bbox_path = _bbox_geojson_path(base_dir)
    if bbox_path:
        bbox_ids = _read_bbox_ids(bbox_path)

    fields_path = _pick_fields_path(base_dir, fields_shp, fields_shp_name_contains)
    if fields_path is None:
        print("  [WARN] could not locate a field polygon shapefile; skip base")
        return

    try:
        fields = _read_fields(fields_path)
    except Exception as e:
        print(f"  [WARN] failed to read fields: {e}; skip base")
        return

    total = len(sized_tifs)
    for i, tif in enumerate(sized_tifs, 1):
        aoi_id = tif.stem
        if bbox_ids and aoi_id not in bbox_ids:
            continue

        with rasterio.open(tif) as src:
            if src.count < 8:
                print(f"  [WARN] {tif.name} has {src.count} bands; expected at least 8")
                continue

            # Split windows
            _write_window_subset(src, dirs["win_a"] / tif.name, [1, 2, 3, 4], overwrite)
            _write_window_subset(src, dirs["win_b"] / tif.name, [5, 6, 7, 8], overwrite)

            # Mask geometry and metadata
            chip_poly = _chip_polygon(src)
            chip_crs = src.crs
            h, w = src.height, src.width
            transform = src.transform
            profile = src.profile

            # Select intersecting fields without clipping
            fields_sel = _select_fields_for_chip(fields, chip_poly, chip_crs)

            # Write masks
            _write_instance_mask(dirs["inst"] / tif.name, fields_sel, h, w, transform, profile, overwrite)
            _write_sem2_mask(dirs["sem2"] / tif.name, fields_sel, h, w, transform, profile, overwrite)
            _write_sem3_mask_vector(dirs["sem3"] / tif.name, fields_sel, h, w, transform, profile, boundary_px, overwrite)

        if (i % 50 == 0) or (i == total):
            print(f"  progress: {i}/{total}")

    print("  done.")


# ----------------------------
# CLI
# ----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split 8-band chips into window_a/window_b and create instance and semantic masks."
    )
    p.add_argument(
        "--base-folder",
        action="append",
        required=True,
        help="Base folder containing a 'sized256' directory with chips. Repeat for multiple bases."
    )
    p.add_argument(
        "--fields-shp",
        default=None,
        help="Optional path to a field polygon shapefile to use for all bases. If omitted, auto-discovers in each base."
    )
    p.add_argument(
        "--fields-shp-name-contains",
        default=None,
        help="When auto-discovering, prefer shapefiles whose name contains this substring."
    )
    p.add_argument(
        "--valid-exts",
        nargs="+",
        default=[".tif", ".tiff"],
        help="Chip filename extensions to include."
    )
    p.add_argument(
        "--boundary-px",
        type=int,
        default=1,
        help="Boundary thickness in pixels for semantic_3class mask."
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs."
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    bases = [Path(b).expanduser().resolve() for b in args.base_folder]
    fields_shp = Path(args.fields_shp).expanduser().resolve() if args.fields_shp else None

    for base in bases:
        process_base(
            base_dir=base,
            fields_shp=fields_shp,
            fields_shp_name_contains=args.fields_shp_name_contains,
            valid_exts=args.valid_exts,
            boundary_px=args.boundary_px,
            overwrite=args.overwrite
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
