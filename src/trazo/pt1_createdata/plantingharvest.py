#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil.relativedelta import relativedelta
import urllib.request

import numpy as np
import geopandas as gpd
import shapely.geometry as sgeom
import rasterio
import rasterio.mask
from rasterio.warp import transform_geom

import pystac
import pystac_client
from pystac.extensions.eo import EOExtension as eo  # noqa: F401 (imported for side effects)
import planetary_computer

import odc.stac
import rioxarray  # noqa: F401 (registers the .rio accessor for xarray)

from tqdm import tqdm


# =====================================================================
# DEFAULTS (OVERRIDABLE VIA CLI)
# =====================================================================

# NODATA filtering behavior for STAC scene filtering
NODATA_FILTER_MODE = "none"  # "none" | "eq0" | "lt"
NODATA_LT = 50.0

# Temporal offsets around SOS/EOS (kept as advanced knobs, documented in help epilog)
PLANTING_TARGET_OFFSET = relativedelta(days=45)
PLANTING_TARGET_TOL_DAYS = 7
PLANTING_PREF_MIN_OFFSET = relativedelta(days=14)
PLANTING_PREF_MAX_OFFSET = relativedelta(days=30)

HARVEST_TARGET_OFFSET = relativedelta(months=1)
HARVEST_TARGET_TOL_DAYS = 7
HARVEST_PREF_MIN_OFFSET = relativedelta(days=14)
HARVEST_PREF_MAX_OFFSET = relativedelta(days=30)

# Bands and geometry
BANDS_OF_INTEREST = ["B04", "B03", "B02", "B08"]
DEFAULT_CHIP_SIZE_PX = 256

# Thread batch size for parallel feature processing
DEFAULT_BBOX_BATCH_SIZE = 10

# Cloud / scene defaults (overridable)
DEFAULT_SCENE_CLOUD_THRESHOLD_PCT = 90.0
DEFAULT_PATCH_CLOUD_THRESHOLD_FRAC = 0.08
DEFAULT_TARGET_CLOUD_MAX_FRAC = 0.01
DEFAULT_PREFERRED_CLOUD_MAX_FRAC = 0.02

# Planting / harvest span and expansion defaults
DEFAULT_PLANTING_SPAN_MONTHS = 2
DEFAULT_HARVEST_SPAN_MONTHS = 2
DEFAULT_MAX_EXPAND_MONTHS_PLANTING = 5
DEFAULT_MAX_EXPAND_MONTHS_HARVEST = 5

# Deg corresponding to 10 m at equator, used if working in EPSG:4326
DEG_10M = 10.0 / 111000.0

# SOS/EOS filenames (downloaded if not present)
SOS_FILENAME = "S2_SOS_WGS84.tif"
EOS_FILENAME = "S2_EOS_WGS84.tif"

# Remote sources
UC_GITHUB_BASE = (
    "https://raw.githubusercontent.com/ucg-uv/research_products/"
    "main/cropcalendars_phase2/smoothed"
)
WRI_GITHUB_BASE = (
    "https://raw.githubusercontent.com/wri/toolkit-for-traceability/main/seasontifs"
)


# =====================================================================
# EXCEPTIONS / SMALL UTILS
# =====================================================================

class WindowEmpty(Exception):
    pass


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def first_jan(year: int) -> datetime:
    return datetime(year, 1, 1)


def doy_to_date(year: int, doy: int) -> datetime:
    d = max(1, min(366, int(doy)))
    return first_jan(year) + relativedelta(days=d - 1)


def parse_chip_name(raw: str) -> str:
    if not raw:
        return "chip_unknown"
    raw = str(raw)
    if raw[0].isalpha():
        return raw
    digits = "".join(ch for ch in raw if ch.isdigit())
    return f"chip_{digits or '0'}"


# =====================================================================
# SOS/EOS DOWNLOAD AND RESOLUTION
# =====================================================================

def _download_file(url: str, dest: Path, verbose: bool = True) -> bool:
    try:
        if verbose:
            print(f"[season] Downloading {url} -> {dest}")
        ensure_dir(dest.parent)
        with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        if verbose:
            print(f"[season] Failed to download {url}: {e}")
        return False


def ensure_season_tifs(season_dir: Path, verbose: bool = True) -> Tuple[Path, Path]:
    """
    Ensure SOS/EOS rasters exist in `season_dir`.

    Order:
      1) If both exist locally, use them.
      2) Try to download from ucg-uv/research_products cropcalendars_phase2/smoothed.
      3) If that fails, try wri/toolkit-for-traceability/seasontifs.
      4) If still missing, raise an error.
    """
    ensure_dir(season_dir)
    sos_path = season_dir / SOS_FILENAME
    eos_path = season_dir / EOS_FILENAME

    if sos_path.exists() and eos_path.exists():
        if verbose:
            print(f"[season] Using existing SOS/EOS rasters in {season_dir}")
        return sos_path, eos_path

    # Try primary ucg-uv repo
    sos_ok = _download_file(f"{UC_GITHUB_BASE}/{SOS_FILENAME}", sos_path, verbose=verbose)
    eos_ok = _download_file(f"{UC_GITHUB_BASE}/{EOS_FILENAME}", eos_path, verbose=verbose)

    if sos_ok and eos_ok:
        return sos_path, eos_path

    # Fallback: wri toolkit-for-traceability repo
    sos_ok = sos_ok or _download_file(f"{WRI_GITHUB_BASE}/{SOS_FILENAME}", sos_path, verbose=verbose)
    eos_ok = eos_ok or _download_file(f"{WRI_GITHUB_BASE}/{EOS_FILENAME}", eos_path, verbose=verbose)

    if sos_ok and eos_ok:
        return sos_path, eos_path

    raise FileNotFoundError(
        "Unable to obtain SOS/EOS rasters from either:\n"
        f"  {UC_GITHUB_BASE}\n"
        f"or\n"
        f"  {WRI_GITHUB_BASE}\n"
        f"Please place {SOS_FILENAME} and {EOS_FILENAME} in {season_dir} manually."
    )


# =====================================================================
# STAC / RASTER HELPERS
# =====================================================================

def stac_client() -> pystac_client.Client:
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def stac_search_items(
    catalog: pystac_client.Client,
    bbox_epsg4326: Tuple[float, float, float, float],
    datetime_range: str,
    scene_cloud_threshold_pct: float,
) -> List[pystac.Item]:
    q: Dict[str, Dict[str, float]] = {"eo:cloud_cover": {"lt": float(scene_cloud_threshold_pct)}}
    if NODATA_FILTER_MODE == "eq0":
        q["s2:nodata_pixel_percentage"] = {"eq": 0}
    elif NODATA_FILTER_MODE == "lt":
        q["s2:nodata_pixel_percentage"] = {"lt": float(NODATA_LT)}

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox_epsg4326,
        datetime=datetime_range,
        query=q,
    )
    return list(search.items())


def scene_contains_box(item: pystac.Item, box_epsg4326: sgeom.Polygon) -> bool:
    scene_bounds = sgeom.shape(item.geometry)
    return scene_bounds.contains(box_epsg4326) or scene_bounds.buffer(1e-12).contains(box_epsg4326)


def chip_cloud_fraction_at_box(item: pystac.Item, box_epsg4326: sgeom.Polygon) -> float:
    scl_url = item.assets["SCL"].href
    with rasterio.open(scl_url) as f:
        geom = sgeom.mapping(box_epsg4326)
        warped_geom = transform_geom("EPSG:4326", f.crs, geom, precision=10)
        out_mask, _ = rasterio.mask.mask(f, [warped_geom], crop=True, filled=True)
        arr = out_mask.squeeze()
    valid = (arr != 0)
    if not np.any(valid):
        raise WindowEmpty("No SCL pixels in window")
    vals = arr[valid]
    cloudy = np.isin(vals, [8, 9])
    return float(np.sum(cloudy)) / float(vals.size)


def extract_mode_doy_from_raster(
    raster_path: str,
    geom_epsg4326: sgeom.base.BaseGeometry,
) -> Optional[int]:
    if not os.path.exists(raster_path):
        return None
    with rasterio.open(raster_path) as src:
        gjson = sgeom.mapping(geom_epsg4326)
        g_reproj = transform_geom("EPSG:4326", src.crs, gjson, precision=10)
        vals_poly = None
        try:
            out, _ = rasterio.mask.mask(
                src,
                [g_reproj],
                crop=True,
                filled=True,
                nodata=src.nodata if src.nodata is not None else 0,
            )
            arr = out[0]
            if src.nodata is not None:
                mask_valid = (arr != src.nodata) & (arr != 0)
            else:
                mask_valid = (arr != 0)
            vals = arr[mask_valid]
            if vals.size > 0:
                vals_poly = vals.astype(np.int32)
        except Exception:
            vals_poly = None

        if vals_poly is not None and vals_poly.size > 0:
            uniq, counts = np.unique(vals_poly, return_counts=True)
            return int(uniq[np.argmax(counts)])

        try:
            centroid = sgeom.shape(geom_epsg4326).centroid
            pt_gj = sgeom.mapping(centroid)
            pt_reproj = transform_geom("EPSG:4326", src.crs, pt_gj, precision=10)
            for val in src.sample([(pt_reproj["coordinates"][0], pt_reproj["coordinates"][1])]):
                v = val[0]
                if src.nodata is not None and v == src.nodata:
                    continue
                if int(v) == 0:
                    continue
                return int(v)
        except Exception:
            return None
    return None


def build_datetime_range_planting(year: int, sos_doy: Optional[int], span_months: int) -> Tuple[str, Optional[datetime], Optional[datetime]]:
    if sos_doy is None:
        return f"{year}-01-01/{year}-12-31", None, None
    start_dt = doy_to_date(year, sos_doy)
    end_dt = start_dt + relativedelta(months=span_months)
    return f"{start_dt.date()}/{end_dt.date()}", start_dt, end_dt


def build_datetime_range_harvest(year: int, eos_doy: Optional[int], span_months: int) -> Tuple[str, Optional[datetime], Optional[datetime]]:
    if eos_doy is None:
        return f"{year}-01-01/{year}-12-31", None, None
    end_dt = doy_to_date(year, eos_doy)
    start_dt = end_dt - relativedelta(months=span_months)
    return f"{start_dt.date()}/{end_dt.date()}", start_dt, end_dt


def odc_load_chip_in_target_crs(
    item: pystac.Item,
    box_src_crs_geom: sgeom.Polygon,
    src_crs: str,
    target_crs: str,
    chip_size_px: int,
):
    if src_crs.upper() != "EPSG:4326":
        box_4326 = sgeom.shape(
            transform_geom(src_crs, "EPSG:4326", sgeom.mapping(box_src_crs_geom), precision=10)
        )
    else:
        box_4326 = box_src_crs_geom

    resolution = 10 if not target_crs.upper().startswith("EPSG:4326") else DEG_10M

    ds = odc.stac.stac_load(
        [item],
        bands=BANDS_OF_INTEREST,
        bbox=tuple(box_4326.bounds),
        crs=target_crs,
        resolution=resolution,
        dtype="uint16",
        resampling="bilinear",
    ).isel(time=0)

    vars_present = [b for b in BANDS_OF_INTEREST if b in ds]
    if len(vars_present) == len(BANDS_OF_INTEREST):
        da = ds[BANDS_OF_INTEREST].to_array("band").assign_coords(band=BANDS_OF_INTEREST)
    else:
        da = ds.to_array("band")
        if da.sizes.get("band", -1) == len(BANDS_OF_INTEREST):
            da = da.assign_coords(band=BANDS_OF_INTEREST)

    geom_target = (
        sgeom.mapping(box_src_crs_geom)
        if src_crs.upper() == target_crs.upper()
        else transform_geom(src_crs, target_crs, sgeom.mapping(box_src_crs_geom), precision=10)
    )
    da = da.rio.write_crs(target_crs, inplace=True)
    da = da.rio.clip([geom_target], target_crs, drop=True)

    lon_name = "x" if "x" in da.coords else ("longitude" if "longitude" in da.coords else None)
    lat_name = "y" if "y" in da.coords else ("latitude" if "latitude" in da.coords else None)
    if lon_name is None or lat_name is None:
        raise ValueError(f"Spatial coords not found in {list(da.coords)}")

    new_x = np.linspace(float(da[lon_name].min()), float(da[lon_name].max()), chip_size_px)
    new_y = np.linspace(float(da[lat_name].min()), float(da[lat_name].max()), chip_size_px)
    da = da.interp({lon_name: new_x, lat_name: new_y})

    if da.dtype != np.uint16:
        da = da.astype(np.uint16)
    return da


def write_multiband_only(da, out_mb: Path, chip_size_px: int):
    ensure_dir(out_mb.parent)
    if "band" not in da.dims:
        da = da.to_array("band")
    try:
        if list(da["band"].values) != BANDS_OF_INTEREST and len(da["band"]) == len(BANDS_OF_INTEREST):
            da = da.assign_coords(band=BANDS_OF_INTEREST)
    except Exception:
        pass

    da.rio.to_raster(
        str(out_mb),
        driver="GTiff",
        dtype="uint16",
        tiled=True,
        blockxsize=chip_size_px,
        blockysize=chip_size_px,
        compress="deflate",
        predictor=2,
    )


def item_datetime(item: pystac.Item) -> datetime:
    dt = item.datetime
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


# =====================================================================
# SELECTION LOGIC
# =====================================================================

def select_with_priorities(
    catalog: pystac_client.Client,
    geom_epsg4326: sgeom.Polygon,
    year: int,
    mode: str,
    sos_doy: Optional[int],
    eos_doy: Optional[int],
    main_span_months: int,
    max_expand_months: int,
    scene_cloud_threshold_pct: float,
    patch_cloud_threshold_frac: float,
    target_cloud_max_frac: float,
    preferred_cloud_max_frac: float,
) -> Optional[pystac.Item]:
    def eval_clouds(items: List[pystac.Item]) -> List[Tuple[pystac.Item, float]]:
        pairs = []
        for it in items:
            try:
                frac = chip_cloud_fraction_at_box(it, geom_epsg4326)
                pairs.append((it, frac))
            except WindowEmpty:
                continue
            except Exception:
                continue
        return pairs

    sos_dt = doy_to_date(year, sos_doy) if sos_doy is not None else None
    eos_dt = doy_to_date(year, eos_doy) if eos_doy is not None else None

    if (mode == "planting" and sos_dt is None) or (mode == "harvest" and eos_dt is None):
        dt_range = f"{year}-01-01/{year}-12-31"
        items = stac_search_items(
            catalog,
            tuple(geom_epsg4326.bounds),
            dt_range,
            scene_cloud_threshold_pct,
        )
        items = [it for it in items if scene_contains_box(it, geom_epsg4326)]
        if not items:
            return None
        pairs = eval_clouds(items)
        if not pairs:
            return None
        it, frac = min(pairs, key=lambda p: p[1])
        if frac <= patch_cloud_threshold_frac:
            return it
        return None

    if mode == "planting":
        target = sos_dt + PLANTING_TARGET_OFFSET
        t0 = (target - timedelta(days=PLANTING_TARGET_TOL_DAYS)).date()
        t1 = (target + timedelta(days=PLANTING_TARGET_TOL_DAYS)).date()
    else:
        target = eos_dt - HARVEST_TARGET_OFFSET
        t0 = (target - timedelta(days=HARVEST_TARGET_TOL_DAYS)).date()
        t1 = (target + timedelta(days=HARVEST_TARGET_TOL_DAYS)).date()

    target_range_str = f"{t0}/{t1}"
    items_target = stac_search_items(
        catalog,
        tuple(geom_epsg4326.bounds),
        target_range_str,
        scene_cloud_threshold_pct,
    )
    items_target = [it for it in items_target if scene_contains_box(it, geom_epsg4326)]
    if items_target:
        items_target_sorted = sorted(
            items_target,
            key=lambda it: abs((item_datetime(it) - target).total_seconds()),
        )
        for it in items_target_sorted:
            try:
                frac = chip_cloud_fraction_at_box(it, geom_epsg4326)
                if frac <= target_cloud_max_frac:
                    return it
            except WindowEmpty:
                continue
            except Exception:
                continue

    if mode == "planting":
        pref_start = sos_dt + PLANTING_PREF_MIN_OFFSET
        pref_end = sos_dt + PLANTING_PREF_MAX_OFFSET
        pref_label = f"{pref_start.date()}/{pref_end.date()}"
    else:
        pref_start = eos_dt - HARVEST_PREF_MAX_OFFSET
        pref_end = eos_dt - HARVEST_PREF_MIN_OFFSET
        pref_label = f"{pref_start.date()}/{pref_end.date()}"

    items_pref = stac_search_items(
        catalog,
        tuple(geom_epsg4326.bounds),
        pref_label,
        scene_cloud_threshold_pct,
    )
    items_pref = [it for it in items_pref if scene_contains_box(it, geom_epsg4326)]
    if items_pref:
        pairs = []
        for it in items_pref:
            try:
                frac = chip_cloud_fraction_at_box(it, geom_epsg4326)
                if frac <= preferred_cloud_max_frac:
                    pairs.append((it, frac))
            except WindowEmpty:
                continue
            except Exception:
                continue
        if pairs:
            it, _ = min(pairs, key=lambda p: p[1])
            return it

    def pick_best_in_span(span_months: int) -> Optional[pystac.Item]:
        if mode == "planting":
            dr, _, _ = build_datetime_range_planting(year, sos_doy, span_months)
        else:
            dr, _, _ = build_datetime_range_harvest(year, eos_doy, span_months)
        items = stac_search_items(
            catalog,
            tuple(geom_epsg4326.bounds),
            dr,
            scene_cloud_threshold_pct,
        )
        items = [it for it in items if scene_contains_box(it, geom_epsg4326)]
        if not items:
            return None
        best = None
        best_frac = None
        for it in items:
            try:
                frac = chip_cloud_fraction_at_box(it, geom_epsg4326)
                if (best is None) or (frac < best_frac):
                    best, best_frac = it, frac
            except WindowEmpty:
                continue
            except Exception:
                continue
        if best is not None and best_frac is not None and best_frac <= patch_cloud_threshold_frac:
            return best
        return None

    chosen = pick_best_in_span(main_span_months)
    if chosen is not None:
        return chosen

    expand = main_span_months
    while expand < max_expand_months:
        expand += 1
        chosen = pick_best_in_span(expand)
        if chosen is not None:
            return chosen

    return None


# =====================================================================
# ID / YEAR RESOLUTION
# =====================================================================

def _resolve_year_for_row(
    gdf: gpd.GeoDataFrame,
    idx: int,
    year_field: Optional[str],
    year_constant: Optional[int],
) -> int:
    if year_field and year_field in gdf.columns:
        val = gdf.loc[idx, year_field]
        return int(val)
    if year_constant is not None:
        return int(year_constant)
    raise ValueError(
        "No year available for row. Provide --year-field or --year-constant."
    )


def _resolve_chip_name(
    gdf: gpd.GeoDataFrame,
    idx: int,
    chip_col: Optional[str],
    shp_stem: str,
    fallback_id_field: Optional[str],
) -> str:
    if chip_col and chip_col in gdf.columns:
        raw = str(gdf.loc[idx, chip_col])
        if raw and raw.strip():
            return parse_chip_name(raw)
    if fallback_id_field and fallback_id_field in gdf.columns:
        raw = str(gdf.loc[idx, fallback_id_field])
        if raw and raw.strip():
            return parse_chip_name(raw)
    return f"{shp_stem}_{idx}"


# =====================================================================
# STACKING
# =====================================================================

def write_stack8(win_a_path: Path, win_b_path: Path, out_path: Path, chip_size_px: int) -> bool:
    if not win_a_path.exists() or not win_b_path.exists():
        return False

    with rasterio.open(win_a_path) as src_a:
        if src_a.count != 4:
            raise ValueError(f"{win_a_path.name} does not have 4 bands")
        prof = src_a.profile.copy()
        data_a = src_a.read()

    with rasterio.open(win_b_path) as src_b:
        if src_b.count != 4:
            raise ValueError(f"{win_b_path.name} does not have 4 bands")
        if src_b.width != prof["width"] or src_b.height != prof["height"]:
            raise ValueError("window_a and window_b chip sizes differ")
        if src_b.crs != prof["crs"]:
            raise ValueError("window_a and window_b CRS differ")
        if src_b.transform != prof["transform"]:
            raise ValueError("window_a and window_b transforms differ")
        data_b = src_b.read()

    prof.update(
        count=8,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=min(prof["width"], chip_size_px),
        blockysize=min(prof["height"], chip_size_px),
        dtype="uint16",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        for i in range(4):
            dst.write(data_a[i], i + 1)
        for i in range(4):
            dst.write(data_b[i], i + 5)
        descs = [
            "A_B04",
            "A_B03",
            "A_B02",
            "A_B08",
            "B_B04",
            "B_B03",
            "B_B02",
            "B_B08",
        ]
        for i, d in enumerate(descs, start=1):
            try:
                dst.set_band_description(i, d)
            except Exception:
                pass
    return True


def sweep_and_stack_all(shp_dir: Path, chip_size_px: int) -> Tuple[int, int, int]:
    """
    After chips are written, scan <shp_dir>/window_a and <shp_dir>/window_b,
    find matching stems, write <shp_dir>/<stem>__stack8.tif
    Returns: (pairs_found, stacks_written, stacks_skipped)
    """
    win_a = shp_dir / "window_a"
    win_b = shp_dir / "window_b"
    if not win_a.exists() or not win_b.exists():
        return 0, 0, 0

    a_map = {p.stem: p for p in win_a.glob("*.tif")}
    b_map = {p.stem: p for p in win_b.glob("*.tif")}

    common = sorted(set(a_map.keys()).intersection(b_map.keys()))
    written = 0
    skipped = 0

    for stem in common:
        out_path = shp_dir / f"{stem}__stack8.tif"
        ok = write_stack8(a_map[stem], b_map[stem], out_path, chip_size_px=chip_size_px)
        if ok:
            written += 1
        else:
            skipped += 1

    return len(common), written, skipped


# =====================================================================
# WORKER
# =====================================================================

def _process_feature_worker(
    shp_name: str,
    geom_src_crs: sgeom.Polygon,
    geom_epsg4326: sgeom.Polygon,
    chip_name: str,
    year_val: int,
    target_crs: str,
    shp_dir: Path,
    out_dir_a: Path,
    out_dir_b: Path,
    sos_raster: str,
    eos_raster: str,
    planting_span_months: int,
    harvest_span_months: int,
    max_expand_planting: int,
    max_expand_harvest: int,
    scene_cloud_threshold_pct: float,
    patch_cloud_threshold_frac: float,
    target_cloud_max_frac: float,
    preferred_cloud_max_frac: float,
    chip_size_px: int,
) -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    errors: List[str] = []

    try:
        catalog = stac_client()
    except Exception as e:
        errors.append(f"{shp_name}::{chip_name}::catalog::{e}")
        return missing, errors

    sos_doy = extract_mode_doy_from_raster(sos_raster, geom_epsg4326)
    eos_doy = extract_mode_doy_from_raster(eos_raster, geom_epsg4326)

    a_file = out_dir_a / f"{chip_name}.tif"
    b_file = out_dir_b / f"{chip_name}.tif"

    try:
        best_a = select_with_priorities(
            catalog=catalog,
            geom_epsg4326=geom_epsg4326,
            year=year_val,
            mode="planting",
            sos_doy=sos_doy,
            eos_doy=eos_doy,
            main_span_months=planting_span_months,
            max_expand_months=max_expand_planting,
            scene_cloud_threshold_pct=scene_cloud_threshold_pct,
            patch_cloud_threshold_frac=patch_cloud_threshold_frac,
            target_cloud_max_frac=target_cloud_max_frac,
            preferred_cloud_max_frac=preferred_cloud_max_frac,
        )
        if best_a is not None:
            da_a = odc_load_chip_in_target_crs(
                best_a,
                geom_src_crs,
                target_crs,
                target_crs,
                chip_size_px=chip_size_px,
            )
            write_multiband_only(da_a, a_file, chip_size_px=chip_size_px)
        else:
            missing.append(f"{shp_name}::{chip_name}::planting")
    except Exception as e:
        errors.append(f"{shp_name}::{chip_name}::planting::{e}")

    try:
        best_b = select_with_priorities(
            catalog=catalog,
            geom_epsg4326=geom_epsg4326,
            year=year_val,
            mode="harvest",
            sos_doy=sos_doy,
            eos_doy=eos_doy,
            main_span_months=harvest_span_months,
            max_expand_months=max_expand_harvest,
            scene_cloud_threshold_pct=scene_cloud_threshold_pct,
            patch_cloud_threshold_frac=patch_cloud_threshold_frac,
            target_cloud_max_frac=target_cloud_max_frac,
            preferred_cloud_max_frac=preferred_cloud_max_frac,
        )
        if best_b is not None:
            da_b = odc_load_chip_in_target_crs(
                best_b,
                geom_src_crs,
                target_crs,
                target_crs,
                chip_size_px=chip_size_px,
            )
            write_multiband_only(da_b, b_file, chip_size_px=chip_size_px)
        else:
            missing.append(f"{shp_name}::{chip_name}::harvest")
    except Exception as e:
        errors.append(f"{shp_name}::{chip_name}::harvest::{e}")

    return missing, errors


# =====================================================================
# MAIN DRIVER FOR ONE SHAPEFILE
# =====================================================================

def run_for_shapefile_parallel(
    shp_path: Path,
    out_root: Path,
    sos_raster: str,
    eos_raster: str,
    chipid_field: Optional[str],
    fallback_id_field: Optional[str],
    year_field: Optional[str],
    year_constant: Optional[int],
    planting_span_months: int,
    harvest_span_months: int,
    max_expand_planting: int,
    max_expand_harvest: int,
    scene_cloud_threshold_pct: float,
    patch_cloud_threshold_frac: float,
    target_cloud_max_frac: float,
    preferred_cloud_max_frac: float,
    chip_size_px: int,
    batch_size: int,
) -> Tuple[List[str], List[str]]:
    t0 = time.time()
    gdf = gpd.read_file(str(shp_path))
    if gdf.empty:
        print(f"{shp_path.name}: empty shapefile")
        return [], []

    cols = {c.lower(): c for c in gdf.columns}
    chip_col = cols.get(chipid_field.lower(), None) if chipid_field else None

    # Ensure aoi_id column exists and is unique
    if "aoi_id" not in gdf.columns:
        gdf["aoi_id"] = np.arange(len(gdf), dtype=np.int64)

    # Ensure year column info exists:
    # If user gave a year_field, keep it; else, create a "year" column from year_constant
    if year_field and year_field in gdf.columns:
        gdf[year_field] = gdf[year_field].astype(int)
    elif year_constant is not None:
        year_field = "year"
        gdf[year_field] = int(year_constant)

    # Write updated grid back to disk (update input grid file)
    gdf.to_file(shp_path)
    print(f"[update] Updated grid {shp_path} with 'aoi_id' and year information.")

    shp_dir = out_root / shp_path.stem
    win_a_dir = shp_dir / "window_a"
    win_b_dir = shp_dir / "window_b"
    ensure_dir(win_a_dir)
    ensure_dir(win_b_dir)

    if gdf.crs is None:
        raise ValueError(f"{shp_path.name}: missing CRS on input shapefile")
    src_crs = gdf.crs.to_string()
    if str(gdf.crs).lower() in ("epsg:4326", "urn:ogc:def:crs:epsg::4326"):
        gdf_4326 = gdf
    else:
        gdf_4326 = gdf.to_crs("EPSG:4326")

    jobs = []
    for idx in range(len(gdf)):
        geom_src = gdf.geometry.iloc[idx]
        if geom_src is None or geom_src.is_empty:
            continue
        geom4326 = gdf_4326.geometry.iloc[idx]
        year_val = _resolve_year_for_row(gdf, idx, year_field, year_constant)
        chip_name = _resolve_chip_name(
            gdf, idx, chip_col, shp_path.stem, fallback_id_field=fallback_id_field
        )
        jobs.append((geom_src, geom4326, chip_name, year_val, src_crs))

    missing_all: List[str] = []
    errors_all: List[str] = []

    if not jobs:
        print(f"{shp_path.name}: no valid geometries")
        return missing_all, errors_all

    with ThreadPoolExecutor(max_workers=batch_size) as ex:
        futures = []
        for geom_src, geom4326, chip_name, year_val, target_crs in jobs:
            fut = ex.submit(
                _process_feature_worker,
                shp_name=shp_path.name,
                geom_src_crs=geom_src,
                geom_epsg4326=geom4326,
                chip_name=chip_name,
                year_val=year_val,
                target_crs=target_crs,
                shp_dir=shp_dir,
                out_dir_a=win_a_dir,
                out_dir_b=win_b_dir,
                sos_raster=sos_raster,
                eos_raster=eos_raster,
                planting_span_months=planting_span_months,
                harvest_span_months=harvest_span_months,
                max_expand_planting=max_expand_planting,
                max_expand_harvest=max_expand_harvest,
                scene_cloud_threshold_pct=scene_cloud_threshold_pct,
                patch_cloud_threshold_frac=patch_cloud_threshold_frac,
                target_cloud_max_frac=target_cloud_max_frac,
                preferred_cloud_max_frac=preferred_cloud_max_frac,
                chip_size_px=chip_size_px,
            )
            futures.append(fut)

        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{shp_path.name} [features]",
        ):
            pass

        for fut in futures:
            try:
                missing, errors = fut.result()
                missing_all.extend(missing)
                errors_all.extend(errors)
            except Exception as e:
                errors_all.append(f"{shp_path.name}::future::{e}")

    pairs, written, skipped = sweep_and_stack_all(shp_dir, chip_size_px=chip_size_px)
    print(f"[STACK] {shp_path.stem}: pairs={pairs}, written={written}, skipped={skipped}")

    dt = time.time() - t0    # seconds
    print(f"Finished {shp_path.name} in {dt:.1f}s")
    return missing_all, errors_all


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    epilog = """
Additional notes on key parameters:

- SOS/EOS rasters:
  The script uses Start-of-Season (SOS) and End-of-Season (EOS) rasters to define
  planting and harvest windows. It will download S2_SOS_WGS84.tif and
  S2_EOS_WGS84.tif into a folder named "eossos tifs" under the input directory
  (or a custom directory if --season-tifs-dir is given). Primary source is:
    ucg-uv/research_products/cropcalendars_phase2/smoothed
  Fallback source:
    wri/toolkit-for-traceability/seasontifs

- NODATA filtering:
  NODATA_FILTER_MODE and NODATA_LT control additional filtering on STAC scenes
  based on s2:nodata_pixel_percentage. By default, this is disabled (mode "none").

- Temporal offsets:
  PLANTING_TARGET_OFFSET, HARVEST_TARGET_OFFSET and their *_TOL_* and *_PREF_*
  counterparts define inner priority windows around SOS/EOS. These are advanced
  knobs; typical users do not need to adjust them.

- Chip layout:
  BANDS_OF_INTEREST = [B04, B03, B02, B08], and chips are saved as 4-band
  uint16 GeoTIFFs for planting and harvest windows separately, plus an 8-band
  stacked version <chip>__stack8.tif combining both windows."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract Sentinel-2 training chips for planting/harvest windows from "
            "grid polygons, using SOS/EOS crop calendars and Planetary Computer STAC."
        ),
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to an input grid shapefile (*.shp) OR a directory containing shapefiles.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Root output directory for chips. If omitted, defaults to the directory of "
            "the input shapefile or the input directory itself."
        ),
    )
    parser.add_argument(
        "--season-tifs-dir",
        default=None,
        help=(
            'Directory where SOS/EOS rasters will be stored/found. '
            'Default: <input-dir>/\"eossos tifs\".'
        ),
    )
    parser.add_argument(
        "--chipid-field",
        default="cellid",
        help="Column name to use as chip ID (default: 'cellid').",
    )
    parser.add_argument(
        "--fallback-id-field",
        default="aoi_id",
        help="Fallback column for chip IDs if chipid-field is missing/empty (default: 'aoi_id').",
    )
    parser.add_argument(
        "--year-field",
        default=None,
        help=(
            "Optional column name containing per-row year. "
            "If omitted, --year-constant is used for all rows."
        ),
    )
    parser.add_argument(
        "--year-constant",
        type=int,
        default=None,
        help=(
            "Year to use for all rows if --year-field is not provided. "
            "If both are omitted, the script will error."
        ),
    )

    # Cloud thresholds etc.
    parser.add_argument(
        "--scene-cloud-threshold-pct",
        type=float,
        default=DEFAULT_SCENE_CLOUD_THRESHOLD_PCT,
        help=(
            "Maximum scene-level cloud cover percent used in STAC search "
            f"(default: {DEFAULT_SCENE_CLOUD_THRESHOLD_PCT})."
        ),
    )
    parser.add_argument(
        "--patch-cloud-threshold-frac",
        type=float,
        default=DEFAULT_PATCH_CLOUD_THRESHOLD_FRAC,
        help=(
            "Maximum allowed fraction of cloud pixels within a chip for general "
            f"acceptance (default: {DEFAULT_PATCH_CLOUD_THRESHOLD_FRAC})."
        ),
    )
    parser.add_argument(
        "--target-cloud-max-frac",
        type=float,
        default=DEFAULT_TARGET_CLOUD_MAX_FRAC,
        help=(
            "Maximum cloud fraction allowed in the tight target-window selection "
            f"(default: {DEFAULT_TARGET_CLOUD_MAX_FRAC})."
        ),
    )
    parser.add_argument(
        "--preferred-cloud-max-frac",
        type=float,
        default=DEFAULT_PREFERRED_CLOUD_MAX_FRAC,
        help=(
            "Maximum cloud fraction allowed in the preferred window selection "
            f"(default: {DEFAULT_PREFERRED_CLOUD_MAX_FRAC})."
        ),
    )

    # Time spans
    parser.add_argument(
        "--planting-span-months",
        type=int,
        default=DEFAULT_PLANTING_SPAN_MONTHS,
        help=f"Initial planting span in months around SOS (default: {DEFAULT_PLANTING_SPAN_MONTHS}).",
    )
    parser.add_argument(
        "--harvest-span-months",
        type=int,
        default=DEFAULT_HARVEST_SPAN_MONTHS,
        help=f"Initial harvest span in months around EOS (default: {DEFAULT_HARVEST_SPAN_MONTHS}).",
    )
    parser.add_argument(
        "--max-expand-months-planting",
        type=int,
        default=DEFAULT_MAX_EXPAND_MONTHS_PLANTING,
        help=(
            "Maximum expansion of planting span in months if no suitable chip is "
            f"found in the initial window (default: {DEFAULT_MAX_EXPAND_MONTHS_PLANTING})."
        ),
    )
    parser.add_argument(
        "--max-expand-months-harvest",
        type=int,
        default=DEFAULT_MAX_EXPAND_MONTHS_HARVEST,
        help=(
            "Maximum expansion of harvest span in months if no suitable chip is "
            f"found in the initial window (default: {DEFAULT_MAX_EXPAND_MONTHS_HARVEST})."
        ),
    )

    # Chip size, batch size
    parser.add_argument(
        "--chip-size",
        type=int,
        default=DEFAULT_CHIP_SIZE_PX,
        help=f"Chip size in pixels (square, default: {DEFAULT_CHIP_SIZE_PX}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BBOX_BATCH_SIZE,
        help=f"Number of features processed in parallel (ThreadPool workers, default: {DEFAULT_BBOX_BATCH_SIZE}).",
    )

    return parser.parse_args()


# =====================================================================
# TOP-LEVEL MAIN
# =====================================================================

def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if args.output_dir is None:
        if input_path.is_file():
            out_root = input_path.parent
        else:
            out_root = input_path
    else:
        out_root = Path(args.output_dir)
    ensure_dir(out_root)

    # Season rasters directory
    if args.season_tifs_dir is None:
        season_dir = (input_path.parent if input_path.is_file() else input_path) / "eossos tifs"
    else:
        season_dir = Path(args.season_tifs_dir)

    sos_path, eos_path = ensure_season_tifs(season_dir, verbose=True)

    # Collect shapefiles
    if input_path.is_file() and input_path.suffix.lower() == ".shp":
        shp_list = [input_path]
    elif input_path.is_dir():
        shp_list = sorted(
            [p for p in input_path.glob("*.shp") if p.is_file()]
        )
    else:
        raise FileNotFoundError(f"Not a shapefile or directory: {input_path}")

    if not shp_list:
        print(f"No shapefiles found in: {input_path}")
        return

    all_missing: List[str] = []
    all_errors: List[str] = []

    for shp in shp_list:
        missing, errors = run_for_shapefile_parallel(
            shp_path=shp,
            out_root=out_root,
            sos_raster=str(sos_path),
            eos_raster=str(eos_path),
            chipid_field=args.chipid_field,
            fallback_id_field=args.fallback_id_field,
            year_field=args.year_field,
            year_constant=args.year_constant,
            planting_span_months=args.planting_span_months,
            harvest_span_months=args.harvest_span_months,
            max_expand_planting=args.max_expand_months_planting,
            max_expand_harvest=args.max_expand_months_harvest,
            scene_cloud_threshold_pct=args.scene_cloud_threshold_pct,
            patch_cloud_threshold_frac=args.patch_cloud_threshold_frac,
            target_cloud_max_frac=args.target_cloud_max_frac,
            preferred_cloud_max_frac=args.preferred_cloud_max_frac,
            chip_size_px=args.chip_size,
            batch_size=args.batch_size,
        )
        all_missing.extend(missing)
        all_errors.extend(errors)

    (out_root / "missing_windows.txt").write_text(
        "\n".join(all_missing), encoding="utf-8"
    )
    (out_root / "errors.txt").write_text(
        "\n".join(all_errors), encoding="utf-8"
    )

    print(f"\nAll done. Missing windows: {len(all_missing)}; Errors: {len(all_errors)}")


if __name__ == "__main__":
    main()
