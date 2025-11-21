#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build a per-chip GeoParquet with exactly one row per chip.

For each <base> folder:
  - Reads chip bboxes from <base>/<base.name>_boundingbox256.geojson
  - Reads a field polygon layer (auto discovery in base or a user path)
  - Joins chips and fields by spatial intersection without clipping fields
  - Produces a GeoParquet with columns:
        aoi_id     string   chip stem
        split      string   train | val | test (deterministic hash)
        has_fields bool     True if any intersecting polygons
        n_fields   int      number of intersecting polygons
        geometry   MultiPolygon of all intersecting field polygons, or None

Notes
  - CRS is normalized to OUTPUT_EPSG
  - Only chips whose stems appear in sized256 window_a are kept if
    --restrict-to-window-a is set
  - Overwrite behavior is controlled by --overwrite
"""

import argparse
import glob
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


# ----------------------------
# Helpers
# ----------------------------

def _region_tag(base_dir: Path) -> str:
    return base_dir.name.lower()


def _bbox_path(base_dir: Path) -> Optional[Path]:
    cand = base_dir / f"{base_dir.name}_boundingbox256.geojson"
    return cand if cand.exists() else None


def _auto_find_fields_shp(base_dir: Path, prefer_contains: Optional[str]) -> Optional[Path]:
    shps = sorted(base_dir.glob("*.shp"))
    if not shps:
        return None
    if prefer_contains:
        hits = [p for p in shps if prefer_contains.lower() in p.stem.lower()]
        if hits:
            return hits[0]
    field_like = [p for p in shps if "field" in p.stem.lower()]
    if field_like:
        return field_like[0]
    return shps[0]


def _ensure_aoi_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = gdf.copy()
    if "aoi_id" in g.columns and g["aoi_id"].notna().any():
        g["aoi_id"] = g["aoi_id"].astype("string")
        return g

    for col in ["id", "name", "tile_id", "grid_id", "stem"]:
        if col in g.columns:
            g["aoi_id"] = g[col].astype("string")
            return g

    # try to derive from any filename-like column
    str_cols = [c for c in g.columns if g[c].dtype == "object" or str(g[c].dtype).startswith("string")]
    for col in str_cols:
        s = g[col].astype("string")
        if s.str.contains(".tif", case=False, na=False).any():
            g["aoi_id"] = (
                s.str.replace("\\\\", "/", regex=True)
                 .str.split("/")
                 .str[-1]
                 .str.replace(".tif", "", case=False, regex=False)
            )
            return g

    raise ValueError("Bounding boxes have no aoi_id and no recognizable fallback column.")


def _to_epsg(gdf: gpd.GeoDataFrame, epsg: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(epsg, allow_override=True)
    if str(gdf.crs).upper() == epsg.upper():
        return gdf
    return gdf.to_crs(epsg)


def _collect_polygon_parts(geom: BaseGeometry) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]  # type: ignore
    if geom.geom_type == "MultiPolygon":
        return [p for p in geom.geoms if p.geom_type == "Polygon"]  # type: ignore
    return []


def _to_multipolygon_or_none(geoms: List[BaseGeometry]) -> Optional[MultiPolygon]:
    parts: List[Polygon] = []
    for g in geoms:
        parts.extend(_collect_polygon_parts(g))
    if len(parts) == 0:
        return None
    if len(parts) == 1:
        return MultiPolygon([parts[0]])
    return MultiPolygon(parts)


def _det_split(aoi_id: str, ratios: Dict[str, float], salt: str) -> str:
    # ratios must sum to 1.0
    t_train = ratios.get("train", 0.0)
    t_val = t_train + ratios.get("val", 0.0)
    h = hashlib.sha1((salt + "::" + str(aoi_id)).encode("utf-8")).hexdigest()
    r = int(h[:8], 16) / 16**8
    if r < t_train:
        return "train"
    elif r < t_val:
        return "val"
    else:
        return "test"


def _list_window_a_stems(base_dir: Path) -> List[str]:
    wa = base_dir / "s2_images" / "window_a"
    if not wa.exists():
        return []
    stems = {Path(p).stem for p in glob.glob(str(wa / "*.tif"))}
    return sorted(stems)


# ----------------------------
# Core
# ----------------------------

def process_one_base(
    base_dir: Path,
    fields_shp: Optional[Path],
    prefer_contains: Optional[str],
    output_epsg: str,
    split_train: float,
    split_val: float,
    split_test: float,
    split_salt: str,
    restrict_to_window_a: bool,
    output_name_tpl: str,
    overwrite: bool
) -> None:
    tag = _region_tag(base_dir)
    print(f"\n=== Building chips parquet for: {base_dir} ===")

    bbox_path = _bbox_path(base_dir)
    if bbox_path is None:
        print("  [WARN] bounding box GeoJSON not found. Expected:", base_dir / f"{tag}_boundingbox256.geojson")
        return

    if fields_shp is None:
        fields_path = _auto_find_fields_shp(base_dir, prefer_contains)
    else:
        fields_path = Path(fields_shp)

    if fields_path is None or not Path(fields_path).exists():
        print("  [WARN] field shapefile not found. Skipping.")
        return

    try:
        gdf_bbox = gpd.read_file(bbox_path)
        gdf_bbox = _ensure_aoi_id(gdf_bbox)
    except Exception as e:
        print(f"  [ERROR] cannot read bbox file: {e}")
        return

    try:
        gdf_fields = gpd.read_file(fields_path)
    except Exception as e:
        print(f"  [ERROR] cannot read fields shapefile: {e}")
        return

    # Normalize geometry types and CRS
    try:
        gdf_fields = gdf_fields[gdf_fields.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    except Exception:
        pass

    gdf_bbox = _to_epsg(gdf_bbox, output_epsg)
    gdf_fields = _to_epsg(gdf_fields, output_epsg)

    # Restrict to chips that actually exist in window_a if requested
    if restrict_to_window_a:
        stems = _list_window_a_stems(base_dir)
        if not stems:
            print("  [WARN] No TIFFs under s2_images/window_a while restrict_to_window_a is set.")
            return
        gdf_bbox = gdf_bbox[gdf_bbox["aoi_id"].astype("string").isin(stems)].copy()

    if gdf_bbox.empty:
        print("  [WARN] No bbox rows to process after filtering. Skipping.")
        return

    print(f"  bbox rows:  {len(gdf_bbox)}")
    print(f"  fields:     {len(gdf_fields)}")
    print(f"  join crs:   {output_epsg}")

    # Spatial join
    try:
        joined = gpd.sjoin(
            gdf_fields,
            gdf_bbox[["aoi_id", "geometry"]],
            how="inner",
            predicate="intersects",
            lsuffix="_field",
            rsuffix="_chip"
        )
        # Determine which aoi_id column to use (from bbox/chip, not fields)
        aoi_col = None
        # Check for chip/right suffix first (this is what we want - the chip aoi_id)
        # Note: geopandas uses double underscores for suffixes
        for c in ["aoi_id__chip", "aoi_id_chip", "aoi_id_right", "index_right", "index__chip"]:
            if c in joined.columns:
                aoi_col = c
                break
        # If no suffix found, check if plain aoi_id exists (only if fields don't have it)
        if aoi_col is None and "aoi_id" in joined.columns:
            # Check if fields also has aoi_id - if so, we need the chip one
            if "aoi_id__field" in joined.columns or "aoi_id_field" in joined.columns:
                # Both exist, prefer the chip one (should have been suffixed)
                print("  [WARN] Both field and chip aoi_id found, but chip suffix not detected. Using plain aoi_id.")
            aoi_col = "aoi_id"
    except TypeError:
        # Fallback API without lsuffix/rsuffix in older versions
        joined = gpd.sjoin(
            gdf_fields,
            gdf_bbox[["aoi_id", "geometry"]],
            how="inner",
            predicate="intersects"
        )
        # In older geopandas, if both have aoi_id, it might be index_right or aoi_id_right
        aoi_col = None
        for c in ["aoi_id_right", "index_right", "aoi_id"]:
            if c in joined.columns:
                aoi_col = c
                break

    if aoi_col is None or aoi_col not in joined.columns:
        print(f"  [ERROR] Could not determine aoi_id column after spatial join.")
        print(f"  Available columns: {list(joined.columns)}")
        return

    # Group field geometries by chip id
    geom_lists = (
        joined.groupby(aoi_col)["geometry"]
              .apply(lambda s: s.tolist())
              .rename("geom_list")
              .to_frame()
    )
    geom_lists.index = geom_lists.index.astype("string")

    # Build base frame of all chips in bbox
    chips = pd.Series(sorted(gdf_bbox["aoi_id"].astype("string").unique()), dtype="string")
    out_df = pd.DataFrame({"aoi_id": chips})
    out_df = out_df.merge(geom_lists, how="left", left_on="aoi_id", right_index=True)
    out_df["geom_list"] = out_df["geom_list"].apply(lambda v: v if isinstance(v, list) else [])

    out_df["n_fields"] = out_df["geom_list"].apply(len)
    out_df["has_fields"] = out_df["n_fields"] > 0
    out_df["geometry"] = out_df["geom_list"].apply(_to_multipolygon_or_none)
    out_df = out_df.drop(columns=["geom_list"])

    # Deterministic split
    ratios = {"train": float(split_train), "val": float(split_val), "test": float(split_test)}
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Splits must sum to 1.0")
    out_df["split"] = out_df["aoi_id"].apply(lambda a: _det_split(a, ratios, split_salt)).astype("string")

    # To GeoDataFrame and write parquet
    out_gdf = gpd.GeoDataFrame(out_df, geometry="geometry", crs=output_epsg)

    out_path = base_dir / output_name_tpl.format(region=_region_tag(base_dir))
    if out_path.exists() and overwrite:
        try:
            out_path.unlink()
        except Exception:
            pass

    out_gdf.to_parquet(out_path, index=False)

    # Simple validation
    chk = gpd.read_parquet(out_path)
    needed = {"aoi_id", "split", "has_fields", "n_fields", "geometry"}
    if not needed.issubset(set(chk.columns)):
        raise RuntimeError("Parquet missing expected columns after write.")
    if not chk["aoi_id"].is_unique:
        raise RuntimeError("aoi_id is not unique in output.")

    print(f"  Wrote: {out_path}")
    print(f"    rows: {len(out_gdf)}")
    print(f"    with fields: {int(out_gdf['has_fields'].sum())} | without fields: {int((~out_gdf['has_fields']).sum())}")
    print(f"    split counts: {out_gdf['split'].value_counts(dropna=False).to_dict()}")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build one-row-per-chip GeoParquet with field MultiPolygons per chip."
    )
    p.add_argument(
        "--base-folder",
        action="append",
        required=True,
        help="Base folder that contains the chip bbox GeoJSON and sized256 window chips."
    )
    p.add_argument(
        "--fields-shp",
        default=None,
        help="Optional explicit field shapefile path. If omitted, the script auto-discovers in each base."
    )
    p.add_argument(
        "--prefer-contains",
        default=None,
        help="When auto-discovering shapefiles, prefer names that contain this substring."
    )
    p.add_argument(
        "--output-epsg",
        default="EPSG:4326",
        help="CRS to project both layers to before the join."
    )
    p.add_argument(
        "--split-train", type=float, default=0.85,
        help="Train ratio. Default 0.85"
    )
    p.add_argument(
        "--split-val", type=float, default=0.15,
        help="Validation ratio. Default 0.15"
    )
    p.add_argument(
        "--split-test", type=float, default=0.0,
        help="Test ratio. Default 0.0"
    )
    p.add_argument(
        "--split-salt",
        default="tkt_split_v1",
        help="Salt for deterministic split hashing."
    )
    p.add_argument(
        "--restrict-to-window-a",
        action="store_true",
        help="If set, only keep chips that exist in s2_images/window_a."
    )
    p.add_argument(
        "--output-name-template",
        default="chips_{region}.parquet",
        help="Filename template for the output parquet. {region} is replaced with base.name.lower()."
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs."
    )
    return p.parse_args()


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args()

    bases = [Path(b).expanduser().resolve() for b in args.base_folder]
    fields = Path(args.fields_shp).expanduser().resolve() if args.fields_shp else None

    for base in bases:
        if not base.exists():
            print(f"[WARN] Missing base folder: {base}")
            continue
        process_one_base(
            base_dir=base,
            fields_shp=fields,
            prefer_contains=args.prefer_contains,
            output_epsg=args.output_epsg,
            split_train=args.split_train,
            split_val=args.split_val,
            split_test=args.split_test,
            split_salt=args.split_salt,
            restrict_to_window_a=args.restrict_to_window_a,
            output_name_tpl=args.output_name_template,
            overwrite=args.overwrite
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
