import os
import math
import argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from shapely import ops as sops

try:
    from shapely.validation import make_valid
    HAS_MAKE_VALID = True
except Exception:
    HAS_MAKE_VALID = False

from pyproj import CRS


def get_brazil_albers_crs() -> CRS:
    """
    Return SIRGAS 2000 / Brazil Albers Equal Area CRS.
    Tries EPSG:5880 first, falls back to explicit PROJ string if EPSG is unavailable.
    """
    try:
        return CRS.from_epsg(5880)
    except Exception:
        proj4 = (
            "+proj=aea +lat_1=-12 +lat_2=-32 +lat_0=0 +lon_0=-54 "
            "+x_0=0 +y_0=0 +datum=SIRGAS2000 +units=m +no_defs"
        )
        return CRS.from_proj4(proj4)


def safe_delete_shapefile(path: str) -> None:
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj"]:
        f = path.replace(".shp", ext)
        if os.path.exists(f):
            os.remove(f)


def pick_utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """
    Choose a WGS84 UTM EPSG zone based on lon/lat.
    Northern hemisphere: 326xx; Southern hemisphere: 327xx.
    """
    zone = int(math.floor((lon + 180) / 6) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def is_geographic_crs(gdf: gpd.GeoDataFrame) -> bool:
    crs = gdf.crs
    if crs is None:
        return False
    try:
        return crs.is_geographic
    except Exception:
        try:
            return crs.axis_info and crs.axis_info[0].unit_name.lower() in ("degree", "degrees")
        except Exception:
            return False


def repair_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Repair invalid geometries robustly:
    1) make_valid if available (Shapely 2.0+)
    2) fallback: buffer(0)
    Drops empties and non polygonal types at the end.
    """
    def _repair(geom):
        if geom is None or geom.is_empty:
            return None
        try:
            if not geom.is_valid:
                if HAS_MAKE_VALID:
                    geom = make_valid(geom)
                else:
                    geom = geom.buffer(0)
        except Exception:
            try:
                geom = geom.buffer(0)
            except Exception:
                return None
        return geom

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(_repair)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf.reset_index(drop=True, inplace=True)
    return gdf


def ensure_projected(
    gdf: gpd.GeoDataFrame,
    input_epsg_if_missing: int | None = None,
    target_projected_epsg: int | None = None,
    use_brazil_albers: bool = True,
) -> gpd.GeoDataFrame:
    """
    Ensure we work in a projected CRS (meters).

    Priority:
    - If CRS missing: require input_epsg_if_missing.
    - If target_projected_epsg is set: reproject to it.
    - Else if use_brazil_albers: reproject to SIRGAS 2000 / Brazil Albers Equal Area.
    - Else: if geographic, choose UTM zone by bounds center; otherwise keep as is.
    """
    if gdf.crs is None:
        if input_epsg_if_missing is None:
            raise ValueError(
                "Input CRS is missing. Set input_epsg_if_missing or fix the .prj on the input file."
            )
        gdf = gdf.set_crs(epsg=input_epsg_if_missing, inplace=False)
        print(f"[INFO] Input CRS was missing. Set to EPSG:{input_epsg_if_missing}.")

    if target_projected_epsg is not None:
        if gdf.crs.to_epsg() != target_projected_epsg:
            gdf = gdf.to_crs(epsg=target_projected_epsg)
            print(f"[INFO] Reprojected to target EPSG:{target_projected_epsg}.")
        else:
            print(f"[INFO] Already in target EPSG:{target_projected_epsg}.")
        return gdf

    if use_brazil_albers:
        brazil_albers = get_brazil_albers_crs()
        try:
            if CRS.from_user_input(gdf.crs).to_wkt() != brazil_albers.to_wkt():
                gdf = gdf.to_crs(brazil_albers)
                try:
                    epsg_val = brazil_albers.to_epsg()
                    epsg_str = f"EPSG:{epsg_val}" if epsg_val else "Brazil_Albers_Equal_Area"
                except Exception:
                    epsg_str = "Brazil_Albers_Equal_Area"
                print(f"[INFO] Reprojected to Brazil Albers Equal Area ({epsg_str}).")
            else:
                print("[INFO] Input already in Brazil Albers Equal Area.")
        except Exception:
            gdf = gdf.to_crs(brazil_albers)
            print("[INFO] Reprojected to Brazil Albers Equal Area (fallback path).")
        return gdf

    if is_geographic_crs(gdf):
        wgs = gdf.to_crs(epsg=4326)
        xmin, ymin, xmax, ymax = wgs.total_bounds
        lon = (xmin + xmax) / 2.0
        lat = (ymin + ymax) / 2.0
        utm_epsg = pick_utm_epsg_from_lonlat(lon, lat)
        gdf = gdf.to_crs(epsg=utm_epsg)
        print(f"[INFO] Geographic input detected. Reprojected to UTM EPSG:{utm_epsg}.")
    else:
        print(f"[INFO] Input is already projected (CRS: {gdf.crs.to_string()}).")

    return gdf


def generate_grid(bounds, cell_size, crs):
    xmin, ymin, xmax, ymax = bounds
    cols = int(math.ceil((xmax - xmin) / cell_size))
    rows = int(math.ceil((ymax - ymin) / cell_size))
    cells = []
    for i in range(cols):
        x0 = xmin + i * cell_size
        x1 = x0 + cell_size
        for j in range(rows):
            y0 = ymin + j * cell_size
            y1 = y0 + cell_size
            cells.append(box(x0, y0, x1, y1))
    return gpd.GeoDataFrame({"geometry": cells}, crs=crs)


def safe_unary_union(geoms):
    """
    Robust union:
    - Try unary_union on repaired geometries.
    - If it fails, union in chunks to isolate bad actors.
    - Final fallback: slightly buffer(0) before union.
    """
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return None

    try:
        return sops.unary_union(geoms)
    except Exception:
        try:
            chunks = 8
            step = max(1, len(geoms) // chunks)
            partials = []
            for i in range(0, len(geoms), step):
                sub = geoms[i:i + step]
                try:
                    partials.append(sops.unary_union(sub))
                except Exception:
                    sub2 = [g.buffer(0) for g in sub]
                    partials.append(sops.unary_union(sub2))
            return sops.unary_union(partials)
        except Exception:
            geoms2 = [g.buffer(0) for g in geoms]
            return sops.unary_union(geoms2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: Create FTW style training grids from field boundary datasets."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input fields file (shp, geojson, gpkg, parquet).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for grid. Defaults to the directory of the input file.",
    )
    parser.add_argument(
        "--cell-size-meters",
        type=float,
        default=2560.0,
        help="Grid cell size in meters (default: 2560).",
    )
    parser.add_argument(
        "--input-epsg-if-missing",
        type=int,
        default=None,
        help="EPSG code to use if input has no CRS defined.",
    )
    parser.add_argument(
        "--target-projected-epsg",
        type=int,
        default=None,
        help="Force reproject to this EPSG. Overrides Brazil Albers and UTM logic.",
    )
    parser.add_argument(
        "--no-brazil-albers",
        action="store_true",
        help="Disable Brazil Albers default and fall back to UTM or existing projection.",
    )
    parser.add_argument(
        "--output-format",
        choices=["shp", "geojson", "gpkg", "parquet"],
        default="shp",
        help="Output format: shp, geojson, gpkg, parquet (default: shp).",
    )
    return parser.parse_args()


def create_grid(
    input_path: str,
    output_dir: str | None = None,
    cell_size_meters: float = 2560.0,
    input_epsg_if_missing: int | None = None,
    target_projected_epsg: int | None = None,
    use_brazil_albers: bool = True,
    output_format: str = "shp",
) -> str:
    """
    Core function for Step 1 training data creation.

    Returns the path of the written grid output file.
    """
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    if output_dir is None:
        output_dir = os.path.dirname(input_path) or os.getcwd()
    else:
        output_dir = os.path.abspath(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Input file: {input_path}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Cell size (m): {cell_size_meters}")
    print(f"[INFO] Output format: {output_format}")

    parcelas = gpd.read_file(input_path)
    print(f"[INFO] Loaded {len(parcelas)} features.")
    print(f"[INFO] Input CRS: {parcelas.crs}")

    parcelas = repair_geometries(parcelas)
    if len(parcelas) == 0:
        raise RuntimeError("After repair, no valid polygonal geometries remain.")
    print(f"[INFO] After repair: {len(parcelas)} polygonal features.")
    print(f"[INFO] Repaired geom types: {parcelas.geom_type.value_counts(dropna=False).to_dict()}")

    parcelas = ensure_projected(
        parcelas,
        input_epsg_if_missing=input_epsg_if_missing,
        target_projected_epsg=target_projected_epsg,
        use_brazil_albers=use_brazil_albers,
    )
    print(f"[INFO] Working CRS: {parcelas.crs.to_string()}")
    print(f"[INFO] Working bounds: {parcelas.total_bounds}")

    union_poly = safe_unary_union(list(parcelas.geometry))
    if union_poly is None or union_poly.is_empty:
        raise RuntimeError("Union produced empty geometry. Check input data and CRS.")
    try:
        union_poly = union_poly.buffer(0)
    except Exception:
        pass

    dissolved = gpd.GeoDataFrame(geometry=[union_poly], crs=parcelas.crs).explode(index_parts=False)
    dissolved = dissolved[dissolved.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    dissolved.reset_index(drop=True, inplace=True)
    print(f"[INFO] Dissolved polygon parts: {len(dissolved)}")

    dissolved_path = os.path.join(output_dir, f"{base_name}_dissolved.shp")
    safe_delete_shapefile(dissolved_path)
    dissolved.to_file(dissolved_path)
    print(f"[OK] Wrote dissolved polygons: {dissolved_path}")

    grid = generate_grid(dissolved.total_bounds, cell_size_meters, parcelas.crs)
    grid["cell_id"] = np.arange(len(grid), dtype=np.int64)
    grid["cell_area"] = grid.geometry.area
    print(f"[INFO] Grid cells: {len(grid)} | CRS: {grid.crs}")

    intersections = grid.geometry.intersection(union_poly)
    grid["covered_area"] = intersections.area.fillna(0.0)
    grid["coverage_pct"] = np.where(
        grid["cell_area"] > 0, 100.0 * grid["covered_area"] / grid["cell_area"], 0.0
    )

    grid_filtered = grid[grid["coverage_pct"] > 10].copy()
    grid_filtered["geometry"] = grid_filtered.geometry.buffer(0)
    grid_filtered = grid_filtered[grid_filtered.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    grid_filtered.reset_index(drop=True, inplace=True)
    print(f"[INFO] Filtered cells: {len(grid_filtered)}")

    if output_format == "shp":
        out_path = os.path.join(output_dir, f"{base_name}_grid.shp")
        safe_delete_shapefile(out_path)
        grid_filtered.to_file(out_path)
    elif output_format == "geojson":
        out_path = os.path.join(output_dir, f"{base_name}_grid.geojson")
        if os.path.exists(out_path):
            os.remove(out_path)
        grid_filtered.to_file(out_path, driver="GeoJSON")
    elif output_format == "gpkg":
        out_path = os.path.join(output_dir, f"{base_name}_grid.gpkg")
        if os.path.exists(out_path):
            os.remove(out_path)
        grid_filtered.to_file(out_path, layer="grid", driver="GPKG")
    elif output_format == "parquet":
        out_path = os.path.join(output_dir, f"{base_name}_grid.parquet")
        grid_filtered.to_parquet(out_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f"[OK] Grid written: {out_path}")

    print("\n=== SANITY ===")
    print("CRS:", grid_filtered.crs)
    print("Bounds:", grid_filtered.total_bounds)
    print("Geom types:", grid_filtered.geom_type.value_counts(dropna=False).to_dict())
    print(grid_filtered.head(1).T)

    return out_path


def main() -> None:
    args = parse_args()
    create_grid(
        input_path=args.input,
        output_dir=args.output_dir,
        cell_size_meters=args.cell_size_meters,
        input_epsg_if_missing=args.input_epsg_if_missing,
        target_projected_epsg=args.target_projected_epsg,
        use_brazil_albers=not args.no_brazil_albers,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()
