import os
os.environ["OGR_ORGANIZE_POLYGONS"] = "ONLY_CCW"   # ESRI ring convention: fast + correct
import sys, time, functools, warnings, numpy as np, pyogrio, geopandas as gpd
warnings.filterwarnings("ignore"); print = functools.partial(print, flush=True)

GDB      = r"D:\WRI\Field Boundaries\Trazo Fields\trazo_fields_v2.gdb"
OUT      = r"D:\WRI\Field Boundaries\Trazo Fields\gpkg_outputs_v2"
AREA_CRS = "ESRI:102033"; MIN_AREA_HA = 0.03

def regen(layer):
    t0 = time.time()
    final = os.path.join(OUT, layer + ".gpkg")
    tmp   = os.path.join(OUT, layer + ".__regen.gpkg")
    for p in (tmp, tmp + "-journal", tmp + "-wal"):
        if os.path.exists(p): os.remove(p)

    # Read only gridcode==1 (drops gridcode=2 boundary polygons at source).
    g = pyogrio.read_dataframe(GDB, layer=layer, where="gridcode = 1")
    g = gpd.GeoDataFrame(g, geometry="geometry")
    if g.crs is None: g = g.set_crs("EPSG:4326")
    n_g1 = len(g)

    # 0.03 ha floor, area measured in the same equal-area CRS Stage F used.
    g = g.to_crs(AREA_CRS)
    g["area_ha"] = g.geometry.area / 1e4
    g = g[g["area_ha"] >= MIN_AREA_HA].copy()
    kept = len(g)
    g = g.to_crs("EPSG:4326")

    g.to_file(tmp, driver="GPKG", layer=layer)
    nout = int(pyogrio.read_info(tmp, layer=layer)["features"])
    assert nout == kept, f"count mismatch {nout} vs {kept}"

    # confirm the floor is now 0.03, not 1 ha
    mn = float((gpd.read_file(tmp, rows=8000).to_crs(AREA_CRS).geometry.area / 1e4).min())
    if os.path.exists(final): os.remove(final)
    os.rename(tmp, final)
    el = time.time() - t0
    print(f"{layer}: gridcode1 {n_g1:,} -> >=0.03ha {kept:,}  min {mn:.5f}ha  "
          f"{'OK' if mn < 0.10 else '*** STILL 1HA ***'}  ({el:.0f}s)")
    return kept

if __name__ == "__main__":
    regen(sys.argv[1])
