"""End-to-end offline smoke path: Step 1.1 gridding through the Step 2 chain."""

import pytest


def test_smoke_runs_end_to_end(tmp_path):
    pytest.importorskip("rasterio", reason="needs the pt2 extra")
    pytest.importorskip("geopandas", reason="needs the core install")
    from trazo.smoke import run

    work = run(tmp_path / "smoke")

    assert (work / "sized256").is_dir()
    assert list((work / "label_masks" / "semantic_3class").glob("*.tif"))
    assert list(work.glob("*.parquet"))


def test_grid_columns_are_shapefile_safe(tmp_path):
    """Shapefile field names truncate at 10 characters.

    The grid columns must survive the default `--output-format shp` unchanged,
    otherwise the documented column names are not the ones users get.
    """
    pytest.importorskip("geopandas", reason="needs the core install")
    import geopandas as gpd

    from trazo.smoke import make_fields, run_gridding

    fields = tmp_path / "fields.shp"
    make_fields(fields, n=8)
    grid_path = run_gridding(fields, tmp_path)

    columns = [c for c in gpd.read_file(grid_path).columns if c != "geometry"]
    too_long = [c for c in columns if len(c) > 10]
    assert not too_long, f"columns truncated by the shapefile driver: {too_long}"
    assert set(columns) == {"chip_id", "chip_area", "cov_area", "cov_pct"}
