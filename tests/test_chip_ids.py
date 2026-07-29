"""Step 1.1 and Step 1.2 have to agree on the chip ID column.

They did not: gridding wrote `cell_id` while the downloader defaulted to
`cellid`, so the default path silently fell through to `aoi_id` and renumbered
every chip. That breaks the correspondence between imagery and the labels drawn
against the original grid, which is expensive and invisible.
"""

import pytest


def test_default_matches_what_gridding_writes():
    pytest.importorskip("pystac_client", reason="needs the pt1 extra")
    from trazo.pt1_createdata.plantingharvest import parse_args

    args = parse_args(["--input", "grid.shp", "--year-constant", "2023"])
    assert args.chipid_field == "chip_id"


@pytest.mark.parametrize(
    "columns,expected",
    [
        (["chip_id", "geometry"], "chip_id"),
        (["CHIP_ID", "geometry"], "CHIP_ID"),          # case-insensitive
        (["cell_id", "geometry"], "cell_id"),          # legacy grids
        (["cellid", "geometry"], "cellid"),            # legacy grids
        (["chipid", "geometry"], "chipid"),
        (["aoi_id", "geometry"], None),                # no chip column at all
        (["geometry"], None),
    ],
)
def test_resolve_chip_column(columns, expected):
    pytest.importorskip("pystac_client", reason="needs the pt1 extra")
    from trazo.pt1_createdata.plantingharvest import resolve_chip_column

    assert resolve_chip_column(columns, "chip_id") == expected


def test_explicit_field_wins_over_legacy():
    pytest.importorskip("pystac_client", reason="needs the pt1 extra")
    from trazo.pt1_createdata.plantingharvest import resolve_chip_column

    columns = ["chip_id", "cell_id", "my_id", "geometry"]
    assert resolve_chip_column(columns, "my_id") == "my_id"


def test_gridding_output_resolves_with_the_default(tmp_path):
    """The real end-to-end guarantee: Step 1.1 output feeds Step 1.2 unflagged."""
    pytest.importorskip("geopandas", reason="needs the core install")
    pytest.importorskip("pystac_client", reason="needs the pt1 extra")
    import geopandas as gpd

    from trazo.pt1_createdata.plantingharvest import resolve_chip_column
    from trazo.smoke import make_fields, run_gridding

    fields = tmp_path / "fields.shp"
    make_fields(fields, n=8)
    grid = gpd.read_file(run_gridding(fields, tmp_path))

    assert resolve_chip_column(grid.columns, "chip_id") == "chip_id"
