"""Random control arm and cross-round exclusion for the active-learning sampler."""

import pytest

pytest.importorskip("rasterio")

from trazo.pt1_createdata.active_sample import load_exclusions, strategy_random


def _records(n):
    return [{"file": f"grid_{i}.tif", "confidence_mean": i / n} for i in range(n)]


def test_random_strategy_is_seeded_and_sized():
    recs = _records(50)
    a = strategy_random(recs, n=10, seed=3)
    assert len(a) == 10 and len({r["file"] for r in a}) == 10
    assert a == strategy_random(recs, n=10, seed=3)
    assert len(strategy_random(recs, n=500, seed=0)) == 50


def test_exclusions_from_dir_and_list(tmp_path):
    d = tmp_path / "round1"
    d.mkdir()
    (d / "grid_1.tif").write_bytes(b"")
    lst = tmp_path / "labelled.txt"
    lst.write_text("some/where/grid_2.tif\n\ngrid_3.tif\n")
    assert load_exclusions([str(d), str(lst)]) == {"grid_1.tif", "grid_2.tif", "grid_3.tif"}
    with pytest.raises(FileNotFoundError):
        load_exclusions([str(tmp_path / "missing")])
