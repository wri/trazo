"""scale_uint16 must scale float reflectance, not truncate it."""

import numpy as np
import pytest

pytest.importorskip("rasterio")

from trazo.pt2_dataprep.scale_uint16 import decide_action


def test_reflectance_floats_are_scaled():
    # A GEE `.divide(10000)` export: these used to be cast to 0.
    assert decide_action(np.dtype("float32"), 0.05, 0.4, 0, 10000) == "scale"


def test_dn_floats_are_cast_and_uint16_skipped():
    assert decide_action(np.dtype("float32"), 120.0, 4000.0, 0, 10000) == "cast"
    assert decide_action(np.dtype("uint16"), 120, 4000, 0, 10000) == "skip"


def test_out_of_range_is_clipped():
    assert decide_action(np.dtype("int16"), -5, 12000, 0, 10000) == "clip"
