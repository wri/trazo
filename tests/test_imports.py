"""Every module must import cleanly.

This is the test that would have caught the `from src.tkt.pt4_train...` imports
that survived the package rename and left Step 4 unrunnable.
"""

import importlib

import pytest

CORE_MODULES = [
    "trazo",
    "trazo.pt1_createdata",
    "trazo.pt1_createdata.gridding",
    "trazo.pt1_createdata.cli",
    "trazo.smoke",
]

PT2_MODULES = [
    "trazo.pt2_dataprep.cli",
    "trazo.pt2_dataprep.pair_stacks",
    "trazo.pt2_dataprep.resize_chips_256",
    "trazo.pt2_dataprep.chips_to_bboxes",
    "trazo.pt2_dataprep.make_masks_and_windows",
    "trazo.pt2_dataprep.build_chips_parquet",
    "trazo.pt2_dataprep.scale_uint16",
]

PT4_MODULES = [
    "trazo.pt4_train.cli",
    "trazo.pt4_train.settings",
    "trazo.pt4_train.models",
    "trazo.pt4_train.losses",
    "trazo.pt4_train.metrics",
    "trazo.pt4_train.utils",
    "trazo.pt4_train.datasets",
    "trazo.pt4_train.datamodules",
    "trazo.pt4_train.trainers",
]

PT3_MODULES = [
    "trazo.pt3_finetune.cli",
    "trazo.pt3_finetune.optimizers",
    "trazo.pt3_finetune.lora",
    "trazo.pt3_finetune.merge",
    "trazo.pt3_finetune.trainers",
]

PT5_MODULES = [
    "trazo.pt5_inference.cli",
    "trazo.pt5_inference.tilepairs",
    "trazo.pt5_inference.tilepairs_advanced",
    "trazo.pt5_inference.tilepairs_tilelist",
    "trazo.pt5_inference.multimodelinference",
    "trazo.pt5_inference.batchinference",
]


@pytest.mark.parametrize("name", CORE_MODULES)
def test_core_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", PT2_MODULES)
def test_pt2_imports(name):
    pytest.importorskip("rasterio", reason="needs the pt2 extra")
    importlib.import_module(name)


@pytest.mark.heavy
@pytest.mark.parametrize("name", PT4_MODULES)
def test_pt4_imports(name):
    pytest.importorskip("torch", reason="needs the pt4 extra")
    importlib.import_module(name)


@pytest.mark.heavy
@pytest.mark.parametrize("name", PT3_MODULES)
def test_pt3_imports(name):
    pytest.importorskip("torch", reason="needs the pt3 extra")
    importlib.import_module(name)


@pytest.mark.heavy
@pytest.mark.parametrize("name", PT5_MODULES)
def test_pt5_imports(name):
    pytest.importorskip("rasterio", reason="needs the pt5 extra")
    importlib.import_module(name)


def test_no_stale_package_name():
    """The package was renamed from `tkt` to `trazo`. Nothing may reference `tkt`."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in src.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and "tkt" in stripped:
                offenders.append(f"{path.relative_to(src)}:{lineno}: {stripped}")
    assert not offenders, "stale `tkt` imports:\n" + "\n".join(offenders)
