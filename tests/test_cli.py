"""The documented CLI entry points must at least start and print help.

`--help` exiting non-zero is the cheapest possible signal that an entry point is
mis-wired, which is exactly how the console scripts broke before.
"""

import subprocess
import sys

import pytest

MODULE_HELP = [
    ("trazo.pt1_createdata.gridding", None),
    ("trazo.pt1_createdata.cli", None),
    ("trazo.pt2_dataprep", "rasterio"),
    ("trazo.pt2_dataprep.pair_stacks", "rasterio"),
    ("trazo.pt2_dataprep.make_masks_and_windows", "rasterio"),
    ("trazo.pt5_inference.cli", "rasterio"),
    ("trazo.smoke", None),
]


@pytest.mark.parametrize("module,needs", MODULE_HELP)
def test_module_help(module, needs):
    if needs:
        pytest.importorskip(needs, reason=f"needs {needs}")
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"`python -m {module} --help` exited {result.returncode}\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
    )
    assert result.stdout.strip(), f"`python -m {module} --help` printed nothing"


def test_pt2_subcommand_dispatch_covers_documented_names():
    """Every subcommand in the usage string must be dispatchable."""
    pytest.importorskip("rasterio", reason="needs the pt2 extra")
    from trazo.pt2_dataprep import cli

    documented = {
        "pair-stacks",
        "resize-256",
        "chips-bboxes",
        "make-masks",
        "chips-parquet",
        "scale-u16",
        "export-hkl",
        "export-zarr",
    }
    assert documented == set(cli._DISPATCH)


def test_pt2_mains_accept_argv():
    """Every Step 2 entry point takes argv, so it is callable in-process."""
    pytest.importorskip("rasterio", reason="needs the pt2 extra")
    import importlib
    import inspect

    modules = [
        "pair_stacks",
        "resize_chips_256",
        "chips_to_bboxes",
        "make_masks_and_windows",
        "build_chips_parquet",
        "scale_uint16",
        "export_hkl",
        "export_zarr",
    ]
    for name in modules:
        try:
            mod = importlib.import_module(f"trazo.pt2_dataprep.{name}")
        except ModuleNotFoundError as exc:  # optional export deps (zarr, hickle)
            pytest.skip(f"{name} needs {exc.name}")
        params = inspect.signature(mod.main).parameters
        assert params, f"trazo.pt2_dataprep.{name}.main() takes no argv"


@pytest.mark.heavy
def test_pt4_fit_forwards_data_and_output_dirs():
    """--data-dir and --output-dir must reach LightningCLI, not be dropped."""
    pytest.importorskip("click", reason="needs the pt4 extra")
    from trazo.pt4_train.cli import build_fit_args

    args = build_fit_args(
        "cfg.yaml",
        ckpt_path="best.ckpt",
        cli_args=("--trainer.max_epochs=1",),
        data_dir="/data/region",
        output_dir="/models/out",
    )
    assert args[0] == "fit"
    assert "--config=cfg.yaml" in args
    assert "--data.init_args.root=/data/region" in args
    assert "--trainer.default_root_dir=/models/out" in args
    assert "--ckpt_path=best.ckpt" in args
    assert "--trainer.max_epochs=1" in args


@pytest.mark.heavy
def test_pt4_fit_args_omit_unset_options():
    pytest.importorskip("click", reason="needs the pt4 extra")
    from trazo.pt4_train.cli import build_fit_args

    args = build_fit_args("cfg.yaml")
    assert args == ["fit", "--config=cfg.yaml"]


@pytest.mark.heavy
def test_pt4_test_command_resolves_the_ftw_entry_point():
    """`trazo-pt4-train test` delegates to ftw-tools.

    ftw-tools installs its packages as `ftw` / `ftw_cli`; the module path the
    code originally imported (`ftw_tools.training.eval`) does not exist in any
    published version, so the command raised ModuleNotFoundError on use.
    """
    pytest.importorskip("ftw_cli", reason="needs the pt4/pt5 extra")
    from ftw_cli.model import test as ftw_test

    assert callable(ftw_test)

    import inspect

    from trazo.pt4_train import cli

    # model_test is a click Command; the function is on .callback
    source = inspect.getsource(cli.model_test.callback)
    assert "ftw_cli.model" in source


@pytest.mark.heavy
def test_ftw_is_importable_with_the_pinned_torchgeo():
    """ftw-tools imports torchgeo.transforms.AugmentationSequential.

    torchgeo removed it in 0.8 while ftw-tools still declares `torchgeo>=0.7`,
    so an unpinned resolve installs a combination where `import ftw` fails.
    The pin exists to prevent that; this asserts the pin is doing its job.
    """
    pytest.importorskip("ftw", reason="needs the pt4/pt5 extra")
    import ftw.datamodules  # noqa: F401
    import ftw.trainers  # noqa: F401
