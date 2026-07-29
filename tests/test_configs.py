"""Every bundled config must actually build a model and a datamodule.

A config that parses as YAML but fails to instantiate is worse than no config:
the user only finds out after installing the whole training stack. This also
pins the override syntax both CLIs generate (`--data.init_args.root=...`), which
is the mechanism that makes `--data-dir` and `--output-dir` mean anything.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.heavy

pytest.importorskip("torch", reason="needs the pt3/pt4 extras")
pytest.importorskip("lightning", reason="needs the pt3/pt4 extras")


def _config_paths():
    import trazo

    root = Path(trazo.__file__).parent
    return sorted((root / "pt3_finetune" / "configs").glob("*.yaml")) + sorted(
        (root / "pt4_train" / "configs").glob("*.yaml")
    )


def test_configs_are_present():
    assert len(_config_paths()) >= 5


@pytest.mark.parametrize("config", _config_paths(), ids=lambda p: p.name)
def test_config_instantiates_and_honours_overrides(config, tmp_path):
    from lightning.pytorch.cli import LightningCLI
    from torchgeo.trainers import BaseTask

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"

    cli = LightningCLI(
        model_class=BaseTask,
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=False,
        save_config_kwargs={"overwrite": True},
        args=[
            "--config", str(config),
            f"--data.init_args.root={data_dir}",
            f"--trainer.default_root_dir={out_dir}",
        ],
    )

    assert cli.model is not None
    assert cli.datamodule is not None
    # The override has to win over the value baked into the config.
    assert str(cli.datamodule.root) == str(data_dir)


@pytest.mark.parametrize(
    "strategy", ["full", "lastlayer", "lora", "upgd"]
)
def test_finetune_config_declares_its_strategy(strategy):
    import yaml

    from trazo.pt3_finetune.cli import default_config_for

    cfg = yaml.safe_load(default_config_for(strategy).read_text(encoding="utf-8"))
    assert cfg["model"]["init_args"]["strategy"] == strategy
    assert cfg["model"]["class_path"] == "trazo.pt3_finetune.trainers.FineTuneTask"
    assert cfg["data"]["class_path"] == "trazo.pt4_train.datamodules.FTWDataModule"
