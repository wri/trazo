import click
import os


def build_fit_args(config, ckpt_path=None, cli_args=(), data_dir=None, output_dir=None):
    """Assemble the argv handed to LightningCLI for a fit run.

    ``data_dir`` and ``output_dir`` become ``--data.init_args.root`` and
    ``--trainer.default_root_dir`` so they override whatever the YAML config
    sets. Offering the flags and then ignoring them is worse than not offering
    them at all, so this is covered by tests.
    """
    args = ["fit", f"--config={config}"] + list(cli_args)
    if data_dir:
        args += [f"--data.init_args.root={data_dir}"]
    if output_dir:
        args += [f"--trainer.default_root_dir={output_dir}"]
    if ckpt_path:
        args += [f"--ckpt_path={ckpt_path}"]
    return args


def fit(config, ckpt_path, cli_args, data_dir=None, output_dir=None):
    """Command to fit the model."""
    from lightning.pytorch.cli import LightningCLI
    from torchgeo.trainers import BaseTask

    print("Running fit command")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cli_args = build_fit_args(
        config, ckpt_path=ckpt_path, cli_args=cli_args,
        data_dir=data_dir, output_dir=output_dir,
    )
    print(f"CLI arguments: {cli_args}")

    rasterio_best_practices = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
    os.environ.update(rasterio_best_practices)

    cli = LightningCLI(
        model_class=BaseTask,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        args=cli_args,
    )
    print("Finished")


@click.group()
def model():
    """Training and testing FTW models."""
    pass


@model.command("fit", help="Fit the model")
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the config file",
)
@click.option(
    "--data-dir", "-d",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Dataset directory. Overrides data.init_args.root in the config.",
)
@click.option(
    "--output-dir", "-o",
    default=None,
    type=click.Path(file_okay=False),
    help=(
        "Directory for logs and checkpoints. Overrides trainer.default_root_dir "
        "in the config."
    ),
)
@click.option(
    "--ckpt_path", "-m",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    show_default=True,
    help="Path to a checkpoint file to resume training from",
)
@click.argument("cli_args", nargs=-1, type=click.UNPROCESSED)
def model_fit(config, data_dir, output_dir, ckpt_path, cli_args):
    """Fit the model using a YAML config."""
    fit(config, ckpt_path, cli_args, data_dir=data_dir, output_dir=output_dir)


@model.command("test", help="Test the model")
@click.option(
    "--model", "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to model checkpoint",
)
@click.option(
    "--countries", "-c",
    multiple=True,
    required=True,
    help="Countries to evaluate on",
)
@click.option(
    "--dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="./data/ftw",
    show_default=True,
    help="Directory of the FTW dataset",
)
@click.option(
    "--gpu",
    type=int,
    default=0,
    show_default=True,
    help="GPU index",
)
def model_test(**kwargs):
    from ftw_tools.training.eval import test
    test(**kwargs)


if __name__ == "__main__":
    model()
