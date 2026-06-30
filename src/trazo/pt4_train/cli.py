import click
import os


def fit(config, ckpt_path, cli_args):
    """Command to fit the model."""
    from lightning.pytorch.cli import LightningCLI
    from torchgeo.trainers import BaseTask

    print("Running fit command")

    cli_args = ["fit", f"--config={config}"] + list(cli_args)
    if ckpt_path:
        cli_args += [f"--ckpt_path={ckpt_path}"]
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
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to dataset directory",
)
@click.option(
    "--output-dir", "-o",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory where logs/checkpoints should be saved",
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
    fit(config, ckpt_path, cli_args)


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
