import click
import yaml
from lightning.pytorch.cli import LightningCLI
import os
from torchgeo.trainers import BaseTask
from src.tkt.pt4_train.settings import ALL_COUNTRIES
COUNTRIES_CHOICE = ALL_COUNTRIES.copy()
COUNTRIES_CHOICE.append("all")
COUNTRIES_CHOICE.append("full_data")


def fit(config, ckpt_path, cli_args):
    """Command to fit the model."""
    print("Running fit command")

    # Construct the arguments for PyTorch Lightning CLI
    cli_args = ["fit", f"--config={config}"] + list(cli_args)

    # If a checkpoint path is provided, append it to the CLI arguments
    if ckpt_path:
        cli_args += [f"--ckpt_path={ckpt_path}"]

    print(f"CLI arguments: {cli_args}")

    # Best practices for Rasterio environment variables
    rasterio_best_practices = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
    os.environ.update(rasterio_best_practices)

    # Run the LightningCLI with the constructed arguments
    cli = LightningCLI(
        model_class=BaseTask,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        args=cli_args,  # Pass the constructed cli_args
    )

    print("Finished")

@click.group()
def model():
    """Training and testing FTW models."""
    pass


#
# ------------------------- MODEL FIT -------------------------
#
@model.command("fit", help="Fit the model")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the config file",
)
@click.option(
    "--data-dir",
    "-d",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to dataset directory",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory where logs/checkpoints should be saved",
)
@click.option(
    "--ckpt_path",
    "-m",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    show_default=True,
    help="Path to a checkpoint file to resume training from",
)
@click.argument(
    "cli_args", nargs=-1, type=click.UNPROCESSED
)
def model_fit(config, data_dir, output_dir, ckpt_path, cli_args):
    """
    EXACT same behavior as original:
    - Loads YAML config
    - Passes config & ckpt_path & cli_args directly to original fit()
    
    NEW:
    - Injects data_dir + output_dir into config before calling fit()
    """

    # Load YAML
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    # Insert CLI args into config (preserves original behavior)
    cfg["data_dir"] = data_dir
    cfg["output_dir"] = output_dir

    # Run the original training function
    fit(cfg, ckpt_path, cli_args)


#
# -------------- MODEL TEST (unchanged from original) ----------
#
@model.command("test", help="Test the model")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to model checkpoint",
)
@click.option(
    "--countries",
    "-c",
    type=click.Choice(COUNTRIES_CHOICE, case_sensitive=False),
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
# ... keep all original test options EXACTLY the same ...
def model_test(**kwargs):
    from ftw_tools.training.eval import test
    test(**kwargs)


if __name__ == "__main__":
    model()





