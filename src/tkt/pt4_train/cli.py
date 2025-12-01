import click
import yaml

from ftw_tools.training.eval import fit as original_fit

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
    original_fit(cfg, ckpt_path, cli_args)


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
