# # Run using !python -m src.tkt.pt4_train.cli fit \
# #     --config "config file path" \
# #     --data-dir "data path" \
# #     --output-dir "path to save files to"

# import sys
# import os
# import argparse
# import yaml
# import tempfile
# from torchgeo.trainers import BaseTask
# from lightning.pytorch.cli import LightningCLI

# # Make sure src/ is importable
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # Import your model + datamodule
# from src.tkt.pt4_train.trainers import CustomSemanticSegmentationTask
# from src.tkt.pt4_train.datamodules import FTWDataModule


# def parse_args():
#     parser = argparse.ArgumentParser(description="Step 4: Model training.")
#     parser.add_argument("--config", required=True)
#     parser.add_argument("--data-dir", default=None)
#     parser.add_argument("--output-dir", default=None)
#     parser.add_argument("--ckpt", default=None)
#     parser.add_argument("subcommand", choices=["fit", "validate", "test", "predict"])
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()

#     # Load YAML config
#     with open(args.config, "r") as f:
#         config = yaml.safe_load(f)

#     # Override paths in config if provided
#     if args.data_dir:
#         config["data"]["root"] = args.data_dir

#     if args.output_dir:
#         config["trainer"]["default_root_dir"] = args.output_dir

#         # Update checkpoint + logger dirs
#         for cb in config["trainer"].get("callbacks", []):
#             if cb.get("class_path") == "lightning.pytorch.callbacks.ModelCheckpoint":
#                 cb["init_args"]["dirpath"] = f"{args.output_dir}/checkpoints"

#         for lg in config["trainer"].get("logger", []):
#             if lg.get("class_path") == "lightning.pytorch.loggers.CSVLogger":
#                 lg["init_args"]["save_dir"] = f"{args.output_dir}/metrics"

#     # Save modified config to a temporary YAML file
#     with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmpfile:
#         yaml.dump(config, tmpfile)
#         tmp_config_path = tmpfile.name

#     # Run LightningCLI with ONLY the subcommand and the modified config
#     cli_args = [args.subcommand, f"--config={tmp_config_path}"]
#     if args.ckpt:
#         cli_args += [f"--ckpt_path={args.ckpt}"]

#     # LightningCLI(
#     #     model_class=CustomSemanticSegmentationTask,
#     #     datamodule_class=FTWDataModule,
#     #     run=True,
#     #     seed_everything_default=7,
#     #     args=cli_args
#     # )
#     LightningCLI(
#         model_class=BaseTask,  
#         datamodule_class=FTWDataModule,
#         seed_everything_default=0,
#         subclass_mode_model=True,
#         subclass_mode_data=True,
#         save_config_kwargs={"overwrite": True},
#         args=cli_args
#     )


# Run using:
# !python -m src.tkt.pt4_train.cli fit \
#     --config "config file path" \
#     --data-dir "data path" \
#     --output-dir "path to save files to"

import sys
import os
import argparse
import yaml
from lightning.pytorch.cli import LightningCLI
from torchgeo.trainers import BaseTask

# Make sure src/ is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your datamodule
from src.tkt.pt4_train.datamodules import FTWDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Step 4: Model training.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data-dir", default=None, help="Override data root directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path to resume from")
    parser.add_argument("subcommand", choices=["fit", "validate", "test", "predict"])
    return parser.parse_args()


def fit(config_path, ckpt_path=None, cli_args=[]):
    """Run the fit command with LightningCLI."""
    print("Running fit command")

    # Construct CLI args
    cli_args = ["fit", f"--config={config_path}"] + list(cli_args)
    if ckpt_path:
        cli_args += [f"--ckpt_path={ckpt_path}"]

    print(f"CLI arguments: {cli_args}")

    # Rasterio best practices
    rasterio_best_practices = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
    os.environ.update(rasterio_best_practices)

    # Run LightningCLI
    cli = LightningCLI(
        model_class=BaseTask,
        datamodule_class=FTWDataModule,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        args=cli_args,
    )

    print("Finished")


if __name__ == "__main__":
    args = parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override data root if provided
    if args.data_dir:
        config["data"]["root"] = args.data_dir

    # Override output dirs if provided
    if args.output_dir:
        if "trainer" in config:
            config["trainer"]["default_root_dir"] = args.output_dir
            for cb in config["trainer"].get("callbacks", []):
                if cb.get("class_path") == "lightning.pytorch.callbacks.ModelCheckpoint":
                    cb["init_args"]["dirpath"] = f"{args.output_dir}/checkpoints"
            for lg in config["trainer"].get("logger", []):
                if lg.get("class_path") == "lightning.pytorch.loggers.CSVLogger":
                    lg["init_args"]["save_dir"] = f"{args.output_dir}/metrics"

    # Save modified config to a temporary YAML file
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmpfile:
        yaml.dump(config, tmpfile)
        tmp_config_path = tmpfile.name

    # Call fit using the old-style function
    fit(tmp_config_path, ckpt_path=args.ckpt, cli_args=[])


















