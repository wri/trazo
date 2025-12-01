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

# src/tkt/pt4_train/cli.py

import sys
import os
import argparse
import yaml
from pathlib import Path
from torch.optim import AdamW
from lightning.pytorch import Trainer
from src.tkt.pt4_train.trainers import CustomSemanticSegmentationTask
from src.tkt.pt4_train.datamodules import FTWDataModule

# Rasterio / GDAL environment variables
rasterio_best_practices = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
    "GDAL_SWATH_SIZE": "200000000",
    "VSI_CURL_CACHE_SIZE": "200000000",
}
os.environ.update(rasterio_best_practices)

def parse_args():
    parser = argparse.ArgumentParser(description="Step 4: Model training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ckpt", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override paths
    if args.data_dir:
        config["data"]["root"] = args.data_dir
    if args.output_dir:
        config["trainer"]["default_root_dir"] = args.output_dir
        # Update checkpoint/logging paths
        for cb in config["trainer"].get("callbacks", []):
            if cb.get("class_path") == "lightning.pytorch.callbacks.ModelCheckpoint":
                cb["init_args"]["dirpath"] = f"{args.output_dir}/checkpoints"
        for lg in config["trainer"].get("logger", []):
            if lg.get("class_path") == "lightning.pytorch.loggers.CSVLogger":
                lg["init_args"]["save_dir"] = f"{args.output_dir}/metrics"

    # Fix learning rate type
    config["model"]["init_args"]["lr"] = float(config["model"]["init_args"]["lr"])

    # Instantiate datamodule and model
    datamodule = FTWDataModule(**config["data"]["init_args"])
    model = CustomSemanticSegmentationTask(**config["model"]["init_args"])

    # Create trainer manually
    trainer = Trainer(**config["trainer"])
    
    # Fit
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt)

