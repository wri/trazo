# import argparse
# import yaml
# import os
# from lightning.pytorch.cli import LightningCLI
# from torchgeo.trainers import BaseTask
# #from ftw_tools.training.datamodules import FTWDataModule
# from src.tkt.pt4_train import FTWDataModule

# def parse_args():
#     parser = argparse.ArgumentParser(description="Step 4: Model training.")
#     parser.add_argument("--config", default=None)
#     parser.add_argument("--data-dir", default=None)
#     parser.add_argument("--output-dir", default=None)
#     parser.add_argument("--ckpt", default=None) #

#     return parser.parse_args()

# # this fit function is copy/pasted from the ftw-baselines eval.py script but with the cli_args param removed
# # def fit(config, ckpt_path, cli_args):
# def fit(config, ckpt_path=None, data_dir=None, output_dir=None):

#     """Command to fit the model."""
#     print("Running fit command")

#     # Construct the arguments for PyTorch Lightning CLI
#     # cli_args = ["fit", f"--config={config}"] + list(cli_args)
#     cli_args = ["fit", f"--config={config}"] 


#     # If a checkpoint path is provided, append it to the CLI arguments
#     if ckpt_path:
#         cli_args += [f"--ckpt_path={ckpt_path}"]

#     print(f"CLI arguments: {cli_args}")

#     # Best practices for Rasterio environment variables
#     rasterio_best_practices = {
#         "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
#         "AWS_NO_SIGN_REQUEST": "YES",
#         "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
#         "GDAL_SWATH_SIZE": "200000000",
#         "VSI_CURL_CACHE_SIZE": "200000000",
#     }
#     os.environ.update(rasterio_best_practices)

#     # Run the LightningCLI with the constructed arguments
#     cli = LightningCLI(
#         model_class=BaseTask,
#         datamodule_class=FTWDataModule,
#         seed_everything_default=0,
#         subclass_mode_model=True,
#         subclass_mode_data=True,
#         save_config_kwargs={"overwrite": True},
#         args=cli_args,  # Pass the constructed cli_args
#         datamodule_defaults={
#             "data_dir": data_dir,
#             "output_dir": output_dir,
#         },
#     )

#     print("Finished")

# def main():
#     args = parse_args()
#     print("[pt4_train] Placeholder CLI.")
#     print("config    :", args.config)
#     print("data-dir  :", args.data_dir)
#     print("output-dir:", args.output_dir)

#     # load config
#     with open(args.config, "r") as f:
#         config = yaml.safe_load(f)

#     # create datamodule for dataloading
#     # datamodule = FTWDataModule(
#     #     data_dir=args.data_dir,
#     #     batch_size=config["training"]["batch_size"],
#     #     num_workers=config["training"].get("num_workers", 4),
#     #     **config.get("datamodule", {})
#     # )
#     fit(
#         config=args.config,
#         ckpt_path=args.ckpt,
#         # cli_args=args.extra_args,
#         data_dir=args.data_dir,
#         output_dir=args.output_dir
#     )



# import argparse
# import yaml
# import os
# from lightning.pytorch.cli import LightningCLI
# from torchgeo.trainers import BaseTask
# from src.tkt.pt4_train.datamodules import FTWDataModule

# # def parse_args():
# #     parser = argparse.ArgumentParser(description="Step 4: Model training.")
# #     parser.add_argument("--config", default=None)
# #     parser.add_argument("--data-dir", default=None)
# #     parser.add_argument("--output-dir", default=None)
# #     parser.add_argument("--ckpt", default=None)
# #     return parser.parse_args()

# def parse_args():
#     parser = argparse.ArgumentParser(description="Step 4: Model training.")
#     parser.add_argument("--config", default=None)
#     parser.add_argument("--data-dir", default=None)
#     parser.add_argument("--output-dir", default=None)
#     parser.add_argument("--ckpt", default=None)
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = parse_args()

#     # Load YAML config
#     with open(args.config, "r") as f:
#         config = yaml.safe_load(f)

#     # Override paths if provided
#     if args.data_dir:
#         config["data"]["dict_kwargs"]["root"] = args.data_dir
#     if args.output_dir:
#         config["trainer"]["default_root_dir"] = args.output_dir
#         # Also update checkpoint/logger dirs if needed
#         if "callbacks" in config["trainer"]:
#             for cb in config["trainer"]["callbacks"]:
#                 if cb.get("class_path") == "lightning.pytorch.callbacks.ModelCheckpoint":
#                     cb["init_args"]["dirpath"] = f"{args.output_dir}/checkpoints"
#         if "logger" in config["trainer"]:
#             for lg in config["trainer"]["logger"]:
#                 if lg.get("class_path") == "lightning.pytorch.loggers.CSVLogger":
#                     lg["init_args"]["save_dir"] = f"{args.output_dir}/metrics"

#     # Launch LightningCLI
#     LightningCLI(
#         model_class=CustomSemanticSegmentationTask,
#         datamodule_class=FTWDataModule,
#         config=config,
#         seed_everything_default=7,
#         run=True,
#     )
# def fit(config_path, ckpt_path=None, data_dir=None, output_dir=None):
#     """Command to fit the model."""
#     print("Running fit command")
#     print(f"Config: {config_path}")
#     if ckpt_path:
#         print(f"Checkpoint: {ckpt_path}")
#     print(f"Data dir: {data_dir}")
#     print(f"Output dir: {output_dir}")

#     # Best practices for Rasterio environment variables
#     rasterio_best_practices = {
#         "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
#         "AWS_NO_SIGN_REQUEST": "YES",
#         "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
#         "GDAL_SWATH_SIZE": "200000000",
#         "VSI_CURL_CACHE_SIZE": "200000000",
#     }
#     os.environ.update(rasterio_best_practices)
    
#     cli_args = ["fit", f"--config={config_path}"]
#     if data_dir:
#         cli_args += [f"--datamodule.data_dir={data_dir}"]
#     if output_dir:
#         cli_args += [f"--datamodule.output_dir={output_dir}"]

#     # Let LightningCLI handle all CLI setup automatically
#     cli = LightningCLI(
#         model_class=BaseTask,
#         datamodule_class=FTWDataModule,
#         seed_everything_default=0,
#         subclass_mode_model=True,
#         subclass_mode_data=True,
#         save_config_kwargs={"overwrite": True},
#         run=False,  # prevents immediate training
#     )

#     # Optionally resume from checkpoint
#     if ckpt_path:
#         cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
#     else:
#         cli.trainer.fit(cli.model, datamodule=cli.datamodule)

#     print("Finished")

# def main():
#     args = parse_args()
#     fit(
#         config_path=args.config,
#         ckpt_path=args.ckpt,
#         data_dir=args.data_dir,
#         output_dir=args.output_dir
#     )

# if __name__ == "__main__":
#     main()
# import sys
# import os
# import argparse
# import yaml
# from lightning.pytorch.cli import LightningCLI
# from torchgeo.trainers import BaseTask
# import tempfile

# # Make sure src/ is importable
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # Import your model + datamodule
# # from src.tkt.pt4_train.trainers import CustomSemanticSegmentationTask
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

#     # Override paths if provided
#     if args.data_dir:
#         config["data"]["dict_kwargs"]["root"] = args.data_dir

#     if args.output_dir:
#         config["trainer"]["default_root_dir"] = args.output_dir

#         # Update checkpoint + logger dirs
#         for cb in config["trainer"].get("callbacks", []):
#             if cb.get("class_path") == "lightning.pytorch.callbacks.ModelCheckpoint":
#                 cb["init_args"]["dirpath"] = f"{args.output_dir}/checkpoints"

#         for lg in config["trainer"].get("logger", []):
#             if lg.get("class_path") == "lightning.pytorch.loggers.CSVLogger":
#                 lg["init_args"]["save_dir"] = f"{args.output_dir}/metrics"

#     # # Run training
#     # LightningCLI(
#     #     model_class=BaseTask,
#     #     datamodule_class=FTWDataModule,
#     #     # config=config,
#     #     run=True,
#     #     seed_everything_default=7,
#     # )
#     # Save modified config to a temporary YAML
#     with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmpfile:
#         yaml.dump(config, tmpfile)
#         tmp_config_path = tmpfile.name
    
#     # Run LightningCLI with the subcommand
#     LightningCLI(
#         model_class=BaseTask,  # or CustomSemanticSegmentationTask
#         datamodule_class=FTWDataModule,
#         run=True,
#         seed_everything_default=7,
#         args=[args.subcommand, f"--config={tmp_config_path}"]
#     )

import sys
import os
import argparse
import yaml
import tempfile
from lightning.pytorch.cli import LightningCLI

# Make sure src/ is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model + datamodule
from src.tkt.pt4_train.trainers import CustomSemanticSegmentationTask
from src.tkt.pt4_train.datamodules import FTWDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Step 4: Model training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("subcommand", choices=["fit", "validate", "test", "predict"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override paths in config if provided
    if args.data_dir:
        config["data"]["dict_kwargs"]["root"] = args.data_dir

    if args.output_dir:
        config["trainer"]["default_root_dir"] = args.output_dir

        # Update checkpoint + logger dirs
        for cb in config["trainer"].get("callbacks", []):
            if cb.get("class_path") == "lightning.pytorch.callbacks.ModelCheckpoint":
                cb["init_args"]["dirpath"] = f"{args.output_dir}/checkpoints"

        for lg in config["trainer"].get("logger", []):
            if lg.get("class_path") == "lightning.pytorch.loggers.CSVLogger":
                lg["init_args"]["save_dir"] = f"{args.output_dir}/metrics"

    # Save modified config to a temporary YAML file
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmpfile:
        yaml.dump(config, tmpfile)
        tmp_config_path = tmpfile.name

    # Run LightningCLI with ONLY the subcommand and the modified config
    cli_args = [args.subcommand, f"--config={tmp_config_path}"]
    if args.ckpt:
        cli_args += [f"--ckpt_path={args.ckpt}"]

    LightningCLI(
        model_class=CustomSemanticSegmentationTask,
        datamodule_class=FTWDataModule,
        run=True,
        seed_everything_default=7,
        args=cli_args
    )












