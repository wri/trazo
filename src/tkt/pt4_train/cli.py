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



import argparse
import yaml
import os
from lightning.pytorch.cli import LightningCLI
from torchgeo.trainers import BaseTask
from src.tkt.pt4_train import FTWDataModule

def parse_args():
    parser = argparse.ArgumentParser(description="Step 4: Model training.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ckpt", default=None)
    return parser.parse_args()

def fit(config_path, ckpt_path=None, data_dir=None, output_dir=None):
    """Command to fit the model."""
    print("Running fit command")
    print(f"Config: {config_path}")
    if ckpt_path:
        print(f"Checkpoint: {ckpt_path}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")

    # Best practices for Rasterio environment variables
    rasterio_best_practices = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
    os.environ.update(rasterio_best_practices)

    # Let LightningCLI handle all CLI setup automatically
    cli = LightningCLI(
        model_class=BaseTask,
        datamodule_class=FTWDataModule,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        datamodule_defaults={
            "data_dir": data_dir,
            "output_dir": output_dir,
        },
        run=False  # prevents immediate training, allows checkpoint to be loaded manually
    )

    # Optionally resume from checkpoint
    if ckpt_path:
        cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
    else:
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    print("Finished")

def main():
    args = parse_args()
    fit(
        config_path=args.config,
        ckpt_path=args.ckpt,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
