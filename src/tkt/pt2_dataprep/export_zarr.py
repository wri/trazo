"""
Export FTW_finaltraining dataset into a SINGLE Zarr store per split.
Optimal for training speed: (N, C, H, W) chunked as (1, C, H, W).

- Overwrites old Zarr stores automatically.
- Uses blosc zstd compression (fastest).
"""

import argparse
from pathlib import Path
import shutil

import torch
import zarr
import numpy as np
from tqdm import tqdm

from src.tkt.pt4_train.datasets import FTW_finaltraining


"""
Export all FTW_finaltraining regions into single Zarr stores per split.
- Creates: train.zarr, val.zarr, test.zarr
- Optimized for training speed: (N, C, H, W) chunked as (1, C, H, W)
- Overwrites old Zarr stores automatically.
- Uses blosc zstd compression (fastest).
"""

import shutil
from pathlib import Path
from tqdm import tqdm

import torch
import zarr
from src.tkt.pt4_train.datasets import FTW_finaltraining

# -----------------------
# CONFIGURATION
# -----------------------
root_dir = "/content/drive/MyDrive/TkT Test Sets/ftw-baselines/ftw-training-data"
output_dir = "/content/drive/MyDrive/TkT Test Sets/ftw-baselines/ftw-zarr"
splits = ["train", "val", "test"]

# List all regions for each split (update if needed)
regions_by_split = {
    "train": [
        "acre_training", "altoparana_training", "amapa_training", "araucaria_training",
        "caatinga_training", "cerrado_training", "chaco_training",
        "humidchaco_training", "maparaguay_training", "matogrosso_10lowestbandpartial",
        "peruvianamazon_training", "pantanal_training", "rondonia_training",
        "uruguayansavannah_training",
    ],
    "val": [
        "acre_validation", "altoparana_validation", "amapa_validation", "araucaria_validation",
        "caatinga_validation", "cerrado_validation", "chaco_validation",
        "humidchaco_validation", "maparaguay_validation", "peruvianamazon_validation",
        "pantanal_validation", "rondonia_validation", "uruguayansavannah_validation",
    ],
    "test": [
        "cerradotraining", "rondonia_testing", "southamericasoy", "soy", "tocantins"
    ],
}

# -----------------------
# FUNCTION TO EXPORT
# -----------------------
def export_split_to_zarr(split: str):
    print(f"\n🚀 Exporting {split} split ...")
    # Collect all samples from all regions
    combined_dataset = []
    for region in regions_by_split[split]:
        print(f"Loading region: {region}")
        ds = FTW_finaltraining(root=root_dir, countries=[region], split=split)
        combined_dataset.extend(ds)

    N = len(combined_dataset)
    print(f"Total samples in {split}: {N}")

    # Infer shape from first sample
    sample0 = combined_dataset[0]
    img0 = sample0["image"].numpy()
    mask0 = sample0["mask"].numpy()
    C, H, W = img0.shape

    # Create Zarr store
    out_path = Path(output_dir) / f"{split}.zarr"
    if out_path.exists():
        shutil.rmtree(out_path)
    store = zarr.open(str(out_path), mode="w")

    img_ds = store.create_dataset(
        "images",
        shape=(N, C, H, W),
        chunks=(1, C, H, W),
        dtype=img0.dtype,
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )
    mask_ds = store.create_dataset(
        "masks",
        shape=(N, H, W),
        chunks=(1, H, W),
        dtype=mask0.dtype,
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )

    # Fill in data
    for idx, sample in enumerate(tqdm(combined_dataset)):
        img_ds[idx] = sample["image"].numpy()
        mask_ds[idx] = sample["mask"].numpy()

    print(f"✅ {split}.zarr created at {out_path}")


# -----------------------
# EXPORT ALL SPLITS
# -----------------------
Path(output_dir).mkdir(parents=True, exist_ok=True)
for split in splits:
    export_split_to_zarr(split)

print("\n All splits exported successfully!")
