"""
Export any FTW dataset (presplit or not) to Zarr format.
"""

import os
from pathlib import Path
import numpy as np
import zarr
from torch.utils.data import Dataset
# from torchgeo.datasets import NonGeoDataset
import torch


def export_to_zarr(dataset: Dataset, out_root: str, split: str):
    """
    Writes image/mask pairs into Zarr store.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    zarr_path = out_root / f"{split}.zarr"
    if zarr_path.exists():
        print(f"Removing previous Zarr store at {zarr_path}")
        import shutil
        shutil.rmtree(zarr_path)

    print(f"Creating Zarr store at: {zarr_path}")
    store = zarr.open(str(zarr_path), mode="w")

    # Probe first sample for shapes
    sample0 = dataset[0]
    C, H, W = sample0["image"].shape
    MH, MW = sample0["mask"].shape

    img_arr = store.create(
        "images",
        shape=(len(dataset), C, H, W),
        chunks=(1, C, H, W),
        dtype=np.float32,
    )

    mask_arr = store.create(
        "masks",
        shape=(len(dataset), MH, MW),
        chunks=(1, MH, MW),
        dtype=np.int64,
    )

    for i in range(len(dataset)):
        s = dataset[i]
        img_arr[i] = s["image"].numpy()
        mask_arr[i] = s["mask"].numpy()

        if i % 500 == 0:
            print(f"  Saved {i}/{len(dataset)} samples ...")

    print(f"Finished writing split '{split}' to {zarr_path}")
    return zarr_path


def export_presplit(train_ds, val_ds, out_root):
    print("Exporting presplit dataset...")

    export_to_zarr(train_ds, out_root, "train")
    export_to_zarr(val_ds, out_root, "val")


def export_unsplit(full_dataset, out_root, train_fraction=0.8):
    print("Exporting unsplit dataset...")

    n = len(full_dataset)
    train_n = int(n * train_fraction)

    indices = torch.randperm(n)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:]

    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)

    export_to_zarr(train_subset, out_root, "train")
    export_to_zarr(val_subset, out_root, "val")

