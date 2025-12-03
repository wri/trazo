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

"""
Export FTW_finaltraining dataset into Zarr stores.
- Automatically splits datasets that are not presplit (train/val/test fractions).
- Uses blosc zstd compression (fastest).
"""

import argparse
from pathlib import Path
import shutil
import random

import torch
import zarr
from tqdm import tqdm
from src.tkt.pt4_train.datasets import FTW_finaltraining


def export_dataset_to_zarr(
    root: str,
    country: str,
    split: str = "train",
    output_dir: str = "data/ftw/zarr",
    temporal_options: str = "stacked",
    load_boundaries: bool = False,
    num_samples: int = -1,
    ignore_sample_fn: str | None = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    """
    Exports a dataset folder to Zarr.
    If the country/folder is presplit (ends with _training/_validation/_testing),
    uses that split directly.
    Otherwise, randomly splits according to train/val/test fractions.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    # Determine if folder is presplit
    presplit = False
    split_suffix = split
    if country.endswith("_training") and split == "train":
        presplit = True
        split_suffix = "training"
    elif country.endswith("_validation") and split == "val":
        presplit = True
        split_suffix = "validation"
    elif country.endswith("_testing") and split == "test":
        presplit = True
        split_suffix = "testing"

    dataset = FTW_finaltraining(
        root=root,
        countries=[country],
        split=split_suffix if presplit else "all",
        temporal_options=temporal_options,
        load_boundaries=load_boundaries,
        num_samples=num_samples,
        ignore_sample_fn=ignore_sample_fn,
    )

    # Determine which indices to use for this split
    N = len(dataset)
    indices = list(range(N))
    if not presplit:
        random.shuffle(indices)
        n_train = int(train_frac * N)
        n_val = int(val_frac * N)
        n_test = N - n_train - n_val
        if split == "train":
            indices = indices[:n_train]
        elif split == "val":
            indices = indices[n_train:n_train + n_val]
        elif split == "test":
            indices = indices[n_train + n_val:]

    # Output path
    out_path = Path(output_dir) / f"{split}.zarr"
    if out_path.exists():
        shutil.rmtree(out_path)

    print(f"Exporting {len(indices)} samples for '{country}' ({split}) → {out_path}")

    # Load 1 sample to infer shapes
    sample0 = dataset[indices[0]]
    img0 = sample0["image"].numpy()
    mask0 = sample0["mask"].numpy()
    C, H, W = img0.shape

    # Create Zarr store
    store = zarr.open(str(out_path), mode="w")
    img_ds = store.create_dataset(
        "images",
        shape=(len(indices), C, H, W),
        chunks=(1, C, H, W),
        dtype=img0.dtype,
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )
    mask_ds = store.create_dataset(
        "masks",
        shape=(len(indices), H, W),
        chunks=(1, H, W),
        dtype=mask0.dtype,
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )

    # Fill in data
    for i, idx in enumerate(tqdm(indices)):
        sample = dataset[idx]
        img_ds[i] = sample["image"].numpy()
        mask_ds[i] = sample["mask"].numpy()

    print(f"✅ Export completed for '{country}' ({split})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--countries", nargs="+", required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="data/ftw/zarr")
    parser.add_argument("--temporal_options", type=str, default="stacked")
    parser.add_argument("--load_boundaries", action="store_true")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--ignore_sample_fn", type=str, default=None)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    args = parser.parse_args()

    for country in args.countries:
        export_dataset_to_zarr(
            root=args.root,
            country=country,
            split=args.split,
            output_dir=args.output_dir,
            temporal_options=args.temporal_options,
            load_boundaries=args.load_boundaries,
            num_samples=args.num_samples,
            ignore_sample_fn=args.ignore_sample_fn,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
