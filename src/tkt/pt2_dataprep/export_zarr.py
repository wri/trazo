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

"""
Export FTW_finaltraining datasets into consolidated Zarr stores per split.
- Handles presplit datasets (e.g., *_training/_validation/_testing) automatically.
- Randomly splits non-presplit datasets according to train/val/test fractions.
- Output: train.zarr, val.zarr, test.zarr in output_dir.
- Optimized for training speed: (N, C, H, W) chunked as (1, C, H, W)
- Uses Blosc ZSTD compression (fast).
"""

import argparse
from pathlib import Path
import shutil
import random

import torch
import zarr
from tqdm import tqdm
from src.tkt.pt4_train.datasets import FTW_finaltraining

def export_all_countries_to_zarr(
    root: str,
    countries: list[str],
    output_dir: str = "data/ftw/zarr",
    temporal_options: str = "stacked",
    load_boundaries: bool = False,
    num_samples: int = -1,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    """
    Exports multiple countries into consolidated train/val/test Zarr stores.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    splits = ["train", "val", "test"]
    split_indices = {s: [] for s in splits}  # global indices for each split
    split_samples = {s: [] for s in splits}  # to store samples temporarily

    for country in countries:
        # Determine if presplit
        presplit = False
        if country.endswith("_training"):
            presplit = True
            country_split_map = {"train": "training"}
        elif country.endswith("_validation"):
            presplit = True
            country_split_map = {"val": "validation"}
        elif country.endswith("_testing"):
            presplit = True
            country_split_map = {"test": "testing"}
        else:
            country_split_map = {}

        # Load full dataset
        dataset = FTW_finaltraining(
            root=root,
            countries=[country],
            split=country_split_map.get("train", "all") if not presplit else list(country_split_map.values())[0],
            temporal_options=temporal_options,
            load_boundaries=load_boundaries,
            num_samples=num_samples,
        )

        N = len(dataset)
        indices = list(range(N))

        if presplit:
            # Presplit dataset: assign all samples to the corresponding split
            if "_training" in country:
                split_samples["train"].extend([dataset[i] for i in indices])
            elif "_validation" in country:
                split_samples["val"].extend([dataset[i] for i in indices])
            elif "_testing" in country:
                split_samples["test"].extend([dataset[i] for i in indices])
        else:
            # Non-presplit dataset: shuffle and split
            random.shuffle(indices)
            n_train = int(train_frac * N)
            n_val = int(val_frac * N)
            n_test = N - n_train - n_val
            split_samples["train"].extend([dataset[i] for i in indices[:n_train]])
            split_samples["val"].extend([dataset[i] for i in indices[n_train:n_train + n_val]])
            split_samples["test"].extend([dataset[i] for i in indices[n_train + n_val:]])

    # Export to Zarr per split
    for split in splits:
        samples = split_samples[split]
        if not samples:
            print(f"No samples for split '{split}', skipping.")
            continue

        out_path = Path(output_dir) / f"{split}.zarr"
        if out_path.exists():
            shutil.rmtree(out_path)

        # Infer shapes from first sample
        img0 = samples[0]["image"].numpy()
        mask0 = samples[0]["mask"].numpy()
        C, H, W = img0.shape
        N = len(samples)

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

        print(f"Exporting {N} samples to {out_path} ...")
        for i, sample in enumerate(tqdm(samples)):
            img_ds[i] = sample["image"].numpy()
            mask_ds[i] = sample["mask"].numpy()
        print(f"✅ Export completed for split '{split}' → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--countries", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="data/ftw/zarr")
    parser.add_argument("--temporal_options", type=str, default="stacked")
    parser.add_argument("--load_boundaries", action="store_true")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    export_all_countries_to_zarr(
        root=args.root,
        countries=args.countries,
        output_dir=args.output_dir,
        temporal_options=args.temporal_options,
        load_boundaries=args.load_boundaries,
        num_samples=args.num_samples,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
