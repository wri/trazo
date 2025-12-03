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


def export_dataset_to_zarr(
    root: str,
    countries: list[str],
    split: str = "train",
    output_dir: str = "data/ftw/zarr",
    temporal_options: str = "stacked",
    load_boundaries: bool = False,
    num_samples: int = -1,
    ignore_sample_fn: str | None = None,
) -> None:

    dataset = FTW_finaltraining(
        root=root,
        countries=countries,
        split=split,
        temporal_options=temporal_options,
        load_boundaries=load_boundaries,
        num_samples=num_samples,
        ignore_sample_fn=ignore_sample_fn,
    )

    out_path = Path(output_dir) / f"{split}.zarr"

    # Delete old zarr store if it exists
    if out_path.exists():
        shutil.rmtree(out_path)

    print(f"Exporting {len(dataset)} samples to {out_path} ...")

    # Load 1 sample to infer shapes
    sample0 = dataset[0]
    img0 = sample0["image"].numpy()
    mask0 = sample0["mask"].numpy()

    N = len(dataset)
    C, H, W = img0.shape

    # Create Zarr store
    store = zarr.open(str(out_path), mode="w")

    # Optimal chunk = 1 sample at a time
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
    for idx in tqdm(range(N)):
        sample = dataset[idx]
        img = sample["image"].numpy()
        mask = sample["mask"].numpy()

        img_ds[idx] = img
        mask_ds[idx] = mask

    print(f"✅ Export completed for split '{split}' → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--countries", nargs="+", required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data/ftw/zarr")
    parser.add_argument("--temporal_options", type=str, default="stacked")
    parser.add_argument("--load_boundaries", action="store_true")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--ignore_sample_fn", type=str, default=None)
    args = parser.parse_args()

    export_dataset_to_zarr(
        root=args.root,
        countries=args.countries,
        split=args.split,
        output_dir=args.output_dir,
        temporal_options=args.temporal_options,
        load_boundaries=args.load_boundaries,
        num_samples=args.num_samples,
        ignore_sample_fn=args.ignore_sample_fn,
    )
