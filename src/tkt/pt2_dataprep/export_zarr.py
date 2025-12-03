#!/usr/bin/env python
"""
Export FTW dataset to Zarr stores.
- Reads window_a and window_b images and a mask for each chip.
- Stacks images along the channel axis (temporal_options="stacked").
- Writes one Zarr store per country.
- Efficient chunking: (1, C, H, W) for images, (1, H, W) for masks.
- Uses Blosc zstd compression.
"""

import argparse
from pathlib import Path
import random

import geopandas as gpd
import numpy as np
import rasterio
import zarr
import numcodecs
from tqdm import tqdm

def find_countries(root: str) -> list[str]:
    """Automatically find country folders: any subfolder containing chips_*.parquet."""
    countries: list[str] = []
    root_path = Path(root)
    if not root_path.exists():
        raise RuntimeError(f"Root directory {root} does not exist")

    for sub in root_path.iterdir():
        if not sub.is_dir():
            continue
        chips_files = list(sub.glob("chips_*.parquet"))
        if chips_files:
            countries.append(sub.name)
    return sorted(countries)

def build_paths(country_root: Path, chip_id: str, mask_type: str):
    window_b_path = country_root / "s2_images" / "window_b" / f"{chip_id}.tif"
    window_a_path = country_root / "s2_images" / "window_a" / f"{chip_id}.tif"
    mask_path = country_root / "label_masks" / mask_type / f"{chip_id}.tif"
    return window_a_path, window_b_path, mask_path

def check_paths(window_a, window_b, mask):
    return window_a.exists() and window_b.exists() and mask.exists()

def export_country_to_zarr(root: str, country: str, mask_type="semantic_3class",
                           output_dir="data/ftw/zarr", overwrite=False, verbose=True):
    country_root = Path(root) / country

    chips_files = list(country_root.glob("chips_*.parquet"))
    if not chips_files:
        raise RuntimeError(f"No chips_*.parquet file found for country {country}")
    chips_fn = chips_files[0]

    if verbose:
        print(f"\nProcessing country: {country}")
        print(f"  Using chips file: {chips_fn}")
        print(f"  Mask type: {mask_type}")

    df = gpd.read_parquet(chips_fn)
    if "aoi_id" not in df.columns:
        raise RuntimeError(f"chips file for {country} does not contain 'aoi_id' column")

    chip_ids = df["aoi_id"].astype(str).tolist()
    if verbose:
        print(f"  Number of chips: {len(chip_ids)}")

    out_path = Path(output_dir) / f"{country}.zarr"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and overwrite:
        if verbose:
            print(f"  Removing existing Zarr store: {out_path}")
        import shutil
        shutil.rmtree(out_path)
    if verbose:
        print(f"  Writing Zarr store to: {out_path}")

    # Read first chip to infer shapes
    for chip_id in chip_ids:
        window_a_path, window_b_path, mask_path = build_paths(country_root, chip_id, mask_type)
        if check_paths(window_a_path, window_b_path, mask_path):
            with rasterio.open(window_b_path) as f_b:
                window_b_img = f_b.read()
            with rasterio.open(window_a_path) as f_a:
                window_a_img = f_a.read()
            with rasterio.open(mask_path) as f_m:
                mask = f_m.read(1)
            C, H, W = np.concatenate([window_b_img, window_a_img], axis=0).shape
            mask_shape = mask.shape
            break
    else:
        raise RuntimeError(f"No valid chips found for country {country}")

    # Create Zarr store
    store = zarr.open(str(out_path), mode="w")
    img_ds = store.create_dataset(
        "images",
        shape=(len(chip_ids), C, H, W),
        chunks=(1, C, H, W),
        dtype=np.float32,
        compressor=numcodecs.Blosc(cname="zstd", clevel=3),
    )
    mask_ds = store.create_dataset(
        "masks",
        shape=(len(chip_ids), *mask_shape),
        chunks=(1, *mask_shape),
        dtype=np.uint8,
        compressor=numcodecs.Blosc(cname="zstd", clevel=3),
    )

    num_written = 0
    num_missing = 0
    for i, chip_id in enumerate(tqdm(chip_ids)):
        window_a_path, window_b_path, mask_path = build_paths(country_root, chip_id, mask_type)
        if not check_paths(window_a_path, window_b_path, mask_path):
            num_missing += 1
            continue

        with rasterio.open(window_b_path) as f_b:
            window_b_img = f_b.read()
        with rasterio.open(window_a_path) as f_a:
            window_a_img = f_a.read()
        with rasterio.open(mask_path) as f_m:
            mask = f_m.read(1)

        image = np.concatenate([window_b_img, window_a_img], axis=0).astype(np.float32)

        img_ds[i] = image
        mask_ds[i] = mask
        num_written += 1

    if verbose:
        print(f"✅ Done: {country}")
        print(f"  Written: {num_written}")
        print(f"  Missing: {num_missing}")

def main():
    parser = argparse.ArgumentParser(description="Export FTW dataset to Zarr stores")
    parser.add_argument("--root", type=str, required=True, help="Root FTW directory")
    parser.add_argument("--countries", nargs="+", default=None, help="List of countries")
    parser.add_argument("--mask-type", type=str, default="semantic_3class")
    parser.add_argument("--output-dir", type=str, default="data/ftw/zarr")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    if args.countries is None:
        countries = find_countries(args.root)
        if verbose:
            print("Discovered countries:", countries)
    else:
        countries = args.countries

    for country in countries:
        export_country_to_zarr(
            root=args.root,
            country=country,
            mask_type=args.mask_type,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            verbose=verbose,
        )

if __name__ == "__main__":
    main()
