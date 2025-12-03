#!/usr/bin/env python
import argparse
from pathlib import Path
import random
import shutil

import geopandas as gpd
import rasterio
import numpy as np
import zarr
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
        if len(chips_files) >= 1:
            countries.append(sub.name)
    return sorted(countries)

def build_paths(country_root: Path, chip_id: str, mask_type: str) -> tuple[Path, Path, Path]:
    """Return paths for window_a, window_b, and mask files."""
    window_b_path = country_root / "s2_images" / "window_b" / f"{chip_id}.tif"
    window_a_path = country_root / "s2_images" / "window_a" / f"{chip_id}.tif"
    mask_path = country_root / "label_masks" / mask_type / f"{chip_id}.tif"
    return window_a_path, window_b_path, mask_path

def check_paths(window_a: Path, window_b: Path, mask: Path) -> bool:
    """Check that all required raster files exist."""
    return window_a.exists() and window_b.exists() and mask.exists()

def export_country_to_zarr(
    root: str,
    country: str,
    mask_type: str,
    output_dir: str = "data/ftw/zarr",
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Export a single country folder to a single Zarr store."""
    country_root = Path(root) / country

    # Find chips_*.parquet
    chips_files = list(country_root.glob("chips_*.parquet"))
    if len(chips_files) == 0:
        raise RuntimeError(f"No chips_*.parquet file found for country {country}")
    if len(chips_files) > 1 and verbose:
        print(f"Multiple chips_*.parquet found for {country}, using: {chips_files[0]}")
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
    if out_path.exists() and overwrite:
        shutil.rmtree(out_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  Writing Zarr store to: {out_path}")

    # Preload first sample to get shapes
    for first_chip in chip_ids:
        win_a, win_b, mask_p = build_paths(country_root, first_chip, mask_type)
        if check_paths(win_a, win_b, mask_p):
            with rasterio.open(win_b) as f_b:
                win_b_img = f_b.read()
            with rasterio.open(win_a) as f_a:
                win_a_img = f_a.read()
            C, H, W = np.concatenate([win_b_img, win_a_img], axis=0).shape
            with rasterio.open(mask_p) as f_m:
                mask_shape = f_m.read(1).shape
            break
    else:
        raise RuntimeError(f"No valid chips found for {country}")

    store = zarr.open(str(out_path), mode="w")
    img_ds = store.create_dataset(
        "images",
        shape=(len(chip_ids), C, H, W),
        chunks=(1, C, H, W),
        dtype=np.float32,
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )
    mask_ds = store.create_dataset(
        "masks",
        shape=(len(chip_ids), *mask_shape),
        chunks=(1, *mask_shape),
        dtype=np.uint8,
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )

    num_written = 0
    num_missing = 0

    for i, chip_id in enumerate(tqdm(chip_ids)):
        win_a, win_b, mask_p = build_paths(country_root, chip_id, mask_type)
        if not check_paths(win_a, win_b, mask_p):
            num_missing += 1
            continue

        with rasterio.open(win_b) as f_b:
            win_b_img = f_b.read()
        with rasterio.open(win_a) as f_a:
            win_a_img = f_a.read()
        img_ds[i] = np.concatenate([win_b_img, win_a_img], axis=0).astype(np.float32)

        with rasterio.open(mask_p) as f_m:
            mask_ds[i] = f_m.read(1)

        num_written += 1

    if verbose:
        print(f"  Done country: {country}")
        print(f"    Written: {num_written}")
        print(f"    Missing: {num_missing}")

def main():
    parser = argparse.ArgumentParser(
        description="Export FTW dataset to Zarr (drop-in replacement for HKL export)."
    )
    parser.add_argument("--root", type=str, required=True, help="Root FTW directory")
    parser.add_argument("--countries", nargs="+", type=str, default=None, help="List of country folders")
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

    if len(countries) == 0:
        raise RuntimeError("No countries found to process")

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

