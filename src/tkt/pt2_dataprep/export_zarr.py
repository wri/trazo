#!/usr/bin/env python
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import zarr
from numcodecs import Blosc

def find_countries(root: str) -> list[str]:
    """Find country folders containing chips_*.parquet."""
    countries: list[str] = []
    root_path = Path(root)
    if not root_path.exists():
        raise RuntimeError(f"Root directory {root} does not exist")
    for sub in root_path.iterdir():
        if not sub.is_dir():
            continue
        if list(sub.glob("chips_*.parquet")):
            countries.append(sub.name)
    return sorted(countries)

def build_paths(country_root: Path, chip_id: str, mask_type: str):
    window_b_path = country_root / "s2_images" / "window_b" / f"{chip_id}.tif"
    window_a_path = country_root / "s2_images" / "window_a" / f"{chip_id}.tif"
    mask_path = country_root / "label_masks" / mask_type / f"{chip_id}.tif"
    return window_a_path, window_b_path, mask_path

def check_paths(window_a: Path, window_b: Path, mask: Path) -> bool:
    return window_a.exists() and window_b.exists() and mask.exists()

def export_to_zarr(root: str, countries: list[str], mask_type: str, output_path: str, overwrite: bool = False, verbose: bool = True):
    """Export all countries into a single Zarr store."""
    output_path = Path(output_path)
    if output_path.exists() and overwrite:
        import shutil
        shutil.rmtree(output_path)

    store = zarr.open(output_path, mode="w")

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    for country in countries:
        country_root = Path(root) / country
        chips_files = list(country_root.glob("chips_*.parquet"))
        if not chips_files:
            if verbose:
                print(f"Skipping {country}, no chips_*.parquet found")
            continue
        chips_fn = chips_files[0]
        df = gpd.read_parquet(chips_fn)
        if "aoi_id" not in df.columns:
            raise RuntimeError(f"chips file for {country} does not contain 'aoi_id' column")
        chip_ids = df["aoi_id"].astype(str).tolist()
        if verbose:
            print(f"Processing {country}: {len(chip_ids)} chips")

        for chip_id in chip_ids:
            window_a_path, window_b_path, mask_path = build_paths(country_root, chip_id, mask_type)
            if not check_paths(window_a_path, window_b_path, mask_path):
                if verbose:
                    print(f"  Skipping {chip_id}: missing one of {window_a_path}, {window_b_path}, {mask_path}")
                continue

            # Read images
            with rasterio.open(window_b_path) as f_b:
                window_b_img = f_b.read()
            with rasterio.open(window_a_path) as f_a:
                window_a_img = f_a.read()
            image = np.concatenate([window_b_img, window_a_img], axis=0).astype(np.float32)

            with rasterio.open(mask_path) as f_m:
                mask = f_m.read(1).astype(np.uint8)

            # Create arrays in Zarr store
            group = store.require_group(country)
            group.array(
                name=f"{chip_id}_image",
                data=image,
                chunks=image.shape,
                dtype=image.dtype,
                compressor=compressor,
                overwrite=overwrite
            )
            group.array(
                name=f"{chip_id}_mask",
                data=mask,
                chunks=mask.shape,
                dtype=mask.dtype,
                compressor=compressor,
                overwrite=overwrite
            )
            if verbose and int(chip_id) % 50 == 0:
                print(f"  Processed {chip_id}")

    if verbose:
        print(f"Done! Zarr store written to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root dataset dir")
    parser.add_argument("--mask-type", type=str, default="semantic_3class")
    parser.add_argument("--output-path", type=str, required=True, help="Output Zarr path, e.g. train.zarr")
    parser.add_argument("--countries", type=str, nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.countries is None:
        countries = find_countries(args.root)
    else:
        countries = args.countries

    if not countries:
        raise RuntimeError("No countries found")

    if verbose:
        print("Countries:", countries)

    export_to_zarr(
        root=args.root,
        countries=countries,
        mask_type=args.mask_type,
        output_path=args.output_path,
        overwrite=args.overwrite,
        verbose=verbose
    )

if __name__ == "__main__":
    main()
