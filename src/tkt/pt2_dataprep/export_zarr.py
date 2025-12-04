#!/usr/bin/env python
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import zarr

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
        if len(chips_files) == 1:
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

def create_zarr_for_country(
    root: str,
    country: str,
    mask_type: str,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Write Zarr files for every chip in a given country folder."""
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
        print(f"  Number of chips (all rows): {len(chip_ids)}")

    zarr_dir = country_root / "zarr"
    zarr_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  Writing Zarr files to: {zarr_dir}")

    num_written = 0
    num_skipped_existing = 0
    num_missing = 0

    for chip_id in chip_ids:
        window_a_path, window_b_path, mask_path = build_paths(country_root, chip_id, mask_type)

        if not check_paths(window_a_path, window_b_path, mask_path):
            num_missing += 1
            if verbose:
                print(f"    Skipping {chip_id}: missing one of {window_a_path}, {window_b_path}, {mask_path}")
            continue

        out_path = zarr_dir / chip_id

        if out_path.exists() and not overwrite:
            num_skipped_existing += 1
            continue

        # Read raster images as [C, H, W]
        with rasterio.open(window_b_path) as f_b:
            window_b_img = f_b.read()
        with rasterio.open(window_a_path) as f_a:
            window_a_img = f_a.read()

        image = np.concatenate([window_b_img, window_a_img], axis=0).astype(np.float32)

        with rasterio.open(mask_path) as f_m:
            mask = f_m.read(1)

        # Create Zarr group for this chip using modern API
        chip_group = zarr.open_group(out_path, mode="w")

        chip_group.create_array(
            name="image",
            shape=image.shape,
            chunks=(1, image.shape[1], image.shape[2]),  # per channel chunking
            dtype=np.float32,
            data=image
        )

        chip_group.create_array(
            name="mask",
            shape=mask.shape,
            chunks=(256, 256),
            dtype=np.int32,
            data=mask
        )

        # Save metadata
        chip_group.attrs["country"] = country
        chip_group.attrs["chip_id"] = chip_id
        chip_group.attrs["mask_type"] = mask_type
        chip_group.attrs["chips_file"] = str(chips_fn)
        chip_group.attrs["window_b_path"] = str(window_b_path)
        chip_group.attrs["window_a_path"] = str(window_a_path)
        chip_group.attrs["mask_path"] = str(mask_path)

        num_written += 1
        if verbose and num_written % 100 == 0:
            print(f"    Written {num_written} Zarr files so far")

    if verbose:
        print(f"  Done country: {country}")
        print(f"    Written:          {num_written}")
        print(f"    Skipped existing: {num_skipped_existing}")
        print(f"    Missing files:    {num_missing}")


def main():
    parser = argparse.ArgumentParser(
        description="Create Zarr files combining window_a, window_b, and a mask type (no splits)."
    )
    parser.add_argument("--root", type=str, required=True, help="Root directory of FTW dataset")
    parser.add_argument("--countries", type=str, nargs="+", default=None, help="List of countries under root")
    parser.add_argument("--mask-type", type=str, default="semantic_3class", help="Mask type folder")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Zarr files")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    args = parser.parse_args()

    verbose = not args.quiet
    countries = args.countries or find_countries(args.root)
    if verbose:
        print("Discovered countries:", countries)

    for country in countries:
        create_zarr_for_country(
            root=args.root,
            country=country,
            mask_type=args.mask_type,
            overwrite=args.overwrite,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()




