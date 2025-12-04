#!/usr/bin/env python
import argparse
from pathlib import Path

import geopandas as gpd
import zarr
import numpy as np
import rasterio


def find_countries(root: str) -> list[str]:
    countries = []
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


def build_paths(country_root: Path, chip_id: str, mask_type: str):
    window_b_path = country_root / "s2_images" / "window_b" / f"{chip_id}.tif"
    window_a_path = country_root / "s2_images" / "window_a" / f"{chip_id}.tif"
    mask_path = country_root / "label_masks" / mask_type / f"{chip_id}.tif"
    return window_a_path, window_b_path, mask_path


def check_paths(window_a, window_b, mask) -> bool:
    return window_a.exists() and window_b.exists() and mask.exists()


def create_zarr_for_country(root, country, mask_type, overwrite=False, verbose=True):

    country_root = Path(root) / country
    chips_files = list(country_root.glob("chips_*.parquet"))

    if len(chips_files) == 0:
        raise RuntimeError(f"No chips_*.parquet file found for {country}")

    chips_fn = chips_files[0]

    if verbose:
        print(f"\nProcessing country: {country}")
        print(f"  Using chips file: {chips_fn}")
        print(f"  Mask type: {mask_type}")

    df = gpd.read_parquet(chips_fn)
    if "aoi_id" not in df.columns:
        raise RuntimeError(f"chips file for {country} does not contain 'aoi_id'")

    chip_ids = df["aoi_id"].astype(str).tolist()

    if verbose:
        print(f"  Number of chips: {len(chip_ids)}")

    zarr_dir = country_root / f"{country}.zarr"

    if zarr_dir.exists() and not overwrite:
        if verbose:
            print(f"  Zarr already exists for {country}, skipping...")
        return

    if verbose:
        print(f"  Writing Zarr to: {zarr_dir}")

    # Collect arrays in memory
    all_images = []
    all_masks = []
    all_meta = []

    num_missing = 0

    for chip_id in chip_ids:
        window_a_path, window_b_path, mask_path = build_paths(country_root, chip_id, mask_type)

        if not check_paths(window_a_path, window_b_path, mask_path):
            num_missing += 1
            if verbose:
                print(f"    Missing files for chip {chip_id}, skipping.")
            continue

        with rasterio.open(window_b_path) as f_b:
            window_b_img = f_b.read()

        with rasterio.open(window_a_path) as f_a:
            window_a_img = f_a.read()

        image = np.concatenate([window_b_img, window_a_img], axis=0).astype(np.float32)

        with rasterio.open(mask_path) as f_m:
            mask = f_m.read(1)

        all_images.append(image)
        all_masks.append(mask)
        all_meta.append({
            "country": country,
            "chip_id": chip_id,
            "mask_type": mask_type,
            "chips_file": str(chips_fn),
            "window_b_path": str(window_b_path),
            "window_a_path": str(window_a_path),
            "mask_path": str(mask_path),
        })

    if len(all_images) == 0:
        if verbose:
            print(f"  No valid chips found for {country}")
        return

    images_array = np.stack(all_images)
    masks_array = np.stack(all_masks)

    # create zarr group
    z = zarr.open(zarr_dir, mode="w")

    C, H, W = images_array.shape[1:]
    z.create_array(
        name = "images",
        data=images_array,
        # chunks=(32, C, H, W),
        # dtype="float32",
    )

    z.create_array(
        name = "masks",
        data=masks_array,
        # chunks=(32, H, W),
        # dtype="uint8",
    )
    # meta_json = np.array([str(m) for m in all_meta], dtype='U')  # convert object -> str
    # z.create_array(name="meta", data=meta_json)
    meta_json = np.array([json.dumps(m) for m in all_meta], dtype='U')
    z.create_array(name="meta", data=meta_json)
    # convert metadata to variable-length strings
    # convert metadata to variable-length UTF-8 strings
    # meta_json = np.array([str(m) for m in all_meta], dtype=object)
    
    # # create Zarr array using variable-length UTF-8 dtype
    # z.create_array(
    #     name="meta",
    #     data=meta_json,
    #     # dtype=zarr.v3.VLenUTF8(),  # <- this is the v3 dtype
    # )


    if verbose:
        print(f"  Done {country}.")
        print(f"    Missing files: {num_missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--mask-type", type=str, default="semantic_3class")
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

    for c in countries:
        create_zarr_for_country(
            root=args.root,
            country=c,
            mask_type=args.mask_type,
            overwrite=args.overwrite,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
















