#!/usr/bin/env python
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import zarr
from numcodecs import Blosc
from tqdm import tqdm


def find_countries(root: str) -> list[str]:
    root_path = Path(root)
    if not root_path.exists():
        raise RuntimeError(f"Root directory {root} does not exist")
    countries = []
    for sub in root_path.iterdir():
        if sub.is_dir() and len(list(sub.glob("chips_*.parquet"))) == 1:
            countries.append(sub.name)
    return sorted(countries)


def build_paths(country_root: Path, chip_id: str, mask_type: str):
    window_b = country_root / "s2_images" / "window_b" / f"{chip_id}.tif"
    window_a = country_root / "s2_images" / "window_a" / f"{chip_id}.tif"
    mask = country_root / "label_masks" / mask_type / f"{chip_id}.tif"
    return window_a, window_b, mask


def create_fast_zarr_for_country(
    root: str,
    country: str,
    mask_type: str,
    overwrite: bool = False,
    verbose: bool = True
):
    country_root = Path(root) / country

    chips_files = list(country_root.glob("chips_*.parquet"))
    if len(chips_files) == 0:
        raise RuntimeError(f"No chips_*.parquet in {country}")
    chips_fn = chips_files[0]

    if verbose:
        print(f"\nProcessing {country}")
        print(f"Using chips file: {chips_fn}")

    df = gpd.read_parquet(chips_fn)
    chip_ids = df["aoi_id"].astype(str).tolist()

    # Discover sample shape from first chip
    first_id = chip_ids[0]
    a, b, m = build_paths(country_root, first_id, mask_type)

    with rasterio.open(b) as fb:
        C, H, W = fb.count, fb.height, fb.width

    # Output Zarr store (one per country)
    zarr_path = country_root / f"{country}.zarr"
    if zarr_path.exists() and not overwrite:
        if verbose: print(f"{zarr_path} exists; skipping")
        return

    if verbose:
        print(f"Writing to: {zarr_path}")

    store = zarr.DirectoryStore(str(zarr_path))
    root_z = zarr.group(store=store, overwrite=True)

    # --- CHUNKING STRATEGY ---
    # Chunk along the chip dimension only (fastest for training)
    img_chunks = (1, C * 2, H, W)
    mask_chunks = (1, H, W)

    # No compression = MAX SPEED (lossless)
    no_compress = None

    # Create arrays
    img_arr = root_z.create(
        name="image",
        shape=(len(chip_ids), C * 2, H, W),
        chunks=img_chunks,
        dtype="float32",
        compressor=no_compress,
        overwrite=True,
    )

    mask_arr = root_z.create(
        name="mask",
        shape=(len(chip_ids), H, W),
        chunks=mask_chunks,
        dtype="uint8",
        compressor=no_compress,
        overwrite=True,
    )

    # Save metadata once
    root_z.attrs.update({
        "country": country,
        "mask_type": mask_type,
        "chips_file": str(chips_fn),
        "total_chips": len(chip_ids)
    })

    # --- WRITE LOOP ---
    for idx, chip_id in enumerate(tqdm(chip_ids, disable=not verbose)):
        a_path, b_path, m_path = build_paths(country_root, chip_id, mask_type)

        if not (a_path.exists() and b_path.exists() and m_path.exists()):
            continue

        with rasterio.open(b_path) as fb:
            img_b = fb.read()
        with rasterio.open(a_path) as fa:
            img_a = fa.read()

        image = np.concatenate([img_b, img_a], axis=0).astype(np.float32)

        with rasterio.open(m_path) as fm:
            mask = fm.read(1)

        img_arr[idx] = image
        mask_arr[idx] = mask

    # Consolidate metadata for fast loading
    zarr.consolidate_metadata(store)

    if verbose:
        print(f"Finished {country}: {len(chip_ids)} chips")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--mask-type", type=str, default="semantic_3class")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root
    mask_type = args.mask_type

    if args.countries is None:
        countries = find_countries(root)
    else:
        countries = args.countries

    for c in countries:
        create_fast_zarr_for_country(
            root=root,
            country=c,
            mask_type=mask_type,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
