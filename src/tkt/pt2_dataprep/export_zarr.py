#!/usr/bin/env python
"""
Export FTW dataset to Zarr stores (Zarr v3 compatible)
"""

import argparse
from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import rasterio
import zarr
import numcodecs
from tqdm import tqdm


def find_countries(root: str) -> list[str]:
    countries: list[str] = []
    root_path = Path(root)
    for sub in root_path.iterdir():
        if not sub.is_dir():
            continue
        if any(sub.glob("chips_*.parquet")):
            countries.append(sub.name)
    return sorted(countries)


def build_paths(country_root: Path, chip_id: str, mask_type: str):
    return (
        country_root / "s2_images" / "window_a" / f"{chip_id}.tif",
        country_root / "s2_images" / "window_b" / f"{chip_id}.tif",
        country_root / "label_masks" / mask_type / f"{chip_id}.tif",
    )


def check_paths(window_a, window_b, mask):
    return window_a.exists() and window_b.exists() and mask.exists()


def export_country_to_zarr(
    root: str,
    country: str,
    mask_type="semantic_3class",
    output_dir="data/ftw/zarr",
    overwrite=False,
    verbose=True,
):
    country_root = Path(root) / country

    # Load chips parquet
    chips_files = list(country_root.glob("chips_*.parquet"))
    if not chips_files:
        raise RuntimeError(f"No chips_*.parquet file found for {country}")
    chips_fn = chips_files[0]
    df = gpd.read_parquet(chips_fn)
    if "aoi_id" not in df.columns:
        raise RuntimeError(f"'aoi_id' column missing in {chips_fn}")

    chip_ids = df["aoi_id"].astype(str).tolist()
    if verbose:
        print(f"Processing {country}: {len(chip_ids)} chips")

    # Output Zarr
    out_path = Path(output_dir) / f"{country}.zarr"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and overwrite:
        shutil.rmtree(out_path)
    if verbose:
        print(f"Writing Zarr store to {out_path}")

    # Read first valid chip to infer shapes
    for chip_id in chip_ids:
        window_a_path, window_b_path, mask_path = build_paths(country_root, chip_id, mask_type)
        if check_paths(window_a_path, window_b_path, mask_path):
            with rasterio.open(window_a_path) as f_a:
                wa = f_a.read()
            with rasterio.open(window_b_path) as f_b:
                wb = f_b.read()
            with rasterio.open(mask_path) as f_m:
                mask = f_m.read(1)
            C, H, W = np.concatenate([wb, wa], axis=0).shape
            break
    else:
        raise RuntimeError(f"No valid chips found for {country}")

    # Zarr store with modern API
    store = zarr.group(store=str(out_path), overwrite=True)
    compressor = numcodecs.Blosc(cname="zstd", clevel=3)
    img_ds = store.create_dataset(
        name="images",
        shape=(len(chip_ids), C, H, W),
        chunks=(1, C, H, W),
        dtype=np.float32,
        compressor=compressor,
    )
    mask_ds = store.create_dataset(
        name="masks",
        shape=(len(chip_ids), *mask.shape),
        chunks=(1, *mask.shape),
        dtype=np.uint8,
        compressor=compressor,
    )

    num_written = 0
    num_missing = 0
    for i, chip_id in enumerate(tqdm(chip_ids)):
        window_a_path, window_b_path, mask_path = build_paths(country_root, chip_id, mask_type)
        if not check_paths(window_a_path, window_b_path, mask_path):
            num_missing += 1
            continue

        with rasterio.open(window_a_path) as f_a:
            wa = f_a.read()
        with rasterio.open(window_b_path) as f_b:
            wb = f_b.read()
        with rasterio.open(mask_path) as f_m:
            mask = f_m.read(1)

        img_ds[i] = np.concatenate([wb, wa], axis=0).astype(np.float32)
        mask_ds[i] = mask
        num_written += 1

    if verbose:
        print(f"✅ {country}: written {num_written}, missing {num_missing}")


def main():
    parser = argparse.ArgumentParser(description="Export FTW to Zarr (v3)")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--mask-type", type=str, default="semantic_3class")
    parser.add_argument("--output-dir", type=str, default="data/ftw/zarr")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    countries = args.countries or find_countries(args.root)
    if verbose:
        print("Countries:", countries)
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
