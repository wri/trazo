#!/usr/bin/env python
"""
Export FTW dataset to Zarr v3 stores (fully compatible).
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
    root_path = Path(root)
    countries = []
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
    chips_files = list(country_root.glob("chips_*.parquet"))
    if not chips_files:
        raise RuntimeError(f"No chips_*.parquet found for {country}")
    chips_fn = chips_files[0]
    df = gpd.read_parquet(chips_fn)
    if "aoi_id" not in df.columns:
        raise RuntimeError(f"'aoi_id' column missing in {chips_fn}")

    chip_ids = df["aoi_id"].astype(str).tolist()
    if verbose:
        print(f"Processing {country}: {len(chip_ids)} chips")

    # Output path
    out_path = Path(output_dir) / f"{country}.zarr"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and overwrite:
        shutil.rmtree(out_path)
    if verbose:
        print(f"Writing Zarr store to {out_path}")

    # Find first valid chip to infer shape
    for chip_id in chip_ids:
        wa_path, wb_path, mask_path = build_paths(country_root, chip_id, mask_type)
        if check_paths(wa_path, wb_path, mask_path):
            with rasterio.open(wa_path) as f_a:
                wa = f_a.read()
            with rasterio.open(wb_path) as f_b:
                wb = f_b.read()
            with rasterio.open(mask_path) as f_m:
                mask = f_m.read(1)
            C, H, W = np.concatenate([wb, wa], axis=0).shape
            break
    else:
        raise RuntimeError(f"No valid chips found for {country}")

    # Create Zarr store (v3)
    store = zarr.group(store=str(out_path), overwrite=True)
    compressor = numcodecs.Blosc(cname="zstd", clevel=3)
    # images array
    store.create_array(
        name="images",
        shape=(len(chip_ids), C, H, W),
        chunks=(1, C, H, W),
        dtype=np.float32,
        compressors=[compressor],
    )
    # masks array
    store.create_array(
        name="masks",
        shape=(len(chip_ids), *mask.shape),
        chunks=(1, *mask.shape),
        dtype=np.uint8,
        compressors=[compressor],
    )

    num_written = 0
    num_missing = 0
    for i, chip_id in enumerate(tqdm(chip_ids)):
        wa_path, wb_path, mask_path = build_paths(country_root, chip_id, mask_type)
        if not check_paths(wa_path, wb_path, mask_path):
            num_missing += 1
            continue
        with rasterio.open(wa_path) as f_a:
            wa = f_a.read()
        with rasterio.open(wb_path) as f_b:
            wb = f_b.read()
        with rasterio.open(mask_path) as f_m:
            mask = f_m.read(1)

        store["images"][i] = np.concatenate([wb, wa], axis=0).astype(np.float32)
        store["masks"][i] = mask
        num_written += 1

    if verbose:
        print(f"✅ {country}: written {num_written}, missing {num_missing}")


def main():
    parser = argparse.ArgumentParser(description="Export FTW to Zarr v3")
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
