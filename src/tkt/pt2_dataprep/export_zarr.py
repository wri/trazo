#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import zarr
from numcodecs import Blosc


def find_countries(root: str) -> list[str]:
    """Auto-detect country folders containing chips_*.parquet."""
    root_path = Path(root)
    countries = []
    for p in root_path.iterdir():
        if p.is_dir() and len(list(p.glob("chips_*.parquet"))) == 1:
            countries.append(p.name)
    return sorted(countries)


def load_image(path: Path) -> np.ndarray:
    with rasterio.open(path) as f:
        return f.read().astype(np.float32)


def load_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as f:
        return f.read(1).astype(np.int16)


def write_split_zarr(split_name: str, samples: list[dict], out_root: Path):
    """Write one split into a Zarr store."""
    zroot = out_root / f"{split_name}.zarr"
    if zroot.exists():
        print(f"[INFO] Overwriting {zroot}")
        for child in zroot.iterdir():
            if child.is_file():
                child.unlink()
            else:
                import shutil
                shutil.rmtree(child)

    compressor = Blosc(cname="zstd", clevel=3)
    z = zarr.open(str(zroot), mode="w")

    n = len(samples)
    c, h, w = samples[0]["image"].shape

    z.create_dataset(
        "images",
        shape=(n, c, h, w),
        chunks=(1, c, h, w),
        dtype="float32",
        compressor=compressor,
    )
    z.create_dataset(
        "masks",
        shape=(n, h, w),
        chunks=(1, h, w),
        dtype="int16",
        compressor=compressor,
    )

    for i, s in enumerate(samples):
        z["images"][i] = s["image"]
        z["masks"][i] = s["mask"]

    print(f"[INFO] Wrote {n} samples to {zroot}")


def process_country(country_root: Path, mask_type: str) -> list[dict]:
    """Load all chips for a single country."""
    chips_file = list(country_root.glob("chips_*.parquet"))[0]
    df = gpd.read_parquet(chips_file)

    chip_ids = df["aoi_id"].astype(str).tolist()
    samples = []

    for chip in chip_ids:
        a = country_root / "s2_images" / "window_a" / f"{chip}.tif"
        b = country_root / "s2_images" / "window_b" / f"{chip}.tif"
        m = country_root / "label_masks" / mask_type / f"{chip}.tif"

        if not (a.exists() and b.exists() and m.exists()):
            continue

        img_a = load_image(a)
        img_b = load_image(b)
        image = np.concatenate([img_b, img_a], axis=0)
        mask = load_mask(m)

        samples.append({"image": image, "mask": mask})

    return samples


def main():
    p = argparse.ArgumentParser(description="Export FTW dataset to unified Zarr format.")
    p.add_argument("--root", required=True, type=str)
    p.add_argument("--countries", nargs="+", default=None)
    p.add_argument("--mask-type", default="semantic_3class")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    root = Path(args.root)
    out_root = root

    if args.countries is None:
        countries = find_countries(root)
    else:
        countries = args.countries

    print("[INFO] Countries:", countries)

    all_samples = []
    for c in countries:
        print(f"[INFO] Processing {c}")
        country_root = root / c
        samples = process_country(country_root, args.mask_type)
        print(f"[INFO] {c}: Loaded {len(samples)} samples")
        all_samples.extend(samples)

    # if dataset has no official splits → everything is train
    write_split_zarr("train", all_samples, out_root)


if __name__ == "__main__":
    main()
