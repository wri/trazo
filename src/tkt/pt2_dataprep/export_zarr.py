#!/usr/bin/env python
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import zarr


def find_countries(root: str) -> list[str]:
    root_path = Path(root)
    return [
        p.name for p in root_path.iterdir()
        if p.is_dir() and len(list(p.glob("chips_*.parquet"))) == 1
    ]


def load_image(path: Path) -> np.ndarray:
    with rasterio.open(path) as f:
        return f.read().astype(np.float32)


def load_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as f:
        return f.read(1).astype(np.int16)


def write_split_zarr(split: str, samples: list[dict], out_root: Path):
    zpath = out_root / f"{split}.zarr"

    if zpath.exists():
        import shutil
        shutil.rmtree(zpath)

    n = len(samples)
    c, h, w = samples[0]["image"].shape

    print(f"[INFO] Creating {zpath} with {n} samples")

    # Open group in v3 mode
    root = zarr.open_group(str(zpath), mode="w", zarr_format=3)

    # Zarr v3 compressors use dict configs
    compressor = {"id": "zstd", "level": 3}

    root.create_array(
        "images",
        shape=(n, c, h, w),
        chunks=(1, c, h, w),
        dtype="float32",
        compressors=[compressor],
    )
    root.create_array(
        "masks",
        shape=(n, h, w),
        chunks=(1, h, w),
        dtype="int16",
        compressors=[compressor],
    )

    for i, s in enumerate(samples):
        root["images"][i] = s["image"]
        root["masks"][i] = s["mask"]

    print(f"[INFO] Finished writing {split}.zarr")


def process_country(country_root: Path, mask_type: str) -> list[dict]:
    chips_file = list(country_root.glob("chips_*.parquet"))[0]
    df = gpd.read_parquet(chips_file)
    samples = []

    for chip_id in df["aoi_id"].astype(str):
        a = country_root / "s2_images" / "window_a" / f"{chip_id}.tif"
        b = country_root / "s2_images" / "window_b" / f"{chip_id}.tif"
        m = country_root / "label_masks" / mask_type / f"{chip_id}.tif"

        if not (a.exists() and b.exists() and m.exists()):
            continue

        img_a = load_image(a)
        img_b = load_image(b)
        img = np.concatenate([img_b, img_a], axis=0)
        mask = load_mask(m)

        samples.append({"image": img, "mask": mask})

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--mask-type", default="semantic_3class")
    args = parser.parse_args()

    root = Path(args.root)

    countries = args.countries or find_countries(str(root))
    print("[INFO] Countries:", countries)

    all_samples = []
    for c in countries:
        print(f"[INFO] Processing {c}")
        country_root = root / c
        samples = process_country(country_root, args.mask_type)
        print(f"[INFO] {c}: Loaded {len(samples)} samples")
        all_samples.extend(samples)

    write_split_zarr("train", all_samples, root)


if __name__ == "__main__":
    main()

