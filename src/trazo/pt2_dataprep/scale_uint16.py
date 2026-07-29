#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scale Sentinel-2 chip TIFFs so outputs are uint16 in [0, 10000].

Rules:
  1) If file is already uint16 and all values in [0, TARGET_MAX] -> SKIP
  2) If values in [0, TARGET_MAX] but dtype is not uint16 -> CAST to uint16
  3) If values look like reflectance decimals in [0, 1] -> SCALE by SCALE_FACTOR then cast to uint16
  4) Otherwise -> CLIP to [TARGET_MIN, TARGET_MAX] then cast to uint16

By default, scans these subfolders inside each base folder:
  - window_a, window_b
  - s2_images/window_a, s2_images/window_b

You can change subdirs with repeated --subdir flags.

Examples
  python -m trazo.pt2_dataprep.scale_uint16 \
    --base-folder /data/regionA --base-folder /data/regionB \
    --write-mode suffix --out-suffix _u16

  python -m trazo.pt2_dataprep.scale_uint16 \
    --base-folder /data/regionA \
    --write-mode inplace --overwrite
"""

import argparse
import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio


DEFAULT_SUBDIRS = [
    "window_a",
    "window_b",
    "s2_images/window_a",
    "s2_images/window_b",
]


def list_tifs(folder: Path) -> List[Path]:
    return sorted([Path(p) for p in glob.glob(str(folder / "*.tif"))]
                  + [Path(p) for p in glob.glob(str(folder / "*.tiff"))])


def finite_min_max(arr: np.ndarray) -> Tuple[float, float]:
    finite = np.isfinite(arr)
    if not finite.any():
        return 0.0, 0.0
    return float(arr[finite].min()), float(arr[finite].max())


def decide_action(dtype: np.dtype, vmin: float, vmax: float,
                  target_min: int, target_max: int) -> str:
    is_int = np.issubdtype(dtype, np.integer)
    is_float = np.issubdtype(dtype, np.floating)

    if vmin >= target_min and vmax <= target_max:
        return "skip" if is_int and dtype == np.uint16 else "cast"

    if is_float and vmax <= 1.0000001 and vmin >= -1e-6:
        return "scale"

    return "clip"


def to_uint16_cast(arr: np.ndarray, target_min: int, target_max: int) -> np.ndarray:
    out = arr.astype(np.float32, copy=False)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=target_max, neginf=target_min)
    np.clip(out, target_min, target_max, out=out)
    return out.astype(np.uint16, copy=False)


def to_uint16_scale(arr: np.ndarray, factor: float, target_min: int, target_max: int) -> np.ndarray:
    out = arr.astype(np.float32, copy=False)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=target_max, neginf=target_min)
    out *= float(factor)
    np.clip(out, target_min, target_max, out=out)
    return out.astype(np.uint16, copy=False)


def to_uint16_clip(arr: np.ndarray, target_min: int, target_max: int) -> np.ndarray:
    out = arr.astype(np.float32, copy=False)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=target_max, neginf=target_min)
    np.clip(out, target_min, target_max, out=out)
    return out.astype(np.uint16, copy=False)


def out_path_for(src_path: Path, write_mode: str, out_suffix: str) -> Path:
    if write_mode == "inplace":
        return src_path
    return src_path.with_name(f"{src_path.stem}{out_suffix}{src_path.suffix}")


def process_one(src_path: Path,
                write_mode: str,
                out_suffix: str,
                overwrite: bool,
                target_min: int,
                target_max: int,
                scale_factor: float) -> Tuple[bool, str]:
    try:
        with rasterio.open(src_path, "r") as src:
            data = src.read()
            vmin, vmax = finite_min_max(data)
            action = decide_action(np.dtype(src.dtypes[0]), vmin, vmax, target_min, target_max)
            dst_path = out_path_for(src_path, write_mode, out_suffix)

            if action == "skip" and write_mode != "inplace":
                # Skip generating a duplicate file if already ideal
                return False, f"[SKIP] {src_path.name} already uint16 in [{target_min},{target_max}]"

            if dst_path.exists() and not overwrite:
                return False, f"[SKIP] Exists (use --overwrite): {dst_path.name}"

            profile = src.profile.copy()
            if "nodata" in profile:
                nd = profile["nodata"]
                if isinstance(nd, float) and (np.isnan(nd) or np.isinf(nd)):
                    profile.pop("nodata", None)

            profile.update({
                "dtype": "uint16",
                "compress": profile.get("compress", "deflate"),
                "predictor": profile.get("predictor", 2),
                "bigtiff": profile.get("bigtiff", "IF_SAFER"),
                "tiled": profile.get("tiled", True),
            })

            if action == "skip":
                # Inplace case
                if write_mode == "inplace":
                    return False, f"[SKIP] Already ideal: {src_path.name}"
                # Else we would have returned earlier
            elif action == "cast":
                out = to_uint16_cast(data, target_min, target_max)
                with rasterio.open(dst_path, "w", **profile) as dst:
                    dst.write(out)
                return True, f"[CAST] {src_path.name} -> {dst_path.name} (min={vmin:.4f}, max={vmax:.4f})"
            elif action == "scale":
                out = to_uint16_scale(data, scale_factor, target_min, target_max)
                with rasterio.open(dst_path, "w", **profile) as dst:
                    dst.write(out)
                return True, f"[SCALE] {src_path.name} -> {dst_path.name} * {scale_factor:g}"
            else:
                out = to_uint16_clip(data, target_min, target_max)
                with rasterio.open(dst_path, "w", **profile) as dst:
                    dst.write(out)
                return True, f"[CLIP] {src_path.name} -> {dst_path.name} to [{target_min},{target_max}]"
    except Exception as e:
        return False, f"[ERROR] {src_path.name} -> {e}"


def scan_subdirs(base: Path, subdirs: List[str]) -> List[Path]:
    targets: List[Path] = []
    for s in subdirs:
        cand = base / s
        if cand.exists() and cand.is_dir():
            targets.append(cand)
    return targets


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scale or cast Sentinel-2 chips to uint16 in [0, 10000]."
    )
    p.add_argument(
        "--base-folder",
        action="append",
        required=True,
        help="Base folder to scan. Repeat for multiple bases."
    )
    p.add_argument(
        "--subdir",
        action="append",
        default=None,
        help="Relative subdir to scan inside base. Repeat to set a custom list. Defaults to standard four."
    )
    p.add_argument(
        "--write-mode",
        choices=["suffix", "inplace"],
        default="suffix",
        help="suffix writes new files next to source with --out-suffix. inplace overwrites."
    )
    p.add_argument(
        "--out-suffix",
        default="_u16",
        help="Suffix to append when write-mode is suffix."
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting when outputs already exist."
    )
    p.add_argument(
        "--target-min",
        type=int,
        default=0,
        help="Lower bound for target range. Default 0."
    )
    p.add_argument(
        "--target-max",
        type=int,
        default=10_000,
        help="Upper bound for target range. Default 10000."
    )
    p.add_argument(
        "--scale-factor",
        type=float,
        default=10_000.0,
        help="Scale factor applied for reflectance decimals. Default 10000.0."
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    bases = [Path(b).expanduser().resolve() for b in args.base_folder]
    subdirs = args.subdir if args.subdir else DEFAULT_SUBDIRS

    print(f"write_mode={args.write_mode} | overwrite={args.overwrite} | target=[{args.target_min},{args.target_max}] uint16 | scale_factor={args.scale_factor:g}")
    print(f"subdirs={subdirs}")

    total = 0
    converted = 0
    skipped = 0
    errors = 0

    for base in bases:
        if not base.exists():
            print(f"[WARN] Missing base: {base}")
            continue

        targets = scan_subdirs(base, subdirs)
        if not targets:
            print(f"[INFO] No target subfolders in: {base}")
            continue

        print(f"\n=== Base: {base} ===")
        for tf in targets:
            tif_list = list_tifs(tf)
            if not tif_list:
                print(f"  [INFO] No TIFFs in: {tf}")
                continue

            print(f"  -> {tf} | {len(tif_list)} files")
            for tif_path in tif_list:
                total += 1
                ok, msg = process_one(
                    tif_path,
                    write_mode=args.write_mode,
                    out_suffix=args.out_suffix,
                    overwrite=args.overwrite,
                    target_min=args.target_min,
                    target_max=args.target_max,
                    scale_factor=args.scale_factor,
                )
                print(f"     {msg}")
                if ok:
                    converted += 1
                else:
                    if msg.startswith("[SKIP]"):
                        skipped += 1
                    elif msg.startswith("[ERROR]"):
                        errors += 1

    print("\nDONE.")
    print(f"  Files found:   {total}")
    print(f"  Converted:     {converted}")
    print(f"  Skipped:       {skipped}")
    print(f"  Errors:        {errors}")


if __name__ == "__main__":
    main()
