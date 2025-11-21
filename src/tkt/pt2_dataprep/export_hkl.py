#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import hickle as hkl
import numpy as np
import rasterio


def _chips_parquet_for_base(base: Path) -> Path:
    """
    Resolve a chips parquet in a single *base* folder.
    Accepts:
      - chips_<something>.parquet in base
      - stacks8/chips_<something>.parquet if user points to the parent
    """
    cands = sorted(base.glob("chips_*.parquet"))
    if cands:
        return cands[0]
    # common case: user passed parent of stacks8
    stacks = base / "stacks8"
    cands = sorted(stacks.glob("chips_*.parquet"))
    if cands:
        return cands[0]
    raise FileNotFoundError(f"No chips_*.parquet found in {base} or {stacks}")


def _chips_parquet_for_country(root: Path, country: str) -> Path:
    fn = root / country / f"chips_{country}.parquet"
    if not fn.exists():
        raise FileNotFoundError(f"Missing {fn}")
    return fn


def _list_ftw_countries(root: Path) -> List[str]:
    out = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if list(p.glob("chips_*.parquet")):
            out.append(p.name)
    return sorted(out)


def _resolve_io_paths(base_or_country: Path, chip_id: str, mask_type: str) -> Tuple[Path, Path, Path]:
    """
    Return (window_a, window_b, mask) for a chip in the Step-2 folder shape:
      <base>/s2_images/window_a/<chip>.tif
      <base>/s2_images/window_b/<chip>.tif
      <base>/label_masks/<mask_type>/<chip>.tif
    Fallbacks if s2_images/* missing:
      <base>/window_a/<chip>.tif and <base>/window_b/<chip>.tif
    """
    wa = base_or_country / "s2_images" / "window_a" / f"{chip_id}.tif"
    wb = base_or_country / "s2_images" / "window_b" / f"{chip_id}.tif"
    if not wa.exists() or not wb.exists():
        wa = base_or_country / "window_a" / f"{chip_id}.tif"
        wb = base_or_country / "window_b" / f"{chip_id}.tif"
    mask = base_or_country / "label_masks" / mask_type / f"{chip_id}.tif"
    return wa, wb, mask


def _exists_all(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


def _stack_images(window_a_path: Path, window_b_path: Path, order: str) -> np.ndarray:
    """
    Read A and B as [C,H,W] each and stack to [2C,H,W].
    order = "b_then_a" or "a_then_b"
    """
    with rasterio.open(window_a_path) as fa:
        img_a = fa.read()
    with rasterio.open(window_b_path) as fb:
        img_b = fb.read()
    if order == "b_then_a":
        img = np.concatenate([img_b, img_a], axis=0)
    else:
        img = np.concatenate([img_a, img_b], axis=0)
    return img.astype(np.float32, copy=False)


def _read_mask(mask_path: Path) -> np.ndarray:
    with rasterio.open(mask_path) as fm:
        return fm.read(1)


def _write_hkl(out_path: Path, image: np.ndarray, mask: np.ndarray, meta: dict) -> None:
    sample = {"image": image, "mask": mask, "meta": meta}
    hkl.dump(sample, out_path, mode="w")


def export_from_base(
    base: Path,
    split: str,
    mask_type: str,
    order: str,
    overwrite: bool,
    quiet: bool,
) -> None:
    chips_path = _chips_parquet_for_base(base)
    df = gpd.read_parquet(chips_path)
    if "aoi_id" not in df or "split" not in df:
        raise RuntimeError(f"{chips_path} must contain columns 'aoi_id' and 'split'.")

    df = df[df["split"] == split]
    chip_ids = df["aoi_id"].astype(str).tolist()

    out_dir = base / "hkl"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"[HKL] Base: {base}")
        print(f"[HKL] Chips file: {chips_path}")
        print(f"[HKL] Split: {split} | Mask: {mask_type} | Order: {order}")
        print(f"[HKL] Output dir: {out_dir}")
        print(f"[HKL] Count: {len(chip_ids)}")

    written = skipped = missing = 0
    for i, chip_id in enumerate(chip_ids, 1):
        wa, wb, m = _resolve_io_paths(base, chip_id, mask_type)
        if not _exists_all(wa, wb, m):
            missing += 1
            if not quiet:
                print(f"  [MISS] {chip_id} missing one of: {wa.name}, {wb.name}, {m.name}")
            continue

        out_path = out_dir / f"{chip_id}.hkl"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        image = _stack_images(wa, wb, order)
        mask = _read_mask(m)

        meta = {
            "chip_id": chip_id,
            "split": split,
            "order": order,
            "mask_type": mask_type,
            "window_a": str(wa),
            "window_b": str(wb),
            "mask": str(m),
        }
        _write_hkl(out_path, image, mask, meta)
        written += 1
        if not quiet and written % 100 == 0:
            print(f"  [..] written {written}")

    if not quiet:
        print(f"[HKL] Done. Written={written} | Skipped={skipped} | Missing={missing}")


def export_from_root(
    root: Path,
    countries: List[str],
    split: str,
    mask_type: str,
    order: str,
    overwrite: bool,
    quiet: bool,
) -> None:
    if not countries:
        countries = _list_ftw_countries(root)
        if not quiet:
            print(f"[HKL] Discovered countries: {countries}")

    for country in countries:
        country_dir = root / country
        chips_path = _chips_parquet_for_country(root, country)
        df = gpd.read_parquet(chips_path)
        if "aoi_id" not in df or "split" not in df:
            raise RuntimeError(f"{chips_path} must contain columns 'aoi_id' and 'split'.")

        df = df[df["split"] == split]
        chip_ids = df["aoi_id"].astype(str).tolist()

        out_dir = country_dir / "hkl"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not quiet:
            print(f"\n[HKL] Country: {country}")
            print(f"[HKL] Chips file: {chips_path}")
            print(f"[HKL] Split: {split} | Mask: {mask_type} | Order: {order}")
            print(f"[HKL] Output dir: {out_dir}")
            print(f"[HKL] Count: {len(chip_ids)}")

        written = skipped = missing = 0
        for chip_id in chip_ids:
            wa, wb, m = _resolve_io_paths(country_dir, chip_id, mask_type)
            if not _exists_all(wa, wb, m):
                missing += 1
                if not quiet:
                    print(f"  [MISS] {chip_id} missing one of: {wa.name}, {wb.name}, {m.name}")
                continue

            out_path = out_dir / f"{chip_id}.hkl"
            if out_path.exists() and not overwrite:
                skipped += 1
                continue

            image = _stack_images(wa, wb, order)
            mask = _read_mask(m)

            meta = {
                "country": country,
                "chip_id": chip_id,
                "split": split,
                "order": order,
                "mask_type": mask_type,
                "window_a": str(wa),
                "window_b": str(wb),
                "mask": str(m),
            }
            _write_hkl(out_path, image, mask, meta)
            written += 1

        if not quiet:
            print(f"[HKL] {country} done. Written={written} | Skipped={skipped} | Missing={missing}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tkt-pt2-dataprep export-hkl",
        description="Create .hkl files by stacking window A/B and pairing with a mask."
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base-folder", type=str,
                      help="Single Step-2 base folder (contains s2_images/, label_masks/, chips_*.parquet).")
    mode.add_argument("--root", type=str,
                      help="FTW-style root containing country subfolders with chips_<country>.parquet.")

    p.add_argument("--countries", nargs="+", default=None,
                   help="Country folder names under --root. If omitted, auto-discover.")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--mask-type", default="semantic_3class",
                   help="Mask subfolder under label_masks/. Default: semantic_3class.")
    p.add_argument("--order", default="b_then_a", choices=["b_then_a", "a_then_b"],
                   help="Channel stacking order. Default: b_then_a.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.base_folder:
        export_from_base(
            base=Path(args.base_folder),
            split=args.split,
            mask_type=args.mask_type,
            order=args.order,
            overwrite=args.overwrite,
            quiet=args.quiet,
        )
        return

    # root mode
    export_from_root(
        root=Path(args.root),
        countries=args.countries or [],
        split=args.split,
        mask_type=args.mask_type,
        order=args.order,
        overwrite=args.overwrite,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
