#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standardize 8 band TIFF stacks to 256x256 chips.

For each input folder:
  - Scan only files directly inside it (no recursion).
  - For each .tif or .tiff with exactly BANDS_REQUIRED bands:
      * If already CROP_W x CROP_H and has no invalids -> copy as is
        into <folder>/sized256/<filename>.
      * If CROP_W x CROP_H but has invalids -> inpaint and write.
      * If larger than or equal to CROP_W x CROP_H:
            - Find 256x256 window with fewest invalids,
              crop to that window and optionally inpaint.
      * If either dimension is smaller than CROP_W or CROP_H:
            - Pad to CROP_W x CROP_H (top left anchored), then inpaint.

Invalids are defined as:
  - For float data: NaN
  - For integer data: nodata value if present

Outputs are written into "<folder>/sized256" with the same filenames.

Typical usage
-------------

As a module:

    from pathlib import Path
    from trazo.pt2_dataprep.resize_chips_256 import resize_folder_to_256

    resize_folder_to_256(
        folder=Path("/data/stacked"),
        bands_required=8,
        crop_width=256,
        crop_height=256,
        overwrite=True,
        use_compression=True,
        compress="deflate",
        blocksize=256,
    )

From CLI:

    python -m tkt.pt2_dataprep.resize_chips_256 \
        --folder /data/stacked \
        --folder /other/stacked \
        --bands-required 8 \
        --crop-width 256 \
        --crop-height 256 \
        --overwrite \
        --use-compression \
        --compress deflate \
        --blocksize 256
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.fill import fillnodata


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_out_profile_like(
    src: rasterio.io.DatasetReader,
    width: int,
    height: int,
    count: Optional[int] = None,
    use_compression: bool = True,
    compress: str = "deflate",
    blocksize: int = 256,
) -> dict:
    """
    Build an output profile similar to src, but with updated size and compression options.
    """
    prof = src.profile.copy()
    prof.update({"driver": "GTiff", "width": width, "height": height})
    if count is not None:
        prof["count"] = count

    if use_compression:
        prof.update(
            {
                "compress": compress,
                "tiled": True,
                "blockxsize": min(blocksize, width),
                "blockysize": min(blocksize, height),
            }
        )
        try:
            dtype_kind = np.dtype(prof.get("dtype", "uint8")).kind
            if dtype_kind in ("f", "i", "u"):
                prof["predictor"] = 2
        except Exception:
            pass
    else:
        prof.pop("compress", None)
        prof.pop("tiled", None)
        prof.pop("blockxsize", None)
        prof.pop("blockysize", None)
        prof.pop("predictor", None)

    return prof


def window_transform(src_transform: Affine, row_off: int, col_off: int) -> Affine:
    """
    Compute a new transform for a window starting at (row_off, col_off).
    """
    return src_transform * Affine.translation(col_off, row_off)


def make_invalid_mask_first_band(
    band: np.ndarray,
    dtype_is_float: bool,
    nodata_value,
) -> np.ndarray:
    """
    Build a boolean mask of invalid pixels from the first band.
    """
    if dtype_is_float:
        return np.isnan(band)
    invalid = np.zeros_like(band, dtype=bool)
    if nodata_value is not None:
        invalid |= band == nodata_value
    return invalid


def integral_image_bool(mask: np.ndarray) -> np.ndarray:
    """
    Build an integral image (summed area table) for a boolean mask.
    """
    return mask.astype(np.uint32).cumsum(axis=0).cumsum(axis=1)


def window_sum_from_integral(
    S: np.ndarray,
    row: int,
    col: int,
    h: int,
    w: int,
) -> int:
    """
    Compute the sum of values in the window using the integral image S.
    """
    r0, c0 = row, col
    r1, c1 = row + h - 1, col + w - 1
    total = S[r1, c1]
    if r0 > 0:
        total -= S[r0 - 1, c1]
    if c0 > 0:
        total -= S[r1, c0 - 1]
    if r0 > 0 and c0 > 0:
        total += S[r0 - 1, c0 - 1]
    return int(total)


def best_window_min_invalid(
    invalid_mask: np.ndarray,
    crop_h: int,
    crop_w: int,
) -> Tuple[int, int, int]:
    """
    Find the row, col for the crop window with the fewest invalid pixels.

    Returns (row_off, col_off, invalid_count).
    """
    H, W = invalid_mask.shape
    if H < crop_h or W < crop_w:
        return 0, 0, int(invalid_mask.sum())

    S = integral_image_bool(invalid_mask)
    best_row, best_col, best_sum = 0, 0, None

    for r in range(0, H - crop_h + 1):
        for c in range(0, W - crop_w + 1):
            s = window_sum_from_integral(S, r, c, crop_h, crop_w)
            if best_sum is None or s < best_sum:
                best_sum = s
                best_row, best_col = r, c
                if best_sum == 0:
                    return best_row, best_col, 0

    return best_row, best_col, int(best_sum if best_sum is not None else 0)


def inpaint_nodata_per_band(
    data: np.ndarray,
    nodata_value,
    max_search_distance: int = 50,
    smoothing_iterations: int = 0,
) -> np.ndarray:
    """
    Fill NaNs or nodata per band using rasterio.fill.fillnodata.

    Preserves dtype:
      - floats remain float
      - integers are round cast back to integer
    """
    out = data.copy()
    bands = out.shape[0]

    for b in range(bands):
        band = out[b]

        if np.issubdtype(band.dtype, np.floating):
            valid_mask = ~np.isnan(band)
            if valid_mask.all():
                continue
            band_float = band.astype(np.float32, copy=True)
            filled = fillnodata(
                band_float,
                mask=valid_mask,
                max_search_distance=max_search_distance,
                smoothing_iterations=smoothing_iterations,
            )
            out[b] = filled
        else:
            if nodata_value is None:
                continue
            valid_mask = band != nodata_value
            if valid_mask.all():
                continue
            band_float = band.astype(np.float32, copy=True)
            band_float[~valid_mask] = np.nan
            filled = fillnodata(
                band_float,
                mask=valid_mask,
                max_search_distance=max_search_distance,
                smoothing_iterations=smoothing_iterations,
            )
            filled_int = np.rint(np.nan_to_num(filled, nan=nodata_value)).astype(
                band.dtype,
                copy=False,
            )
            out[b] = filled_int

    return out


def process_one_tif(
    src_path: Path,
    out_dir: Path,
    bands_required: int,
    crop_w: int,
    crop_h: int,
    overwrite: bool,
    use_compression: bool,
    compress: str,
    blocksize: int,
    max_search_distance: int,
    smoothing_iterations: int,
) -> str:
    """
    Process a single TIFF and write the standardized 256x256 result into out_dir.

    Returns a short status string describing the action:
      "keep", "fill_only", "crop_only", "crop_then_fill",
      "pad_then_fill", "skip_bands", or "skip_exists".
    """
    out_path = out_dir / src_path.name
    if out_path.exists() and not overwrite:
        return "skip_exists"

    with rasterio.open(src_path) as src:
        bands = src.count
        if bands != bands_required:
            return "skip_bands"

        width, height = src.width, src.height
        dtype_is_float = np.issubdtype(np.dtype(src.dtypes[0]), np.floating)
        nodata_value = src.nodata

        band1 = src.read(1)
        invalid_mask = make_invalid_mask_first_band(
            band1,
            dtype_is_float,
            nodata_value,
        )
        total_invalid = int(invalid_mask.sum())

        if width == crop_w and height == crop_h:
            if total_invalid == 0:
                prof = build_out_profile_like(
                    src,
                    width,
                    height,
                    src.count,
                    use_compression=use_compression,
                    compress=compress,
                    blocksize=blocksize,
                )
                prof["transform"] = src.transform
                prof["crs"] = src.crs
                out_dir.mkdir(parents=True, exist_ok=True)
                with rasterio.open(out_path, "w", **prof) as dst:
                    dst.write(src.read())
                return "keep"
            else:
                data = src.read()
                filled = inpaint_nodata_per_band(
                    data,
                    nodata_value,
                    max_search_distance=max_search_distance,
                    smoothing_iterations=smoothing_iterations,
                )
                prof = build_out_profile_like(
                    src,
                    width,
                    height,
                    src.count,
                    use_compression=use_compression,
                    compress=compress,
                    blocksize=blocksize,
                )
                prof["transform"] = src.transform
                prof["crs"] = src.crs
                out_dir.mkdir(parents=True, exist_ok=True)
                with rasterio.open(out_path, "w", **prof) as dst:
                    dst.write(filled)
                return "fill_only"

        can_crop = width >= crop_w and height >= crop_h

        if can_crop:
            row_off, col_off, best_invalid = best_window_min_invalid(
                invalid_mask,
                crop_h,
                crop_w,
            )
            win = Window(
                col_off=col_off,
                row_off=row_off,
                width=crop_w,
                height=crop_h,
            )
            data = src.read(window=win)
            do_fill = best_invalid > 0

            if do_fill:
                data = inpaint_nodata_per_band(
                    data,
                    nodata_value,
                    max_search_distance=max_search_distance,
                    smoothing_iterations=smoothing_iterations,
                )
                action = "crop_then_fill"
            else:
                action = "crop_only"

            prof = build_out_profile_like(
                src,
                crop_w,
                crop_h,
                src.count,
                use_compression=use_compression,
                compress=compress,
                blocksize=blocksize,
            )
            prof["transform"] = window_transform(src.transform, row_off, col_off)
            prof["crs"] = src.crs
            out_dir.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **prof) as dst:
                dst.write(data)
            return action

        H, W = height, width
        data = src.read()
        bands = data.shape[0]

        if np.issubdtype(data.dtype, np.floating):
            canvas = np.full((bands, crop_h, crop_w), np.nan, dtype=data.dtype)
        else:
            fill_val = nodata_value if nodata_value is not None else 0
            canvas = np.full((bands, crop_h, crop_w), fill_val, dtype=data.dtype)

        canvas[:, 0:H, 0:W] = data
        canvas_filled = inpaint_nodata_per_band(
            canvas,
            nodata_value,
            max_search_distance=max_search_distance,
            smoothing_iterations=smoothing_iterations,
        )

        prof = build_out_profile_like(
            src,
            crop_w,
            crop_h,
            src.count,
            use_compression=use_compression,
            compress=compress,
            blocksize=blocksize,
        )
        prof["transform"] = src.transform
        prof["crs"] = src.crs
        out_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(canvas_filled)
        return "pad_then_fill"


def resize_folder_to_256(
    base-folder: Path,
    bands_required: int = 8,
    crop_width: int = 256,
    crop_height: int = 256,
    overwrite: bool = True,
    use_compression: bool = True,
    compress: str = "deflate",
    blocksize: int = 256,
    max_search_distance: int = 50,
    smoothing_iterations: int = 0,
) -> None:
    """
    Resize and clean all eligible TIFFs directly inside a folder to crop_width x crop_height.

    Outputs are written into "<folder>/sized256".
    """
    if not base-folder.exists() or not base-folder.is_dir():
        print(f"[WARN] Missing or not a directory: {base-folder}")
        return

    out_dir = base-folder / "sized256"
    safe_mkdir(out_dir)

    seen = 0
    processed = 0
    kept = 0
    filled = 0
    cropped = 0
    cropfilled = 0
    padded = 0
    skipped_bands = 0
    skipped_exists = 0

    for entry in sorted(base-folder.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in (".tif", ".tiff"):
            continue

        seen += 1
        try:
            action = process_one_tif(
                src_path=entry,
                out_dir=out_dir,
                bands_required=bands_required,
                crop_w=crop_width,
                crop_h=crop_height,
                overwrite=overwrite,
                use_compression=use_compression,
                compress=compress,
                blocksize=blocksize,
                max_search_distance=max_search_distance,
                smoothing_iterations=smoothing_iterations,
            )
            if action == "skip_bands":
                skipped_bands += 1
            elif action == "skip_exists":
                skipped_exists += 1
            else:
                processed += 1
                if action == "keep":
                    kept += 1
                elif action == "fill_only":
                    filled += 1
                elif action == "crop_only":
                    cropped += 1
                elif action == "crop_then_fill":
                    cropfilled += 1
                elif action == "pad_then_fill":
                    padded += 1
        except Exception as e:
            print(f"[ERROR] {entry} -> {e}")

    print(f"\n=== {base-folder} ===")
    print(
        f"Seen TIFF files: {seen}\n"
        f"Processed: {processed} | Kept: {kept} | Fill only: {filled} | "
        f"Crop: {cropped} | Crop+Fill: {cropfilled} | Pad+Fill: {padded}\n"
        f"Skipped (not {bands_required} bands): {skipped_bands} | "
        f"Skipped (exists, overwrite={overwrite}): {skipped_exists}\n"
        f"Outputs in: {out_dir}"
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize 8 band TIFF stacks to 256x256 chips. "
            "Writes outputs into <base-folder>/sized256."
        )
    )

    parser.add_argument(
        "--base-folder",
        action="append",
        required=True,
        help="Base Folder containing 8 band stacks (can be provided multiple times).",
    )
    parser.add_argument(
        "--bands-required",
        type=int,
        default=8,
        help="Only process TIFFs with this many bands (default: 8).",
    )
    parser.add_argument(
        "--crop-width",
        type=int,
        default=256,
        help="Target chip width in pixels (default: 256).",
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=256,
        help="Target chip height in pixels (default: 256).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs in sized256 (default: False).",
    )
    parser.add_argument(
        "--use-compression",
        action="store_true",
        help="Use compressed tiled GeoTIFF outputs (default: False).",
    )
    parser.add_argument(
        "--compress",
        default="deflate",
        help='Compression method when --use-compression is set (default: "deflate").',
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=256,
        help="Block size for tiled outputs when using compression (default: 256).",
    )
    parser.add_argument(
        "--max-search-distance",
        type=int,
        default=50,
        help="Max search distance for fillnodata (default: 50).",
    )
    parser.add_argument(
        "--smoothing-iterations",
        type=int,
        default=0,
        help="Smoothing iterations for fillnodata (default: 0).",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    folders = [Path(f).expanduser().resolve() for f in args.folder]

    for folder in folders:
        resize_folder_to_256(
            folder=base-folder,
            bands_required=args.bands_required,
            crop_width=args.crop_width,
            crop_height=args.crop_height,
            overwrite=args.overwrite,
            use_compression=args.use_compression,
            compress=args.compress,
            blocksize=args.blocksize,
            max_search_distance=args.max_search_distance,
            smoothing_iterations=args.smoothing_iterations,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

