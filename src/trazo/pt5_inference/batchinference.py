#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 5: Batch inference over a folder of tiles using the FTW CLI.

This script assumes:
  - You already have stacked Sentinel-2 chips (e.g. AP_<tile>.tif) in a folder.
  - The 'ftw' CLI is installed and on PATH.
  - You have a FTW model checkpoint (.ckpt) you want to use.

Basic usage:

    python -m tkt.pt5_inference.batch_inference \
        --input-dir /path/to/tiles \
        --model-checkpoint /path/to/model.ckpt

By default this will:
  - Look for all *.tif files in --input-dir.
  - Create an output folder: <input-dir>/inf_output
  - For each tile, run:
      ftw inference run <input-tif> -f -o <output-dir>/<name>-inf.tif --gpu 0 -m <checkpoint>
  - Skip tiles whose output already exists, unless --force is given.
  - Optionally polygonize the outputs if --polygonize is passed.

You can also override:
  - --pattern          (glob pattern for input tiles, default: *.tif)
  - --gpu              (GPU index, default: 0)
  - --output-dir       (where to write outputs; default: <input-dir>/inf_output)
  - --polygonize       (also run `ftw inference polygonize` on each inferred raster)
  - --force            (overwrite existing outputs)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_cmd(cmd: List[str]) -> None:
    """Run a shell command and stream stdout/stderr."""
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 5: Batch FTW inference over a folder of tiles."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing input tiles (e.g. AP_<tile>.tif)."
    )
    parser.add_argument(
        "--model-checkpoint",
        required=True,
        help="Path to FTW model checkpoint (.ckpt) to use for inference."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for inference results. "
             "Defaults to <input-dir>/inf_output if not provided."
    )
    parser.add_argument(
        "--pattern",
        default="*.tif",
        help="Glob pattern for input tiles in --input-dir (default: '*.tif')."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to pass to `ftw inference run` (default: 0)."
    )
    parser.add_argument(
        "--polygonize",
        action="store_true",
        help="If set, also run `ftw inference polygonize` for each inferred raster."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs. If not set, existing outputs are skipped."
    )

    return parser.parse_args(argv)


def main(argv: list | None = None) -> None:
    args = parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    model_ckpt = Path(args.model_checkpoint).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    if not model_ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt}")

    if args.output_dir is None:
        output_dir = input_dir / "inf_output"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    tiles = sorted(input_dir.glob(args.pattern))
    if not tiles:
        print(f"No input tiles found in {input_dir} matching pattern '{args.pattern}'.")
        return

    print(f"[INFO] Input directory:  {input_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Model checkpoint: {model_ckpt}")
    print(f"[INFO] GPU index:        {args.gpu}")
    print(f"[INFO] Tiles to process: {len(tiles)}")
    print(f"[INFO] Polygonize:       {args.polygonize}")
    print(f"[INFO] Force overwrite:  {args.force}")
    print("")

    for tile_path in tiles:
        stem = tile_path.stem  # e.g. "AP_TILEID"
        out_raster = output_dir / f"{stem}-inf.tif"
        out_vector = output_dir / f"{stem}-inf.parquet"

        # Step 1: Inference
        if out_raster.exists() and not args.force:
            print(f"[SKIP] Inference output already exists for {stem}: {out_raster}")
        else:
            print(f"[RUN] Inference for {stem}")
            cmd_run = [
                "ftw",
                "inference",
                "run",
                str(tile_path),
                "-f",
                "-o",
                str(out_raster),
                "--gpu",
                str(args.gpu),
                "-m",
                str(model_ckpt),
            ]
            run_cmd(cmd_run)
            print(f"[OK] Inference complete: {out_raster}")

        # Step 2: Polygonize (optional)
        if args.polygonize:
            if out_vector.exists() and not args.force:
                print(f"[SKIP] Polygonized output already exists for {stem}: {out_vector}")
            else:
                print(f"[RUN] Polygonize for {stem}")
                cmd_poly = [
                    "ftw",
                    "inference",
                    "polygonize",
                    str(out_raster),
                    "--out",
                    str(out_vector),
                ]
                run_cmd(cmd_poly)
                print(f"[OK] Polygonized output: {out_vector}")

        print("")

    print("[DONE] All tiles processed.")


if __name__ == "__main__":
    main(sys.argv[1:])
