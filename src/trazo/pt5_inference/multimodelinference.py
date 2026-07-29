#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 5 utility: run FTW inference across multiple checkpoints and stacks.

This script does one thing:

  * For every TIFF stack in an input directory (typically the 8-band stacks
    written by trazo.pt5_inference.tilepairs) and for every model checkpoint
    in a checkpoint directory, run:

        ftw inference run <stack> -f -o <output> --gpu <gpu> -m <checkpoint>

  and organize the outputs by checkpoint, e.g.:

      <output_base>/
        <checkpoint_stem_1>/
            S2_Stack_..._inf.tif
            ...
        <checkpoint_stem_2>/
            S2_Stack_..._inf.tif
            ...

Polygonization is intentionally not done here.

Example:

    python -m trazo.pt5_inference.multimodelinference \
        --input-dir /path/to/tilepairs/stacks \
        --checkpoint-dir /path/to/checkpoints \
        --output-dir /path/to/inf_output \
        --gpu-id 0 \
        --overwrite

or via the CLI wrapper (if wired):

    trazo-pt5-infer batch-infer \
        --input-dir /path/to/tilepairs/stacks \
        --checkpoint-dir /path/to/checkpoints \
        --output-dir /path/to/inf_output
"""

import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def run_cmd(cmd: List[str]) -> int:
    """
    Run a subprocess command with live stdout/err printing.
    Returns the process return code.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for line in iter(proc.stdout.readline, ""):
            if line == "" and proc.poll() is not None:
                break
            print(line, end="")
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        raise
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass


def check_ftw_available() -> None:
    """
    Ensure the `ftw` CLI is available on PATH.
    """
    if shutil.which("ftw") is None:
        raise RuntimeError(
            "The 'ftw' CLI was not found. Activate your environment and ensure it's installed."
        )


def sanitize_checkpoint_name(ckpt_path: Path) -> str:
    """
    Turn a checkpoint file path into a safe folder name.
    For now, just use the filename stem.
    """
    return ckpt_path.stem


def infer_one_image_with_checkpoint(
    tif_path: Path,
    ckpt_path: Path,
    out_tif_path: Path,
    gpu_id: int = 0,
    overwrite: bool = False,
    extra_args: Optional[List[str]] = None,
) -> bool:
    """
    Run FTW inference for a single TIFF with a single checkpoint.

    Returns True if a new output was written, False if skipped.
    """
    out_tif_path.parent.mkdir(parents=True, exist_ok=True)

    if out_tif_path.exists() and not overwrite:
        print(f"[SKIP] Output exists: {out_tif_path}")
        return False

    cmd = [
        "ftw",
        "inference",
        "run",
        str(tif_path),
        "-f",
        "-o",
        str(out_tif_path),
        "--gpu",
        str(gpu_id),
        "-m",
        str(ckpt_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n[RUN ] Model {ckpt_path.stem} on {tif_path.name}")
    rc = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"Inference failed ({rc}) for {tif_path}")

    print(f"[OK  ] Wrote: {out_tif_path}")
    return True


def batch_ftw_inference(
    input_dir: Path,
    checkpoint_dir: Path,
    output_base: Path,
    gpu_id: int,
    overwrite: bool,
    recursive: bool,
    tif_glob: str,
    ckpt_glob: str,
    inference_extra_args: Optional[List[str]],
) -> None:
    """
    Main batch loop: for each checkpoint and each TIFF in input_dir,
    run inference and write output into <output_base>/<ckpt_stem>/.
    """
    check_ftw_available()

    if recursive:
        tif_iter = input_dir.rglob(tif_glob)
    else:
        tif_iter = input_dir.glob(tif_glob)

    tifs = sorted([p for p in tif_iter if p.is_file()])
    ckpts = sorted([p for p in checkpoint_dir.glob(ckpt_glob) if p.is_file()])

    if not tifs:
        raise FileNotFoundError(f"No TIFFs matching '{tif_glob}' found in {input_dir}")
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints matching '{ckpt_glob}' found in {checkpoint_dir}"
        )

    output_base.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Starting batch at {datetime.now().isoformat(timespec='seconds')}")
    print(f"[INFO] Found {len(tifs)} TIFF(s), {len(ckpts)} checkpoint(s).")
    print(f"[INFO] Input dir:      {input_dir}")
    print(f"[INFO] Checkpoint dir: {checkpoint_dir}")
    print(f"[INFO] Output base:    {output_base}")
    print(f"[INFO] Recursive:      {recursive}")
    print(f"[INFO] Overwrite:      {overwrite}")
    print("")

    for ckpt in ckpts:
        ckpt_name = sanitize_checkpoint_name(ckpt)
        ckpt_out_dir = output_base / ckpt_name
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[INFO] === Model: {ckpt_name} ===")
        print(f"[INFO] Output dir for this model: {ckpt_out_dir}")

        for tif in tifs:
            out_tif = ckpt_out_dir / f"{tif.stem}_inf.tif"
            infer_one_image_with_checkpoint(
                tif,
                ckpt,
                out_tif,
                gpu_id=gpu_id,
                overwrite=overwrite,
                extra_args=inference_extra_args,
            )

    print("\n[DONE] Batch complete.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FTW inference over a directory of 8-band stacks "
            "for all checkpoints in a directory."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Directory containing input TIFF stacks (e.g. output from "
            "trazo.pt5_inference.tilepairs). These are typically 8-band stacks "
            "with window_a and window_b combined."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing model checkpoints (*.ckpt).",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        default=None,
        help=(
            "Base output directory. A subfolder per checkpoint will be created "
            "here. Defaults to '<input-dir>/inf_output' if not provided."
        ),
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU index to use for inference (default: 0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs instead of skipping.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for TIFFs under --input-dir.",
    )
    parser.add_argument(
        "--tif-glob",
        default="*.tif",
        help="Glob pattern for input TIFFs (default: '*.tif').",
    )
    parser.add_argument(
        "--ckpt-glob",
        default="*.ckpt",
        help="Glob pattern for checkpoints (default: '*.ckpt').",
    )
    parser.add_argument(
        "--inference-extra-args",
        nargs=argparse.REMAINDER,
        default=None,
        help=(
            "Any extra arguments to pass through to 'ftw inference run'. "
            "Everything after this flag is appended as-is to the ftw command."
        ),
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()

    if args.output_dir is None:
        output_base = input_dir / "inf_output"
    else:
        output_base = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    print(f"[CFG] Input dir:      {input_dir}")
    print(f"[CFG] Checkpoint dir: {checkpoint_dir}")
    print(f"[CFG] Output base:    {output_base}")
    print(f"[CFG] GPU ID:         {args.gpu_id}")
    print(f"[CFG] Overwrite:      {args.overwrite}")
    print(f"[CFG] Recursive:      {args.recursive}")
    print(f"[CFG] TIF glob:       {args.tif_glob}")
    print(f"[CFG] CKPT glob:      {args.ckpt_glob}")
    print(f"[CFG] Extra args:     {args.inference_extra_args}")
    print("")

    batch_ftw_inference(
        input_dir=input_dir,
        checkpoint_dir=checkpoint_dir,
        output_base=output_base,
        gpu_id=args.gpu_id,
        overwrite=args.overwrite,
        recursive=args.recursive,
        tif_glob=args.tif_glob,
        ckpt_glob=args.ckpt_glob,
        inference_extra_args=args.inference_extra_args,
    )


if __name__ == "__main__":
    main()
