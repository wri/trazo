#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pt1_createdata/active_sample.py

Integrated active-learning sampling pipeline for the trazo package.

Runs FTW model inference from a user-supplied checkpoint, scores every chip
against a local agricultural binary raster, and selects chips to re-label
using four complementary strategies.

Four sampling strategies
------------------------
A  lc_discrepancy_all
       LC-raster vs. model predictions counting field boundaries AND interiors
       (all non-background pixels, i.e. ``predictions != 0``).
       Selects chips with the largest absolute discrepancy.

B  lc_discrepancy_interiors
       Same discrepancy metric, but using only field-interior predictions
       (``predictions == interior_class``, default class 1).
       Selects chips with the largest absolute discrepancy.

C  low_confidence
       Chips where the model is least certain (lowest mean softmax
       max-probability over all pixels in the chip).  No LC filter.

D  low_confidence_ag20pct
       Same as C, but restricted to chips where the AG raster flags at least
       20 % of the chip area as agricultural (configurable via
       ``--ag-threshold``).

User-defined inputs
-------------------
--chips-dir       : directory of Sentinel-2 chip GeoTIFFs (model inputs)
--checkpoint      : path to FTW model .ckpt file
--ag-raster       : path to agricultural land-cover binary GeoTIFF
                    pixel value == 1  →  agricultural
--output-dir      : root directory for all outputs
--n-per-strategy  : chips to select per strategy (default 100)

CLI example
-----------
::

    trazo-active-sample \\
        --chips-dir       /data/grids_tiff \\
        --checkpoint      /models/my_model.ckpt \\
        --ag-raster       /data/ag_binary.tif \\
        --output-dir      /data/active_learning \\
        --n-per-strategy  100

Output layout
-------------
::

    <output-dir>/
        scores.json                        full per-chip score table
        lc_discrepancy_all.json            metadata for strategy A selection
        lc_discrepancy_all/                chip TIFFs for strategy A
        lc_discrepancy_interiors.json      metadata for strategy B selection
        lc_discrepancy_interiors/          chip TIFFs for strategy B
        low_confidence.json                metadata for strategy C selection
        low_confidence/                    chip TIFFs for strategy C
        low_confidence_ag20pct.json        metadata for strategy D selection
        low_confidence_ag20pct/            chip TIFFs for strategy D
        inference/preds/                   (optional) prediction TIFFs
        inference/conf/                    (optional) confidence TIFFs

FTW model class convention
--------------------------
The default FTW model produces three classes::

    0  background
    1  field interior   ← ``--interior-class`` (default)
    2  field boundary

If your checkpoint uses a different ordering, pass ``--interior-class``
accordingly.
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm


# ============================================================================
# Module-level constants / defaults
# ============================================================================

#: FTW field interior prediction class (background=0, interior=1, boundary=2)
DEFAULT_INTERIOR_CLASS: int = 1

#: Minimum fraction of the chip that the AG raster must flag as agricultural
#: for a chip to be eligible under strategy D.
DEFAULT_AG_THRESHOLD: float = 0.20

#: Default number of chips to select per strategy.
DEFAULT_N: int = 100

#: Chips with more than this fraction of zero pixels are considered padded and
#: are skipped (they sit on the edge of the AOI and contain mostly nodata).
DEFAULT_MAX_ZERO_FRAC: float = 0.15

#: Input upsampling factor for model inference (matches FTW training config).
DEFAULT_RESIZE_FACTOR: int = 2

#: GPU index to use.  ``None`` → CPU.
DEFAULT_GPU: Optional[int] = 0

#: Pixel value in the AG binary raster that means "agricultural".
AG_BINARY_VALUE: int = 1


# ============================================================================
# Internal type alias
# ============================================================================

ScoreRecord = Dict  # one dict per chip


# ============================================================================
# Inference helpers
# ============================================================================

def _load_model(checkpoint_path: Path, gpu: Optional[int]):
    """
    Load a FTW segmentation model from a Lightning checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.ckpt`` file.
    gpu:
        GPU index.  ``None`` forces CPU inference.

    Returns
    -------
    (model, device) tuple — model is in eval mode and moved to ``device``.
    """
    try:
        import torch
        from ftw.trainers import CustomSemanticSegmentationTask
    except ImportError as exc:
        raise ImportError(
            "Inference requires 'ftw-tools' and PyTorch.  Install with:\n"
            "    pip install ftw-tools torch\n"
            f"Original error: {exc}"
        ) from exc

    if gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        if gpu is not None:
            print(
                f"[active_sample] WARNING: GPU {gpu} requested but CUDA unavailable. "
                "Falling back to CPU.",
                file=sys.stderr,
            )
        device = torch.device("cpu")

    print(f"[active_sample] Loading checkpoint: {checkpoint_path}")
    print(f"[active_sample] Device: {device}")

    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        str(checkpoint_path), map_location="cpu"
    )
    task.freeze()
    model = task.model.eval().to(device)
    return model, device


def _preprocess(chip_data: np.ndarray) -> "torch.Tensor":  # noqa: F821
    """
    Normalise a raw chip array and return a float32 tensor.

    Uses FTW's ``preprocess`` transform if available; falls back to simple
    uint16 → float32 scaling (divide by 10 000, standard for Sentinel-2
    surface reflectance values).
    """
    import torch

    image = torch.from_numpy(chip_data).float()
    sample = {"image": image}

    try:
        from ftw.datamodules import preprocess as ftw_preprocess
        sample = ftw_preprocess(sample)
        return sample["image"]
    except ImportError:
        # Fallback normalisation for Sentinel-2 uint16 values.
        return image / 10_000.0


def _infer_chip(
    model,
    device,
    chip_data: np.ndarray,
    resize_factor: int = DEFAULT_RESIZE_FACTOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run FTW model inference on a single chip.

    Parameters
    ----------
    model:
        Loaded FTW model (eval mode, on ``device``).
    device:
        torch.device.
    chip_data:
        Raw chip array, shape ``(bands, H, W)``.
    resize_factor:
        The model input is upsampled by this factor before inference and the
        outputs are downsampled back to the original chip resolution.
        This matches the FTW super-resolution inference pipeline.

    Returns
    -------
    predictions : np.ndarray, shape (H, W), dtype uint8
        Per-pixel argmax class.
    confidence : np.ndarray, shape (H, W), dtype float32
        Per-pixel softmax max-probability (higher = more confident).
    """
    import torch
    import torch.nn.functional as F

    h, w = chip_data.shape[1], chip_data.shape[2]

    # Preprocess → add batch dim → move to device
    image = _preprocess(chip_data).unsqueeze(0).to(device)  # (1, C, H, W)

    # Upsample for super-resolution inference
    if resize_factor != 1:
        image = F.interpolate(
            image,
            scale_factor=resize_factor,
            mode="bilinear",
            align_corners=False,
        )

    with torch.inference_mode():
        logits = model(image)                            # (1, K, H', W')
        probs = logits.softmax(dim=1)                    # (1, K, H', W')
        conf_map, _ = probs.max(dim=1, keepdim=True)     # (1, 1, H', W')
        pred_map = logits.argmax(dim=1, keepdim=True).float()  # (1, 1, H', W')

    # Downsample predictions (nearest to preserve class labels) and confidence
    if resize_factor != 1:
        pred_map = F.interpolate(pred_map, size=(h, w), mode="nearest")
        conf_map = F.interpolate(
            conf_map, size=(h, w), mode="bilinear", align_corners=False
        )

    predictions = pred_map.squeeze().int().cpu().numpy().astype(np.uint8)  # (H, W)
    confidence = conf_map.squeeze().cpu().numpy().astype(np.float32)       # (H, W)
    return predictions, confidence


# ============================================================================
# AG raster helper
# ============================================================================

def _get_ag_mask(
    chip_bounds,
    chip_crs,
    chip_transform,
    h: int,
    w: int,
    ag_src: rasterio.DatasetReader,
) -> Tuple[float, np.ndarray]:
    """
    Reproject the agricultural binary raster into the chip's pixel grid.

    Returns
    -------
    (ag_frac, ag_mask)
        ag_frac : fraction of chip pixels that are agricultural.
        ag_mask : boolean array, shape (H, W).
    """
    dst = np.zeros((h, w), dtype=ag_src.dtypes[0])
    reproject(
        source=rasterio.band(ag_src, 1),
        destination=dst,
        src_transform=ag_src.transform,
        src_crs=ag_src.crs,
        dst_transform=chip_transform,
        dst_crs=chip_crs,
        resampling=Resampling.nearest,
    )
    ag_mask = dst == AG_BINARY_VALUE
    ag_frac = float(ag_mask.sum()) / (h * w)
    return ag_frac, ag_mask


# ============================================================================
# Core pipeline: inference → scoring
# ============================================================================

def run_pipeline(
    chips_dir: Path,
    checkpoint_path: Path,
    ag_raster_path: Path,
    output_dir: Path,
    interior_class: int = DEFAULT_INTERIOR_CLASS,
    max_zero_frac: float = DEFAULT_MAX_ZERO_FRAC,
    resize_factor: int = DEFAULT_RESIZE_FACTOR,
    gpu: Optional[int] = DEFAULT_GPU,
    save_inference: bool = False,
    verbose: bool = True,
) -> List[ScoreRecord]:
    """
    Run inference over all chips in ``chips_dir``, score each chip against
    the AG raster, and return a list of per-chip score records.

    Parameters
    ----------
    chips_dir:
        Directory of Sentinel-2 chip GeoTIFFs (model inputs).
    checkpoint_path:
        Path to the FTW ``.ckpt`` checkpoint.
    ag_raster_path:
        Path to the agricultural binary GeoTIFF.
        Pixels with value ``AG_BINARY_VALUE`` (1) are considered agricultural.
    output_dir:
        Root output directory.  Inference artefacts (if ``save_inference``)
        are written under ``output_dir/inference/``.
    interior_class:
        Prediction class index for field interiors (FTW default = 1).
    max_zero_frac:
        Chips with more than this fraction of zero pixels are skipped.
    resize_factor:
        Input upsampling factor passed to :func:`_infer_chip`.
    gpu:
        GPU index.  ``None`` → CPU.
    save_inference:
        If ``True``, write per-chip prediction and confidence TIFFs to
        ``output_dir/inference/preds/`` and ``output_dir/inference/conf/``.
    verbose:
        Show a tqdm progress bar and summary statistics.

    Returns
    -------
    List of :data:`ScoreRecord` dicts, one per scored chip, with keys:

    * ``file``                – chip filename
    * ``confidence_mean``     – mean softmax confidence over all chip pixels
    * ``confidence_std``      – std of softmax confidence over all chip pixels
    * ``confidence_mean_ag``  – mean softmax confidence over AG pixels only
                                (``nan`` if no AG pixels in chip)
    * ``lc_ag_frac``          – fraction of chip flagged as ag by the LC raster
    * ``pred_frac_all``       – predicted crop fraction (boundaries + interiors)
    * ``pred_frac_interiors`` – predicted crop fraction (interiors only)
    * ``metric_all``          – ``lc_ag_frac - pred_frac_all``
    * ``metric_interiors``    – ``lc_ag_frac - pred_frac_interiors``
    """
    chip_files = sorted(chips_dir.glob("*.tif"))
    if not chip_files:
        raise FileNotFoundError(f"No .tif files found in chips_dir: {chips_dir}")

    # Optional inference output directories
    preds_dir = output_dir / "inference" / "preds"
    conf_dir = output_dir / "inference" / "conf"
    if save_inference:
        preds_dir.mkdir(parents=True, exist_ok=True)
        conf_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, device = _load_model(checkpoint_path, gpu)

    # Open AG raster (held open for the duration for efficiency)
    ag_src = rasterio.open(ag_raster_path)

    records: List[ScoreRecord] = []
    skipped_padding = 0
    skipped_errors = 0

    iterator = (
        tqdm(chip_files, desc="Inference + scoring", unit="chip")
        if verbose
        else chip_files
    )

    try:
        for chip_path in iterator:
            fname = chip_path.name

            # ── Load chip ──────────────────────────────────────────────
            try:
                with rasterio.open(chip_path) as src:
                    chip_data = src.read()           # (bands, H, W)
                    chip_profile = src.profile.copy()
                    chip_crs = src.crs
                    chip_bounds = src.bounds
                    chip_transform = src.transform
                    h, w = src.height, src.width
            except Exception as exc:
                if verbose:
                    print(f"[active_sample] WARNING: could not read {fname}: {exc}")
                skipped_errors += 1
                continue

            # ── Padding filter ─────────────────────────────────────────
            n_bands = chip_data.shape[0]
            zero_frac = float((chip_data == 0).sum()) / (n_bands * h * w)
            if zero_frac > max_zero_frac:
                skipped_padding += 1
                continue

            # ── Model inference ────────────────────────────────────────
            try:
                predictions, confidence = _infer_chip(
                    model, device, chip_data, resize_factor=resize_factor
                )
            except Exception as exc:
                if verbose:
                    print(
                        f"[active_sample] WARNING: inference failed for {fname}: {exc}"
                    )
                skipped_errors += 1
                continue

            # ── (Optional) save inference artefacts ────────────────────
            if save_inference:
                _save_raster(predictions[np.newaxis], chip_profile, preds_dir / fname,
                             dtype="uint8")
                _save_raster(confidence[np.newaxis], chip_profile, conf_dir / fname,
                             dtype="float32")

            # ── AG raster mask ─────────────────────────────────────────
            try:
                ag_frac, ag_mask = _get_ag_mask(
                    chip_bounds, chip_crs, chip_transform, h, w, ag_src
                )
            except Exception as exc:
                if verbose:
                    print(
                        f"[active_sample] WARNING: AG reproject failed for {fname}: {exc}"
                    )
                skipped_errors += 1
                continue

            # ── Compute all metrics ────────────────────────────────────

            # Strategy A/B: LC discrepancy
            pred_frac_all = float((predictions != 0).sum()) / (h * w)
            pred_frac_int = float((predictions == interior_class).sum()) / (h * w)
            metric_all = ag_frac - pred_frac_all
            metric_int = ag_frac - pred_frac_int

            # Strategy C/D: model confidence
            conf_mean_all = float(confidence.mean())
            conf_std_all = float(confidence.std())
            conf_mean_ag = (
                float(confidence[ag_mask].mean()) if ag_mask.any() else float("nan")
            )

            records.append(
                {
                    "file": fname,
                    "confidence_mean": conf_mean_all,
                    "confidence_std": conf_std_all,
                    "confidence_mean_ag": conf_mean_ag,
                    "lc_ag_frac": float(ag_frac),
                    "pred_frac_all": pred_frac_all,
                    "pred_frac_interiors": pred_frac_int,
                    "metric_all": float(metric_all),
                    "metric_interiors": float(metric_int),
                }
            )

    finally:
        ag_src.close()

    if verbose:
        print(
            f"\n[active_sample] Scored {len(records)} chips.  "
            f"Skipped: {skipped_padding} (padding), "
            f"{skipped_errors} (read/inference errors)."
        )

    return records


def _save_raster(
    data: np.ndarray,
    profile: dict,
    path: Path,
    dtype: str,
) -> None:
    """Write a single-band array to a GeoTIFF, reusing the chip's georef."""
    profile = profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        compress="lzw",
        nodata=None,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


# ============================================================================
# Sampling strategies
# ============================================================================

def strategy_lc_discrepancy_all(
    records: List[ScoreRecord],
    n: int = DEFAULT_N,
    seed: Optional[int] = None,
) -> List[ScoreRecord]:
    """
    **Strategy A** — LC discrepancy vs. model predictions (boundaries + interiors).

    Metric: ``lc_ag_frac - pred_frac_all``  where ``pred_frac_all`` counts all
    non-background pixels (field interiors + field boundaries).

    Selects the ``n`` chips with the largest *absolute* discrepancy so that
    both "model misses ag" (positive metric) and "model over-predicts ag"
    (negative metric) are equally represented.

    Parameters
    ----------
    records:
        Output of :func:`run_pipeline`.
    n:
        Number of chips to select.
    seed:
        Random seed (used only if ``n`` exceeds the number of candidates,
        which should not happen in practice).

    Returns
    -------
    List of selected :data:`ScoreRecord` dicts sorted by |metric| descending.
    """
    if seed is not None:
        random.seed(seed)
    return sorted(records, key=lambda r: abs(r["metric_all"]), reverse=True)[:n]


def strategy_lc_discrepancy_interiors(
    records: List[ScoreRecord],
    n: int = DEFAULT_N,
    seed: Optional[int] = None,
) -> List[ScoreRecord]:
    """
    **Strategy B** — LC discrepancy vs. field-interior predictions only.

    Metric: ``lc_ag_frac - pred_frac_interiors``  where ``pred_frac_interiors``
    counts only pixels predicted as the field-interior class (default class 1).

    This strategy focuses on the "filled interior" of fields and ignores
    boundary artefacts that can inflate the predicted crop fraction.

    Parameters
    ----------
    records:
        Output of :func:`run_pipeline`.
    n:
        Number of chips to select.
    seed:
        Random seed.

    Returns
    -------
    List of selected :data:`ScoreRecord` dicts sorted by |metric| descending.
    """
    if seed is not None:
        random.seed(seed)
    return sorted(
        records, key=lambda r: abs(r["metric_interiors"]), reverse=True
    )[:n]


def strategy_low_confidence(
    records: List[ScoreRecord],
    n: int = DEFAULT_N,
) -> List[ScoreRecord]:
    """
    **Strategy C** — Lowest model confidence (no LC filter).

    Selects the ``n`` chips where the model's mean softmax max-probability
    over all pixels is smallest — i.e. the model is most uncertain.

    Parameters
    ----------
    records:
        Output of :func:`run_pipeline`.
    n:
        Number of chips to select.

    Returns
    -------
    List of selected :data:`ScoreRecord` dicts sorted by ``confidence_mean``
    ascending.
    """
    return sorted(records, key=lambda r: r["confidence_mean"])[:n]


def strategy_low_confidence_ag20pct(
    records: List[ScoreRecord],
    n: int = DEFAULT_N,
    ag_threshold: float = DEFAULT_AG_THRESHOLD,
) -> List[ScoreRecord]:
    """
    **Strategy D** — Lowest confidence chips where AG covers ≥ ``ag_threshold``.

    First filters to chips where ``lc_ag_frac >= ag_threshold`` (default 0.20,
    i.e. at least 20 % of the chip area is agricultural according to the LC
    raster), then selects the ``n`` least confident.

    This avoids flagging low-confidence chips in bare-soil or urban areas
    where the model uncertainty carries no crop-labelling value.

    Parameters
    ----------
    records:
        Output of :func:`run_pipeline`.
    n:
        Number of chips to select.
    ag_threshold:
        Minimum ``lc_ag_frac`` to be eligible (default 0.20 = 20 %).

    Returns
    -------
    List of selected :data:`ScoreRecord` dicts sorted by ``confidence_mean``
    ascending.
    """
    eligible = [r for r in records if r["lc_ag_frac"] >= ag_threshold]
    if not eligible:
        print(
            f"[active_sample] WARNING: no chips with lc_ag_frac >= {ag_threshold:.0%}. "
            "Strategy D returns empty selection.  "
            "Try lowering --ag-threshold.",
            file=sys.stderr,
        )
        return []
    return sorted(eligible, key=lambda r: r["confidence_mean"])[:n]


# ============================================================================
# Export helper
# ============================================================================

def export_chips(
    records: List[ScoreRecord],
    chips_dir: Path,
    out_dir: Path,
    verbose: bool = True,
) -> int:
    """
    Copy the chip TIFFs referenced in ``records`` to ``out_dir``.

    Parameters
    ----------
    records:
        List of score records — each must have a ``"file"`` key.
    chips_dir:
        Source directory containing the original chip TIFFs.
    out_dir:
        Destination directory (created if it does not exist).
    verbose:
        Print a summary line on completion.

    Returns
    -------
    Number of files successfully copied.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for rec in records:
        src = chips_dir / rec["file"]
        dst = out_dir / rec["file"]
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        elif verbose:
            print(
                f"[active_sample] WARNING: source chip not found, skipping: {src}",
                file=sys.stderr,
            )
    if verbose:
        print(f"[active_sample]   → {copied} chips copied to {out_dir}")
    return copied


# ============================================================================
# Internal helper: save selection + (optionally) export chips
# ============================================================================

def _save_selection(
    name: str,
    selected: List[ScoreRecord],
    chips_dir: Path,
    output_dir: Path,
    no_export: bool,
) -> None:
    """Persist a selection's metadata JSON and (optionally) export chip TIFFs."""
    json_path = output_dir / f"{name}.json"
    with open(json_path, "w") as jf:
        json.dump(selected, jf, indent=2)
    print(f"[active_sample] {name}: {len(selected)} chips → {json_path}")
    if not no_export:
        export_chips(selected, chips_dir, output_dir / name, verbose=True)


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for :func:`main`."""
    p = argparse.ArgumentParser(
        prog="trazo-active-sample",
        description=(
            "Active-learning chip sampling for trazo.  "
            "Runs FTW model inference from a checkpoint, scores chips against "
            "an agricultural binary raster, and exports chips selected by four "
            "complementary strategies."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ────────────────────────────────────────────────────────
    req = p.add_argument_group("Required inputs")
    req.add_argument(
        "--chips-dir",
        required=True,
        metavar="PATH",
        help="Directory of Sentinel-2 chip GeoTIFFs (model inputs, *.tif).",
    )
    req.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to the FTW model checkpoint (.ckpt).",
    )
    req.add_argument(
        "--ag-raster",
        required=True,
        metavar="PATH",
        help=(
            "Agricultural land-cover binary GeoTIFF.  "
            "Pixels with value 1 are treated as agricultural."
        ),
    )
    req.add_argument(
        "--output-dir",
        required=True,
        metavar="PATH",
        help="Root directory for scores, selection JSONs, and exported chips.",
    )

    # ── Sampling ────────────────────────────────────────────────────────
    samp = p.add_argument_group("Sampling")
    samp.add_argument(
        "--n-per-strategy",
        type=int,
        default=DEFAULT_N,
        metavar="N",
        help="Number of chips to select per strategy.",
    )
    samp.add_argument(
        "--ag-threshold",
        type=float,
        default=DEFAULT_AG_THRESHOLD,
        metavar="FRAC",
        help=(
            "Strategy D: minimum fraction of the chip area that the AG raster "
            "must flag as agricultural for a chip to be eligible.  "
            "E.g. 0.20 = at least 20 %% of the chip must be ag."
        ),
    )
    samp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    # ── Inference ───────────────────────────────────────────────────────
    inf_grp = p.add_argument_group("Inference")
    inf_grp.add_argument(
        "--gpu",
        type=int,
        default=DEFAULT_GPU,
        metavar="IDX",
        help="GPU index to use.  Pass -1 to force CPU inference.",
    )
    inf_grp.add_argument(
        "--resize-factor",
        type=int,
        default=DEFAULT_RESIZE_FACTOR,
        metavar="N",
        help=(
            "Upsample factor for model inference (matches FTW training config). "
            "The chip is upsampled by this factor before being passed to the "
            "model and the predictions are downsampled back."
        ),
    )
    inf_grp.add_argument(
        "--interior-class",
        type=int,
        default=DEFAULT_INTERIOR_CLASS,
        metavar="IDX",
        help=(
            "Prediction class index for field interiors.  "
            "Used by strategy B (lc_discrepancy_interiors).  "
            "FTW default: 0=background, 1=interior, 2=boundary."
        ),
    )
    inf_grp.add_argument(
        "--save-inference",
        action="store_true",
        help=(
            "Save per-chip prediction and confidence TIFFs to "
            "<output-dir>/inference/preds/ and <output-dir>/inference/conf/.  "
            "Useful for debugging or re-running sampling without re-running inference."
        ),
    )

    # ── Quality filters ─────────────────────────────────────────────────
    filt = p.add_argument_group("Chip quality filters")
    filt.add_argument(
        "--max-zero-frac",
        type=float,
        default=DEFAULT_MAX_ZERO_FRAC,
        metavar="FRAC",
        help=(
            "Skip chips with more than this fraction of zero pixels.  "
            "Chips near the edge of the AOI are often padded with nodata zeros."
        ),
    )

    # ── Output control ──────────────────────────────────────────────────
    p.add_argument(
        "--no-export",
        action="store_true",
        help=(
            "Only compute and save scores and selection JSONs; do not copy "
            "chip TIFFs to the per-strategy subdirectories."
        ),
    )

    return p


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for ``trazo-active-sample``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    chips_dir = Path(args.chips_dir).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    ag_path = Path(args.ag_raster).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # ── Input validation ────────────────────────────────────────────────
    errors = []
    if not chips_dir.is_dir():
        errors.append(f"chips-dir not found or not a directory: {chips_dir}")
    if not ckpt_path.exists():
        errors.append(f"checkpoint not found: {ckpt_path}")
    elif ckpt_path.suffix != ".ckpt":
        errors.append(f"checkpoint must be a .ckpt file, got: {ckpt_path.suffix}")
    if not ag_path.exists():
        errors.append(f"ag-raster not found: {ag_path}")
    if errors:
        for e in errors:
            print(f"[active_sample] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    gpu = None if args.gpu < 0 else args.gpu

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n[active_sample] ─── Configuration ───────────────────────────────")
    print(f"  chips-dir        : {chips_dir}")
    print(f"  checkpoint       : {ckpt_path}")
    print(f"  ag-raster        : {ag_path}")
    print(f"  output-dir       : {output_dir}")
    print(f"  n per strategy   : {args.n_per_strategy}")
    print(f"  ag threshold (D) : {args.ag_threshold:.0%}")
    print(f"  interior class   : {args.interior_class}")
    print(f"  resize factor    : {args.resize_factor}×")
    print(f"  gpu              : {'CPU' if gpu is None else gpu}")
    print(f"  save inference   : {args.save_inference}")
    print()

    # ── 1. Run inference + scoring ──────────────────────────────────────
    records = run_pipeline(
        chips_dir=chips_dir,
        checkpoint_path=ckpt_path,
        ag_raster_path=ag_path,
        output_dir=output_dir,
        interior_class=args.interior_class,
        max_zero_frac=args.max_zero_frac,
        resize_factor=args.resize_factor,
        gpu=gpu,
        save_inference=args.save_inference,
        verbose=True,
    )

    if not records:
        print(
            "[active_sample] No chips were scored.  "
            "Check input directories, filters, and the AG raster coverage.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 2. Save full score table ────────────────────────────────────────
    scores_path = output_dir / "scores.json"
    with open(scores_path, "w") as jf:
        json.dump(records, jf, indent=2)
    print(f"\n[active_sample] Scores saved → {scores_path}  ({len(records)} chips)")

    n = args.n_per_strategy

    # ── 3. Strategy A: LC discrepancy (boundaries + interiors) ─────────
    print("\n[active_sample] ─── Strategy A: lc_discrepancy_all ─────────────")
    sel_a = strategy_lc_discrepancy_all(records, n=n, seed=args.seed)
    _save_selection("lc_discrepancy_all", sel_a, chips_dir, output_dir, args.no_export)

    # ── 4. Strategy B: LC discrepancy (interiors only) ─────────────────
    print("\n[active_sample] ─── Strategy B: lc_discrepancy_interiors ───────")
    sel_b = strategy_lc_discrepancy_interiors(records, n=n, seed=args.seed)
    _save_selection(
        "lc_discrepancy_interiors", sel_b, chips_dir, output_dir, args.no_export
    )

    # ── 5. Strategy C: Lowest confidence (all chips) ────────────────────
    print("\n[active_sample] ─── Strategy C: low_confidence ─────────────────")
    sel_c = strategy_low_confidence(records, n=n)
    _save_selection("low_confidence", sel_c, chips_dir, output_dir, args.no_export)

    # ── 6. Strategy D: Lowest confidence where AG ≥ threshold ───────────
    print(
        f"\n[active_sample] ─── Strategy D: low_confidence_ag{int(args.ag_threshold*100)}pct ─"
    )
    sel_d = strategy_low_confidence_ag20pct(
        records, n=n, ag_threshold=args.ag_threshold
    )
    _save_selection(
        "low_confidence_ag20pct", sel_d, chips_dir, output_dir, args.no_export
    )

    print("\n[active_sample] Done.")


if __name__ == "__main__":
    main()
