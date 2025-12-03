import argparse
from pathlib import Path
import shutil
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, Subset

# ---------------------
# Utility
# ---------------------

def clear_if_exists(path: Path, overwrite: bool):
    if path.exists():
        if overwrite:
            print(f"[INFO] Removing existing {path}")
            shutil.rmtree(path)
        else:
            raise RuntimeError(f"{path} exists. Use --overwrite to replace.")


# ---------------------
# Zarr Export Function
# ---------------------

def export_to_zarr(dataset: Dataset, out_root: Path, split: str, overwrite: bool):
    out_root.mkdir(parents=True, exist_ok=True)

    zarr_path = out_root / f"{split}.zarr"
    clear_if_exists(zarr_path, overwrite)

    print(f"[INFO] Creating Zarr store for '{split}' at: {zarr_path}")

    store = zarr.open(str(zarr_path), mode="w")

    # Probe first item
    sample0 = dataset[0]
    C, H, W = sample0["image"].shape
    MH, MW = sample0["mask"].shape

    img_arr = store.create(
        "images",
        shape=(len(dataset), C, H, W),
        chunks=(1, C, H, W),
        dtype=np.float32,
    )

    mask_arr = store.create(
        "masks",
        shape=(len(dataset), MH, MW),
        chunks=(1, MH, MW),
        dtype=np.int64,
    )

    for i in range(len(dataset)):
        s = dataset[i]
        img_arr[i] = s["image"].numpy()
        mask_arr[i] = s["mask"].numpy()

        if i % 100 == 0:
            print(f"[INFO]   {i}/{len(dataset)} saved...")

    print(f"[INFO] Finished '{split}' Zarr export.")
    return zarr_path


# ---------------------
# Split Handling
# ---------------------

def export_unsplit(full_dataset, out_root, overwrite):
    print("[INFO] Exporting unsplit dataset (automatic 80/20 split)")

    n = len(full_dataset)
    train_n = int(n * 0.8)

    idx = torch.randperm(n)
    train_ds = Subset(full_dataset, idx[:train_n])
    val_ds   = Subset(full_dataset, idx[train_n:])

    export_to_zarr(train_ds, out_root, "train", overwrite)
    export_to_zarr(val_ds, out_root, "val", overwrite)


def export_presplit(train_ds, val_ds, out_root, overwrite):
    print("[INFO] Exporting presplit dataset")

    export_to_zarr(train_ds, out_root, "train", overwrite)
    export_to_zarr(val_ds, out_root, "val", overwrite)


# ---------------------
# CLI + MAIN
# ---------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--mask-type", type=str, default="semantic_3class")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)

    print(f"[INFO] Loading dataset root: {root}")
    print(f"[INFO] Countries: {args.countries}")
    print(f"[INFO] Mask type: {args.mask_type}")

    # -----------------------------
    # Load your FTW_raw dataset here
    # (this part depends on your original FTW dataset loader)
    # -----------------------------

    from tkt.pt2_dataprep.datasets import FTW  # your original dataset

    print("[INFO] Initializing FTW dataset...")

    ds = FTW(
        root=str(root),
        countries=args.countries,
        mask_type=args.mask_type,
        split="all",      # load everything
        load_boundaries=False,
        load_edges=False,
    )

    print(f"[INFO] Dataset contains {len(ds)} total samples.")

    # -----------------------------
    # If dataset is already presplit
    # -----------------------------
    if hasattr(ds, "train") and hasattr(ds, "val"):
        export_presplit(ds.train, ds.val, Path("data/ftw/zarr"), args.overwrite)
    else:
        # cerrrado2, etc.
        export_unsplit(ds, Path("data/ftw/zarr"), args.overwrite)


if __name__ == "__main__":
    main()


