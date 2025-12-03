import os
from pathlib import Path
import zarr
import torch
from torch import Tensor
from torch.utils.data import Dataset


class FTW_finaltraining(Dataset):
    """Ultra-fast dataset loader for single-file Zarr dataset."""

    def __init__(
        self,
        root: str = "data/ftw/zarr",
        split: str = "train",
        transforms=None,
        verbose: bool = True,
    ):
        self.root = root
        self.split = split
        self.transforms = transforms

        zarr_path = Path(root) / f"{split}.zarr"

        if verbose:
            print(f"📂 Loading Zarr store: {zarr_path}")

        self.z = zarr.open(str(zarr_path), mode="r")

        self.images = self.z["images"]
        self.masks = self.z["masks"]

        self.length = self.images.shape[0]

        if verbose:
            print(f"✅ Loaded {self.length} samples for split '{split}'")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.images[idx]).float()
        mask = torch.from_numpy(self.masks[idx]).long()

        sample = {"image": img, "mask": mask}

        if self.transforms:
            sample = self.transforms(sample)
            sample["mask"] = sample["mask"].long()

        return sample
