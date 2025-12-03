"""
FTW Zarr-based dataset loader
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
import zarr


class FTW_finaltraining(Dataset):
    """
    Loads pre-exported Zarr datasets for FTW.
    Raw-file arguments remain for compatibility but are ignored.
    """

    def __init__(
        self,
        root: str = "data/ftw/zarr",
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        num_samples: int = -1,
        verbose: bool = True,
        **kwargs,  # ignore old parameters
    ):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        zarr_path = self.root / f"{split}.zarr"
        if verbose:
            print(f"Loading Zarr dataset from: {zarr_path}")

        self.store = zarr.open(str(zarr_path), mode="r")

        self.images = self.store["images"]
        self.masks = self.store["masks"]

        self.length = len(self.images)
        if num_samples > 0:
            self.length = min(self.length, num_samples)

        if verbose:
            print(f"Loaded {self.length} samples from Zarr.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        image = torch.from_numpy(self.images[idx])
        mask = torch.from_numpy(self.masks[idx])

        sample = {"image": image, "mask": mask}

        if self.transforms:
            sample = self.transforms(sample)

        return sample
