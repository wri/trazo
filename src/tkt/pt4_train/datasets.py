"""FTW dataset."""
import hickle as hkl
import os
import random
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from torch.utils.data import Dataset
import zarr
class FTW_Zarr(Dataset):
    """
    Optimized FTW dataset that loads pre-exported Zarr data.
    All raw-file parameters are kept for compatibility but ignored.
    """

    def __init__(
        self,
        root: str = "data/ftw/zarr",
        countries: Sequence[str] | str | None = None,  # UNUSED, kept for compatibility
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        checksum: bool = False,        # UNUSED
        load_boundaries: bool = False, # UNUSED
        load_edges: bool = False,      # UNUSED
        temporal_options: str = "stacked",  # UNUSED
        swap_order: bool = False,           # UNUSED
        num_samples: int = -1,
        ignore_sample_fn: Optional[str] = None, # UNUSED
        verbose: bool = True,
    ) -> None:

        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.num_samples = num_samples

        zarr_path = self.root / f"{split}.zarr"

        if verbose:
            print(f"Loading Zarr dataset from: {zarr_path}")

        self.store = zarr.open(str(zarr_path), mode="r")

        self.images = self.store["image"]
        self.masks = self.store["mask"]

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
