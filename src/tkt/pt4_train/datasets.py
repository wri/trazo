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

from src.tkt.pt4_train.settings import ALL_COUNTRIES, TEMPORAL_OPTIONS

from src.tkt.pt4_train.utils import validate_checksums

import json

import zarr
import geopandas as gpd

from src.tkt.pt4_train.settings import ALL_COUNTRIES, TEMPORAL_OPTIONS
from src.tkt.pt4_train.utils import validate_checksums


class FTW_finaltraining(NonGeoDataset):
    valid_splits = ["train", "val", "test"]
    def __init__(
        self,
        root: str = "data/ftw",
        countries: Sequence[str] | str | None = None,
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        checksum: bool = False,
        load_boundaries: bool = False,
        load_edges: bool = False,
        temporal_options: str = "stacked",
        swap_order: bool = False,
        num_samples: int = -1,
        ignore_sample_fn: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize a new FTW dataset instance.

        Args:
            root: root directory where dataset can be found, this should contain the
                country folder
            countries: the countries to load the dataset from, e.g. "france"
            split: string specifying what split to load (e.g. "train", "val", "test")
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            load_boundaries: if True, load the 3 class masks with boundaries
            load_edges: if True, load the edge masks
            temporal_options : for ablation study, valid option are (stacked, windowA,
                windowB, median, rgb, random_window)
            swap_order: if True, swap the order of temporal data (i.e. use window A first)
            ignore_sample_fn: path to a filename with a list of samples to ignore
        Raises:
            AssertionError: if ``countries`` argument is invalid
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if data is not found, or checksums don't match
        """
        self.root = root

        if countries is None:
            raise ValueError("Please specify the countries to load the dataset from")

        if temporal_options not in TEMPORAL_OPTIONS:
            raise ValueError(f"Invalid temporal option {temporal_options}")

        if isinstance(countries, str):
            countries = [countries]
        countries = [country.lower() for country in countries]
        for country in countries:
            assert country in ALL_COUNTRIES, f"Invalid country {country}"

        self.countries = countries
        assert split in self.valid_splits
        self.transforms = transforms
        self.checksum = checksum
        self.load_boundaries = load_boundaries
        self.load_edges = load_edges
        self.temporal_options = temporal_options
        self.num_samples = num_samples

        if swap_order:
            if temporal_options not in ("stacked", "rgb"):
                raise ValueError(
                    "Can only use swap_order with temporal_options stacked or rgb"
                )
        self.swap_order = swap_order

        if verbose:
            if self.load_boundaries:
                print("Loading 3 Class Masks, with Boundaries")
            else:
                print("Loading 2 Class Masks, without Boundaries")
            print("Temporal option: ", temporal_options)
            if swap_order:
                print("Using window A first, then window B")
            else:
                print("Using window B first, then window A")
            if self.load_edges:
                print("Loading edge masks")

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found at root directory or corrupted.  Download dataset with `ftw data download`"
            )

        if checksum:
            assert self._checksum(), "Checksum of dataset does not match"

        # Load split selections
        self.samples = []   # list of (country, idx_in_country)

        for country in self.countries:
            country_dir = Path(self.root) / country

            chips_file = list(country_dir.glob("chips_*.parquet"))
            if len(chips_file) != 1:
                raise RuntimeError(f"{country}: missing chips_*.parquet")

            df = gpd.read_parquet(chips_file[0])
            df_split = df[df["split"] == split]
            aoi_ids = df_split["aoi_id"].astype(str).tolist()

            # Convert AOI_ids → index in Zarr
            # We rely on the "meta" inside the Zarr
            zarr_path = country_dir / f"{country}.zarr"
            if not zarr_path.exists():
                raise RuntimeError(f"Missing {zarr_path}")

            z = zarr.open(zarr_path, mode="r")


            meta = [json.loads(m) for m in z["meta"][:]]
            chip_ids_in_zarr = [m["chip_id"] for m in meta]

            for chip in aoi_ids:
                if chip in chip_ids_in_zarr:
                    idx = chip_ids_in_zarr.index(chip)
                    self.samples.append((country, idx))

        # Sample reduction
        if self.num_samples > 0:
            import random
            self.samples = random.sample(
                self.samples, min(self.num_samples, len(self.samples))
            )

        if verbose:
            print(f"Loaded {len(self.samples)} samples for split={split}")


        self.images_cache = {}
        self.masks_cache = {}
        for country in self.countries:
            country_dir = Path(self.root) / country
            z = zarr.open(country_dir / f"{country}.zarr", mode="r")
            self.images_cache[country] = z["images"]
            self.masks_cache[country] = z["masks"]
        

    def _check_integrity(self):
        errors = []
    
        for country in self.countries:
            country_dir = Path(self.root) / country
            zarr_path = country_dir / f"{country}.zarr"
            if not zarr_path.exists():
                errors.append(f"Missing Zarr file: {zarr_path}")
                continue
    
            z = zarr.open(zarr_path, mode="r")
    
            # check "meta", "images", "masks" exist
            for key in ["images", "masks", "meta"]:
                if key not in z:
                    errors.append(f"Missing key in zarr: {key} in {zarr_path}")
    
        if errors:
            raise RuntimeError(
                "Dataset integrity check failed:\n" + "\n".join(errors)
            )
        else:
            print("Integrity check passed")
            return True

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Load a single sample {country, idx} from the .zarr store."""
        country, array_idx = self.samples[index]

        # Load from RAM cache
        img = torch.as_tensor(self.images_cache[country][array_idx], dtype=torch.float32)
        mask = torch.as_tensor(self.masks_cache[country][array_idx], dtype=torch.long)


        sample = {"image": img, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

