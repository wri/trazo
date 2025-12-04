import os
import random
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
import zarr
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from src.tkt.pt4_train.settings import ALL_COUNTRIES, TEMPORAL_OPTIONS

class FTW_finaltraining(NonGeoDataset):
    valid_splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str,
        countries: Sequence[str] | str | None = None,
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        num_samples: int = -1,
        verbose: bool = True,
    ):
        self.root = root
        if countries is None:
            raise ValueError("Specify countries to load")
        if isinstance(countries, str):
            countries = [countries]
        countries = [c.lower() for c in countries]
        for c in countries:
            assert c in ALL_COUNTRIES, f"Invalid country {c}"
        self.countries = countries
        self.split = split
        self.transforms = transforms
        self.num_samples = num_samples

        self.filenames = []
        all_filenames = []

        for country in self.countries:
            country_root = Path(self.root) / country
            zarr_dir = country_root / "zarr"
            if not zarr_dir.exists():
                raise RuntimeError(f"Missing Zarr directory for {country}: {zarr_dir}")

            for chip_path in zarr_dir.iterdir():
                if chip_path.is_dir():
                    all_filenames.append({"zarr": str(chip_path)})

        if self.num_samples == -1:
            self.filenames = all_filenames
        else:
            self.filenames = random.sample(all_filenames, min(self.num_samples, len(all_filenames)))

        if verbose:
            print(f"Selected {len(self.filenames)} samples from Zarr dataset")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        file_info = self.filenames[index]
        zarr_path = file_info["zarr"]
        sample = zarr.open(zarr_path, mode="r")

        image = torch.from_numpy(sample["image"][:]).float()
        mask = torch.from_numpy(sample["mask"][:]).long()

        out = {"image": image, "mask": mask}
        if self.transforms is not None:
            out = self.transforms(out)
            out["mask"] = out["mask"].long()
        return out


