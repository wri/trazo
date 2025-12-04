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

#from ftw_tools.settings import ALL_COUNTRIES, TEMPORAL_OPTIONS
from src.tkt.pt4_train.settings import ALL_COUNTRIES, TEMPORAL_OPTIONS

# from ftw_tools.utils import validate_checksums
from src.tkt.pt4_train.utils import validate_checksums




import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
import zarr
import geopandas as gpd

from src.tkt.pt4_train.settings import ALL_COUNTRIES, TEMPORAL_OPTIONS
from src.tkt.pt4_train.utils import validate_checksums


class FTW_finaltraining(NonGeoDataset):
    # self.filenames = # point to your new directory with the te4nsors
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

        # ------------------------
        # Load split selections
        # ------------------------
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

            # meta = list(z["meta"][:])
            # chip_ids_in_zarr = [m["chip_id"] for m in meta]
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

        # Cache all zarr stores
        self.zarr_stores = {}
        for country in self.countries:
            country_dir = Path(self.root) / country
            self.zarr_stores[country] = zarr.open(country_dir / f"{country}.zarr", mode="r")

    # def _check_integrity(self) -> bool:
    #     """Check that HKL files exist for the selected countries."""
    #     for country in self.countries:
    #         country_root = os.path.join(self.root, country)
    #         hkl_dir = os.path.join(country_root, "hkl")
    #         if not os.path.exists(hkl_dir):
    #             print(f"Country {country} is missing hkl directory: {hkl_dir}")
    #             return False

    #         hkl_files = list(Path(hkl_dir).glob("*.hkl"))
    #         if len(hkl_files) == 0:
    #             print(f"No hkl files found in {hkl_dir}")
    #             return False

    #     return True
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
            print("Integrity check passed ✓")
            return True

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Load a single sample {country, idx} from the .zarr store."""
        country, array_idx = self.samples[index]

        z = self.zarr_stores[country]


        # ----------------------------
        # Load from Zarr
        # ----------------------------
        img = torch.from_numpy(z["images"][array_idx]).float()
        mask = torch.from_numpy(z["masks"][array_idx]).long()

        sample = {"image": img, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
    # def __getitem__(self, index: int) -> dict[str, Tensor]:
    #     """Return an index within the dataset.

    #     Args:
    #         index: index to return

    #     Returns:
    #         dictionary containing "image" and "mask" PyTorch tensors
    #     """
    #     file_name = self.filenames[index]
    #     hkl_path = file_name["hkl"]

    #     sample = hkl.load(hkl_path)   # ← returns dict: {"image": ..., "mask": ...}
    #     # Convert image and mask to PyTorch tensors
    #     sample["image"] = torch.from_numpy(sample["image"]).float()   # keep image as float
    #     sample["mask"] = torch.from_numpy(sample["mask"]).long()      # mask must be long for CrossEntropyLoss

    #     if self.transforms is not None:
    #         sample = self.transforms(sample)
    #         sample["mask"] = sample["mask"].long()

    #     return sample
# import json
# from pathlib import Path
# from typing import Any, Callable, Optional, Sequence

# import numpy as np
# import torch
# from torch import Tensor
# from torchgeo.datasets import NonGeoDataset
# import zarr
# import geopandas as gpd

# from src.tkt.pt4_train.settings import ALL_COUNTRIES, TEMPORAL_OPTIONS
# from src.tkt.pt4_train.utils import validate_checksums

# class FTW_finaltraining(NonGeoDataset):
#     valid_splits = ["train", "val", "test"]

#     def __init__(
#         self,
#         root: str = "data/ftw",
#         countries: Sequence[str] | str | None = None,
#         split: str = "train",
#         transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
#         checksum: bool = False,
#         load_boundaries: bool = False,
#         load_edges: bool = False,
#         temporal_options: str = "stacked",
#         swap_order: bool = False,
#         num_samples: int = -1,
#         ignore_sample_fn: Optional[str] = None,
#         verbose: bool = True,
#     ) -> None:
#         super().__init__()
#         self.root = Path(root)

#         if countries is None:
#             raise ValueError("Please specify the countries to load the dataset from")
#         if isinstance(countries, str):
#             countries = [countries]
#         countries = [c.lower() for c in countries]
#         for c in countries:
#             assert c in ALL_COUNTRIES, f"Invalid country {c}"

#         if temporal_options not in TEMPORAL_OPTIONS:
#             raise ValueError(f"Invalid temporal option {temporal_options}")

#         self.countries = countries
#         self.transforms = transforms
#         self.checksum = checksum
#         self.load_boundaries = load_boundaries
#         self.load_edges = load_edges
#         self.temporal_options = temporal_options
#         self.swap_order = swap_order
#         self.num_samples = num_samples

#         if verbose:
#             print(f"Loading split '{split}' for countries {countries}")

#         # ----------------------------
#         # Open Zarr files once
#         # ----------------------------
#         self.zarr_files = {}
#         self.samples = []  # list of (country, array_idx)

#         for country in self.countries:
#             country_dir = self.root / country
#             zarr_path = country_dir / f"{country}.zarr"
#             if not zarr_path.exists():
#                 raise RuntimeError(f"Missing Zarr file: {zarr_path}")

#             z = zarr.open(zarr_path, mode="r")
#             # Quick integrity check
#             for key in ["images", "masks", "meta"]:
#                 if key not in z:
#                     raise RuntimeError(f"Missing key '{key}' in {zarr_path}")

#             self.zarr_files[country] = z

#             # Load chips split
#             chips_file = list(country_dir.glob("chips_*.parquet"))
#             if len(chips_file) != 1:
#                 raise RuntimeError(f"{country}: missing chips_*.parquet")
#             df = gpd.read_parquet(chips_file[0])
#             df_split = df[df["split"] == split]
#             aoi_ids = df_split["aoi_id"].astype(str).tolist()

#             meta = [json.loads(m) for m in list(z["meta"][:])]
#             chip_ids_in_zarr = [m["chip_id"] for m in meta]

#             for chip_id in aoi_ids:
#                 if chip_id in chip_ids_in_zarr:
#                     array_idx = chip_ids_in_zarr.index(chip_id)
#                     self.samples.append((country, array_idx))

#         # Sample reduction
#         if self.num_samples > 0:
#             import random
#             self.samples = random.sample(
#                 self.samples, min(self.num_samples, len(self.samples))
#             )

#         if verbose:
#             print(f"Loaded {len(self.samples)} samples for split={split}")

#     def __len__(self) -> int:
#         return len(self.samples)

#     def __getitem__(self, index: int) -> dict[str, Tensor]:
#         country, array_idx = self.samples[index]
#         z = self.zarr_files[country]

#         # Load single sample
#         img_np: np.ndarray = z["images"][array_idx]  # [C, H, W]
#         mask_np: np.ndarray = z["masks"][array_idx]  # [H, W]

#         # Convert to torch
#         img = torch.from_numpy(img_np.copy()).float()
#         mask = torch.from_numpy(mask_np.copy()).long()

#         sample = {"image": img, "mask": mask}

#         if self.transforms is not None:
#             sample = self.transforms(sample)

#         return sample
