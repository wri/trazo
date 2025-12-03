import torch
from torch.utils.data import Dataset
from pathlib import Path
import zarr


class FTWZarr(Dataset):
    """Minimal dataset that reads FTW data from Zarr."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms=None,
        num_samples: int = -1,
        verbose: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        zpath = self.root / f"{split}.zarr"
        if verbose:
            print(f"[INFO] Loading Zarr: {zpath}")

        store = zarr.open(str(zpath), mode="r")
        self.images = store["images"]
        self.masks = store["masks"]

        self.length = len(self.images)
        if num_samples > 0:
            self.length = min(self.length, num_samples)

        if verbose:
            print(f"[INFO] Loaded {self.length} samples.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        mask = torch.from_numpy(self.masks[idx])

        sample = {"image": image, "mask": mask}

        if self.transforms:
            sample = self.transforms(sample)

        return sample
