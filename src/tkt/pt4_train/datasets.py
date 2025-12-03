import torch
from torch.utils.data import Dataset
from pathlib import Path
import zarr


class FTWZarr(Dataset):
    def __init__(self, root: str, split="train", transforms=None, num_samples=-1):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        zpath = self.root / f"{split}.zarr"
        print(f"[INFO] Loading Zarr store: {zpath}")

        group = zarr.open_group(str(zpath), mode="r")

        self.images = group["images"]
        self.masks = group["masks"]

        self.length = len(self.images)
        if num_samples > 0:
            self.length = min(self.length, num_samples)

        print(f"[INFO] Loaded {self.length} samples")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx])
        mask = torch.from_numpy(self.masks[idx])

        sample = {"image": img, "mask": mask}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

