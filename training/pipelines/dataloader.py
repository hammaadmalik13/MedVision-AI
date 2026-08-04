"""PyTorch Dataset for preprocessed BraTS data."""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, cache_dir: Path, subject_ids: list[str] | None = None, transform=None) -> None:
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.files = sorted(self.cache_dir.glob("*_preprocessed.pt"))
        if subject_ids:
            self.files = [f for f in self.files if any(sid in f.stem for sid in subject_ids)]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = torch.load(self.files[idx], weights_only=False)
        sample = {"image": data["image"].float(), "label": data["label"].long()}
        if self.transform:
            sample = self.transform(sample)
        return sample
