"""
Dataset loaders for the NexaDoc-50k synthetic corpus.

The published corpus is already split by the generator and already at the
training resolution:

    deshadow/{train,val,test}/{input,target}/*.jpg  (256x256)
    deshadow/{train,val,test}/mask/*.png            (256x256, 0/255)
    upsample/{train,val,test}/input/*.jpg           (256x256)
    upsample/{train,val,test}/target/*.jpg          (768x1024)

Do NOT reuse ``create_dataloaders`` from ``src.data.dataset`` for this data:
that factory performs an image-level random split and resizes to 192x256.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.float32) / 255.0


def _read_mask(path: Path, threshold: int = 127) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > threshold).astype(np.float32)


def _list_stems(directory: Path) -> list[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    stems = sorted(
        p.stem for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not stems:
        raise ValueError(f"No images found in {directory}")
    return stems


def _pairwise_flip(
    arrays: Tuple[np.ndarray, ...], rng: random.Random
) -> Tuple[np.ndarray, ...]:
    """Apply the same horizontal/vertical flip to every array."""
    if rng.random() < 0.5:
        arrays = tuple(np.ascontiguousarray(a[:, ::-1]) for a in arrays)
    if rng.random() < 0.5:
        arrays = tuple(np.ascontiguousarray(a[::-1, :]) for a in arrays)
    return arrays


class NexaDocDeshadowDataset(Dataset):
    """Official-split 256x256 deshadow pairs.

    Returns a dict with:
        input:  float32 (3, 256, 256)
        target: float32 (3, 256, 256)
        mask:   float32 (1, 256, 256), 1 = cast shadow
        name:   file stem
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        augment: bool = False,
        mask_threshold: int = 127,
    ):
        self.root = Path(root)
        self.split = split
        self.augment = bool(augment) and split == "train"
        self.mask_threshold = int(mask_threshold)

        base = self.root / "deshadow" / split
        self.input_dir = base / "input"
        self.target_dir = base / "target"
        self.mask_dir = base / "mask"

        self.names = _list_stems(self.input_dir)
        missing = [
            name
            for name in self.names
            if not (self.target_dir / f"{name}.jpg").is_file()
            or not (self.mask_dir / f"{name}.png").is_file()
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} samples in {self.input_dir} lack target/mask "
                f"(examples: {preview})"
            )

        # Sample one image to fail fast on the wrong layout.
        sample = _read_rgb(self.input_dir / f"{self.names[0]}.jpg")
        if sample.shape != (256, 256, 3):
            raise ValueError(
                f"Expected 256x256x3 deshadow input, got {sample.shape} "
                f"from {self.input_dir}"
            )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        image = _read_rgb(self.input_dir / f"{name}.jpg")
        target = _read_rgb(self.target_dir / f"{name}.jpg")
        mask = _read_mask(
            self.mask_dir / f"{name}.png", threshold=self.mask_threshold
        )

        if target.shape != image.shape:
            raise ValueError(
                f"input/target shape mismatch for {name}: "
                f"{image.shape} vs {target.shape}"
            )
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"mask shape mismatch for {name}: {mask.shape} vs {image.shape[:2]}"
            )

        if self.augment:
            image, target, mask = _pairwise_flip(
                (image, target, mask[..., None]), random
            )
            mask = mask[..., 0]

        return {
            "input": torch.from_numpy(image.transpose(2, 0, 1).copy()),
            "target": torch.from_numpy(target.transpose(2, 0, 1).copy()),
            "mask": torch.from_numpy(mask[None].copy()),
            "name": name,
        }


class NexaDocUpsampleDataset(Dataset):
    """Official-split upsample pairs: 256x256 -> 768x1024.

    The published ``upsample/input`` is byte-identical to the corresponding
    ``deshadow/target``.  It is therefore the correct input for a second-stage
    super-resolution / refinement model.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        augment: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.augment = bool(augment) and split == "train"

        base = self.root / "upsample" / split
        self.input_dir = base / "input"
        self.target_dir = base / "target"
        self.names = _list_stems(self.input_dir)

        missing = [
            name
            for name in self.names
            if not (self.target_dir / f"{name}.jpg").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} upsample inputs lack targets "
                f"(examples: {', '.join(missing[:5])})"
            )

        sample_in = _read_rgb(self.input_dir / f"{self.names[0]}.jpg")
        sample_tgt = _read_rgb(self.target_dir / f"{self.names[0]}.jpg")
        if sample_in.shape != (256, 256, 3):
            raise ValueError(
                f"Expected 256x256x3 upsample input, got {sample_in.shape}"
            )
        if sample_tgt.shape != (1024, 768, 3):
            raise ValueError(
                f"Expected 768x1024 upsample target, got {sample_tgt.shape}"
            )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        image = _read_rgb(self.input_dir / f"{name}.jpg")
        target = _read_rgb(self.target_dir / f"{name}.jpg")

        if self.augment:
            image, target = _pairwise_flip((image, target), random)

        return {
            "input": torch.from_numpy(image.transpose(2, 0, 1).copy()),
            "target": torch.from_numpy(target.transpose(2, 0, 1).copy()),
            "name": name,
        }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    *,
    distributed: bool = False,
    drop_last: bool = False,
    pin_memory: Optional[bool] = None,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """Build a loader, optionally with a DistributedSampler.

    Returns ``(loader, sampler)``; the caller must call
    ``sampler.set_epoch(epoch)`` before each epoch.
    """
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset, shuffle=shuffle, drop_last=drop_last
        )
        shuffle = False

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    kwargs = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs,
    )
    return loader, sampler
