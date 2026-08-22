"""
Mixed-dataset module for Stage 1 training (FSDSRD, RDD, SD7K, A-OSR).

This module is imported by train.py (NOT run directly). It provides:

- MixedShadowRemovalDataset : reads the preprocessed merged flat folder
  (Data/Mixed_Stage1/{input,target,mask}) and applies augmentation to ALL
  samples (every dataset tag: FSDSRD, RDD, SD7K, A-OSR).
- MixedBatchSampler          : per-dataset shuffle queues. Each dataset has its
  own shuffled index queue; batches are composed by popping a fixed count from
  each queue (5 FSDSRD + 4 RDD + 5 SD7K + 2 A-OSR = 16). When a queue empties
  it is reshuffled and refilled ("no repeat until exhausted, then repeat").
  An epoch is one full pass through the largest dataset.
- create_mixed_dataloaders() : factory returning (train_loader, val_loader).

The merged folder is produced offline by scripts/mixed_dataset.py, so training
does no resize/merge work (no CPU load during training).
"""
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
import cv2

from src.data.dataset import _resolve_data_folder, _has_test_split

# Dataset tags -> per-batch counts (must sum to batch_size).
# Augmentation is applied to ALL datasets (not just A-OSR).
# AOSR is the largest (10/32) since it contains real shadow datasets
# (OSR + Kligler + Jung) that need the most representation.
# SynDoc_Wild / SynDoc_Wild_3D are large synthetic datasets (6120/5590).
# CLEAN = input=target, mask=black (identity/no-shadow). BLACK = all black.
DEFAULT_RATIOS = {
    "AOSR": 10,
    "SynDoc_Wild": 4,
    "SynDoc_Wild_3D": 4,
    "FSDSRD": 4,
    "RDD": 4,
    "SD7K": 4,
    "CLEAN": 1,
    "BLACK": 1,
}


def _tag_of(name: str) -> str:
    """
    Return the dataset tag from a prefixed filename.

    Handles multi-word tags (e.g. 'SynDoc_Wild_0001.png' -> 'SynDoc_Wild').
    Uses longest-prefix matching against the known tags so that tags that are
    prefixes of each other (e.g. 'SynDoc_Wild' vs 'SynDoc_Wild_3D') resolve
    correctly.
    """
    known = sorted(DEFAULT_RATIOS.keys(), key=len, reverse=True)
    for tag in known:
        if name.startswith(tag + "_"):
            return tag
    # Fallback: first underscore-delimited token (single-word tags).
    return name.split("_", 1)[0]


class MixedShadowRemovalDataset(Dataset):
    """
    Dataset over the merged flat folder. Augmentation is applied to ALL
    samples (every dataset tag) on the train split.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        input_resolution: Optional[Tuple[int, int]] = None,
        augment: bool = True,
        illumination_strength: float = 0.1,
        shadow_color_shift: float = 0.05,
        rotation_range: float = 0.0,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.input_resolution = input_resolution
        # augment flag only matters for the train split; per-sample gating is
        # applied in __getitem__.
        self.augment = augment and (split == "train")
        self.illumination_strength = illumination_strength
        self.shadow_color_shift = shadow_color_shift
        self.rotation_range = rotation_range

        data_folder = _resolve_data_folder(self.root_dir, split)

        self.input_dir = data_folder / "input"
        self.target_dir = data_folder / "target"
        self.mask_dir = data_folder / "mask"

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")

        self.image_names = sorted([
            f.name for f in self.input_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])

        if not self.image_names:
            raise ValueError(f"No images found in {self.input_dir}")

        # Group indices by dataset tag (for the sampler and for stats).
        self.tag_to_indices: Dict[str, List[int]] = {}
        for i, name in enumerate(self.image_names):
            self.tag_to_indices.setdefault(_tag_of(name), []).append(i)

    def __len__(self) -> int:
        return len(self.image_names)

    def get_tag_counts(self) -> Dict[str, int]:
        """Return {tag: num_samples} for the whole dataset."""
        return {tag: len(idxs) for tag, idxs in self.tag_to_indices.items()}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name = self.image_names[idx]

        try:
            input_bgr = cv2.imread(str(self.input_dir / name))
            target_bgr = cv2.imread(str(self.target_dir / name))

            if input_bgr is None or target_bgr is None:
                print(f"[WARNING] Failed to load {name}")
                return self.__getitem__((idx + 1) % len(self))

            input_np = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            target_np = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        except Exception as e:
            print(f"[WARNING] Failed to load {name}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # Load mask if available
        mask_np = None
        mask_path = self.mask_dir / name
        if mask_path.exists():
            try:
                mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_bgr is not None:
                    if self.input_resolution is not None:
                        expected_w, expected_h = self.input_resolution
                        mask_h, mask_w = mask_bgr.shape[:2]
                        if (mask_h, mask_w) != (expected_h, expected_w):
                            mask_bgr = cv2.resize(mask_bgr, (expected_w, expected_h),
                                                  interpolation=cv2.INTER_LINEAR)
                    mask_np = mask_bgr.astype(np.float32) / 255.0
            except Exception:
                mask_np = None

        # Augment ALL samples (and only on the train split).
        if self.augment:
            input_np, target_np, mask_np = self._augment(input_np, target_np, mask_np)

        # Resize to target resolution if needed (preprocessed data usually matches).
        if self.input_resolution is not None:
            expected_w, expected_h = self.input_resolution
            if input_np.shape[:2] != (expected_h, expected_w):
                input_np = cv2.resize(input_np, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
                target_np = cv2.resize(target_np, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
                if mask_np is not None:
                    mask_np = cv2.resize(mask_np, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)

        input_tensor = torch.from_numpy(input_np.transpose((2, 0, 1)).copy()).float()
        target_tensor = torch.from_numpy(target_np.transpose((2, 0, 1)).copy()).float()

        if mask_np is not None:
            if mask_np.ndim == 2:
                mask_tensor = torch.from_numpy(mask_np.copy()).unsqueeze(0).float()
            else:
                mask_tensor = torch.from_numpy(mask_np.transpose((2, 0, 1)).copy()).float()
        else:
            mask_tensor = torch.ones(1, input_tensor.shape[1], input_tensor.shape[2])

        return {
            "input": input_tensor,
            "target": target_tensor,
            "mask": mask_tensor,
            "name": name,
        }

    def _augment(self, input_img, target_img, mask):
        """Apply augmentation transforms (same recipe as ShadowRemovalDataset)."""
        # 1. Random horizontal flip
        if random.random() > 0.5:
            input_img = np.fliplr(input_img).copy()
            target_img = np.fliplr(target_img).copy()
            if mask is not None:
                mask = np.fliplr(mask).copy()

        # 2. Random vertical flip
        if random.random() > 0.5:
            input_img = np.flipud(input_img).copy()
            target_img = np.flipud(target_img).copy()
            if mask is not None:
                mask = np.flipud(mask).copy()

        # 3. Illumination variation (both input and target)
        if self.illumination_strength > 0:
            illum_factor = 1.0 + random.uniform(-self.illumination_strength, self.illumination_strength)
            input_img = np.clip(input_img * illum_factor, 0, 1)
            target_img = np.clip(target_img * illum_factor, 0, 1)

        # 4. Shadow color modification (only shadow regions, input only)
        if self.shadow_color_shift > 0 and mask is not None:
            shadow_mask = np.clip(mask, 0, 1)
            if shadow_mask.ndim == 3 and shadow_mask.shape[0] == 1:
                shadow_mask = shadow_mask[0]
            color_shift = np.random.uniform(
                -self.shadow_color_shift, self.shadow_color_shift, size=(1, 1, 3)
            ).astype(np.float32)
            # color_shift (1,1,3) * shadow_mask (H,W,1) -> (H,W,3), matches HWC input_img.
            input_img = input_img + color_shift * shadow_mask[:, :, None]
            input_img = np.clip(input_img, 0, 1)

        # 5. Small rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            h, w = input_img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            input_img = cv2.warpAffine(input_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            target_img = cv2.warpAffine(target_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        return input_img, target_img, mask


class MixedBatchSampler(Sampler):
    """
    Per-dataset shuffle-queue sampler.

    Each dataset has its own shuffled queue of indices. For every batch we pop
    `ratios[tag]` indices from each queue. When a queue empties it is reshuffled
    and refilled, so there is no repetition within a dataset until it is
    exhausted. An epoch is one full pass through the largest dataset.
    """

    def __init__(
        self,
        dataset: MixedShadowRemovalDataset,
        ratios: Optional[Dict[str, int]] = None,
        batch_size: int = 16,
        generator: Optional[torch.Generator] = None,
        allowed_indices: Optional[List[int]] = None,
    ):
        self.dataset = dataset
        self.ratios = dict(ratios or DEFAULT_RATIOS)
        self.batch_size = batch_size
        self.generator = generator if generator is not None else torch.Generator()

        # Validate ratios sum to batch_size
        total = sum(self.ratios.values())
        if total != batch_size:
            raise ValueError(
                f"Ratios {self.ratios} sum to {total}, but batch_size is {batch_size}. "
                f"They must be equal."
            )

        # Build per-tag index pools from the dataset's tag grouping.
        # If allowed_indices is given (e.g. train-only indices), restrict pools to it.
        allowed = set(allowed_indices) if allowed_indices is not None else None
        self.tag_pools: Dict[str, List[int]] = {}
        for tag, idxs in dataset.tag_to_indices.items():
            if tag in self.ratios:
                pool = idxs if allowed is None else [i for i in idxs if i in allowed]
                self.tag_pools[tag] = list(pool)

        # Missing tags (dataset doesn't contain a configured tag) -> empty pool.
        for tag in self.ratios:
            self.tag_pools.setdefault(tag, [])

        # Per-tag shuffle queues (refilled lazily).
        self._queues: Dict[str, List[int]] = {tag: [] for tag in self.ratios}

        # Epoch length = full pass through the largest dataset.
        self._epoch_len = max(
            (len(self.tag_pools[tag]) // self.ratios[tag]) * self.batch_size
            for tag in self.ratios
            if self.ratios[tag] > 0 and len(self.tag_pools[tag]) > 0
        )

    def __len__(self) -> int:
        return self._epoch_len // self.batch_size

    def _refill(self, tag: str) -> None:
        """Reshuffle and refill the queue for a tag."""
        pool = self.tag_pools[tag]
        if not pool:
            return
        indices = list(pool)
        if self.generator is not None:
            perm = torch.randperm(len(indices), generator=self.generator).tolist()
            indices = [indices[i] for i in perm]
        else:
            random.shuffle(indices)
        self._queues[tag] = indices

    def _pop(self, tag: str, n: int) -> List[int]:
        """Pop n indices from a tag's queue, refilling as needed."""
        out = []
        while len(out) < n:
            if not self._queues[tag]:
                self._refill(tag)
            if not self._queues[tag]:
                # Empty pool for this tag; nothing to draw.
                break
            out.append(self._queues[tag].pop())
        return out

    def __iter__(self):
        # Reset queues at the start of each epoch.
        self._queues = {tag: [] for tag in self.ratios}

        for _ in range(len(self)):
            batch = []
            for tag, count in self.ratios.items():
                batch.extend(self._pop(tag, count))
            # Shuffle within the batch so dataset order isn't predictable.
            if self.generator is not None:
                perm = torch.randperm(len(batch), generator=self.generator).tolist()
                batch = [batch[i] for i in perm]
            else:
                random.shuffle(batch)
            yield batch


def create_mixed_dataloaders(
    root_dir: str,
    input_resolution: Tuple[int, int] = (192, 256),
    batch_size: int = 16,
    num_workers: int = 0,
    val_split: float = 0.1,
    ratios: Optional[Dict[str, int]] = None,
    augment: bool = True,
    illumination_strength: float = 0.1,
    shadow_color_shift: float = 0.05,
    rotation_range: float = 0.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders for mixed-dataset Stage 1 training.

    Returns:
        (train_loader, val_loader). No test_loader is produced (the merged
        folder is flat and evaluation is a separate post-training step).
    """
    ratios = dict(ratios or DEFAULT_RATIOS)

    full_dataset = MixedShadowRemovalDataset(
        root_dir=root_dir,
        split="train",
        input_resolution=input_resolution,
        augment=augment,
        illumination_strength=illumination_strength,
        shadow_color_shift=shadow_color_shift,
        rotation_range=rotation_range,
    )

    # Carve a val slice from each dataset (same ratios).
    val_indices = []
    train_indices = []
    for tag, idxs in full_dataset.tag_to_indices.items():
        n_val = int(len(idxs) * val_split)
        if n_val < 1 and len(idxs) > 0:
            n_val = 1  # ensure every dataset contributes at least one val sample
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(idxs), generator=gen).tolist()
        val_indices.extend([idxs[i] for i in perm[:n_val]])
        train_indices.extend([idxs[i] for i in perm[n_val:]])

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Disable augmentation for validation (A-OSR val samples load raw).
    full_dataset.augment = False

    use_persistent = num_workers > 0

    train_sampler = MixedBatchSampler(
        dataset=full_dataset,
        ratios=ratios,
        batch_size=batch_size,
        generator=torch.Generator().manual_seed(seed),
        allowed_indices=train_indices,
    )

    train_loader = DataLoader(
        full_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    counts = full_dataset.get_tag_counts()
    print(f"[Mixed DataLoader] Train: {len(train_indices)}, Val: {len(val_indices)}")
    print(f"[Mixed DataLoader] Per-dataset counts: {counts}")
    print(f"[Mixed DataLoader] Batch composition: {ratios} (sum={sum(ratios.values())})")
    print(f"[Mixed DataLoader] Batches/epoch: {len(train_loader)}")

    return train_loader, val_loader
