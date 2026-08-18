from pathlib import Path
from typing import Tuple, Optional, Dict
import random
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2


# =============================================================================
# Stage 1: Low-Resolution Dataset (192×256)
# =============================================================================

class ShadowRemovalDataset(Dataset):
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        input_resolution: Optional[Tuple[int, int]] = None,
        augment: bool = True,
        illumination_strength: float = 0.2,
        shadow_color_shift: float = 0.1,
        rotation_range: float = 5.0,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.input_resolution = input_resolution
        self.augment = augment and (split == "train")
        self.illumination_strength = illumination_strength
        self.shadow_color_shift = shadow_color_shift
        self.rotation_range = rotation_range
        
        # Determine data folder
        data_folder = "train" if split in ["train", "val"] else "test"
        
        self.input_dir = self.root_dir / data_folder / "input"
        self.target_dir = self.root_dir / data_folder / "target"
        self.mask_dir = self.root_dir / data_folder / "mask"
        
        # Validate directories exist
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")
        
        # Get image list
        self.image_names = sorted([
            f.name for f in self.input_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])
        
        if not self.image_names:
            raise ValueError(f"No images found in {self.input_dir}")
        
        # Track resolution validation warnings
        self.mismatched_resolutions = {}
        self._validate_first_images()
    
    def __len__(self) -> int:
        return len(self.image_names)
    
    def _validate_first_images(self) -> None:
        """Check resolution of first 3 images and warn if mismatches detected."""
        if self.input_resolution is None:
            return
        
        # Config format: input_resolution = [W, H]
        expected_w, expected_h = self.input_resolution
        
        # Check first 3 images
        mismatch_count = 0
        for name in self.image_names[:3]:
            try:
                img = cv2.imread(str(self.input_dir / name))
                if img is not None:
                    h, w = img.shape[:2]
                    if (h, w) != (expected_h, expected_w):
                        mismatch_count += 1
                        self.mismatched_resolutions[name] = (w, h)
            except Exception:
                pass
        
        # Only log on first dataset creation (train split)
        if self.split == "train":
            print(f"\n{'='*80}")
            print(f"Dataset Resolution Validation (from config: input_resolution={self.input_resolution})")
            print(f"{'='*80}")
            print(f"  Loaded: {len(self.image_names)} images")
            print(f"  Expected: {expected_w}×{expected_h} (W×H)")
            print(f"  Split: {self.split}")
            
            if mismatch_count > 0:
                warnings.warn(
                    f"Found {mismatch_count}/3 sampled images with mismatched resolution. "
                    f"On-the-fly resizing will be applied automatically.",
                    UserWarning
                )
                print(f"  Status: MISMATCHES DETECTED ({mismatch_count}/3)")
                print(f"  Action: Applying on-the-fly resizing during training")
            else:
                print(f"  Status: All images match expected resolution ✓")
            print(f"{'='*80}\n")
    
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Dict with keys:
                - 'input': shadow image (3, H, W) in [0, 1]
                - 'target': shadow-free image (3, H, W) in [0, 1]
                - 'mask': shadow mask (1, H, W) in [0, 1], or ones if not available
                - 'name': filename
        """
        name = self.image_names[idx]
        
        # Load images directly with OpenCV (faster than PIL, less RAM)
        # cv2.imread returns BGR format by default
        try:
            input_bgr = cv2.imread(str(self.input_dir / name))
            target_bgr = cv2.imread(str(self.target_dir / name))
            
            if input_bgr is None or target_bgr is None:
                print(f"[WARNING] Failed to load {name}")
                return self.__getitem__((idx + 1) % len(self))
            
            # Validate and warn about resolution mismatches
            self._validate_image_resolution(name, input_bgr, target_bgr)
            
            # Convert BGR to RGB
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
                    # Validate mask resolution
                    if self.input_resolution is not None:
                        # Config format: input_resolution = [W, H]
                        expected_w, expected_h = self.input_resolution
                        mask_h, mask_w = mask_bgr.shape[:2]
                        if (mask_h, mask_w) != (expected_h, expected_w):
                            mask_bgr = cv2.resize(mask_bgr, (expected_w, expected_h), 
                                                interpolation=cv2.INTER_LINEAR)
                    mask_np = mask_bgr.astype(np.float32) / 255.0
            except Exception:
                mask_np = None
        
        # Apply augmentation
        if self.augment:
            input_np, target_np, mask_np = self._augment(input_np, target_np, mask_np)
        
        # Resize to target resolution (if images aren't pre-resized)
        if self.input_resolution is not None:
            # Config format: input_resolution = [W, H]
            expected_w, expected_h = self.input_resolution
            # Only resize if dimensions don't match (saves CPU if using pre-resized dataset)
            if input_np.shape[:2] != (expected_w, expected_h):
                input_np = cv2.resize(input_np, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
                target_np = cv2.resize(target_np, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
                if mask_np is not None:
                    mask_np = cv2.resize(mask_np, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensors (C, H, W)
        # Use .copy() to ensure contiguous memory (prevents PyTorch warnings)
        input_tensor = torch.from_numpy(input_np.transpose((2, 0, 1)).copy()).float()
        target_tensor = torch.from_numpy(target_np.transpose((2, 0, 1)).copy()).float()
        
        if mask_np is not None:
            if mask_np.ndim == 2:
                mask_tensor = torch.from_numpy(mask_np.copy()).unsqueeze(0).float()
            else:
                mask_tensor = torch.from_numpy(mask_np.transpose((2, 0, 1)).copy()).float()
        else:
            # Default mask (all ones)
            mask_tensor = torch.ones(1, input_tensor.shape[1], input_tensor.shape[2])
        
        return {
            "input": input_tensor,
            "target": target_tensor,
            "mask": mask_tensor,
            "name": name,
        }
    
    def _augment(
        self,
        input_img: np.ndarray,
        target_img: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Apply augmentation transforms (paper alignment: LP-IOANet + WACV 2023)."""
        
        # 1. Random horizontal flip (paper: WACV 2023 uses random horizontal flipping)
        if random.random() > 0.5:
            input_img = np.fliplr(input_img).copy()
            target_img = np.fliplr(target_img).copy()
            if mask is not None:
                mask = np.fliplr(mask).copy()
        
        # 2. Random vertical flip (extra robustness for document orientation)
        if random.random() > 0.5:
            input_img = np.flipud(input_img).copy()
            target_img = np.flipud(target_img).copy()
            if mask is not None:
                mask = np.flipud(mask).copy()
        
        # 3. Illumination variation (paper: apply illumination augmentation)
        #    Applied to both input and target to simulate lighting changes.
        if self.illumination_strength > 0:
            illum_factor = 1.0 + random.uniform(-self.illumination_strength, self.illumination_strength)
            input_img = np.clip(input_img * illum_factor, 0, 1)
            target_img = np.clip(target_img * illum_factor, 0, 1)
        
        # 4. Shadow color modification (paper: "we also modify the colour values of the shadows")
        #    Only the shadow regions (where mask > 0) get a color/illumination shift,
        #    making the model robust to varied shadow colors. Applied to input only.
        if self.shadow_color_shift > 0 and mask is not None:
            # Shadow mask in [0,1]; shift color of shadow regions
            shadow_mask = np.clip(mask, 0, 1)
            if shadow_mask.ndim == 3 and shadow_mask.shape[0] == 1:
                shadow_mask = shadow_mask[0]  # (H, W)
            # Per-channel color shift
            color_shift = np.random.uniform(
                -self.shadow_color_shift, self.shadow_color_shift, size=(3, 1, 1)
            ).astype(np.float32)
            # Apply shift only in shadow regions
            input_img = input_img + color_shift * shadow_mask[None, :, :]
            input_img = np.clip(input_img, 0, 1)
        
        # 5. Small rotation (document perspective robustness)
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            h, w = input_img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            input_img = cv2.warpAffine(input_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            target_img = cv2.warpAffine(target_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # 6. Random crop (paper: WACV 2023 resizes 286x286 then random crops to 256x256)
        #    Applied after rotation to simulate the paper's crop-based augmentation.
        if self.input_resolution is not None:
            expected_w, expected_h = self.input_resolution
            h, w = input_img.shape[:2]
            # Only crop if image is larger than target (crop to a random window)
            if h > expected_h and w > expected_w:
                top = random.randint(0, h - expected_h)
                left = random.randint(0, w - expected_w)
                input_img = input_img[top:top + expected_h, left:left + expected_w]
                target_img = target_img[top:top + expected_h, left:left + expected_w]
                if mask is not None:
                    mask = mask[top:top + expected_h, left:left + expected_w]
        
        return input_img, target_img, mask
    
    def _validate_image_resolution(self, name: str, input_bgr: np.ndarray, 
                                   target_bgr: np.ndarray) -> None:
        """Check and warn about resolution mismatches. Does NOT modify images."""
        if self.input_resolution is None:
            return
        
        # Config format: input_resolution = [W, H]
        expected_w, expected_h = self.input_resolution
        input_h, input_w = input_bgr.shape[:2]
        target_h, target_w = target_bgr.shape[:2]
        
        if (input_h, input_w) != (expected_h, expected_w):
            warnings.warn(
                f"Input '{name}' has resolution {input_w}×{input_h}, "
                f"expected {expected_w}×{expected_h}. On-the-fly resizing will be applied.",
                UserWarning
            )
        
        if (target_h, target_w) != (expected_h, expected_w):
            warnings.warn(
                f"Target '{name}' has resolution {target_w}×{target_h}, "
                f"expected {expected_w}×{expected_h}. On-the-fly resizing will be applied.",
                UserWarning
            )


def create_dataloaders(
    root_dir: str,
    input_resolution: Tuple[int, int] = (192, 256),
    batch_size: int = 16,
    num_workers: int = 0,
    val_split: float = 0.08,
    augment: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create standard dataloaders WITHOUT mixed pool sampling.
    
    Uses regular PyTorch DataLoader with random shuffling.
    Does NOT use sampler.py - this is a simple baseline loader.
    
    For mixed pool training (20% small + 80% large), use create_mixed_dataloaders() instead.
    
    Args:
        root_dir: Dataset root directory
        input_resolution: Target resolution [W, H]
        batch_size: Batch size for all loaders (default: 16)
        num_workers: Number of data loading workers
        val_split: Validation split ratio (default: 0.08 = 8%)
        augment: Enable augmentation for training
    
    Returns:
        (train_loader, val_loader, test_loader)
    
    Note:
        ⚠ This function does NOT use the mixed pool sampler from sampler.py
        To use sampler.py → use create_mixed_dataloaders() instead
    """
    
    full_dataset = ShadowRemovalDataset(
        root_dir=root_dir,
        split="train",
        input_resolution=input_resolution,
        augment=augment,
        **kwargs
    )
    
    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Disable augmentation for validation
    if hasattr(val_dataset.dataset, 'augment'):
        val_dataset.dataset.augment = False
    
    # Create test dataset
    test_dataset = ShadowRemovalDataset(
        root_dir=root_dir,
        split="test",
        input_resolution=input_resolution,
        augment=False,
        **kwargs
    )
    
    # Create dataloaders with optimized settings for large datasets
    # persistent_workers: Keeps workers alive between epochs (critical for Windows)
    # prefetch_factor: Number of batches to load in advance (2 = lower RAM)
    use_persistent = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=use_persistent,  # Keep workers alive (faster, less CPU spikes)
        prefetch_factor=2 if num_workers > 0 else None  # Lower RAM usage
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    print(f"[DataLoader] Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

# =============================================================================
# Stage 2: High-Resolution Dataset (768×1024)
# =============================================================================

class HighResolutionDataset(Dataset):
    """
    High-resolution dataset for Stage 2 (LPTN-Lite) training.
    
    Unlike Stage 1 which uses low-res 192×256 images, Stage 2 requires
    full high-resolution 768×1024 images for refinement.
    
    This dataset:
    1. Loads full-res images (768×1024)
    2. Applies augmentation at full resolution
    3. On-the-fly creates Laplacian pyramid levels (for validation purposes)
    4. Returns (high_res_input, high_res_target, low_res_hint)
    
    Note: Stage 2 trains exclusively on dataset (high-res triplets available)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        input_resolution: Optional[Tuple[int, int]] = None,
        augment: bool = True,
        illumination_strength: float = 0.15,
        shadow_color_shift: float = 0.08,
        rotation_range: float = 3.0,
    ):
        """
        Args:
            root_dir: Dataset root directory (should be for Stage 2)
            split: 'train', 'val', or 'test'
            input_resolution: Target resolution [W, H]. Default is [1024, 768] for Stage 2
            augment: Enable augmentation (only for train split)
            illumination_strength: Strength of illumination variation
            shadow_color_shift: Strength of shadow color shift
            rotation_range: Maximum rotation angle in degrees
        """
        self.root_dir = Path(root_dir)
        self.split = split
        
        # Stage 2 default: 768×1024 (H×W stored as [W, H])
        if input_resolution is None:
            input_resolution = (768, 1024)
        self.input_resolution = input_resolution
        
        self.augment = augment and (split == "train")
        self.illumination_strength = illumination_strength
        self.shadow_color_shift = shadow_color_shift
        self.rotation_range = rotation_range
        
        # Determine data folder
        data_folder = "train" if split in ["train", "val"] else "test"
        
        self.input_dir = self.root_dir / data_folder / "input"
        self.target_dir = self.root_dir / data_folder / "target"
        self.mask_dir = self.root_dir / data_folder / "mask"
        
        # Validate directories exist
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")
        
        # Get image list
        self.image_names = sorted([
            f.name for f in self.input_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])
        
        if not self.image_names:
            raise ValueError(f"No images found in {self.input_dir}")
    
    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a high-resolution sample.
        
        Returns:
            Dict with keys:
                - 'input': shadow image (3, H, W) in [0, 1]
                - 'target': shadow-free image (3, H, W) in [0, 1]
                - 'mask': shadow mask (1, H, W) in [0, 1], or ones if not available
                - 'name': filename
        """
        name = self.image_names[idx]
        
        try:
            input_bgr = cv2.imread(str(self.input_dir / name))
            target_bgr = cv2.imread(str(self.target_dir / name))
            
            if input_bgr is None or target_bgr is None:
                print(f"[WARNING] Failed to load {name}")
                return self.__getitem__((idx + 1) % len(self))
            
            # Convert BGR to RGB
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
                    # Resize mask to target resolution if needed
                    expected_w, expected_h = self.input_resolution
                    mask_h, mask_w = mask_bgr.shape[:2]
                    if (mask_h, mask_w) != (expected_h, expected_w):
                        mask_bgr = cv2.resize(mask_bgr, (expected_w, expected_h), 
                                            interpolation=cv2.INTER_LINEAR)
                    mask_np = mask_bgr.astype(np.float32) / 255.0
            except Exception:
                mask_np = None
        
        # Apply augmentation (at full resolution)
        if self.augment:
            input_np, target_np, mask_np = self._augment(input_np, target_np, mask_np)
        
        # Resize to target resolution (Stage 2: 1024×768)
        if self.input_resolution is not None:
            expected_w, expected_h = self.input_resolution
            # Check if resize needed (input_np is H×W, config is [W, H])
            if input_np.shape[:2] != (expected_h, expected_w):
                input_np = cv2.resize(input_np, (expected_w, expected_h), 
                                    interpolation=cv2.INTER_LINEAR)
                target_np = cv2.resize(target_np, (expected_w, expected_h), 
                                     interpolation=cv2.INTER_LINEAR)
                if mask_np is not None:
                    mask_np = cv2.resize(mask_np, (expected_w, expected_h), 
                                       interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensors (C, H, W)
        input_tensor = torch.from_numpy(input_np.transpose((2, 0, 1)).copy()).float()
        target_tensor = torch.from_numpy(target_np.transpose((2, 0, 1)).copy()).float()
        
        if mask_np is not None:
            if mask_np.ndim == 2:
                mask_tensor = torch.from_numpy(mask_np.copy()).unsqueeze(0).float()
            else:
                mask_tensor = torch.from_numpy(mask_np.transpose((2, 0, 1)).copy()).float()
        else:
            # Default mask (all ones)
            mask_tensor = torch.ones(1, input_tensor.shape[1], input_tensor.shape[2])
        
        return {
            "input": input_tensor,
            "target": target_tensor,
            "mask": mask_tensor,
            "name": name,
        }
    
    def _augment(
        self,
        input_img: np.ndarray,
        target_img: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Apply augmentation transforms at high resolution."""
        
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
        
        # 3. Illumination variation (apply to both input and target)
        if self.illumination_strength > 0:
            illum_factor = 1.0 + random.uniform(-self.illumination_strength, self.illumination_strength)
            input_img = np.clip(input_img * illum_factor, 0, 1)
            target_img = np.clip(target_img * illum_factor, 0, 1)
        
        # 4. Shadow color modification (paper: "we also modify the colour values of the shadows")
        #    Only the shadow regions (where mask > 0) get a color/illumination shift.
        if self.shadow_color_shift > 0 and mask is not None:
            shadow_mask = np.clip(mask, 0, 1)
            if shadow_mask.ndim == 3 and shadow_mask.shape[0] == 1:
                shadow_mask = shadow_mask[0]  # (H, W)
            color_shift = np.random.uniform(
                -self.shadow_color_shift, self.shadow_color_shift, size=(3, 1, 1)
            ).astype(np.float32)
            input_img = input_img + color_shift * shadow_mask[None, :, :]
            input_img = np.clip(input_img, 0, 1)
        
        # 5. Small rotation (subtle, to preserve text quality)
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            h, w = input_img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            input_img = cv2.warpAffine(input_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            target_img = cv2.warpAffine(target_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return input_img, target_img, mask


def create_stage2_dataloaders(
    root_dir: str,
    input_resolution: Tuple[int, int] = (768, 1024),
    batch_size: int = 4,
    num_workers: int = 0,
    val_split: float = 0.1,
    augment: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for Stage 2 training (high-resolution).
    
    Args:
        root_dir: Dataset root directory (high-res dataset)
        input_resolution: Target resolution [W, H] (default: [768, 1024] for 768×1024 images)
        batch_size: Batch size (typically 4 for high-res due to memory)
        num_workers: Number of data loading workers
        val_split: Validation split ratio (default 0.1 = 10%)
        augment: Enable augmentation for training
        **kwargs: Additional arguments for HighResolutionDataset
    
    Returns:
        (train_loader, val_loader, test_loader)
    
    Note:
        Batch size is typically 4 (or smaller) for 768×1024 images
        due to VRAM constraints. See Plan.md Phase 6 for memory management.
    """
    
    # Create full training dataset
    full_dataset = HighResolutionDataset(
        root_dir=root_dir,
        split="train",
        input_resolution=input_resolution,
        augment=augment,
        **kwargs
    )
    
    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Disable augmentation for validation
    if hasattr(val_dataset.dataset, 'augment'):
        val_dataset.dataset.augment = False
    
    # Create test dataset
    test_dataset = HighResolutionDataset(
        root_dir=root_dir,
        split="test",
        input_resolution=input_resolution,
        augment=False,
        **kwargs
    )
    
    # Create dataloaders
    use_persistent = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    print(f"[Stage 2 DataLoader] Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")
    print(f"[Stage 2 DataLoader] Resolution: {input_resolution[0]}×{input_resolution[1]} (W×H)")
    print(f"[Stage 2 DataLoader] Batch Size: {batch_size}")
    
    return train_loader, val_loader, test_loader