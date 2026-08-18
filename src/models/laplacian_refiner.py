"""
Laplacian Pyramid Refinement Network (LPTN-Lite)

Implements Stage 2 of DocShadow-Lite as described in Plan.md Phase 2-5.

Architecture:
- Depthwise Separable Convolutions for efficiency
- Two refinement blocks (384×512 and 768×1024)
- Mask prediction via attention mechanisms
- Pyramid-based frequency band refinement

Key Constraint: 1.47 GFLOPs (vs 22.8 in full version)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List


# =============================================================================
# Depthwise Separable Convolution Block
# =============================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution: [3×3 Depthwise] → [1×1 Pointwise]
    
    This is the core building block for LPTN-Lite, reducing parameters and
    computations while maintaining representational capacity.
    
    Flow: [Depthwise 3×3] → [BN] → [Leaky ReLU] → [Pointwise 1×1] → [BN]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()
        
        # Depthwise convolution: groups = in_channels
        # Each group handles one input channel independently
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # Pointwise convolution: 1×1 to expand/contract channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise path
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        
        # Pointwise path
        x = self.pointwise(x)
        x = self.bn2(x)
        return x


# =============================================================================
# Refinement Block (Mask Prediction Network)
# =============================================================================

class RefinementBlock(nn.Module):
    """
    Refinement Block for a single pyramid level.
    
    Purpose: Predict a spatial mask M_i that modulates the high-frequency
    residual L_i. The mask acts as a "gain control" to brighten text edges
    trapped in shadows.
    
    Architecture:
    1. Depthwise Separable Conv blocks to process residual + upsampled low-res
    2. Output: Sigmoid-activated mask (values in [0, 1])
    
    Input: Concatenation of:
        - High-frequency residual L_i (from pyramid decomposition)
        - Upsampled low-res output (from IOANet or previous level)
    Output: Mask M_i (same spatial dimensions, 1 channel)
    """
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        num_blocks: int = 2,
        mask_scale: float = 1.0
    ):
        """
        Args:
            in_channels: Input channels (typically 6 = 3 for residual + 3 for upsampled low-res)
            base_channels: Number of internal channels for depthwise separable blocks
            num_blocks: Number of depthwise separable blocks to stack
            mask_scale: Scaling factor for mask output. Default 1.0 → mask ∈ [0,1].
                       Set to 2.0 to enable contrast amplification → mask ∈ [0,2].
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.mask_scale = mask_scale
        
        # Normalize mixed-range concatenated input (residuals [-1,1] + upsampled low-res [0,1])
        # This stabilizes BatchNorm in depthwise separable blocks
        self.input_norm = nn.BatchNorm2d(in_channels)
        
        # Initial projection to base_channels
        self.proj_in = DepthwiseSeparableConv(
            in_channels, base_channels,
            kernel_size=3, stride=1, padding=1
        )
        
        # Stack of depthwise separable blocks
        self.blocks = nn.ModuleList([
            DepthwiseSeparableConv(
                base_channels, base_channels,
                kernel_size=3, stride=1, padding=1
            )
            for _ in range(num_blocks)
        ])
        
        # Output projection to 1 channel (mask)
        # Final sigmoid activation produces values in [0, 1], scaled by mask_scale
        self.proj_out = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        # Apply mask_scale to shift sigmoid range [0,1] → [0, mask_scale]
        # mask_scale=1.0: suppress only (standard Sigmoid)
        # mask_scale=2.0: allow both suppression and amplification (contrast restoration)
        self.register_buffer('mask_scale_factor', torch.tensor(mask_scale))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Concatenated input (B, in_channels, H, W)
        Returns:
            Mask M_i (B, 1, H, W) with values in [0, 1] (or [0, mask_scale] if scaled)
        """
        # Normalize mixed-range concatenation (residuals [-1,1] + upsampled guidance [0,1])
        x = self.input_norm(x)
        
        # Project input to base_channels
        x = self.proj_in(x)
        
        # Stack of blocks with residual connections where possible
        for block in self.blocks:
            x = block(x) + x  # Residual connection
        
        # Project to mask (single channel, sigmoid)
        mask = self.proj_out(x)
        
        # Apply scaling: shift sigmoid range [0,1] → [0, mask_scale]
        # Allows optional contrast amplification beyond standard suppression
        mask = mask * self.mask_scale_factor
        
        return mask


# =============================================================================
# Laplacian Pyramid Decomposition & Recomposition
# =============================================================================

class LaplacianDecomposer(nn.Module):
    """
    Decompose an image into Laplacian pyramid levels.
    
    Given an image at resolution H×W, creates three pyramid levels:
    - Level 2 (Low): H/4 × W/4 (downsampled from Level 1)
    - Level 1 (Mid): H/2 × W/2 (downsampled from Level 0)
    - Level 0 (High): H × W (original)
    
    Residuals (Laplacian coefficients):
    - L_0 = I_0 - Upsample(I_1)
    - L_1 = I_1 - Upsample(I_2)
    - L_2 = I_2 (low-frequency base)
    """
    
    @staticmethod
    def decompose(image: torch.Tensor, num_levels: int = 3) -> Dict[str, torch.Tensor]:
        """
        Decompose image into Laplacian pyramid.
        
        Args:
            image: Input image (B, 3, H, W)
            num_levels: Number of pyramid levels (typically 3)
        
        Returns:
            Dictionary with:
                - 'levels': List of images at each resolution [I_0, I_1, I_2]
                - 'residuals': List of Laplacian coefficients [L_0, L_1, L_2]
                - 'shapes': List of spatial shapes at each level
        """
        levels = []
        shapes = []
        current = image
        
        # Build pyramid by successive downsampling
        for level in range(num_levels):
            levels.append(current)
            shapes.append(current.shape[-2:])
            
            if level < num_levels - 1:
                # Downsample for next level (using 2x average pooling)
                current = F.avg_pool2d(current, kernel_size=2, stride=2)
        
        # Compute Laplacian residuals: L_i = I_i - Upsample(I_{i+1})
        residuals = []
        for i in range(num_levels - 1):
            # Upsample level i+1 to match level i's spatial dimensions
            # NOTE: Using 'nearest' for the reconstruction path (paper alignment).
            # LP-IOANet Figure 1 uses Nearest x2 upsampling for the reconstruction
            # path to preserve high-frequency text edges.
            upsampled = F.interpolate(
                levels[i + 1],
                size=levels[i].shape[-2:],
                mode='nearest'
            )
            # Compute residual
            residual = levels[i] - upsampled
            residuals.append(residual)
        
        # Lowest level is just the image (not a residual)
        residuals.append(levels[-1])
        
        return {
            'levels': levels,
            'residuals': residuals,
            'shapes': shapes
        }
    
    @staticmethod
    def recompose(refined_residuals: List[torch.Tensor], num_levels: int = 3) -> torch.Tensor:
        """
        Reconstruct image from refined Laplacian pyramid.
        
        Given refined residuals [L'_0, L'_1, L'_2], reconstruct via:
        I'_0 = L'_0 + Upsample(I'_1)
        I'_1 = L'_1 + Upsample(I'_2)
        I'_2 = L'_2
        
        Args:
            refined_residuals: List of refined Laplacian coefficients [L'_0, L'_1, L'_2]
            num_levels: Number of pyramid levels
        
        Returns:
            Reconstructed image at full resolution (same as L'_0)
        """
        # Start from the lowest level
        current = refined_residuals[-1]  # L'_2
        
        # Progressively upsample and add residuals (bottom-up)
        for i in range(num_levels - 2, -1, -1):
            # Upsample current to match residual i's spatial dimensions
            # Using 'nearest' (paper alignment): LP-IOANet reconstruction path
            # uses Nearest x2 upsampling to preserve high-frequency text edges.
            upsampled = F.interpolate(
                current,
                size=refined_residuals[i].shape[-2:],
                mode='nearest'
            )
            # Add residual
            current = refined_residuals[i] + upsampled
        
        return current


# =============================================================================
# Laplacian Refiner (Stage 2 Main Module)
# =============================================================================

class LaplacianRefiner(nn.Module):
    """
    Laplacian Pyramid Refinement Network (LPTN-Lite).
    
    Takes the low-resolution output from frozen IOANet and uses it to guide
    high-resolution refinement via Laplacian pyramid decomposition.
    
    Architecture:
    1. Decompose high-res input into Laplacian pyramid (3 levels)
    2. Refine ONLY the intermediate level (384×512) with a RefinementBlock:
       - Concatenate residual with upsampled lower-res output
       - Pass through RefinementBlock to predict mask
       - Apply mask to residual: L'_i = L_i ⊗ M_i
    3. The high-res level (768×1024) residual is passed through unchanged
    4. Recompose refined pyramid into high-res output
    
    Paper alignment (arXiv 2303.12862): "The residual refinement network
    operates on the intermediate resolution of (384×512)." The 768×1024 output
    is obtained by upsampling the refined 384×512 result, NOT by a separate
    refinement at 768×1024.
    
    Flow for 768×1024 input:
    - Level 2 (Low): 192×256 (fed to frozen IOANet)
    - Level 1 (Mid): 384×512 (refine with RefinementBlock)
    - Level 0 (High): 768×1024 (residual passed through unchanged)
    """
    
    def __init__(
        self,
        base_channels: int = 32,
        num_levels: int = 3,
        refine_blocks: int = 2,
        mask_scale: float = 1.0
    ):
        """
        Args:
            base_channels: Base channel count for refinement blocks (budget: 32 or 16)
            num_levels: Number of pyramid levels (typically 3)
            refine_blocks: Number of depthwise separable blocks per refinement network
            mask_scale: Mask scaling factor. Default 1.0 (suppress only). Set to 2.0 for contrast boost.
        """
        super().__init__()
        
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.mask_scale = mask_scale
        self.decomposer = LaplacianDecomposer()
        
        # Paper alignment: the residual refinement network operates ONLY at the
        # intermediate resolution (384×512). So we create exactly ONE refinement
        # block (for the intermediate level), not one per level.
        self.refine_blocks = nn.ModuleList([
            RefinementBlock(
                in_channels=6,
                base_channels=base_channels,
                num_blocks=refine_blocks,
                mask_scale=mask_scale  # Pass mask_scale to each refinement block
            )
        ])
    
    def forward(
        self,
        high_res_input: torch.Tensor,
        low_res_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Refine high-resolution image using low-resolution IOANet output.
        
        Args:
            high_res_input: Input image at 768×1024 (B, 3, H, W)
            low_res_output: Shadow-free output from IOANet at 192×256 (B, 3, H/4, W/4)
        
        Returns:
            Refined high-res output at 768×1024 (B, 3, H, W)
        """
        # =====================================================================
        # Step 1: Decompose high-res input into Laplacian pyramid
        # =====================================================================
        pyramid = self.decomposer.decompose(high_res_input, num_levels=self.num_levels)
        residuals = pyramid['residuals']  # [L_0, L_1, L_2]
        
        # =====================================================================
        # Step 2: Refine residuals using frozen IOANet output
        # =====================================================================
        # Paper alignment: the residual refinement network operates ONLY at the
        # intermediate resolution (384×512). The high-res level (768×1024)
        # residual is passed through unchanged.
        #
        # Pyramid levels (for 768×1024 input):
        #   Level 2 (Low):  192×256  -> IOANet output (I'_2)
        #   Level 1 (Mid):  384×512  -> refined with RefinementBlock (I'_1)
        #   Level 0 (High): 768×1024 -> residual passed through unchanged (I'_0)
        
        # Start from IOANet's low-res output (lowest level, unchanged)
        current_reconstruction = low_res_output  # I'_2 at 192×256
        
        # Refine the intermediate level (Level 1, 384×512)
        mid_level = self.num_levels - 2  # index of the intermediate level
        
        # Upsample previous reconstruction to match the intermediate level
        upsampled_guidance = F.interpolate(
            current_reconstruction,
            size=residuals[mid_level].shape[-2:],
            mode='nearest'
        )
        
        # Concatenate residual and upsampled guidance
        # Input to refiner: (B, 6, H_i, W_i)
        concat_input = torch.cat(
            [residuals[mid_level], upsampled_guidance],
            dim=1
        )
        
        # Predict mask for the intermediate level using RefinementBlock
        # mask is in [0, 1] (sigmoid output)
        mask = self.refine_blocks[0](concat_input)  # (B, 1, H_i, W_i)
        
        # Apply mask to residual (element-wise multiplication)
        refined_mid_residual = residuals[mid_level] * mask
        
        # Reconstruct the intermediate level
        mid_reconstruction = upsampled_guidance + refined_mid_residual  # I'_1 at 384×512
        
        # =====================================================================
        # Step 3: Recompose to high-res output
        # =====================================================================
        # The high-res level (Level 0, 768×1024) residual is passed through
        # unchanged (no refinement at 768×1024, per paper).
        upsampled_high = F.interpolate(
            mid_reconstruction,
            size=residuals[0].shape[-2:],
            mode='nearest'
        )
        output = residuals[0] + upsampled_high  # I'_0 at 768×1024
        
        # CRITICAL FIX: Clamp output to valid image range [0, 1]
        # Without this, numerical instability from uninitialized residuals can cause
        # output ranges like [-6.6, 8.4] which breaks gradient flow and loss computation.
        # This is expected behavior with random weights in first few batches.
        output = torch.clamp(output, 0.0, 1.0)
        
        return output


# =============================================================================
# LP-IOANet: Full Stage 2 Model
# =============================================================================

class LPIOANet(nn.Module):
    """
    Complete Stage 2 model: Frozen IOANet + Laplacian Pyramid Refiner.
    
    This module combines:
    1. Frozen IOANet (loaded from Stage 1 checkpoint)
    2. LaplacianRefiner (trainable, processes high-res image)
    
    Forward pass:
    1. Downsample input to 192×256
    2. Pass through frozen IOANet → low-res shadow-free output
    3. Pass high-res input + low-res output to LaplacianRefiner
    4. Return high-res refined output
    """
    
    def __init__(
        self,
        ioanet_model: nn.Module,
        base_channels: int = 32,
        num_levels: int = 3,
        refine_blocks: int = 2,
        mask_scale: float = 1.0
    ):
        """
        Args:
            ioanet_model: Trained IOANet module (will be frozen)
            base_channels: Base channel count for refinement blocks
            num_levels: Number of pyramid levels
            refine_blocks: Number of depthwise separable blocks per refinement network
            mask_scale: Mask scaling factor (1.0=suppress only, 2.0=allow contrast boost)
        """
        super().__init__()
        
        self.ioanet = ioanet_model
        self.refiner = LaplacianRefiner(
            base_channels=base_channels,
            num_levels=num_levels,
            refine_blocks=refine_blocks,
            mask_scale=mask_scale  # Risk 2 fix: Pass mask_scale to LaplacianRefiner
        )
        
        # Freeze IOANet parameters
        for param in self.ioanet.parameters():
            param.requires_grad = False
        
        # Ensure IOANet is in eval mode (fixes batch norm stats)
        self.ioanet.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for full Stage 2 model.
        
        Args:
            x: Input image at 768×1024 (B, 3, H, W)
        
        Returns:
            Refined high-res output at 768×1024 (B, 3, H, W)
        """
        # Get original dimensions
        _, _, h, w = x.shape
        
        # Downsample to 192×256 for IOANet
        # Standard: 768/4 = 192, 1024/4 = 256
        target_low_res_h = h // 4
        target_low_res_w = w // 4
        
        x_low_res = F.interpolate(
            x,
            size=(target_low_res_h, target_low_res_w),
            mode='bilinear',
            align_corners=True
        )
        with torch.no_grad():
            low_res_output = self.ioanet(x_low_res)
        
        # Refine using Laplacian pyramid
        high_res_output = self.refiner(x, low_res_output)
        
        return high_res_output


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and non-trainable parameters.
    
    Returns:
        Dictionary with:
            - 'total': Total parameters
            - 'trainable': Trainable parameters
            - 'frozen': Non-trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen
    }
