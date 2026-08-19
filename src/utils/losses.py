"""
Loss Functions for Shadow Removal

Implementation based on Plan.md Phase E:
- L1 Loss: Pixel-level accuracy
- LPIPS (VGG-16): Perceptual quality
- Total Loss: 10 × L1 + 5 × LPIPS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import lpips


# =============================================================================
# Phase E: Shadow Removal Loss Function
# =============================================================================

class ShadowRemovalLoss(nn.Module):
    """
    Loss function for shadow removal.
    
    Implements Equation 1 from Plan.md:
        Loss = 10 × L1 + 5 × LPIPS
    
    Where:
        - L1: Mean Absolute Error (pixel-level accuracy)
        - LPIPS: Learned Perceptual Image Patch Similarity (VGG-16 based)
    """
    
    def __init__(
        self,
        l1_weight: float = 10.0,
        lpips_weight: float = 5.0,
        device: str = "cuda"
    ):
        """
        Args:
            l1_weight: Weight for L1 loss (default: 10.0)
            lpips_weight: Weight for LPIPS perceptual loss (default: 5.0)
            device: Device for LPIPS model
        """
        super().__init__()
        
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        
        # Initialize LPIPS with AlexNet backbone (paper alignment)
        # The paper cites LPIPS [20]; the standard/default backbone is AlexNet.
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        
        # Freeze LPIPS parameters (we don't train it)
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
        
        self.lpips_fn.eval()
    
    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            output: Predicted shadow-free image (B, 3, H, W) in [0, 1]
            target: Ground truth shadow-free image (B, 3, H, W) in [0, 1]

        Returns:
            Dictionary with keys:
                - 'total': Total weighted loss
                - 'l1': L1 loss value
                - 'lpips': LPIPS loss value
        """
        # =====================================================================
        # L1 Loss Calculation (paper alignment: plain L1, no shadow weighting)
        # =====================================================================
        # The paper uses plain L1 loss (Equation 1). No shadow-aware weighting.
        l1_loss = F.l1_loss(output, target)
        
        # =====================================================================
        # LPIPS Loss Calculation (Perceptual Quality)
        # =====================================================================
        # LPIPS expects images in range [-1, 1], so we need to normalize
        output_normalized = output * 2.0 - 1.0
        target_normalized = target * 2.0 - 1.0
        
        # LPIPS returns (B, 1, 1, 1), we take mean across batch
        lpips_loss = self.lpips_fn(output_normalized, target_normalized).mean()
        
        # =====================================================================
        # Total Objective (Equation 1 from Plan.md Phase E - Enhanced)
        # =====================================================================
        total_loss = self.l1_weight * l1_loss + self.lpips_weight * lpips_loss
        
        return {
            'total': total_loss,
            'l1': l1_loss,
            'lpips': lpips_loss
        }



# =============================================================================
# Metrics Calculator
# =============================================================================

class MetricsCalculator:
    """Calculate evaluation metrics for shadow removal."""
    
    @staticmethod
    def compute_ssim(output: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
        """
        Compute SSIM (Structural Similarity Index).
        
        Args:
            output: Predicted image (B, 3, H, W) in [0, 1]
            target: Ground truth image (B, 3, H, W) in [0, 1]
            window_size: Size of the gaussian window (default: 11)
        
        Returns:
            SSIM value in [0, 1] (higher is better)
        """
        with torch.no_grad():
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            # Create gaussian window
            def gaussian_window(size, sigma=1.5):
                coords = torch.arange(size, dtype=torch.float32) - size // 2
                g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
                g = g / g.sum()
                return g.unsqueeze(0) * g.unsqueeze(1)
            
            window = gaussian_window(window_size).to(output.device)
            window = window.unsqueeze(0).unsqueeze(0)
            window = window.expand(output.size(1), 1, window_size, window_size)
            
            # Compute means
            mu1 = F.conv2d(output, window, padding=window_size // 2, groups=output.size(1))
            mu2 = F.conv2d(target, window, padding=window_size // 2, groups=target.size(1))
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            # Compute variances and covariance
            sigma1_sq = F.conv2d(output * output, window, padding=window_size // 2, groups=output.size(1)) - mu1_sq
            sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=target.size(1)) - mu2_sq
            sigma12 = F.conv2d(output * target, window, padding=window_size // 2, groups=output.size(1)) - mu1_mu2
            
            # Compute SSIM
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return ssim_map.mean().item()
    
    @staticmethod
    def compute_all(output: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            output: Predicted image (B, 3, H, W) in [0, 1]
            target: Ground truth image (B, 3, H, W) in [0, 1]
        
        Returns:
            Dictionary with keys:
                - 'mae': Mean Absolute Error
                - 'mse': Mean Squared Error
                - 'psnr': Peak Signal-to-Noise Ratio (dB)
                - 'rmse': Root Mean Squared Error
                - 'ssim': Structural Similarity Index
        """
        with torch.no_grad():
            # MAE (Mean Absolute Error)
            mae = F.l1_loss(output, target).item()
            
            # MSE (Mean Squared Error)
            mse_tensor = F.mse_loss(output, target)
            mse = mse_tensor.item()
            
            # RMSE (Root Mean Squared Error)
            rmse = torch.sqrt(mse_tensor).item()
            
            # PSNR (Peak Signal-to-Noise Ratio)
            # PSNR = 10 * log10(MAX^2 / MSE)
            # For images in [0, 1], MAX = 1.0
            if mse < 1e-10:
                psnr = 100.0  # Perfect reconstruction
            else:
                psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item()
            
            # SSIM (Structural Similarity Index)
            ssim = MetricsCalculator.compute_ssim(output, target)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'psnr': psnr,
            'ssim': ssim
        }


class RegionBasedMetrics:
    """Compute metrics for different regions: overall, shadow, non-shadow."""
    
    @staticmethod
    def compute(
        output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        threshold: float = 0.3
    ) -> Dict[str, float]:
        """
        Compute region-based metrics.
        
        Args:
            output: Predicted image (B, 3, H, W) in [0, 1]
            target: Ground truth image (B, 3, H, W) in [0, 1]
            mask: Shadow mask (B, 1, H, W) in [0, 1]
            threshold: Threshold to binarize mask
        
        Returns:
            Dictionary with region-specific metrics
        """
        with torch.no_grad():
            # Binarize mask
            binary_mask = (mask > threshold).float()
            
            # Shadow region metrics
            shadow_pixels = binary_mask.sum()
            if shadow_pixels > 0:
                shadow_mae = (torch.abs(output - target) * binary_mask).sum() / shadow_pixels
                shadow_mse = ((output - target) ** 2 * binary_mask).sum() / shadow_pixels
                shadow_psnr = 10 * torch.log10(1.0 / (shadow_mse + 1e-10))
            else:
                shadow_mae = 0.0
                shadow_psnr = 100.0
            
            # Non-shadow region metrics
            non_shadow_mask = 1.0 - binary_mask
            non_shadow_pixels = non_shadow_mask.sum()
            if non_shadow_pixels > 0:
                non_shadow_mae = (torch.abs(output - target) * non_shadow_mask).sum() / non_shadow_pixels
                non_shadow_mse = ((output - target) ** 2 * non_shadow_mask).sum() / non_shadow_pixels
                non_shadow_psnr = 10 * torch.log10(1.0 / (non_shadow_mse + 1e-10))
            else:
                non_shadow_mae = 0.0
                non_shadow_psnr = 100.0
            
            # Overall metrics
            overall_mae = torch.abs(output - target).mean()
            overall_mse = ((output - target) ** 2).mean()
            overall_psnr = 10 * torch.log10(1.0 / (overall_mse + 1e-10))
        
        return {
            'overall_mae': overall_mae.item(),
            'overall_psnr': overall_psnr.item(),
            'shadow_mae': shadow_mae.item() if isinstance(shadow_mae, torch.Tensor) else shadow_mae,
            'shadow_psnr': shadow_psnr.item() if isinstance(shadow_psnr, torch.Tensor) else shadow_psnr,
            'non_shadow_mae': non_shadow_mae.item() if isinstance(non_shadow_mae, torch.Tensor) else non_shadow_mae,
            'non_shadow_psnr': non_shadow_psnr.item() if isinstance(non_shadow_psnr, torch.Tensor) else non_shadow_psnr,
        }


