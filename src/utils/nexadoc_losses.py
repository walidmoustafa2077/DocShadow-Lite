"""Losses for the NexaDoc training pipeline."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LPIPS(nn.Module):
    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = bool(enabled)
        if self.enabled:
            import lpips

            self.net = lpips.LPIPS(net="alex")
            for param in self.net.parameters():
                param.requires_grad = False
            self.net.eval()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return output.new_zeros(())
        # LPIPS expects [-1, 1].
        return self.net(output * 2.0 - 1.0, target * 2.0 - 1.0).mean()


class DeshadowLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        mask_weight: float = 2.0,
        lpips_weight: float = 0.1,
        use_lpips: bool = True,
    ):
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.mask_weight = float(mask_weight)
        self.lpips_weight = float(lpips_weight)
        self.lpips = _LPIPS(use_lpips and lpips_weight > 0)

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        l1 = F.l1_loss(output, target)

        if mask is None:
            masked_l1 = l1
        else:
            mask = (mask > 0.5).to(output.dtype)
            denom = mask.sum() * output.shape[1]
            masked_l1 = ((output - target).abs() * mask).sum() / denom.clamp_min(1.0)

        lpips_value = self.lpips(output, target)
        total = (
            self.l1_weight * l1
            + self.mask_weight * masked_l1
            + self.lpips_weight * lpips_value
        )
        return {
            "total": total,
            "l1": l1.detach(),
            "masked_l1": masked_l1.detach(),
            "lpips": lpips_value.detach(),
        }


class SRLoss(nn.Module):
    def __init__(self, l1_weight: float = 1.0, lpips_weight: float = 0.05):
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.lpips_weight = float(lpips_weight)
        self.lpips = _LPIPS(lpips_weight > 0)

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        l1 = F.l1_loss(output, target)
        lpips_value = self.lpips(output, target)
        total = self.l1_weight * l1 + self.lpips_weight * lpips_value
        return {
            "total": total,
            "l1": l1.detach(),
            "lpips": lpips_value.detach(),
        }
