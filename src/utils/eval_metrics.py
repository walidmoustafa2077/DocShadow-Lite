"""Dataset-level metrics with correct aggregation.

The legacy ``MetricsCalculator.compute_all`` averages per-batch PSNR/SSIM,
which makes results depend on batch size.  These accumulators sum pixel errors
and per-image SSIM, then divide once, so PSNR is identical for batch size 1
and 128.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.distributed as dist

from .losses import MetricsCalculator


class MetricAccumulator:
    def __init__(self):
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.ssim_sum = 0.0
        self.numel = 0
        self.images = 0

    @torch.no_grad()
    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        output = output.detach().float()
        target = target.detach().float()
        diff = output - target
        self.abs_sum += diff.abs().sum().item()
        self.sq_sum += diff.pow(2).sum().item()
        self.numel += diff.numel()
        for i in range(output.size(0)):
            self.ssim_sum += MetricsCalculator.compute_ssim(
                output[i : i + 1], target[i : i + 1]
            )
        self.images += output.size(0)

    def reduce(self, device: torch.device | None = None) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        values = torch.tensor(
            [self.abs_sum, self.sq_sum, self.ssim_sum, self.numel, self.images],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        self.abs_sum, self.sq_sum, self.ssim_sum, self.numel, self.images = (
            values.tolist()
        )

    def compute(self) -> Dict[str, float]:
        if self.numel == 0:
            return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "psnr": 0.0, "ssim": 0.0}
        mae = self.abs_sum / self.numel
        mse = self.sq_sum / self.numel
        psnr = 100.0 if mse < 1e-10 else 10.0 * torch.log10(
            torch.tensor(1.0 / mse)
        ).item()
        ssim = self.ssim_sum / max(self.images, 1)
        return {
            "mae": mae,
            "mse": mse,
            "rmse": mse ** 0.5,
            "psnr": psnr,
            "ssim": ssim,
        }


class RegionAccumulator:
    """Overall / shadow / non-shadow errors, normalized per pixel (not 3x)."""

    def __init__(self):
        self.abs_all = 0.0
        self.sq_all = 0.0
        self.n_all = 0
        self.abs_shadow = 0.0
        self.sq_shadow = 0.0
        self.n_shadow = 0
        self.abs_clean = 0.0
        self.sq_clean = 0.0
        self.n_clean = 0

    @torch.no_grad()
    def update(
        self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> None:
        output = output.detach().float()
        target = target.detach().float()
        diff = (output - target).abs()
        sq = (output - target) ** 2

        mask = (mask.detach().float() > 0.5).expand_as(diff)
        clean = ~mask

        self.abs_all += diff.sum().item()
        self.sq_all += sq.sum().item()
        self.n_all += diff.numel()

        self.abs_shadow += (diff * mask).sum().item()
        self.sq_shadow += (sq * mask).sum().item()
        self.n_shadow += int(mask.sum().item())

        self.abs_clean += (diff * clean).sum().item()
        self.sq_clean += (sq * clean).sum().item()
        self.n_clean += int(clean.sum().item())

    def reduce(self, device: torch.device | None = None) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        values = torch.tensor(
            [
                self.abs_all,
                self.sq_all,
                self.n_all,
                self.abs_shadow,
                self.sq_shadow,
                self.n_shadow,
                self.abs_clean,
                self.sq_clean,
                self.n_clean,
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        (
            self.abs_all,
            self.sq_all,
            self.n_all,
            self.abs_shadow,
            self.sq_shadow,
            self.n_shadow,
            self.abs_clean,
            self.sq_clean,
            self.n_clean,
        ) = values.tolist()

    @staticmethod
    def _psnr(mse: float) -> float:
        if mse <= 1e-10:
            return 100.0
        return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()

    def compute(self) -> Dict[str, float]:
        all_mae = self.abs_all / self.n_all if self.n_all else 0.0
        all_mse = self.sq_all / self.n_all if self.n_all else 0.0
        shadow_mae = (
            self.abs_shadow / self.n_shadow if self.n_shadow else 0.0
        )
        shadow_mse = (
            self.sq_shadow / self.n_shadow if self.n_shadow else 0.0
        )
        clean_mae = self.abs_clean / self.n_clean if self.n_clean else 0.0
        clean_mse = self.sq_clean / self.n_clean if self.n_clean else 0.0
        return {
            "overall_mae": all_mae,
            "overall_psnr": self._psnr(all_mse),
            "shadow_mae": shadow_mae,
            "shadow_psnr": self._psnr(shadow_mse),
            "nonshadow_mae": clean_mae,
            "nonshadow_psnr": self._psnr(clean_mse),
        }
