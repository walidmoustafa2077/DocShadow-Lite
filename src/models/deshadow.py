"""IOANet variant for the NexaDoc corpus.

Keeps the existing MobileNetV2 IOANet backbone (per the chosen baseline), but
enforces the documented [0, 1] output contract.  The original ``IOANet`` is
left untouched because its tests explicitly assert an unclamped output.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn

from .ioanet import IOANet


class DeshadowNet(IOANet):
    """IOANet with a clamped image-space output."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        pretrained: bool = True,
        clamp_output: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            pretrained=pretrained,
        )
        self.clamp_output = bool(clamp_output)

    def forward(
        self, x: torch.Tensor, return_debug: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        if return_debug:
            output, debug = super().forward(x, return_debug=True)
            if self.clamp_output:
                output = output.clamp(0.0, 1.0)
            return output, debug

        output = super().forward(x, return_debug=False)
        if self.clamp_output:
            output = output.clamp(0.0, 1.0)
        return output
