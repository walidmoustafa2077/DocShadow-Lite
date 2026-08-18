"""
IOANet for Document Shadow Removal

Implementation based on Plan.md Phase B, C, D:
- Phase B: Coordinate Attention Module (Ref [12])
- Phase C: MobileNetV2 Encoder + FB-Decoder
- Phase D: IOA Forward Pass (Input Modulation + Backbone + Output Modulation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import torchvision.models as models


# =============================================================================
# Phase B: Coordinate Attention Module (Ref [12])
# =============================================================================

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module (Ref [12])
    
    Captures long-range horizontal and vertical illumination gradients
    characteristic of document shadows.
    
    Steps:
    1. Squeeze (Spatial Encoding): Global Average Pooling along Width and Height
    2. Excitation (Feature Interaction): Concatenate, Conv, Activation, Split
    3. Recalibration: Conv to restore channels, Sigmoid, Multiply with input
    """
    
    def __init__(self, in_channels: int, reduction: int = 4):
        """
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio for bottleneck (default: 4 for complex shadow shapes)
        """
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Intermediate channel count (bottleneck)
        # Lower reduction (4 vs 16) = higher capacity for wrinkled shadow patterns
        hidden_channels = max(in_channels, 16)        

        # Excitation: Shared convolution for feature interaction
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.SiLU(inplace=True)  # Swish activation
        
        # Recalibration: Separate convolutions for H and W attention
        self.conv_h = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map (B, C, H, W)
        Returns:
            Attention-modulated feature map (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # =====================================================================
        # Step 1: Squeeze (Spatial Encoding)
        # =====================================================================
        # Pool along width to get (B, C, H, 1)
        x_h = F.adaptive_avg_pool2d(x, (H, 1))
        # Pool along height to get (B, C, 1, W)
        x_w = F.adaptive_avg_pool2d(x, (1, W))
        
        # =====================================================================
        # Step 2: Excitation (Feature Interaction)
        # =====================================================================
        # Concatenate along width: (B, C, H, 1) + (B, C, 1, W) → (B, C, H, W+1)
        # We need to transpose x_w to concat along height dimension
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, W, 1)
        
        # Concatenate: (B, C, H, 1) cat (B, C, W, 1) → (B, C, H+W, 1)
        y = torch.cat([x_h, x_w], dim=2)
        
        # Apply 1x1 conv to reduce channels (feature interaction)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split back into H and W components
        y_h, y_w = torch.split(y, [H, W], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)  # Transpose back: (B, hidden, 1, W)
        
        # =====================================================================
        # Step 3: Recalibration
        # =====================================================================
        # Apply 1x1 conv to restore original channel depth
        # Plain sigmoid (paper alignment): no gain factor, matches Coordinate Attention source
        a_h = torch.sigmoid(self.conv_h(y_h))  # (B, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(y_w))  # (B, C, 1, W)
        
        # Multiply original feature map with attention weights
        out = x * a_h * a_w
        
        return out


# =============================================================================
# Phase C: IOANet Backbone (MobileNetV2 Encoder + FB-Decoder)
# =============================================================================

class InvertedResidual(nn.Module):
    """Inverted Residual Block (MobileNetV2 building block)"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise linear projection
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2Encoder(nn.Module):
    """MobileNetV2-based Encoder (Ref [11])"""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Extract feature layers (encoder blocks)
        # MobileNetV2 structure: features[0-18] contains all layers
        self.conv1 = mobilenet.features[0:2]   # Output: 16 channels, stride 2
        self.conv2 = mobilenet.features[2:4]   # Output: 24 channels, stride 2
        self.conv3 = mobilenet.features[4:7]   # Output: 32 channels, stride 2
        self.conv4 = mobilenet.features[7:14]  # Output: 96 channels, stride 2
        self.conv5 = mobilenet.features[14:18] # Output: 320 channels, stride 2
        
        # Paper alignment: KEEP BatchNorm in the pretrained MobileNetV2 encoder.
        # Converting to GroupNorm would discard the ImageNet-pretrained running
        # statistics and change the normalization behavior the weights were
        # trained under, undermining the pretraining benefit.
        
        # Channel dimensions for skip connections
        self.out_channels = [16, 24, 32, 96, 320]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: Input image (B, 3, H, W)
        Returns:
            Tuple of feature maps at different scales (skip connections)
        """
        c1 = self.conv1(x)    # 1/2
        c2 = self.conv2(c1)   # 1/4
        c3 = self.conv3(c2)   # 1/8
        c4 = self.conv4(c3)   # 1/16
        c5 = self.conv5(c4)   # 1/32
        
        return c1, c2, c3, c4, c5


class FBDecoder(nn.Module):
    """
    Feature Boosting Decoder (FB-Decoder)
    
    Uses depthwise separable convolutions for upsampling to minimize GFLOPs.
    Applies 1x1 convolutions to align skip connection channels (boosting).
    """
    
    def __init__(self, encoder_channels: list):
        super().__init__()
        
        # Decoder channel progression (going from bottleneck to output)
        decoder_channels = [256, 128, 64, 32, 16]
        
        # Feature boosting: 1x1 convs to align encoder skip connections
        self.boost5 = nn.Conv2d(encoder_channels[4], decoder_channels[0], 1)  # 320 -> 256
        self.boost4 = nn.Conv2d(encoder_channels[3], decoder_channels[1], 1)  # 96 -> 128
        self.boost3 = nn.Conv2d(encoder_channels[2], decoder_channels[2], 1)  # 32 -> 64
        self.boost2 = nn.Conv2d(encoder_channels[1], decoder_channels[3], 1)  # 24 -> 32
        self.boost1 = nn.Conv2d(encoder_channels[0], decoder_channels[4], 1)  # 16 -> 16
        
        # Upsampling blocks (depthwise separable convolutions)
        self.up5 = self._make_upsample_block(decoder_channels[0], decoder_channels[1])  # 256 -> 128
        self.up4 = self._make_upsample_block(decoder_channels[1] * 2, decoder_channels[2])  # 256 -> 64
        self.up3 = self._make_upsample_block(decoder_channels[2] * 2, decoder_channels[3])  # 128 -> 32
        self.up2 = self._make_upsample_block(decoder_channels[3] * 2, decoder_channels[4])  # 64 -> 16
        self.up1 = self._make_upsample_block(decoder_channels[4] * 2, decoder_channels[4])  # 32 -> 16
        
        # Final prediction head: 3-channel residual map
        self.final = nn.Sequential(
            nn.Conv2d(decoder_channels[4], decoder_channels[4], 3, 1, 1),
            nn.BatchNorm2d(decoder_channels[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[4], 3, 3, 1, 1),  # Output: unbounded residual correction
            # No activation - allow unbounded residuals (positive & negative) for shadow/color correction
        )
    
    def _make_upsample_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create upsampling block using depthwise separable convolution"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
    
    def forward(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, 
                c4: torch.Tensor, c5: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c1, c2, c3, c4, c5: Encoder feature maps (skip connections)
        Returns:
            3-channel residual map (B, 3, H, W)
        """
        # Boost encoder features (align channels)
        c1_boost = self.boost1(c1)
        c2_boost = self.boost2(c2)
        c3_boost = self.boost3(c3)
        c4_boost = self.boost4(c4)
        c5_boost = self.boost5(c5)
        
        # Decoder with skip connections
        # Paper alignment: Nearest-Neighbor x2 upsampling preserves text edges
        x = self.up5(F.interpolate(c5_boost, scale_factor=2, mode='nearest'))
        x = torch.cat([x, c4_boost], dim=1)
        
        x = self.up4(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = torch.cat([x, c3_boost], dim=1)
        
        x = self.up3(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = torch.cat([x, c2_boost], dim=1)
        
        x = self.up2(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = torch.cat([x, c1_boost], dim=1)
        
        x = self.up1(F.interpolate(x, scale_factor=2, mode='nearest'))
        
        # Final prediction head
        residual = self.final(x)
        
        return residual


# =============================================================================
# Phase D: IOANet Assembly (Input-Output Attention Network)
# =============================================================================

class IOANet(nn.Module):
    """
    IOANet: Input-Output Attention Network for Document Shadow Removal.
    
    Architecture (paper alignment, arXiv 2303.12862):
    1. Input Attention (LRA): Coordinate Attention on input image (parallel branch)
    2. Backbone Execution: MobileNetV2 Encoder + FB-Decoder on RAW input
    3. Output Attention (LDRA): Coordinate Attention on the 3-channel residual
    4. Long Residual Summation: I_out = LDRA(R(x)) + LRA(x)
    """
    
    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 3,
        pretrained: bool = True,
        width_mult: float = 1.0
    ):
        super().__init__()
        
        # Input Attention (LRA): Coordinate Attention on input (parallel branch)
        self.input_attention = CoordinateAttention(in_channels=3, reduction=4)
        
        # Backbone: MobileNetV2 Encoder + FB-Decoder
        self.encoder = MobileNetV2Encoder(pretrained=pretrained)
        self.decoder = FBDecoder(encoder_channels=self.encoder.out_channels)
        
        # Output Attention (LDRA): Coordinate Attention on the 3-channel residual
        # Paper alignment: I_out = LDRA(R(x)) + LRA(x)
        # LDRA (output attention) operates in image space on the 3-channel residual
        self.output_attention = CoordinateAttention(in_channels=3, reduction=4)
        
        # Phase F: Store intermediate outputs for debugging
        self.debug_outputs = {}
    
    def forward(self, x: torch.Tensor, return_debug: bool = False) -> torch.Tensor:
        """
        Forward pass implementing the IOA mechanism (paper alignment).
        
        Args:
            x: Input shadow image (B, 3, H, W) in [0, 1]
            return_debug: If True, return debug outputs (LRA, LDRA, Residual)
        Returns:
            Shadow-free image (B, 3, H, W) in [0, 1]
            If return_debug=True: Tuple[output, debug_dict]
        """
        # =====================================================================
        # Step 1: Input Attention (LRA) - Parallel Branch
        # =====================================================================
        # Paper alignment: input attention (LRA) is executed CONCURRENTLY with
        # the backbone, NOT fed into it. Non-shadow areas are copied through
        # the long residual connection.
        x_attended = self.input_attention(x)
        
        # Phase F: Store M_in for visualization
        if return_debug:
            # Visualize attention by computing difference
            self.debug_outputs['input_attention_map'] = (x_attended - x).abs().mean(dim=1, keepdim=True)
        
        # =====================================================================
        # Step 2: Backbone Execution (Encoder-Decoder) on RAW input
        # =====================================================================
        # Paper alignment: backbone operates on the raw input x (parallel to LRA)
        c1, c2, c3, c4, c5 = self.encoder(x)
        
        # Decoder: Reconstruct shadow correction signal (residual)
        residual = self.decoder(c1, c2, c3, c4, c5)
        
        # Phase F: Store residual for visualization
        if return_debug:
            self.debug_outputs['residual'] = residual.clone()
        
        # =====================================================================
        # Step 3: Output Attention (LDRA) + Long Residual Summation
        # =====================================================================
        # Paper alignment: I_out = LDRA(R(x)) + LRA(x)
        # Output attention (LDRA) is applied to the 3-channel residual in image
        # space, then summed with the input-attention branch (LRA).
        output = self.output_attention(residual) + x_attended
        
        if return_debug:
            self.debug_outputs['output_attention_map'] = (self.output_attention(residual) - residual).abs().mean(dim=1, keepdim=True)
        
        if return_debug:
            return output, self.debug_outputs
        
        return output


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_info(model: nn.Module, name: str = "Model") -> str:
    """Get formatted model info string."""
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    return f"{name}:\n  Total: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M"


def compute_gflops(model: nn.Module, input_size: tuple = (1, 3, 192, 256)) -> float:
    """
    Compute GFLOPs for the model.
    
    Phase F: Mobile Compatibility Check
    Target: < 1.47 GFLOPs (as per Plan.md)
    
    Args:
        model: The model to analyze
        input_size: Input tensor size (B, C, H, W)
    
    Returns:
        GFLOPs (Giga Floating Point Operations)
    """
    try:
        from thop import profile
        import torch
        
        model.eval()
        input_tensor = torch.randn(input_size)
        
        # Move to same device as model
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        gflops = flops / 1e9
        
        return gflops
    except ImportError:
        print("[Warning] thop not installed. Cannot compute GFLOPs. Install with: pip install thop")
        return -1.0
    except Exception as e:
        print(f"[Warning] Failed to compute GFLOPs: {e}")
        return -1.0


def validate_model_efficiency(model: nn.Module, target_gflops: float = 1.47) -> Dict[str, bool]:
    """
    Phase F: Engineering Sanity Checks
    
    Validates:
    1. Mobile Compatibility: GFLOPs < target
    2. Parameter Count: Should be low (MobileNetV2 is lightweight)
    
    Args:
        model: The IOANet model
        target_gflops: Maximum allowed GFLOPs (default: 1.47)
    
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    # Check parameter count
    total_params = count_parameters(model, trainable_only=False)
    results['param_count_ok'] = total_params < 10e6  # Less than 10M parameters
    results['total_params'] = total_params
    
    # Check GFLOPs
    gflops = compute_gflops(model)
    if gflops > 0:
        results['gflops_ok'] = gflops < target_gflops
        results['gflops'] = gflops
    else:
        results['gflops_ok'] = None  # Could not compute
        results['gflops'] = -1.0
    
    return results

