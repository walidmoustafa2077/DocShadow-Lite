# DocShadow-Lite

A PyTorch-based document shadow removal system featuring Input-Output Attention Networks (IOANet) for Stage 1 and Laplacian Pyramid Refinement (LP-IOANet) for Stage 2 processing. Single-stage or two-stage shadow-to-shadow-free image translation on scanned documents with on-the-fly resolution handling.

## Overview

DocShadow-Lite provides flexible document shadow removal:
- **Stage 1 (IOANet)**: 192×256 resolution lightweight shadow removal
- **Stage 2 (LP-IOANet)**: 768×1024 high-resolution refinement with Laplacian pyramid

Both stages support flexible input resizing—images are automatically resized to target resolution during training, eliminating the need for separate pre-processed datasets per stage.

### Key Features

- **Dual-Stage Architecture**: IOANet (Stage 1) + LP-IOANet (Stage 2) for progressive refinement
- **Flexible Resolution Handling**: On-the-fly resizing supports any input resolution
- **Single root_dir**: Both stages use same dataset root with automatic multi-scale pyramid generation
- **Lightweight**: MobileNetV2 backbone (~3.10M parameters)
- **Efficient Training**: Fast data loading with on-the-fly downsampling for Stage 2
- **Comprehensive Metrics**: MAE, PSNR, SSIM with region-based analysis (shadow vs non-shadow)
- **Dataset Support**: Kligler and SynDoc datasets (any resolution supported)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd DocShadow-Lite

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# or source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- torch>=2.0.0
- torchvision>=0.15.0
- numpy>=1.21.0
- opencv-python>=4.5.0
- PyYAML>=5.4.0
- tqdm>=4.60.0
- Pillow>=8.0.0
- lpips>=0.1.4
- tensorboard>=2.10.0

## Quick Start

### Training

```bash
# Train Stage 1 (IOANet at 192×256)
python train.py --config configs/config.yaml --stage 1

# Train Stage 2 (LP-IOANet at 768×1024)
python train.py --config configs/config.yaml --stage 2
```

Configuration is controlled via `configs/config.yaml`:
- Single `dataset.root_dir` for all stages (automatic resizing on-the-fly)
- Model hyperparameters (channels, refinement blocks)
- Training settings (epochs, batch size, learning rate)
- Loss weights
- Augmentation parameters
- Stage 1: expects 192×256, resizes if needed
- Stage 2: loads full resolution, downsamples on-the-fly to create pyramid

### Evaluation

```bash
# Evaluate on test set
python evaluate.py --model outputs/checkpoints/stage1/best_model.pth --config configs/config.yaml
```

### Inference

```bash
# Run inference on single/batch images (Stage 1)
python inference.py --input path/to/image.jpg --model outputs/checkpoints/stage1/best_model.pth --config configs/config.yaml
```

### Handling Different Input Resolutions

No separate preprocessing needed! The system automatically handles on-the-fly resizing:

```bash
# Stage 1: Any resolution → resized to 192×256
# Stage 2: Any resolution → upsampled to 768×1024 via Laplacian pyramid

# Optional: Pre-cache dataset to 768×1024 for faster Stage 2 training (one-time)
python scripts/preprocess_dataset.py --input Data/RawDataset --output Data/Cached_Dataset --size 768 1024
```

## Project Structure

```
├── train.py                      # Main training script (Stage 1 & 2)
├── evaluate.py                   # Evaluation on test set
├── inference.py                  # Single/batch image inference
├── configs/
│   └── config.yaml              # Central configuration file
├── src/
│   ├── models/
│   │   ├── ioanet.py            # IOANet architecture (Stage 1)
│   │   └── laplacian_refiner.py # Laplacian refiner (Stage 2)
│   ├── data/
│   │   └── dataset.py           # Dataset loaders
│   └── utils/
│       └── losses.py            # Loss functions & metrics
├── scripts/
│   └── preprocess_dataset.py    # Dataset preprocessing utility
├── Data/                        # Dataset storage
|    └── SynDoc_Wild/                 # Output dataset
|       ├── train/
|       │   ├── input/               # Shadowed + degraded images
|       │   ├── target/              # Clean ground truth
|       │   └── mask/                # Binary shadow masks
|       └── test/
|           └── input/, target/, mask/
└── outputs/                     # Training outputs (auto-generated)
    ├── checkpoints/             # Model checkpoints
    ├── logs/                    # TensorBoard logs
    └── samples/                 # Debug visualizations
```

## Architecture Details

### Two-Stage Pipeline

#### Stage 1: IOANet (192×256)

Paper-aligned implementation (arXiv 2303.12862):

1. **Input Attention (LRA)**: Coordinate Attention on the raw input, executed as a **parallel branch** (concurrent with the backbone)
2. **Backbone**: MobileNetV2 encoder + Feature Boosting (FB) decoder, running on the **raw input**
3. **Decoder**: Progressive upsampling with skip connections (1×1 "boost" convolutions)
4. **Output Attention (LDRA)**: Coordinate Attention on the 3-channel residual
5. **Long Residual Summation**: `output = LDRA(R(x)) + LRA(x)` (no clamping, no mask gating)

#### Stage 2: LP-IOANet (768×1024)

- **HR-Only Loading**: Loads full resolution images
- **On-the-Fly Pyramid**: Creates 192×256 (low) → 384×512 (mid) → 768×1024 (high) pyramid
- **Frozen IOANet**: Stage 1 model is frozen (`requires_grad=False`, `eval()`)
- **Laplacian Refinement**: A single depthwise-separable refinement block at 384×512 predicts a mask, which is upsampled and applied at 768×1024
- **Benefits**: No folder structure mismatches, perfect scale relationships, efficient GPU usage (~1.47 GFLOPs)

### Loss Function

**Stage 1 (IOANet)**:
- Combined L1 + LPIPS loss: `10 × L1 + 5 × LPIPS`
- LPIPS uses a frozen AlexNet backbone

**Stage 2 (LP-IOANet)**:
- L1 loss only (no LPIPS)

### Metrics

- **MAE** (target < 0.02): Mean absolute error
- **PSNR** (target > 28 dB): Peak signal-to-noise ratio
- **SSIM** (target > 0.95): Structural similarity
- Region-based analysis for shadow vs non-shadow areas

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
dataset:
  root_dir: "Data/your_dataset"
  num_workers: 4

data:
  # Stage 1 resolution (IOANet input)
  input_resolution: [192, 256]      # [W, H]

  # Stage 2 resolution (LP-IOANet output)
  input_resolution_stage2: [768, 1024]  # [W, H]

  augmentation:
    enabled: true
    illumination_strength: 0.1
    shadow_color_shift: 0.05
    rotation_range: 0

model:
  stage1:
    input_channels: 3
    output_channels: 3
    pretrained: true
    checkpoint: "outputs/checkpoints/stage1/best_model.pth"

  stage2:
    base_channels: 16
    num_levels: 3
    refine_blocks: 2

training:
  stage1:
    epochs: 1000
    batch_size: 32
    learning_rate: 0.0002
    losses:
      l1_weight: 10.0
      lpips_weight: 5.0

  stage2:
    epochs: 200
    batch_size: 16
    learning_rate: 0.0002
    losses:
      l1_weight: 1.0
```

### Dataset Structure

Both Stage 1 and Stage 2 use the same directory structure:

```
Data/YourDataset/
├── train/
│   ├── input/      # Shadow images (any resolution)
│   ├── target/     # Shadow-free images (same resolution as input)
│   └── mask/       # Shadow masks (optional)
└── test/
    ├── input/
    ├── target/
    └── mask/
```

## Training Tips

- **Single dataset**: Use one `root_dir` for both stages (automatic on-the-fly resizing)
- **No preprocessing needed**: System handles resolution mismatch automatically
- **Optional optimization**: Pre-cache dataset to 768×1024 for 10-15% Stage 2 speedup
- **Early stopping**: Training halts after 50 epochs (Stage 1) / 30 epochs (Stage 2) without validation improvement
- **Validation frequency**: Validation runs every 5 epochs (Stage 1) / 2 epochs (Stage 2)
- **Debug samples**: Side-by-side visualizations saved to `outputs/samples/stage{N}/debug/`
- **Memory**: Stage 2 uses batch_size=16 (config default) for 768×1024

## Results

Expected metric ranges from domain expertise:
- MAE: < 0.02 (shadows nearly invisible)
- PSNR: > 28 dB (high visual quality)
- SSIM: > 0.95 (structural preservation)

## Development

### Extending Loss Function

1. Add new loss term in `src/utils/losses.py`
2. Add weight parameter to `configs/config.yaml`
3. Update logging in `train.py`

### Adding New Metrics

1. Add static method to `MetricsCalculator` in `src/utils/losses.py`
2. Update `compute_all()` to call new metric
3. Log results in `evaluate.py` or `train.py`
