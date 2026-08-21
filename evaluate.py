import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
import cv2
import yaml
from tqdm import tqdm

from src.models.ioanet import IOANet
from src.models.laplacian_refiner import LPIOANet
from src.data.dataset import ShadowRemovalDataset, HighResolutionDataset
from torch.utils.data import DataLoader
from src.utils.losses import MetricsCalculator, RegionBasedMetrics


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda", stage: int = 1):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        device: Target device
        stage: 1 for IOANet, 2 for LPIOANet

    Returns:
        Tuple of (model, config)
    """

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if stage == 2:
        # Load Stage 1 IOANet (frozen) + Stage 2 LPIOANet
        stage1_config = config["model"]["stage1"]
        stage2_config = config["model"]["stage2"]

        ioanet = IOANet(
            in_channels=stage1_config["input_channels"],
            out_channels=stage1_config["output_channels"],
            pretrained=False
        )

        # Load Stage 1 checkpoint (needed to build the frozen IOANet inside LPIOANet)
        stage1_ckpt = config["model"]["stage1"].get("checkpoint")
        if not stage1_ckpt or not Path(stage1_ckpt).exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found: {stage1_ckpt}. "
                f"Stage 2 evaluation requires the Stage 1 model to build LPIOANet."
            )
        ckpt1 = torch.load(stage1_ckpt, map_location=device, weights_only=False)
        if isinstance(ckpt1, dict) and 'model_state_dict' in ckpt1:
            ioanet.load_state_dict(ckpt1['model_state_dict'])
        elif isinstance(ckpt1, dict) and 'state_dict' in ckpt1:
            ioanet.load_state_dict(ckpt1['state_dict'])
        else:
            ioanet.load_state_dict(ckpt1)
        ioanet = ioanet.to(device)
        ioanet.eval()

        model = LPIOANet(
            ioanet_model=ioanet,
            base_channels=stage2_config.get("base_channels", 16),
            num_levels=stage2_config.get("num_levels", 3),
            refine_blocks=stage2_config.get("refine_blocks", 3)
        )

        # Load Stage 2 checkpoint (the refiner weights)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()

        return model, config

    # Stage 1: IOANet
    stage1_config = config["model"]["stage1"]

    model = IOANet(
        in_channels=stage1_config["input_channels"],
        out_channels=stage1_config["output_channels"],
        pretrained=False
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    return model, config


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cuda"
) -> dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained IOANet model
        test_loader: Test data loader
        device: Target device
    
    Returns:
        Dict with metrics: mae, psnr, ssim (overall/shadow/non-shadow)
    """
    model.eval()
    
    metrics_sum = defaultdict(float)
    region_metrics_sum = defaultdict(float)
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Stage 1: Simple input/target
            input_img = batch["input"].to(device)
            target_img = batch["target"].to(device)
            mask = batch["mask"].to(device)
            
            output = model(input_img)
            
            # Overall metrics
            metrics = MetricsCalculator.compute_all(output, target_img)
            for k, v in metrics.items():
                metrics_sum[k] += v * output.size(0)
            
            # Region-based metrics
            region_metrics = RegionBasedMetrics.compute(output, target_img, mask)
            for k, v in region_metrics.items():
                region_metrics_sum[k] += v * output.size(0)
            
            num_samples += output.size(0)
    
    # Average
    results = {}
    for k, v in metrics_sum.items():
        results[k] = v / num_samples
    for k, v in region_metrics_sum.items():
        results[k] = v / num_samples
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate shadow removal model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--stage", type=int, default=1, help="Stage (1 for IOANet, 2 for LPIOANet)")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate on (test/train/val)")
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OK] Device: {device}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print(f"[OK] Loading model from: {args.model}")
    model, _ = load_model(args.model, args.config, device, stage=args.stage)
    
    # Create test dataloader
    if args.stage == 2:
        resolution = tuple(config["data"]["input_resolution_stage2"])
        dataset_cls = HighResolutionDataset
    else:
        resolution = tuple(config["data"]["input_resolution"])
        dataset_cls = ShadowRemovalDataset
    
    test_dataset = dataset_cls(
        root_dir=config["dataset"]["root_dir"],
        split=args.split,
        input_resolution=resolution,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"[OK] Test samples: {len(test_dataset)}")
    print(f"[OK] Resolution: {resolution}")
    print("-" * 60)
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # Print results (matching paper format)
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS")
    print("=" * 60)
    
    print("\n[Overall Metrics]")
    print(f"  MAE:  {results['mae']:.4f}")
    print(f"  PSNR: {results['psnr']:.2f} dB")
    print(f"  SSIM: {results['ssim']:.4f}")
    
    print("\n[Region-Based Metrics]")
    print(f"  MAE (Overall):    {results['overall_mae']:.4f}")
    print(f"  MAE (Shadow):     {results['shadow_mae']:.4f}")
    print(f"  MAE (Non-Shadow): {results['non_shadow_mae']:.4f}")
    
    # Paper-style format (MAE: overall/non-shadow/shadow)
    print("\n[Paper Format (MAE: overall/non-shadow/shadow)]")
    mae_str = f"{results['overall_mae']*100:.4f} / {results['non_shadow_mae']*100:.4f} / {results['shadow_mae']*100:.4f}"
    print(f"  MAE: {mae_str}")
    print(f"  PSNR: {results['psnr']:.2f}")
    
    print("\n" + "=" * 60)
    
    # Target comparison
    print("\n[Target Comparison]")
    if results['mae'] < 0.02:
        print("  [OK] MAE < 0.02 (EXCELLENT)")
    elif results['mae'] < 0.05:
        print("  [~] MAE < 0.05 (GOOD)")
    else:
        print("  [X] MAE > 0.05 (NEEDS IMPROVEMENT)")
    
    if results['psnr'] > 28:
        print("  [OK] PSNR > 28 dB (HIGH QUALITY)")
    elif results['psnr'] > 25:
        print("  [~] PSNR > 25 dB (ACCEPTABLE)")
    else:
        print("  [X] PSNR < 25 dB (LOW QUALITY)")


if __name__ == "__main__":
    main()
