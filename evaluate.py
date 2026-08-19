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
from src.data.dataset import ShadowRemovalDataset
from torch.utils.data import DataLoader
from src.utils.losses import MetricsCalculator, RegionBasedMetrics


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """
    Load trained IOANet model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        device: Target device
    
    Returns:
        Tuple of (model, config)
    """
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load IOANet
    stage1_config = config["model"]["stage1"]
    
    model = IOANet(
        in_channels=stage1_config["input_channels"],
        out_channels=stage1_config["output_channels"],
        pretrained=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OK] Device: {device}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print(f"[OK] Loading model from: {args.model}")
    model, _ = load_model(args.model, args.config, device)
    
    # Create test dataloader
    resolution = tuple(config["data"]["input_resolution"])
    test_dataset = ShadowRemovalDataset(
        root_dir=config["dataset"]["root_dir"],
        split="test",
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
