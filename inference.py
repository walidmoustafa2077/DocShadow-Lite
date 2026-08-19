import argparse
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import yaml

from src.models.ioanet import IOANet


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


def process_image_stage1(
    model: torch.nn.Module,
    image_path: str,
    resolution: tuple = None,
    device: str = "cuda"
) -> tuple:
    """
    Process a single image through Stage 1 IOANet.
    
    Returns:
        (input_np, output_np, inference_time_ms)
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (W, H)
    
    # Resize if resolution specified
    if resolution is not None:
        w, h = resolution  # Config format is [W, H]
        img = img.resize((w, h), Image.BILINEAR)  # PIL expects (W, H)
    
    # Convert to tensor
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        output = model(img_tensor)
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Convert back to numpy
    input_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    output_np = output[0].permute(1, 2, 0).cpu().numpy()
    output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
    
    return input_np, output_np, inference_time




def main():
    parser = argparse.ArgumentParser(description="Run shadow removal inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default="outputs/inference", help="Output directory")
    parser.add_argument("--dataset-root", type=str, default=None, help="Dataset root (for loading target/mask)")
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OK] Device: {device}")
    
    # Load model
    print(f"[OK] Loading model from: {args.model}")
    model, config = load_model(args.model, args.config, device)
    
    # Resolution
    resolution = tuple(config["data"]["input_resolution"])
    print(f"[OK] Resolution: {resolution}")
    
    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = list(input_path.glob("*.[jp][pn][g]")) + list(input_path.glob("*.jpeg"))
    
    # Limit to 50 images
    image_files = image_files[:50]
    
    # Determine dataset structure for target/mask
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    if dataset_root is None and input_path.is_dir():
        if input_path.parent.name in ["input", "test", "train"]:
            if input_path.parent.parent.name in ["test", "train"]:
                dataset_root = input_path.parent.parent.parent
            else:
                dataset_root = input_path.parent.parent
    
    print(f"[OK] Processing {len(image_files)} images...")
    if dataset_root:
        print(f"[OK] Dataset root: {dataset_root}")
    print("-" * 60)
    
    total_time = 0
    for img_path in image_files:
        input_np, output_np, inference_time = process_image_stage1(
            model, str(img_path), resolution, device
        )
        
        total_time += inference_time
        
        # Try to load target and mask if available
        target_np = None
        mask_np = None
        
        if dataset_root:
            if "test" in str(img_path).lower():
                split = "test"
            else:
                split = "train"
            
            target_path = dataset_root / split / "target" / img_path.name
            mask_path = dataset_root / split / "mask" / img_path.name
            
            if target_path.exists():
                target_img = cv2.imread(str(target_path))
                if target_img is not None:
                    target_np = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                    # Resize to match output resolution
                    if target_np.shape[:2] != output_np.shape[:2]:
                        target_np = cv2.resize(target_np, (output_np.shape[1], output_np.shape[0]))
            
            if mask_path.exists():
                mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    mask_np = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
                    if mask_np.shape[:2] != output_np.shape[:2]:
                        mask_np = cv2.resize(mask_np, (output_np.shape[1], output_np.shape[0]))
        
        # Create comparison image: Input | Mask | Target | Output
        panels = [input_np]
        labels = ["Input"]
        if mask_np is not None:
            panels.append(mask_np)
            labels.append("Mask")
        if target_np is not None:
            panels.append(target_np)
            labels.append("Target")
        panels.append(output_np)
        labels.append("Output")
        
        comparison = np.concatenate(panels, axis=1)
        
        # Add labels
        h, w = output_np.shape[:2]
        panel_width = w
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        
        for i, label in enumerate(labels):
            x = i * panel_width + 10
            y = 20
            cv2.putText(comparison, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Save comparison
        output_path = output_dir / f"{img_path.stem}_comparison.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
        # Also save just the output
        output_only_path = output_dir / f"{img_path.stem}_output.png"
        cv2.imwrite(str(output_only_path), cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR))
        
        print(f"  {img_path.name} → {output_path.name} ({inference_time:.1f}ms)")
    
    print("-" * 60)
    avg_time = total_time / len(image_files) if image_files else 0
    if avg_time > 0:
        print(f"[OK] Average inference time: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)")
    print(f"[OK] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
