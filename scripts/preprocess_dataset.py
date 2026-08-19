import os
import argparse
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


def process_single_image(args):
    f, current_dest, target_size_cv, quality, type_dir = args
    try:
        img = cv2.imread(str(f))
        if img is None:
            return False, f"Failed to read: {f.name}"
        
        img_resized = cv2.resize(img, target_size_cv, interpolation=cv2.INTER_LINEAR)
        
        save_path = current_dest / f.name
        # Only pass JPEG quality for JPEG outputs; PNG/BMP encoders don't
        # understand IMWRITE_JPEG_QUALITY and would emit "unsupported key" warnings.
        if type_dir == "mask" or save_path.suffix.lower() in [".png", ".bmp"]:
            cv2.imwrite(str(save_path), img_resized)
        else:
            cv2.imwrite(str(save_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        return True, None
    except Exception as e:
        return False, f"{f.name}: {e}"


def resize_and_save(source_dir, dest_dir, target_size, quality=95, fraction=1.0):
    """
    Resize images to target_size.
    
    Args:
        target_size: tuple of (WIDTH, HEIGHT) following W×H paper convention
        fraction: Fraction of the dataset to process (0.0 < fraction <= 1.0).
                  Useful when disk space is limited. A deterministic subset is
                  sampled (sorted order, evenly spaced) so input/target/mask
                  stay aligned across the three folders.
    """
    src_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not src_path.exists():
        print(f"[ERROR] Source directory not found: {source_dir}")
        return
    
    # target_size is (W, H), convert to OpenCV format (W, H) for cv2.resize
    target_size_cv = (target_size[0], target_size[1])
    
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"[OK] Resizing images to {target_size[0]}×{target_size[1]} (W×H)")
    print(f"[OK] Source: {source_dir}")
    print(f"[OK] Destination: {dest_dir}")
    print(f"[OK] Parallel workers: {num_workers}")
    print()
    
    total_processed = 0
    total_failed = 0
    
    # Detect structure: split (train/test) vs flat (input/mask/target directly)
    has_splits = (src_path / "train").exists() or (src_path / "test").exists()
    if has_splits:
        splits = ["train", "test"]
        print(f"[OK] Detected split structure (train/test)")
    else:
        splits = [None]
        print(f"[OK] Detected flat structure (no train/test split)")
    
    for split in splits:
        for type_dir in ["input", "target", "mask"]:
            if split is not None:
                current_src = src_path / split / type_dir
                current_dest = dest_path / split / type_dir
                desc = f"{split}/{type_dir}"
            else:
                current_src = src_path / type_dir
                current_dest = dest_path / type_dir
                desc = type_dir
            
            if not current_src.exists():
                print(f"[SKIP] {desc} not found")
                continue
            
            current_dest.mkdir(parents=True, exist_ok=True)
            
            extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
            files = []
            for ext in extensions:
                files.extend(list(current_src.glob(ext)))
            
            if not files:
                print(f"[SKIP] {desc} - no images found")
                continue
            
            # Sample a deterministic subset if fraction < 1.0.
            # Sorted order + evenly-spaced indices keeps input/target/mask
            # aligned (same filenames selected across all three folders).
            if fraction < 1.0:
                files = sorted(files)
                n_keep = max(1, int(round(len(files) * fraction)))
                if n_keep < len(files):
                    step = len(files) / n_keep
                    indices = [int(i * step) for i in range(n_keep)]
                    files = [files[i] for i in indices]
            
            print(f"[Processing] {desc} - {len(files)} images (fraction={fraction})")
            
            tasks = [(f, current_dest, target_size_cv, quality, type_dir) for f in files]
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_single_image, task): task for task in tasks}
                
                with tqdm(total=len(files), desc=desc) as pbar:
                    for future in as_completed(futures):
                        success, error_msg = future.result()
                        if success:
                            total_processed += 1
                        else:
                            if error_msg:
                                print(f"[WARNING] {error_msg}")
                            total_failed += 1
                        pbar.update(1)
            
            print()
    
    print("=" * 80)
    print(f"[COMPLETE] Preprocessing finished!")
    print(f"  Processed: {total_processed} images")
    if total_failed > 0:
        print(f"  Failed: {total_failed} images")
    print(f"  Output: {dest_dir}")
    print()
    print("[NEXT] Update config.yaml root_dir to:")
    print(f'  root_dir: "{dest_dir}"')
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Pre-resize dataset for efficient training")
    parser.add_argument("--source", type=str, required=True, 
                        help="Source dataset path")
    parser.add_argument("--dest", type=str, default="Data/Cached_Stage1",
                        help="Destination path for resized images")
    parser.add_argument("--size", nargs=2, type=int, default=None,
                        help="Target size as WIDTH HEIGHT (W×H paper convention, optional)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality (1-100, default: 95)")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of the dataset to process (0.0 < fraction <= 1.0, "
                             "e.g. 0.5 for half, 0.25 for a quarter). Default: 1.0 (all)")
    
    args = parser.parse_args()
    
    # Validate fraction
    if not (0.0 < args.fraction <= 1.0):
        parser.error("--fraction must be in the range (0.0, 1.0]")
    
    if args.fraction < 1.0:
        print(f"[OK] Processing {args.fraction:.0%} of the dataset (subset mode)")
    
    # Load target size from config if not provided via CLI
    if args.size is None:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            input_resolution = config.get('data', {}).get('input_resolution', [192, 256])
            target_size = tuple(input_resolution)
            print(f"[OK] Loaded input_resolution from config: {target_size}")
        except Exception as e:
            print(f"[WARNING] Failed to load config ({e}). Using default: (192, 256)")
            target_size = (192, 256)
    else:
        target_size = tuple(args.size)
        print(f"[OK] Using CLI argument --size: {target_size}")
    
    resize_and_save(args.source, args.dest, target_size, args.quality, args.fraction)


if __name__ == "__main__":
    main()
