"""
Mixed-dataset preprocessing script (Stage 1).

Merges multiple shadow-removal datasets into a single flat folder
(Data/Mixed_Stage1/{input,target,mask}) at the Stage 1 resolution (192×256),
prefixing every filename with its dataset tag so the training-time dataset
class can identify which samples belong to each dataset.

Run ONCE, offline, before training. This keeps resize/merge work out of the
training loop so there is no CPU load during training.

Usage:
    python scripts/mixed_dataset.py \
        --source FSDSRD=path/to/fsdsrd \
        --source RDD=path/to/rdd \
        --source SD7K=path/to/sd7k \
        --source AOSR=path/to/osr \
        --source SynDoc_Wild=path/to/syndoc_wild \
        --source SynDoc_Wild_3D=path/to/syndoc_wild_3d \
        --dest Data/Mixed_Stage1 \
        --size 192 256

Each --source is `TAG=PATH`. The tag becomes the filename prefix (e.g. AOSR_).
Multi-word tags (e.g. SynDoc_Wild_3D) are supported — the training-time
dataset class uses longest-prefix matching to resolve them.
Each source path should contain {input,target,mask} subfolders (flat layout).
"""
import argparse
import random
import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Dataset tags -> filename prefixes. All datasets are augmented at training
# time, so no single tag needs special handling here.

# Subfolders expected inside each source dataset (flat layout).
TYPE_DIRS = ["input", "target", "mask"]


def process_single_image(args):
    """Resize one image and save it to the merged destination."""
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


def merge_dataset(source_dir, dest_dir, tag, target_size, quality=95, fraction=1.0):
    """
    Resize one dataset's {input,target,mask} into the merged dest folder,
    prefixing filenames with `tag_` so input/target/mask stay aligned.

    fraction: Fraction of the dataset to process (0.0 < fraction <= 1.0).
              A deterministic subset is sampled (sorted order, evenly spaced)
              so input/target/mask stay aligned across the three folders.
    """
    src_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not src_path.exists():
        print(f"[ERROR] Source directory not found: {source_dir}")
        return 0, 0

    target_size_cv = (target_size[0], target_size[1])
    num_workers = max(1, multiprocessing.cpu_count() - 1)

    total_processed = 0
    total_failed = 0

    for type_dir in TYPE_DIRS:
        current_src = src_path / type_dir
        current_dest = dest_path / type_dir
        desc = f"{tag}/{type_dir}"

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
        # Sorted order + evenly-spaced indices keeps input/target/mask aligned
        # (same filenames selected across all three folders).
        if fraction < 1.0:
            files = sorted(files)
            n_keep = max(1, int(round(len(files) * fraction)))
            if n_keep < len(files):
                step = len(files) / n_keep
                indices = [int(i * step) for i in range(n_keep)]
                files = [files[i] for i in indices]

        # Prefix filenames with the dataset tag so input/target/mask stay aligned
        # AND the training dataset can identify A-OSR samples by prefix.
        tasks = []
        for f in files:
            prefixed_name = f"{tag}_{f.name}"
            tasks.append((f, current_dest, target_size_cv, quality, type_dir, prefixed_name))

        print(f"[Processing] {desc} - {len(files)} images (fraction={fraction})")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_prefixed, task): task for task in tasks}
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

    return total_processed, total_failed


def _process_prefixed(args):
    """Like process_single_image but writes to a prefixed filename."""
    f, current_dest, target_size_cv, quality, type_dir, prefixed_name = args
    try:
        img = cv2.imread(str(f))
        if img is None:
            return False, f"Failed to read: {f.name}"

        img_resized = cv2.resize(img, target_size_cv, interpolation=cv2.INTER_LINEAR)

        save_path = current_dest / prefixed_name
        if type_dir == "mask" or save_path.suffix.lower() in [".png", ".bmp"]:
            cv2.imwrite(str(save_path), img_resized)
        else:
            cv2.imwrite(str(save_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, quality])

        return True, None
    except Exception as e:
        return False, f"{f.name}: {e}"


def generate_synthetic_samples(dest_dir, target_size, quality=95, seed=42,
                               clean_count=0, black_count=0):
    """
    Generate synthetic "no-shadow" samples into the merged dest folder.

    Two new tags are produced from the already-merged real samples:

    - CLEAN_* : pick a real target image, set input = target and mask = all
      black (0). Teaches the model that a document with no shadow should be
      passed through unchanged (identity).
    - BLACK_* : input = target = mask = all black. Teaches the model that a
      fully black region should not be brightened / no shadow invented.

    Both are regularizers, so they should be a clear minority of the dataset.

    Returns (clean_written, black_written).
    """
    dest_path = Path(dest_dir)
    target_dir = dest_path / "target"
    if not target_dir.exists():
        print("[SKIP] Synthetic samples: no target folder to source from")
        return 0, 0

    # Source real targets (any tag) to build CLEAN samples from.
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
    real_targets = []
    for ext in extensions:
        real_targets.extend(list(target_dir.glob(ext)))
    real_targets = sorted(real_targets)

    rng = random.Random(seed)
    w, h = target_size

    clean_written = 0
    black_written = 0

    # --- CLEAN: input = target, mask = black ---
    if clean_count > 0:
        clean_input_dir = dest_path / "input"
        clean_target_dir = dest_path / "target"
        clean_mask_dir = dest_path / "mask"
        clean_input_dir.mkdir(parents=True, exist_ok=True)
        clean_target_dir.mkdir(parents=True, exist_ok=True)
        clean_mask_dir.mkdir(parents=True, exist_ok=True)

        for i in range(clean_count):
            if not real_targets:
                break
            src = rng.choice(real_targets)
            img = cv2.imread(str(src))
            if img is None:
                continue
            img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            name = f"CLEAN_{i:04d}.png"
            cv2.imwrite(str(clean_input_dir / name), img_resized)
            cv2.imwrite(str(clean_target_dir / name), img_resized)
            black_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(str(clean_mask_dir / name), black_mask)
            clean_written += 1

    # --- BLACK samples: input = target = mask = all black ---
    if black_count > 0:
        black_input_dir = dest_path / "input"
        black_target_dir = dest_path / "target"
        black_mask_dir = dest_path / "mask"
        black_input_dir.mkdir(parents=True, exist_ok=True)
        black_target_dir.mkdir(parents=True, exist_ok=True)
        black_mask_dir.mkdir(parents=True, exist_ok=True)

        black_img = np.zeros((h, w, 3), dtype=np.uint8)
        black_mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(black_count):
            name = f"BLACK_{i:04d}.png"
            cv2.imwrite(str(black_input_dir / name), black_img)
            cv2.imwrite(str(black_target_dir / name), black_img)
            cv2.imwrite(str(black_mask_dir / name), black_mask)
            black_written += 1

    return clean_written, black_written


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple shadow-removal datasets into one flat folder for mixed training"
    )
    parser.add_argument("--source", action="append", required=True, metavar="TAG=PATH",
                        help="Source dataset as TAG=PATH (repeatable). TAG becomes filename prefix.")
    parser.add_argument("--dest", type=str, default="Data/Mixed_Stage1",
                        help="Destination path for merged resized images")
    parser.add_argument("--size", nargs=2, type=int, default=None,
                        help="Target size as WIDTH HEIGHT (W×H paper convention, optional)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file (for default size)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality (1-100, default: 95)")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of each dataset to process (0.0 < fraction <= 1.0, "
                             "e.g. 0.5 for half, 0.25 for a quarter). Default: 1.0 (all)")
    parser.add_argument("--clean-count", type=int, default=0,
                        help="Number of synthetic CLEAN samples to generate "
                             "(input=target, mask=black). Teaches identity/no-shadow.")
    parser.add_argument("--black-count", type=int, default=0,
                        help="Number of synthetic BLACK samples to generate "
                             "(input=target=mask=black). Teaches black-region guard.")
    parser.add_argument("--synthetic-seed", type=int, default=42,
                        help="Random seed for synthetic sample selection (default: 42)")

    args = parser.parse_args()

    # Validate fraction
    if not (0.0 < args.fraction <= 1.0):
        parser.error("--fraction must be in the range (0.0, 1.0]")

    if args.fraction < 1.0:
        print(f"[OK] Processing {args.fraction:.0%} of each dataset (subset mode)")

    # Parse TAG=PATH pairs
    sources = {}
    for item in args.source:
        if "=" not in item:
            parser.error(f"--source must be TAG=PATH, got: {item}")
        tag, path = item.split("=", 1)
        sources[tag.strip()] = path.strip()

    if not sources:
        parser.error("At least one --source is required")

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

    print("=" * 80)
    print("MIXED DATASET PREPROCESSING")
    print("=" * 80)
    print(f"[OK] Target size: {target_size[0]}×{target_size[1]} (W×H)")
    print(f"[OK] Destination: {args.dest}")
    print(f"[OK] Datasets:")
    for tag, path in sources.items():
        print(f"      {tag}: {path} (AUGMENTED at train time)")
    print()

    total_processed = 0
    total_failed = 0
    for tag, path in sources.items():
        processed, failed = merge_dataset(path, args.dest, tag, target_size, args.quality, args.fraction)
        total_processed += processed
        total_failed += failed

    # Generate synthetic no-shadow samples (CLEAN + BLACK) as regularizers.
    if args.clean_count > 0 or args.black_count > 0:
        print("=" * 80)
        print("SYNTHETIC NO-SHADOW SAMPLES")
        print("=" * 80)
        print(f"[OK] CLEAN (input=target, mask=black): {args.clean_count}")
        print(f"[OK] BLACK (input=target=mask=black): {args.black_count}")
        syn_clean, syn_black = generate_synthetic_samples(
            args.dest, target_size, args.quality,
            seed=args.synthetic_seed,
            clean_count=args.clean_count,
            black_count=args.black_count,
        )
        total_processed += syn_clean + syn_black
        print(f"[OK] CLEAN written: {syn_clean}, BLACK written: {syn_black}")
        print()

    print("=" * 80)
    print(f"[COMPLETE] Mixed preprocessing finished!")
    print(f"  Processed: {total_processed} images")
    if total_failed > 0:
        print(f"  Failed: {total_failed} images")
    print(f"  Output: {args.dest}")
    print()
    print("[NEXT] Update config.yaml dataset.mixed.root_dir to:")
    print(f'  root_dir: "{args.dest}"')
    print("=" * 80)


if __name__ == "__main__":
    main()
