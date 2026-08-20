"""
Mixed-dataset preprocessing script (Stage 1).

Merges multiple shadow-removal datasets into a single flat folder
(Data/Mixed_Stage1/{input,target,mask}) at the Stage 1 resolution (192×256),
prefixing every filename with its dataset tag so the training-time dataset
class can identify which samples belong to A-OSR (the only dataset that gets
augmentation).

Run ONCE, offline, before training. This keeps resize/merge work out of the
training loop so there is no CPU load during training.

Usage:
    python scripts/mixed_dataset.py \
        --source FSDSRD=path/to/fsdsrd \
        --source RDD=path/to/rdd \
        --source SD7K=path/to/sd7k \
        --source AOSR=path/to/osr \
        --dest Data/Mixed_Stage1 \
        --size 192 256

Each --source is `TAG=PATH`. The tag becomes the filename prefix (e.g. AOSR_).
Each source path should contain {input,target,mask} subfolders (flat layout).
"""
import argparse
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Dataset tags -> filename prefixes. A-OSR is the ONLY dataset that gets
# augmentation at training time, so it must be identifiable by prefix.
AUGMENTED_TAG = "AOSR"

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


def merge_dataset(source_dir, dest_dir, tag, target_size, quality=95):
    """
    Resize one dataset's {input,target,mask} into the merged dest folder,
    prefixing filenames with `tag_` so input/target/mask stay aligned.
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

        # Prefix filenames with the dataset tag so input/target/mask stay aligned
        # AND the training dataset can identify A-OSR samples by prefix.
        tasks = []
        for f in files:
            prefixed_name = f"{tag}_{f.name}"
            tasks.append((f, current_dest, target_size_cv, quality, type_dir, prefixed_name))

        print(f"[Processing] {desc} - {len(files)} images")

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

    args = parser.parse_args()

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
        aug_note = " (AUGMENTED at train time)" if tag == AUGMENTED_TAG else ""
        print(f"      {tag}: {path}{aug_note}")
    print()

    total_processed = 0
    total_failed = 0
    for tag, path in sources.items():
        processed, failed = merge_dataset(path, args.dest, tag, target_size, args.quality)
        total_processed += processed
        total_failed += failed

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
