"""Evaluate the NexaDoc-50k two-stage pipeline on the official test split.

Stage 1 is measured on deshadow/test (input/target/mask, 256x256).
Stage 2 is measured on upsample/test (input 256x256, target 768x1024).

Usage:
    python evaluate_nexadoc.py --stage 1 --checkpoint outputs/nexadoc/stage1/best.pt
    python evaluate_nexadoc.py --stage 2 --checkpoint outputs/nexadoc/stage2/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.data.nexadoc_dataset import NexaDocDeshadowDataset, NexaDocUpsampleDataset, make_loader
from src.models.deshadow import DeshadowNet
from src.models.sr_refiner import SRUNet
from src.utils.eval_metrics import MetricAccumulator, RegionAccumulator
from train_nexadoc import extract_model_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, choices=(1, 2), default=1)
    p.add_argument("--config", default=str(ROOT / "configs" / "nexadoc.yaml"))
    p.add_argument("--checkpoint", default=None,
                   help="default: outputs/nexadoc/stage{1,2}/best.pt")
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--save-samples", type=int, default=8)
    p.add_argument("--device", default=None)
    return p.parse_args()


def build_model(stage: int, cfg: dict) -> torch.nn.Module:
    if stage == 1:
        return DeshadowNet(pretrained=False, clamp_output=bool(cfg.get("clamp_output", True)))
    return SRUNet(base_channels=int(cfg.get("base_channels", 32)),
                  clamp_output=bool(cfg.get("clamp_output", True)))


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().clamp(0, 1).numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return np.ascontiguousarray((arr[..., ::-1] * 255.0 + 0.5).astype(np.uint8))


def panel(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    h = max(im.shape[0] for im in images)
    resized = []
    for im in images:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            im = cv2.resize(im, (int(round(im.shape[1] * scale)), h),
                            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        resized.append(im)
    out = np.concatenate(resized, axis=1)
    for i, text in enumerate(labels):
        x = sum(im.shape[1] for im in resized[:i]) + 6
        cv2.putText(out, text, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cfg = config[f"stage{args.stage}"]
    data_root = Path(args.data_root or config["data"]["root"])
    ckpt_path = Path(args.checkpoint or ROOT / f"outputs/nexadoc/stage{args.stage}/best.pt")
    out_dir = Path(args.output_dir or ROOT / f"outputs/nexadoc/eval_stage{args.stage}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.stage == 1:
        dataset = NexaDocDeshadowDataset(data_root, "test", augment=False)
        batch_size = args.batch_size or 16
    else:
        dataset = NexaDocUpsampleDataset(data_root, "test", augment=False)
        batch_size = args.batch_size or 1
    loader, _ = make_loader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, drop_last=False)

    model = build_model(args.stage, cfg).to(device).eval()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(extract_model_state(checkpoint, prefer_ema=True))
    model.eval()
    print(f"Loaded {ckpt_path} on {device}; test samples: {len(dataset)}")

    metrics = MetricAccumulator()
    regions = RegionAccumulator() if args.stage == 1 else None
    saved = 0
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
            outputs = outputs.float()
            targets = targets.float()
            metrics.update(outputs, targets)
            if regions is not None:
                regions.update(outputs, targets, batch["mask"].to(device).float())

            if saved < args.save_samples:
                for i in range(outputs.shape[0]):
                    if saved >= args.save_samples:
                        break
                    stem = str(batch["name"][i])
                    inp = to_uint8(inputs[i])
                    tgt = to_uint8(targets[i])
                    out = to_uint8(outputs[i])
                    if args.stage == 1:
                        images = [inp, tgt, out, np.abs(inp.astype(np.float32) - out.astype(np.float32)).clip(0, 255).astype(np.uint8)]
                        labels = ["input", "target", "prediction", "abs diff"]
                    else:
                        images = [inp, tgt, out]
                        labels = ["input 256", "target 1024x768", "prediction"]
                    cv2.imwrite(str(out_dir / f"{stem}_comparison.png"), panel(images, labels))
                    saved += 1

    result = metrics.compute()
    if regions is not None:
        result.update(regions.compute())
    result["stage"] = args.stage
    result["checkpoint"] = str(ckpt_path)
    result["samples"] = metrics.images
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {out_dir / 'metrics.json'} and {saved} comparison image(s)")


if __name__ == "__main__":
    main()
