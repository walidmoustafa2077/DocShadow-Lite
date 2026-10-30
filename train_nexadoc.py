"""Train the NexaDoc-50k deshadow (stage 1) and SR (stage 2) models.

Examples
--------
Single GPU / CPU smoke test::

    python train_nexadoc.py --stage 1 --smoke --data-root D:/Temp/nexadoc50k

Kaggle 2xT4::

    torchrun --nproc_per_node=2 train_nexadoc.py --stage 1 \
        --config configs/nexadoc.yaml --data-root /kaggle/input/nexadoc-50k

    torchrun --nproc_per_node=2 train_nexadoc.py --stage 2 \
        --config configs/nexadoc.yaml --data-root /kaggle/input/nexadoc-50k
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nexadoc_dataset import (  # noqa: E402
    NexaDocDeshadowDataset,
    NexaDocUpsampleDataset,
    make_loader,
)
from src.models.deshadow import DeshadowNet  # noqa: E402
from src.models.ioanet import count_parameters  # noqa: E402
from src.models.sr_refiner import SRUNet  # noqa: E402
from src.utils.eval_metrics import MetricAccumulator, RegionAccumulator  # noqa: E402
from src.utils.nexadoc_losses import DeshadowLoss, SRLoss  # noqa: E402


# ---------------------------------------------------------------------------
# Distributed / environment
# ---------------------------------------------------------------------------


def init_distributed() -> Tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        rank, local_rank = 0, 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--config", type=str, default="configs/nexadoc.yaml")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Tuple[dict, dict]:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.data_root:
        config["data"]["root"] = args.data_root

    stage = config[f"stage{args.stage}"]
    if args.output_dir:
        stage["output_dir"] = args.output_dir
    if args.epochs is not None:
        stage["epochs"] = args.epochs
    if args.batch_size is not None:
        stage["batch_size"] = args.batch_size
    if args.backbone is not None and args.stage == 1:
        stage["backbone"] = args.backbone
    if args.seed is not None:
        stage["seed"] = args.seed
    if args.no_lpips:
        stage["lpips_weight"] = 0.0
    if args.no_pretrained and args.stage == 1:
        stage["pretrained"] = False
    if args.smoke:
        stage["epochs"] = 1
        stage["batch_size"] = min(int(stage["batch_size"]), 2)
        stage["val_batch_size"] = min(int(stage.get("val_batch_size", 2)), 2)
        stage["pretrained"] = False
        stage["amp"] = torch.cuda.is_available()
    return config, stage


# ---------------------------------------------------------------------------
# Models / checkpoints
# ---------------------------------------------------------------------------


def build_model(stage: int, cfg: dict, device: torch.device) -> nn.Module:
    if stage == 1:
        backbone = cfg.get("backbone", "mobilenet_v2")
        if backbone not in ("mobilenet_v2", "mobilenetv2"):
            raise ValueError(
                f"Backbone {backbone!r} is not implemented yet. "
                "Use 'mobilenet_v2' for this run."
            )
        model = DeshadowNet(
            pretrained=bool(cfg.get("pretrained", True)),
            clamp_output=bool(cfg.get("clamp_output", True)),
        )
    else:
        model = SRUNet(
            base_channels=int(cfg.get("base_channels", 32)),
            clamp_output=bool(cfg.get("clamp_output", True)),
        )
    return model.to(device)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _clean_state_dict(state: dict) -> dict:
    if any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    return state


def extract_model_state(checkpoint: dict, prefer_ema: bool = False) -> dict:
    keys = ("ema", "model", "model_state_dict", "state_dict")
    if not prefer_ema:
        keys = ("model", "model_state_dict", "state_dict", "ema")
    for key in keys:
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return _clean_state_dict(candidate)
    return _clean_state_dict(checkpoint)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler=None,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = extract_model_state(checkpoint)
    model.load_state_dict(state)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(unwrap(model)).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = unwrap(model).state_dict()
        target = self.shadow.state_dict()
        for key, value in target.items():
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(source[key].detach(), alpha=1.0 - self.decay)
            else:
                value.copy_(source[key])

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(_clean_state_dict(state))


# ---------------------------------------------------------------------------
# Metrics / training loops
# ---------------------------------------------------------------------------


def autocast_context(enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_optimizer(model: nn.Module, cfg: dict) -> AdamW:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return AdamW(
        [
            {"params": decay, "weight_decay": float(cfg["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(cfg["lr"]),
    )


def make_scheduler(optimizer: AdamW, cfg: dict) -> LambdaLR:
    warmup = int(cfg.get("warmup_epochs", 0))
    epochs = int(cfg["epochs"])
    min_ratio = float(cfg.get("min_lr_ratio", 0.02))

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler,
    ema: ModelEMA,
    device: torch.device,
    stage: int,
    cfg: dict,
    epoch: int,
    rank: int,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    amp = bool(cfg.get("amp", True))
    clip = float(cfg.get("grad_clip", 0.0))
    totals: Dict[str, float] = defaultdict(float)
    batches = 0

    progress = tqdm(
        loader,
        desc=f"train {epoch + 1}/{cfg['epochs']}",
        disable=not is_main(rank),
        dynamic_ncols=True,
    )
    for step, batch in enumerate(progress):
        if max_batches is not None and step >= max_batches:
            break
        image = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True) if stage == 1 else None

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(amp):
            output = model(image)
            losses = criterion(output, target, mask)

        scaler.scale(losses["total"]).backward()
        if clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        for key, value in losses.items():
            totals[key] += float(value.detach())
        batches += 1
        if is_main(rank) and step % 10 == 0:
            progress.set_postfix(
                loss=f"{totals['total'] / max(batches, 1):.4f}",
                l1=f"{totals['l1'] / max(batches, 1):.4f}",
            )

    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    stage: int,
    amp: bool,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    metrics = MetricAccumulator()
    regions = RegionAccumulator() if stage == 1 else None
    loss_sum = 0.0
    batches = 0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        image = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True) if stage == 1 else None

        with autocast_context(amp):
            output = model(image)
        output = output.float()
        metrics.update(output, target)
        if regions is not None and mask is not None:
            regions.update(output, target, mask)

        losses = criterion(output, target, mask)
        loss_sum += float(losses["total"])
        batches += 1

    result = metrics.compute()
    result["loss"] = loss_sum / max(batches, 1)
    if regions is not None:
        result.update(regions.compute())
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config, cfg = load_config(args)
    rank, world_size, local_rank, device = init_distributed()
    seed_everything(int(cfg.get("seed", args.seed)) + rank)

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    stage = args.stage
    root = config["data"]["root"]
    workers = int(config["data"].get("num_workers", 4))
    if args.smoke:
        workers = 0

    Dataset = NexaDocDeshadowDataset if stage == 1 else NexaDocUpsampleDataset

    def build_dataset(split: str, augment: bool):
        return Dataset(root, split=split, augment=augment)

    train_dataset = build_dataset("train", True)
    val_dataset = build_dataset("val", False)
    test_dataset = build_dataset("test", False)

    if is_main(rank):
        model = build_model(stage, cfg, device)
        print(f"Stage {stage} model: {count_parameters(model, False):,} params")
        print(
            f"Data: train={len(train_dataset):,} val={len(val_dataset):,} "
            f"test={len(test_dataset):,} root={root}"
        )
        print(f"Device: {device} | world_size={world_size} | rank={rank}")
    else:
        model = build_model(stage, cfg, device)

    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    amp = bool(cfg.get("amp", True)) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    start_epoch = 0
    best_score = -float("inf")
    epochs_no_improve = 0
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        best_score = float(resume_checkpoint.get("best_score", best_score))
        epochs_no_improve = int(resume_checkpoint.get("epochs_no_improve", 0))
        if is_main(rank):
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    ema = ModelEMA(model, decay=float(cfg.get("ema_decay", 0.999)))
    if resume_checkpoint is not None and isinstance(resume_checkpoint.get("ema"), dict):
        ema.load_state_dict(resume_checkpoint["ema"])
    criterion = (
        DeshadowLoss(
            l1_weight=float(cfg.get("l1_weight", 1.0)),
            mask_weight=float(cfg.get("mask_weight", 2.0)),
            lpips_weight=float(cfg.get("lpips_weight", 0.1)),
        )
        if stage == 1
        else SRLoss(
            l1_weight=float(cfg.get("l1_weight", 1.0)),
            lpips_weight=float(cfg.get("lpips_weight", 0.05)),
        )
    ).to(device)

    train_loader, train_sampler = make_loader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=workers,
        distributed=world_size > 1,
        drop_last=True,
        prefetch_factor=int(config["data"].get("prefetch_factor", 2)),
    )
    val_loader = None
    test_loader = None
    if is_main(rank):
        val_loader, _ = make_loader(
            val_dataset,
            batch_size=int(cfg.get("val_batch_size", cfg["batch_size"])),
            shuffle=False,
            num_workers=workers,
            distributed=False,
            drop_last=False,
        )
        test_loader, _ = make_loader(
            test_dataset,
            batch_size=int(cfg.get("val_batch_size", cfg["batch_size"])),
            shuffle=False,
            num_workers=workers,
            distributed=False,
            drop_last=False,
        )

    writer = SummaryWriter(str(out_dir / "tb")) if is_main(rank) else None
    metrics_path = out_dir / "metrics.jsonl"
    if is_main(rank):
        with open(out_dir / "config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        try:
            git_sha = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            git_sha = "unknown"
        (out_dir / "run_info.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "git_sha": git_sha,
                    "world_size": world_size,
                    "torch": torch.__version__,
                    "config": config,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    epochs = int(cfg["epochs"])
    val_interval = int(cfg.get("val_interval", 1))
    save_interval = int(cfg.get("save_interval", 5))
    patience = int(cfg.get("patience", 20))
    max_train_batches = 2 if args.smoke else None
    max_val_batches = 2 if args.smoke else None

    for epoch in range(start_epoch, epochs):
        started = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            ema,
            device,
            stage,
            cfg,
            epoch,
            rank,
            max_batches=max_train_batches,
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if is_main(rank):
            line = {
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": train_metrics["total"],
                "train_l1": train_metrics["l1"],
                "seconds": round(time.time() - started, 1),
            }
            if writer is not None:
                for key, value in train_metrics.items():
                    writer.add_scalar(f"train/{key}", value, epoch + 1)
                writer.add_scalar("train/lr", lr, epoch + 1)

            should_validate = (epoch + 1) % val_interval == 0 or epoch + 1 == epochs
            if should_validate:
                val_metrics = validate(
                    unwrap(model),
                    val_loader,
                    criterion,
                    device,
                    stage,
                    amp,
                    max_batches=max_val_batches,
                )
                ema_metrics = validate(
                    ema.shadow,
                    val_loader,
                    criterion,
                    device,
                    stage,
                    amp,
                    max_batches=max_val_batches,
                )
                line.update({f"val_{k}": v for k, v in val_metrics.items()})
                line.update({f"ema_{k}": v for k, v in ema_metrics.items()})
                if writer is not None:
                    for key, value in val_metrics.items():
                        writer.add_scalar(f"val/{key}", value, epoch + 1)
                    for key, value in ema_metrics.items():
                        writer.add_scalar(f"ema/{key}", value, epoch + 1)

                score = max(val_metrics["psnr"], ema_metrics["psnr"])
                if score > best_score:
                    best_score = score
                    epochs_no_improve = 0
                    save_checkpoint(
                        {
                            "model": unwrap(model).state_dict(),
                            "ema": ema.shadow.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict(),
                            "epoch": epoch + 1,
                            "best_score": best_score,
                            "epochs_no_improve": epochs_no_improve,
                            "config": config,
                            "stage": stage,
                        },
                        out_dir / "best.pt",
                    )
                else:
                    epochs_no_improve += 1
                print(
                    f"epoch {epoch + 1}: val psnr {val_metrics['psnr']:.3f} "
                    f"ssim {val_metrics['ssim']:.4f} mae {val_metrics['mae']:.4f} "
                    f"| ema psnr {ema_metrics['psnr']:.3f} "
                    f"ssim {ema_metrics['ssim']:.4f} | best {best_score:.3f}"
                )

            print(json.dumps(line))
            with open(metrics_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")

            if (epoch + 1) % save_interval == 0 or epoch + 1 == epochs:
                save_checkpoint(
                    {
                        "model": unwrap(model).state_dict(),
                        "ema": ema.shadow.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch + 1,
                        "best_score": best_score,
                        "epochs_no_improve": epochs_no_improve,
                        "config": config,
                        "stage": stage,
                    },
                    out_dir / "last.pt",
                )
                save_checkpoint(
                    {
                        "model": unwrap(model).state_dict(),
                        "ema": ema.shadow.state_dict(),
                        "epoch": epoch + 1,
                        "best_score": best_score,
                        "epochs_no_improve": epochs_no_improve,
                        "config": config,
                        "stage": stage,
                    },
                    out_dir / f"epoch_{epoch + 1:04d}.pt",
                )

        if world_size > 1:
            dist.barrier()

        if world_size > 1:
            stop = torch.tensor(
                [1 if epochs_no_improve >= patience else 0],
                device=device,
            )
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if stop.item():
                break
        elif epochs_no_improve >= patience:
            if is_main(rank):
                print(f"Early stopping after {epochs_no_improve} epochs without improvement")
            break

    if is_main(rank):
        # Final test with the best EMA weights, falling back to the last model.
        best_path = out_dir / "best.pt"
        if best_path.is_file():
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
            state = checkpoint.get("ema") or checkpoint.get("model")
            unwrap(model).load_state_dict(state)
        test_metrics = validate(
            unwrap(model),
            test_loader,
            criterion,
            device,
            stage,
            amp,
            max_batches=max_val_batches,
        )
        print("test:", json.dumps(test_metrics, indent=2))
        (out_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8"
        )
        if writer is not None:
            for key, value in test_metrics.items():
                writer.add_scalar(f"test/{key}", value)
            writer.close()

    if world_size > 1:
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
