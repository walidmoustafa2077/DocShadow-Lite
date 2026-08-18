import argparse
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore', category=UserWarning)

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import cv2

from src.models.ioanet import IOANet, get_model_info
from src.models.laplacian_refiner import LPIOANet, LaplacianRefiner, count_parameters
from src.data.dataset import create_dataloaders, create_stage2_dataloaders
from src.utils.losses import ShadowRemovalLoss, MetricsCalculator


class Trainer:
    """Stage 1 IOANet Trainer (192×256 resolution)."""
    
    def __init__(self, config_path: str, debug: bool = True, stage: int = 1, 
                 resume_checkpoint: str = None, finetune: bool = False):
        self.debug = debug
        self.config_path = config_path
        self.stage = stage
        self.resume_checkpoint = resume_checkpoint
        self.finetune = finetune
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Command-line arg takes precedence over config
        self.resume_checkpoint = resume_checkpoint or self.config.get("training", {}).get("stage1", {}).get("resume_checkpoint")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup_directories()
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = 50
        
        self._print_header()
    
    def _setup_directories(self):
        base_dir = Path(self.config["output"]["checkpoint_dir"])
        
        self.checkpoint_dir = base_dir / f"stage{self.stage}"
        self.log_dir = Path(self.config["output"]["log_dir"]) / f"stage{self.stage}"
        self.sample_dir = Path(self.config["output"]["sample_output_dir"]) / f"stage{self.stage}"
        self.debug_dir = self.sample_dir / "debug"
        
        for d in [self.checkpoint_dir, self.log_dir, self.sample_dir, self.debug_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
    
    def _print_header(self):
        print("\n" + "=" * 100)
        print(f"{'IOANet TRAINING - STAGE 1':^100}")
        print("=" * 100)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[OK] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("[!] Running on CPU (training will be slow)")
        
        print(f"[OK] Device: {self.device}")
        print(f"[OK] Config: {self.config_path}")
        print(f"[OK] Debug: {self.debug}")
        if self.resume_checkpoint:
            mode = "FINE-TUNE" if self.finetune else "RESUME"
            print(f"[OK] Resume Checkpoint: {self.resume_checkpoint} ({mode} mode)")
        print("=" * 100)
    
    def _print_metrics_guide(self):
        print("\n" + "-" * 100)
        print("METRICS GUIDE")
        print("-" * 100)
        print("  MAE (Mean Absolute Error)")
        print("    - Range: 0-1 (pixel error on normalized scale)")
        print("    - Target: < 0.02 (shadows barely visible)")
        print("    - Start: ~0.08 (shadows clearly visible)")
        print()
        print("  PSNR (Peak Signal-to-Noise Ratio)")
        print("    - Target: > 28 dB (high quality)")
        print("    - Start: ~20 dB (visible artifacts)")
        print()
        print("  SSIM (Structural Similarity Index)")
        print("    - Range: 0-1 (structural similarity)")
        print("    - Target: > 0.95 (excellent quality)")
        print("    - Start: ~0.85 (moderate quality)")
        print()
        print("  Training Indicators:")
        print("    [+] = Validation improved > model saved")
        print("    [-] = No improvement > patience counter +1")
        print("-" * 100 + "\n")
    
    def _create_model(self) -> nn.Module:
        stage_config = self.config["model"]["stage1"]
        
        pretrained = stage_config.get("pretrained", True)
        
        model = IOANet(
            in_channels=stage_config["input_channels"],
            out_channels=stage_config["output_channels"],
            pretrained=pretrained
        )
        
        model = model.to(self.device)
        print(get_model_info(model, "Stage 1 Model"))
        
        return model
    
    def _load_checkpoint_with_mode(self, model: nn.Module, optimizer, scheduler) -> int:
        """
        Load Stage 1 checkpoint with optional fine-tuning mode.
        
        Logic:
        - If finetune=True: Load Weights ONLY, Reset Epoch/Optimizer (continue learning fresh)
        - If finetune=False: Load Weights + Optimizer + Scheduler + Epoch (Resume from crash)
        
        Returns:
            start_epoch: Starting epoch for training loop
        """
        start_epoch = 0
        
        if not self.resume_checkpoint:
            return start_epoch
        
        checkpoint_path = Path(self.resume_checkpoint)
        if not checkpoint_path.exists():
            print(f"\n[!] Checkpoint not found: {self.resume_checkpoint}")
            return start_epoch
        
        print(f"\n[Checkpoint Load]")
        print(f"  Path: {self.resume_checkpoint}")
        print(f"  Mode: {'FINE-TUNE (weights only, reset optimizer)' if self.finetune else 'RESUME (full state recovery)'}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # =====================================================================
        # STEP 1: Load Model Weights
        # =====================================================================
        try:
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  [OK] Model: Loaded model_state_dict")
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print(f"  [OK] Model: Loaded state_dict")
            else:
                # Direct state dict (old format)
                model.load_state_dict(checkpoint if not isinstance(checkpoint, dict) else checkpoint)
                print(f"  [OK] Model: Loaded weights (legacy format)")
        except Exception as e:
            print(f"  [!] Error loading model weights: {e}")
            return start_epoch
        
        # =====================================================================
        # STEP 2: Load Training State (Conditional)
        # =====================================================================
        if self.finetune:
            # FINE-TUNE MODE: Reset optimizer and epoch
            print(f"  [OK] Optimizer: Reset (fine-tuning mode)")
            print(f"  [OK] Epoch: Reset to 0 (fine-tuning mode)")
            print(f"  [!] Note: Starting fresh training with loaded weights")
            start_epoch = 0
            self.best_val_loss = float('inf')  # Reset best loss for new phase
            self.patience_counter = 0
        else:
            # RESUME MODE: Load optimizer and training state
            if isinstance(checkpoint, dict):
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print(f"  [OK] Optimizer: Loaded state_dict")
                    except Exception as e:
                        print(f"  [!] Error loading optimizer: {e}")
                
                if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        print(f"  [OK] Scheduler: Loaded state_dict")
                    except Exception as e:
                        print(f"  [!] Error loading scheduler: {e}")
                
                # Restore epoch and loss
                start_epoch = checkpoint.get('epoch', 0)
                self.best_val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', float('inf')))
                self.patience_counter = checkpoint.get('patience_counter', 0)
                
                print(f"  [OK] Training State: Epoch={start_epoch}, Best Loss={self.best_val_loss:.4f}, Patience={self.patience_counter}")
            else:
                print(f"  [!] Legacy checkpoint format - starting from epoch 0")
        
        print()
        return start_epoch
    
    def _save_debug_images(self, epoch: int, input_img: torch.Tensor, 
                          target_img: torch.Tensor, output: torch.Tensor,
                          batch_idx: int = 0, sample_num: int = 1,
                          debug_outputs: dict = None) -> Path:
        """
        Save debug visualizations (Phase F).
        
        If debug_outputs is provided (every 10 epochs), visualize:
        - Input Image
        - M_in (Input Attention)
        - M_out (Output Attention)
        - Residual (what the backbone predicted)
        - Final Output
        """
        if not self.debug:
            return None
        
        inp_np = (input_img[batch_idx].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        tgt_np = (target_img[batch_idx].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out_np = (output[batch_idx].permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        h, w = inp_np.shape[:2]
        label_height = 30
        
        def add_label(img, text):
            labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
            labeled[label_height:, :, :] = img
            cv2.putText(labeled, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return labeled
        
        # Phase F: Enhanced visualization every 10 epochs
        if debug_outputs is not None and (epoch % 10 == 0):
            # Extract debug outputs
            residual = debug_outputs.get('residual', None)
            input_attn = debug_outputs.get('input_attention_map', None)
            output_attn = debug_outputs.get('output_attention_map', None)
            
            images_to_concat = []
            
            # Input
            images_to_concat.append(add_label(inp_np, "Input (Shadow)"))
            
            # M_in visualization (if available)
            if input_attn is not None:
                attn_np = (input_attn[batch_idx, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                attn_colored = cv2.applyColorMap(attn_np, cv2.COLORMAP_JET)
                images_to_concat.append(add_label(attn_colored, "M_in (Input Attn)"))
            
            # Residual visualization
            if residual is not None:
                # Residual is in range [-1, 1], normalize to [0, 255]
                res_np = (residual[batch_idx].permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5
                res_np = res_np.clip(0, 255).astype(np.uint8)
                images_to_concat.append(add_label(res_np, "Residual"))
            
            # M_out visualization (if available)
            if output_attn is not None:
                out_attn_np = (output_attn[batch_idx, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                out_attn_colored = cv2.applyColorMap(out_attn_np, cv2.COLORMAP_JET)
                images_to_concat.append(add_label(out_attn_colored, "M_out (Output Attn)"))
            
            # Output
            images_to_concat.append(add_label(out_np, "Output (Predicted)"))
            
            # Target
            images_to_concat.append(add_label(tgt_np, "Target (Clean)"))
            
            combined = np.concatenate(images_to_concat, axis=1)
            
            save_path = self.debug_dir / f"epoch_{epoch:04d}_detailed_sample{sample_num}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        else:
            # Standard visualization (3 images)
            inp_labeled = add_label(inp_np, "Input (Shadow)")
            tgt_labeled = add_label(tgt_np, "Target (Clean)")
            out_labeled = add_label(out_np, "Output (Predicted)")
            
            combined = np.concatenate([inp_labeled, tgt_labeled, out_labeled], axis=1)
            
            save_path = self.debug_dir / f"epoch_{epoch:04d}_sample{sample_num}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        return save_path
    
    def train(self):
        print("\n" + "=" * 100)
        print(f"{'IOANet Training - Stage 1 (192×256)':^100}")
        print("=" * 100)
        
        config = self.config["training"]["stage1"]
        
        resolution = tuple(self.config["data"]["input_resolution"])
        aug_cfg = self.config["data"].get("augmentation", {})
        train_loader, val_loader, test_loader = create_dataloaders(
            root_dir=self.config["dataset"]["root_dir"],
            input_resolution=resolution,
            batch_size=config["batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
            augment=True,
            illumination_strength=aug_cfg.get("illumination_strength", 0.1),
            shadow_color_shift=aug_cfg.get("shadow_color_shift", 0.05),
            rotation_range=aug_cfg.get("rotation_range", 5.0),
        )
        
        model = self._create_model()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        # Paper alignment: no LR scheduler (LP-IOANet paper uses constant LR with Adam)
        scheduler = None
        
        # Load checkpoint if resuming
        start_epoch = self._load_checkpoint_with_mode(model, optimizer, scheduler)
        
        loss_fn = ShadowRemovalLoss(
            l1_weight=config["losses"]["l1_weight"],
            lpips_weight=config["losses"]["lpips_weight"],
            shadow_weight_multiplier=config["losses"].get("shadow_weight_multiplier", 2.0),
            device=str(self.device)
        )
        
        print(f"\n[Training Config]")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  LR Scheduler: None (paper uses constant LR)")
        print(f"  Loss: L1×{config['losses']['l1_weight']} + LPIPS×{config['losses']['lpips_weight']} (VGG)")
        print(f"  Train Batches: {len(train_loader)}")
        print(f"  Val Batches: {len(val_loader)}")
        print(f"  Shadow-Aware L1 Weighting: Enabled (2.0x in shadowed regions)")
        
        if start_epoch > 0:
            print(f"  Resuming from Epoch: {start_epoch + 1}")
        
        self._print_metrics_guide()
        
        print("Starting training...\n")
        
        for epoch in range(start_epoch, config["epochs"]):
            model.train()
            train_losses = defaultdict(float)
            train_metrics = defaultdict(float)
            
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1:3d}",
                leave=True,
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )
            
            for batch_idx, batch in enumerate(pbar):
                input_img = batch["input"].to(self.device)
                target_img = batch["target"].to(self.device)
                mask = batch["mask"].to(self.device)  # Shadow mask for region weighting
                
                output = model(input_img)
                
                # Pass input_img for shadow-aware L1 weighting (2.0x in shadowed regions)
                losses = loss_fn(output, target_img, input_img=input_img, mask=mask)
                
                optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                for k, v in losses.items():
                    train_losses[k] += v.item()
                
                with torch.no_grad():
                    metrics = MetricsCalculator.compute_all(output, target_img)
                    for k, v in metrics.items():
                        train_metrics[k] += v
                
                # Update tqdm postfix with concise values
                pbar.set_postfix({
                    'Loss': f"{losses['total'].item():.3f}",
                    'MAE': f"{metrics['mae']:.4f}",
                    'PSNR': f"{metrics['psnr']:.1f}",
                    'SSIM': f"{metrics['ssim']:.3f}"
                })

            num_batches = len(train_loader)
            for k in train_losses:
                train_losses[k] /= num_batches
            for k in train_metrics:
                train_metrics[k] /= num_batches
            
            for k, v in train_losses.items():
                self.writer.add_scalar(f"train/loss_{k}", v, epoch)
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            self.writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch final training metrics
            print(
                f"[Train] Epoch {epoch+1:3d} | "
                f"Loss: {train_losses['total']:.4f} (l1={train_losses['l1']:.4f}, lpips={train_losses['lpips']:.4f}) | "
                f"MAE: {train_metrics['mae']:.4f} | PSNR: {train_metrics['psnr']:.2f} dB | SSIM: {train_metrics['ssim']:.3f}"
            )
            
            if (epoch + 1) % config["validation_interval"] == 0:
                model.eval()
                val_losses = defaultdict(float)
                val_metrics = defaultdict(float)
                val_samples = []
                
                # Phase F: Enable debug outputs every 10 epochs
                enable_debug = (epoch + 1) % 10 == 0
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        input_img = batch["input"].to(self.device)
                        target_img = batch["target"].to(self.device)
                        mask = batch["mask"].to(self.device)  # Shadow mask for region weighting
                        
                        # Phase F: Get debug outputs every 10 epochs for visualization
                        if enable_debug and batch_idx == 0:
                            output, debug_outputs = model(input_img, return_debug=True)
                        else:
                            output = model(input_img, return_debug=False)
                            debug_outputs = None
                        
                        # Pass input_img for shadow-aware L1 weighting (2.0x in shadowed regions)
                        losses = loss_fn(output, target_img, input_img=input_img, mask=mask)
                        
                        for k, v in losses.items():
                            val_losses[k] += v.item()
                        
                        metrics = MetricsCalculator.compute_all(output, target_img)
                        for k, v in metrics.items():
                            val_metrics[k] += v
                        
                        if batch_idx < 8:
                            val_samples.append((input_img, target_img, output, debug_outputs if batch_idx == 0 else None))
                
                num_val_batches = len(val_loader)
                for k in val_losses:
                    val_losses[k] /= num_val_batches
                for k in val_metrics:
                    val_metrics[k] /= num_val_batches
                
                for k, v in val_losses.items():
                    self.writer.add_scalar(f"val/loss_{k}", v, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)
                
                val_loss = val_losses["total"]
                improved = val_loss < self.best_val_loss
                status = "[+]" if improved else "[-]"
                
                print(
                    f"{status} Epoch {epoch+1:3d} | "
                    f"Loss: {val_loss:.4f} (l1={val_losses['l1']:.4f}, lpips={val_losses['lpips']:.4f}) | "
                    f"MAE: {val_metrics['mae']:.4f} | PSNR: {val_metrics['psnr']:.2f} dB | SSIM: {val_metrics['ssim']:.3f}"
                )
                
                if self.debug and val_samples:
                    for i, (inp, tgt, out, dbg) in enumerate(val_samples[:8]):
                        path = self._save_debug_images(epoch + 1, inp, tgt, out, sample_num=i + 1, debug_outputs=dbg)
                        if path:
                            print(f"    ► Sample {i+1} saved: {path.name}")
                
                if improved:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    save_path = self.checkpoint_dir / "best_model.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'val_loss': val_loss,
                        'best_val_loss': self.best_val_loss,
                        'patience_counter': self.patience_counter
                    }, save_path)
                    print(f"    ► Best model saved: {save_path.name}")
                else:
                    self.patience_counter += 1
                    print(f"    (No improvement - patience: {self.patience_counter}/{self.early_stopping_patience})")
                    
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"\n[!] Early stopping triggered at epoch {epoch + 1}")
                        break
            
            if (epoch + 1) % config["checkpoint_interval"] == 0:
                ckpt_path = self.checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': self.best_val_loss,
                    'best_val_loss': self.best_val_loss,
                    'patience_counter': self.patience_counter
                }, ckpt_path)
        
        print("\n" + "=" * 100)
        print(f"[OK] Training Complete!")
        print(f"[OK] Best Model: {self.checkpoint_dir / 'best_model.pth'}")
        print(f"[OK] Best Val Loss: {self.best_val_loss:.4f}")
        print(f"[OK] TensorBoard Logs: {self.log_dir}")
        print("=" * 100 + "\n")
        
        self.writer.close()


# =============================================================================
# Stage 2: LPTN-Lite Trainer (768×1024 resolution)
# =============================================================================

class Stage2Trainer:
    """
    Stage 2 LPTN-Lite Trainer (768×1024 high-resolution refinement).
    
    Trains the Laplacian Pyramid Refinement Network to refine high-resolution
    images using the frozen Stage 1 (IOANet) as guidance.
    
    Key differences from Stage 1:
    - Input: 768×1024 high-res images
    - IOANet: Frozen (no gradients)
    - Loss: L1 only (no LPIPS)
    - Epochs: 200 (vs 1000 for Stage 1)
    - Dataset: only (requires high-res triplets)
    - Memory: Lower batch size (4 instead of 32) due to high-resolution processing
    
    Flow:
    1. Load trained IOANet from Stage 1
    2. Freeze IOANet parameters
    3. Create LaplacianRefiner (trainable)
    4. Train on high-res image pairs
    5. Validate and save checkpoints
    """
    
    def __init__(
        self,
        config_path: str,
        debug: bool = True,
        resume_checkpoint: str = None,
        stage1_checkpoint: str = None
    ):
        """
        Args:
            config_path: Path to config.yaml
            debug: Enable debug image saving
            resume_checkpoint: Optional checkpoint to resume from
            stage1_checkpoint: Optional override for Stage 1 checkpoint path (uses config if not provided)
        """
        self.debug = debug
        self.config_path = config_path
        self.resume_checkpoint = resume_checkpoint
        self.stage = 2
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load Stage 1 checkpoint path from config, or use override if provided
        self.stage1_checkpoint = stage1_checkpoint or self.config["model"]["stage1"]["checkpoint"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup_directories()
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = 30  # Earlier stopping for Stage 2 (200 epochs total)
        
        self._print_header()
    
    def _setup_directories(self):
        base_dir = Path(self.config["output"]["checkpoint_dir"])
        
        self.checkpoint_dir = base_dir / f"stage{self.stage}"
        self.log_dir = Path(self.config["output"]["log_dir"]) / f"stage{self.stage}"
        self.sample_dir = Path(self.config["output"]["sample_output_dir"]) / f"stage{self.stage}"
        self.debug_dir = self.sample_dir / "debug"
        
        for d in [self.checkpoint_dir, self.log_dir, self.sample_dir, self.debug_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
    
    def _print_header(self):
        print("\n" + "=" * 100)
        print(f"{'LPTN-LITE TRAINING - STAGE 2 (768×1024)':^100}")
        print("=" * 100)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[OK] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("[!] Running on CPU (training will be VERY slow)")
        
        print(f"[OK] Device: {self.device}")
        print(f"[OK] Config: {self.config_path}")
        print(f"[OK] Stage 1 Checkpoint: {self.stage1_checkpoint}")
        print(f"[OK] Debug: {self.debug}")
        if self.resume_checkpoint:
            print(f"[OK] Resume Checkpoint: {self.resume_checkpoint}")
        print("=" * 100)
    
    def _load_stage1_model(self) -> IOANet:
        """Load and freeze Stage 1 (IOANet) model."""
        print(f"\n[Stage 1 Model Loading]")
        
        stage1_config = self.config["model"]["stage1"]
        ioanet = IOANet(
            in_channels=stage1_config["input_channels"],
            out_channels=stage1_config["output_channels"],
            pretrained=False  # Don't use ImageNet pretraining; load from checkpoint
        )
        
        # Load Stage 1 checkpoint
        checkpoint_path = Path(self.stage1_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {self.stage1_checkpoint}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        try:
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                ioanet.load_state_dict(checkpoint['model_state_dict'])
                print(f"  [OK] Loaded model_state_dict from {checkpoint_path.name}")
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                ioanet.load_state_dict(checkpoint['state_dict'])
                print(f"  [OK] Loaded state_dict from {checkpoint_path.name}")
            else:
                ioanet.load_state_dict(checkpoint if not isinstance(checkpoint, dict) else checkpoint)
                print(f"  [OK] Loaded weights (legacy format)")
        except Exception as e:
            raise RuntimeError(f"Failed to load Stage 1 weights: {e}")
        
        ioanet = ioanet.to(self.device)
        
        # Freeze all IOANet parameters
        for param in ioanet.parameters():
            param.requires_grad = False
        
        # Ensure IOANet is in eval mode (keeps batch norm stats frozen)
        ioanet.eval()
        
        print(f"  [OK] Model: Frozen (requires_grad=False)")
        print(f"  [OK] Mode: eval() (batch norm statistics frozen)")
        
        return ioanet
    
    def _create_model(self, ioanet: IOANet) -> LPIOANet:
        """Create Stage 2 model with frozen IOANet and trainable LaplacianRefiner."""
        stage2_config = self.config["model"]["stage2"]
        
        model = LPIOANet(
            ioanet_model=ioanet,
            base_channels=stage2_config.get("base_channels", 32),
            num_levels=stage2_config.get("num_levels", 3),
            refine_blocks=stage2_config.get("refine_blocks", 2),
            mask_scale=stage2_config.get("mask_scale", 1.0)  # Risk 2 fix: Pass mask_scale from config
        )
        
        model = model.to(self.device)
        
        # Print model information
        print(f"\n[Stage 2 Model Summary]")
        total_params = count_parameters(model)
        print(f"  Total Parameters: {total_params['total']:,}")
        print(f"  Trainable Parameters: {total_params['trainable']:,}")
        print(f"  Frozen Parameters: {total_params['frozen']:,}")
        
        return model
    
    def _load_checkpoint(self, model: nn.Module, optimizer) -> int:
        """Load Stage 2 checkpoint for resuming training."""
        start_epoch = 0
        
        if not self.resume_checkpoint:
            return start_epoch
        
        checkpoint_path = Path(self.resume_checkpoint)
        if not checkpoint_path.exists():
            print(f"\n[!] Checkpoint not found: {self.resume_checkpoint}")
            return start_epoch
        
        print(f"\n[Stage 2 Checkpoint Load]")
        print(f"  Path: {self.resume_checkpoint}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        try:
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  [OK] Model: Loaded model_state_dict")
            else:
                model.load_state_dict(checkpoint if not isinstance(checkpoint, dict) else checkpoint)
                print(f"  [OK] Model: Loaded weights")
        except Exception as e:
            print(f"  [!] Error loading model weights: {e}")
            return start_epoch
        
        # Load optimizer state
        if isinstance(checkpoint, dict):
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f"  [OK] Optimizer: Loaded state_dict")
                except Exception as e:
                    print(f"  [!] Error loading optimizer: {e}")
            
            # Restore training state
            start_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', float('inf')))
            self.patience_counter = checkpoint.get('patience_counter', 0)
            
            print(f"  [OK] Training State: Epoch={start_epoch}, Best Loss={self.best_val_loss:.4f}, Patience={self.patience_counter}")
        
        print()
        return start_epoch
    
    def _save_debug_images(
        self,
        epoch: int,
        input_img: torch.Tensor,
        target_img: torch.Tensor,
        output: torch.Tensor,
        batch_idx: int = 0,
        sample_num: int = 1
    ) -> Path:
        """Save debug visualization."""
        if not self.debug:
            return None
        
        inp_np = (input_img[batch_idx].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        tgt_np = (target_img[batch_idx].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out_np = (output[batch_idx].permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        h, w = inp_np.shape[:2]
        label_height = 30
        
        def add_label(img, text):
            labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
            labeled[label_height:, :, :] = img
            cv2.putText(labeled, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return labeled
        
        inp_labeled = add_label(inp_np, "Input (Shadow)")
        tgt_labeled = add_label(tgt_np, "Target (Clean)")
        out_labeled = add_label(out_np, "Output (Refined)")
        
        combined = np.concatenate([inp_labeled, tgt_labeled, out_labeled], axis=1)
        
        save_path = self.debug_dir / f"epoch_{epoch:04d}_sample{sample_num}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        return save_path
    
    def train(self):
        print("\n" + "=" * 100)
        print(f"{'LPTN-Lite Training - Stage 2 (768×1024)':^100}")
        print("=" * 100)
        
        config = self.config["training"]["stage2"]
        
        # Load data
        resolution = tuple(self.config["data"]["input_resolution_stage2"])
        aug_cfg = self.config["data"].get("augmentation_stage2", {})
        train_loader, val_loader, test_loader = create_stage2_dataloaders(
            root_dir=self.config["dataset"]["root_dir"],
            input_resolution=resolution,
            batch_size=config["batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
            augment=True,
            illumination_strength=aug_cfg.get("illumination_strength", 0.15),
            shadow_color_shift=aug_cfg.get("shadow_color_shift", 0.08),
            rotation_range=aug_cfg.get("rotation_range", 3.0),
        )
        
        # Load Stage 1 and create Stage 2 model
        ioanet = self._load_stage1_model()
        model = self._create_model(ioanet)
        
        # Only train LaplacianRefiner parameters (IOANet is frozen)
        optimizer = torch.optim.Adam(
            model.refiner.parameters(),
            lr=config["learning_rate"]
        )
        
        # Paper alignment: no LR scheduler (constant LR with Adam)
        scheduler = None
        
        # Load checkpoint if resuming
        start_epoch = self._load_checkpoint(model, optimizer)
        
        # Stage 2 uses L1 loss only (Phase 4 from Plan.md)
        print(f"\n[Stage 2 Loss Configuration]")
        print(f"  Loss: L1 Only (no LPIPS)")
        print(f"  Rationale: IOANet already solved perceptual structure; Stage 2 refines high-res pixel intensities")
        
        # Create simple L1 loss function
        loss_fn = nn.L1Loss()
        
        print(f"\n[Training Config]")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  LR Scheduler: None (paper uses constant LR)")
        print(f"  Train Batches: {len(train_loader)}")
        print(f"  Val Batches: {len(val_loader)}")
        print(f"  Dataset: {self.config['dataset'].get('root_dir')}")
        print(f"  Memory: Using torch.no_grad() for frozen IOANet (saves VRAM)")
        
        if start_epoch > 0:
            print(f"  Resuming from Epoch: {start_epoch + 1}")
        
        print("\nStarting training...\n")
        
        for epoch in range(start_epoch, config["epochs"]):
            model.refiner.train()  # Only refiner is trainable
            train_losses = []
            train_metrics = defaultdict(float)
            
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1:3d}",
                leave=True,
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )
            
            for batch_idx, batch in enumerate(pbar):
                input_img = batch["input"].to(self.device)
                target_img = batch["target"].to(self.device)
                
                # Forward pass through LP-IOANet
                # IOANet forward is wrapped in torch.no_grad() inside LPIOANet.forward()
                output = model(input_img)
                
                # L1 loss
                loss = loss_fn(output, target_img)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
                
                with torch.no_grad():
                    metrics = MetricsCalculator.compute_all(output, target_img)
                    for k, v in metrics.items():
                        train_metrics[k] += v
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'MAE': f"{metrics['mae']:.4f}",
                    'PSNR': f"{metrics['psnr']:.1f}",
                    'SSIM': f"{metrics['ssim']:.3f}"
                })
            
            avg_train_loss = np.mean(train_losses)
            num_batches = len(train_loader)
            for k in train_metrics:
                train_metrics[k] /= num_batches
            
            # TensorBoard logging
            self.writer.add_scalar("train/loss", avg_train_loss, epoch)
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            self.writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
            
            print(
                f"[Train] Epoch {epoch+1:3d} | "
                f"Loss: {avg_train_loss:.4f} | "
                f"MAE: {train_metrics['mae']:.4f} | "
                f"PSNR: {train_metrics['psnr']:.2f} dB | "
                f"SSIM: {train_metrics['ssim']:.3f}"
            )
            
            # Validation
            if (epoch + 1) % config["validation_interval"] == 0:
                model.refiner.eval()
                val_losses = []
                val_metrics = defaultdict(float)
                val_samples = []
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        input_img = batch["input"].to(self.device)
                        target_img = batch["target"].to(self.device)
                        
                        output = model(input_img)
                        loss = loss_fn(output, target_img)
                        
                        val_losses.append(loss.item())
                        
                        metrics = MetricsCalculator.compute_all(output, target_img)
                        for k, v in metrics.items():
                            val_metrics[k] += v
                        
                        if batch_idx < 4:
                            val_samples.append((input_img, target_img, output))
                
                avg_val_loss = np.mean(val_losses)
                num_val_batches = len(val_loader)
                for k in val_metrics:
                    val_metrics[k] /= num_val_batches
                
                # TensorBoard logging
                self.writer.add_scalar("val/loss", avg_val_loss, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)
                
                improved = avg_val_loss < self.best_val_loss
                status = "[+]" if improved else "[-]"
                
                print(
                    f"{status} Epoch {epoch+1:3d} | "
                    f"Loss: {avg_val_loss:.4f} | "
                    f"MAE: {val_metrics['mae']:.4f} | "
                    f"PSNR: {val_metrics['psnr']:.2f} dB | "
                    f"SSIM: {val_metrics['ssim']:.3f}"
                )
                
                # Save debug images
                if self.debug and val_samples:
                    for i, (inp, tgt, out) in enumerate(val_samples[:4]):
                        path = self._save_debug_images(epoch + 1, inp, tgt, out, sample_num=i + 1)
                        if path:
                            print(f"    ► Sample {i+1} saved: {path.name}")
                
                # Save checkpoint
                if improved:
                    self.best_val_loss = avg_val_loss
                    self.patience_counter = 0
                    
                    save_path = self.checkpoint_dir / "best_model.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'val_loss': avg_val_loss,
                        'best_val_loss': self.best_val_loss,
                        'patience_counter': self.patience_counter
                    }, save_path)
                    print(f"    ► Best model saved: {save_path.name}")
                else:
                    self.patience_counter += 1
                    print(f"    (No improvement - patience: {self.patience_counter}/{self.early_stopping_patience})")
                    
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"\n[!] Early stopping triggered at epoch {epoch + 1}")
                        break
            
            # Periodic checkpointing
            if (epoch + 1) % config["checkpoint_interval"] == 0:
                ckpt_path = self.checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': self.best_val_loss,
                    'best_val_loss': self.best_val_loss,
                    'patience_counter': self.patience_counter
                }, ckpt_path)
        
        print("\n" + "=" * 100)
        print(f"[OK] Stage 2 Training Complete!")
        print(f"[OK] Best Model: {self.checkpoint_dir / 'best_model.pth'}")
        print(f"[OK] Best Val Loss: {self.best_val_loss:.4f}")
        print(f"[OK] TensorBoard Logs: {self.log_dir}")
        print("=" * 100 + "\n")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train IOANet or LPTN-Lite for document shadow removal")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--stage", type=int, default=1, help="Training stage (1 or 2, default: 1)")
    parser.add_argument("--debug", action="store_true", default=True, help="Save debug images")
    parser.add_argument("--no-debug", action="store_false", dest="debug", help="Disable debug images")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--finetune", action="store_true", default=False, help="Fine-tune from checkpoint (Stage 1 only)")
    parser.add_argument("--stage1-checkpoint", type=str, default=None, help="Override Stage 1 checkpoint path (uses config if not provided)")
    
    args = parser.parse_args()
    
    if args.stage == 1:
        trainer = Trainer(args.config, debug=args.debug, stage=args.stage, 
                         resume_checkpoint=args.resume, finetune=args.finetune)
        trainer.train()
    elif args.stage == 2:
        trainer = Stage2Trainer(
            config_path=args.config,
            debug=args.debug,
            resume_checkpoint=args.resume,
            stage1_checkpoint=args.stage1_checkpoint
        )
        trainer.train()
    else:
        print(f"[ERROR] Invalid stage: {args.stage}. Must be 1 or 2.")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
