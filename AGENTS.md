# AGENTS.md — DocShadow-Lite fast-context handoff

> Read this first. It gives a new session the repo map, the real behavior (not the docs), exact commands, and the known traps. Full evidence is in `AUDIT_REPORT.md`.
>
> **2026-10-28 update — new retrain pipeline.** A self-contained pipeline for the
> public `nexadoc-50k` dataset now exists: `train_nexadoc.py`,
> `evaluate_nexadoc.py`, `configs/nexadoc.yaml`, `src/data/nexadoc_dataset.py`,
> `src/models/deshadow.py`, `src/models/sr_refiner.py`,
> `src/utils/eval_metrics.py`, `src/utils/nexadoc_losses.py`, and
> `tests/test_nexadoc_pipeline.py`. **Use these, not the legacy `train.py`.**
> The legacy code below is kept for reference and is still buggy. See
> `NEXADOC.md` for the dataset facts, Kaggle 2×T4 commands, and baselines.

## 1. What this project is

### 1a. New pipeline (nexadoc-50k)

- **Stage 1** — `DeshadowNet(IOANet)` at **256×256**, MobileNetV2 backbone kept,
  clamped output. 1,975,871 params. Official 40k/5k/5k deshadow split.
- **Stage 2** — `SRUNet`, a full-resolution residual U-Net (base 16, 1,233,315
  params) for **256² → 768×1024**, trained on the 10k `upsample` split. It
  replaces the old 1,879-param LP-IOANet, which cannot do super-resolution.
- Loss: Stage 1 `L1 + 2·masked-L1 + 0.1·LPIPS(AlexNet)`; Stage 2 pure `L1` (LPIPS off for best PSNR).
- Metrics: global, batch-invariant MAE/MSE/RMSE/PSNR/SSIM + correct per-pixel
  region metrics (`src/utils/eval_metrics.py`).
- Training: DDP `torchrun --nproc_per_node=2`, AMP, AdamW + cosine, EMA 0.999,
  resume, official splits (no `random_split`), config/RNG-safe checkpoints.
- Tests: `tests/test_nexadoc_pipeline.py` (7 tests). Full suite is now 68 tests.

Quick start (details in `NEXADOC.md`):

```bash
torchrun --nproc_per_node=2 train_nexadoc.py --stage 1 --data-root /kaggle/input/nexadoc-50k
torchrun --nproc_per_node=2 train_nexadoc.py --stage 2 --data-root /kaggle/input/nexadoc-50k
python evaluate_nexadoc.py --stage 1 --data-root /kaggle/input/nexadoc-50k
```

### 1b. Legacy pipeline (Kligler / old checkpoints)

The rest of this document describes the legacy two-stage code.

Two-stage document shadow removal:

1. **Stage 1 — IOANet** (`src/models/ioanet.py`): MobileNetV2 encoder + lightweight decoder + coordinate attention, 192×256 (W×H), predicts a residual added to the input. **1,975,871 params.** Output is **not clamped** (intentional; tests assert it).
2. **Stage 2 — LP-IOANet** (`src/models/laplacian_refiner.py`): input downsampled /4 (768×1024 → 192×256), frozen Stage 1 runs there, then a Laplacian refiner works back up at 768×1024 (W×H). Refiner has only **1,879 trainable params**. Output clamped to [0,1].

Loss: `10·L1 + 5·LPIPS(AlexNet)` (Stage 1); Stage 2 uses plain L1 on the refiner.
Metrics: MAE, MSE, RMSE, PSNR, SSIM (`src/utils/losses.py`), plus region metrics (buggy — see §8).

## 2. Current state (2026-10-28)

- Audit complete. Legacy source is **unchanged**; new NexaDoc pipeline added (see §1a).
- Tests: **68 passed, 46 warnings, 22.94 s** (CPU) via `.venv\Scripts\python.exe -m pytest tests -q`.
- **Legacy checkpoints are pre-fix artifacts** trained with augmentation silently disabled (see §8). Do not use them as the new baseline.
- The tracked `README.md` is **deleted in the working tree**; recover with `git show HEAD:README.md`. `NEXADOC.md` documents the new pipeline; `docs/` is empty.
- Git works only with an explicit safe.directory (see §3).

## 3. Environment (Windows, PowerShell)

```powershell
# ALWAYS use the venv interpreter; bare `python` is the Windows Store stub
.\.venv\Scripts\python.exe --version        # Python 3.14.5
.\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
# 2.14.1+cpu 0.29.1+cpu False
```

- No CUDA. All training/eval here is CPU-slow. Cloud training was done on Kaggle P100 (notebook).
- Other versions: numpy 2.5.3, opencv 5.0.0, lpips installed.
- Git repo has “dubious ownership” (different SID). Prefix git commands:

```powershell
git -c safe.directory="D:/Archive/Side Projects/Projects/DocShadow-Lite" status
```

- Remote: `https://github.com/walidmoustafa2077/DocShadow-Lite.git`; HEAD `73bfc8f` (2026-08-22).
- Working tree: `D README.md`, `M .gitignore`, `M evaluate.py`, `M inference.py`, `M requirements.txt`; untracked `Data/` (276 MB) and the audit/export scripts. Only 18 files tracked.
- `Data/Mixed_Stage1` does **not** exist, so `--mixed` cannot run as-is.

## 4. Repo map

```
train.py                     # Stage 1 Trainer + Stage 2 Stage2Trainer + CLI (1049 lines)
evaluate.py                  # metrics + sample images; supports fp32/fp16/int8 dequant
inference.py                 # single-image / folder inference, comparison PNGs
configs/config.yaml          # all hyperparameters (single source of truth)
requirements.txt             # unpinned; no onnx
src/
  models/ioanet.py           # CoordinateAttention, MobileNetV2Encoder, FBDecoder, IOANet
  models/laplacian_refiner.py# LaplacianDecomposer, RefinementBlock, LPIOANet
  data/dataset.py            # ShadowRemovalDataset (192×256), HighResolutionDataset (768×1024), factories
  data/mixed_dataset.py      # tag-aware mixed dataset + MixedBatchSampler (needs flat merged data)
  utils/losses.py            # ShadowRemovalLoss, MetricsCalculator, RegionBasedMetrics
scripts/
  mixed_dataset.py           # merge TAG=PATH sources into flat Data/Mixed_Stage1
  preprocess_dataset.py      # resize/cache a dataset (used by Kaggle notebook)
  audit_checkpoints.py       # integrity + quality audit → outputs/audit/
  compare_stage1.py          # v1/v2/v3 on Data/Kligler/test2 (87 pairs)
  export_compact.py          # fp32/fp16/int8 strip-optimizer export (int8 broken)
  export_onnx.py             # Stage 1/2 ONNX export
  export_onnx_combined.py    # combined pipeline ONNX (duplicates forward)
tests/                       # 5 test modules, 61 tests; git-ignored (!)
notebook/doc-shadow-lite.ipynb  # Kaggle P100 workflow
Data/Kligler/                # train 272, test 28, test2 87 (see §6)
outputs/                     # checkpoints, samples, eval images, audit
```

No `README.md` in the working tree, no `LICENSE`, no `pyproject.toml`/`setup.py`, no CI/lint config, no `Plan.md` (many comments reference it).

## 5. Commands

```powershell
# Train Stage 1 (defaults from config; on CPU this is effectively a smoke test)
.\.venv\Scripts\python.exe train.py --stage 1
.\.venv\Scripts\python.exe train.py --stage 1 --mixed        # requires Data/Mixed_Stage1
.\.venv\Scripts\python.exe train.py --stage 1 --resume outputs/checkpoints_v2/stage1/best_model.pth
.\.venv\Scripts\python.exe train.py --stage 1 --finetune outputs/checkpoints_v2/stage1/best_model.pth
.\.venv\Scripts\python.exe train.py --stage 1 --lr 1e-4

# Train Stage 2 (needs a Stage 1 checkpoint; config model.stage1.checkpoint by default)
.\.venv\Scripts\python.exe train.py --stage 2
.\.venv\Scripts\python.exe train.py --stage 2 --stage1-checkpoint outputs/checkpoints_v2/stage1/best_model.pth

# Evaluate
.\.venv\Scripts\python.exe evaluate.py --model outputs/checkpoints_v2/stage1/best_model.pth --split test --batch-size 1
.\.venv\Scripts\python.exe evaluate.py --model outputs/checkpoints_v1/stage2/best_model.pth --stage 2 --split test
.\.venv\Scripts\python.exe evaluate.py --model outputs/deploy/stage1_int8.pth --dtype int8

# Inference
.\.venv\Scripts\python.exe inference.py --stage 1 --model outputs/checkpoints_v2/stage1/best_model.pth --input path\to\img.jpg --output outputs\inference
.\.venv\Scripts\python.exe inference.py --stage 2 --model outputs/checkpoints_v1/stage2/best_model.pth --input path\to\img.jpg --dataset-root Data/Kligler --preserve-aspect

# Tests
.\.venv\Scripts\python.exe -m pytest tests -q

# Checkpoint audit (writes outputs/audit/checkpoint_audit.{json,md}; CPU-slow for Stage 2)
.\.venv\Scripts\python.exe scripts\audit_checkpoints.py --split test
```

Note: `--lr`/`--finetune` are accepted but **ignored for Stage 2**. `train.py` has no `--seed`.

## 6. Data layout

```
Data/Kligler/
  train/{input,target,mask}/   272 each   (22 document prefixes: natiCHW04/05, CPO01-08, CPR01-05, CSY01-07)
  test/{input,target,mask}/     28 each   (CHW01-03 — disjoint from train, good)
  test2/{input,target}/         87 each   (rect* — unrelated; no masks; NOT reachable by split resolver)
```

- Train images are large and varied (e.g. 1986×2750, 1017×1445, 2510×3306, 2034×3582); all are stretched to 192×256 (aspect distortion).
- `create_dataloaders` splits **images**, so val shares documents with train → validation leakage.
- `_resolve_data_folder` maps any split other than `train`/`val` to `root/test`; `--split test2` silently evaluates `test` (28 images), not the 87-pair set.
- `scripts/compare_stage1.py` is the only script that reads `test2` directly.
- Datasets resize masks **before** augmentation (bug C2) and use `INTER_LINEAR` for masks in the merge scripts.

## 7. Checkpoints / artifacts

| Path | What | Notes |
|---|---|---|
| `outputs/checkpoints_v1/stage1/best_model.pth` | Stage 1 v1, 24 MB | epoch 285, val 0.1799 |
| `outputs/checkpoints_v1/stage2/best_model.pth` | Stage 2 v1, 8 MB | epoch 75, val 0.0159 |
| `outputs/checkpoints_v2/stage1/best_model.pth` | Stage 1 v2, 24 MB | epoch 170, val 0.1774; referenced by config |
| `outputs/checkpoints_v3/stage1/best_model.pth` | Stage 1 v3, 24 MB | epoch 155, val 0.2182; **not in git** |
| `outputs/deploy/stage1_fp16.pth` | fp16, 4 MB | ≈ v2 numerically |
| `outputs/deploy/stage1_int8.pth` | int8, 2 MB | **broken for inference**; needs dequant |
| `outputs/audit/checkpoint_audit.{json,md}` | prior audit | metrics unreliable, see §8 |
| `outputs/eval_vis*`, `outputs/inference_cell_pic_doc`, `outputs/samples` | visuals | 107 PNG, ~263 MB |

All checkpoints contain `model_state_dict, optimizer_state_dict, epoch, val_loss, best_val_loss, patience_counter` (Stage 2 adds the frozen IOANet weights). No scheduler, config, RNG, or git SHA.

Approx quality from the prior audit (unreliable — batch-dependent PSNR, 3× region bug):
v3 0.0398 / 27.41 / 0.9305; v2 0.0402 / 27.31 / 0.9369; fp16 same; v1 0.0446 / 26.32 / 0.9346; Stage 2 0.0452 / 26.62 / 0.9045; int8 0.2251 / 11.46 / 0.4381.

## 8. Known bugs / traps (details + evidence in AUDIT_REPORT.md)

**Critical — do not trust current training/metrics:**

1. **Augmentation is silently off everywhere.** Factories set `val_dataset.dataset.augment = False` on the shared dataset (`dataset.py:381-382`, `699-701`; `mixed_dataset.py:380-381`). Verified: train and val both `augment=False`. All shipped checkpoints trained without augmentation.
2. **Re-enabling augmentation crashes.** Mask is resized to 192×256 before `_augment`, while input/target are full-res → `ValueError: operands could not be broadcast (1017,1714,3) vs (256,192,3)`. Fix ordering: resize all three first, then augment.
3. **Region metrics 3× wrong.** Numerator sums 3 channels, denominator counts 1 (`losses.py:236-251`) → shadow MAE/MSE 3× high, shadow PSNR ~4.77 dB low. Verified 0.060519 vs correct 0.020173.
4. **PSNR depends on batch size** (per-batch means): 26.71 dB at batch 8 vs 28.26 dB at batch 1 for v2/test. Rank comparisons (v3 67.43 vs v2 67.38) are noise.

**High:**

5. **Val leakage:** image-level `random_split`, documents appear in both train and val.
6. **Stage 2 is tiny:** 1,879 trainable params; `num_levels` hardcoded to 3 in `forward`; one `RefinementBlock` despite the plural name.
7. **int8 unusable:** export quantizes every fp32 tensor incl. BatchNorm; `inference.py` has no dequant → `Unexpected key(s) ..._scale`.
8. **`weights_only=False`** in evaluate/inference/scripts (pickle RCE). `train.py` omits it (safe default on torch ≥2.6) — inconsistent.
9. **`test2` silently maps to `test`**; 87 pairs are dead data.
10. **Stage 1 unclamped** but documented/exported as [0,1]; LPIPS gets `y*2-1`; evaluate uses raw output while saving clips.

**Medium/low:** SSIM every training batch (`train.py:415`); global `warnings.filterwarnings("ignore")`; no seeding/config snapshot; test loader unused; README/doc drift (3.10M params, early-stop semantics, CLI names); masks resized linearly/JPEG; merge subsets per folder (desync risk); duplicated checkpoint logic 5×; `tests/` git-ignored but `Data/` not; deprecated `pretrained=`; double attention in debug; recursive `__getitem__` fallback; `--lr` ignored for Stage 2; dead config/`Plan.md` refs; inference caps at 50 files, misses `.bmp`/uppercase, split guessed by `"test"` substring.

## 9. Kaggle notebook reality check

`notebook/doc-shadow-lite.ipynb` clones the GitHub repo and patches a config. It preprocesses only the **192×256** cache (`fast_data`); the 768×1024 block is commented out. It actually runs **Stage 1 only**; Stage 2 would train/evaluate on 192×256 images upsampled to 768×1024. Stage 2 batch is 7 (markdown says 6), epochs 60. Do not assume the notebook reproduces the config in `configs/config.yaml`.

## 10. If you continue this work

Order of operations (full rationale in `AUDIT_REPORT.md` §6):

1. Fix augmentation propagation + resize/augment order; add a test that the train loader is stochastic.
2. Fix region metric normalization; add numeric tests.
3. Fix metric aggregation to global sums; assert PSNR identical for batch 1 and 8.
4. Re-run `evaluate.py` and regenerate `outputs/audit/`.
5. Replace image-level split with document/prefix-level split.
6. Decide and enforce the Stage 1 `[0,1]` contract.
7. Fix/remove int8; switch to `weights_only=True`; centralize checkpoint I/O.
8. Seed everything and store config/RNG/git SHA in checkpoints.
9. Restore/correct README; un-ignore `tests/`; ignore `Data/`.

**Do not** cite current PSNR/SSIM/shadow numbers as final, and do not compare Stage 1 (192×256) with Stage 2 (768×1024) on one leaderboard.

## 11. Handy facts

- Stage 1 params: 1,975,871. Stage 2 trainable: 1,879. Total Stage 2 file has 466 tensors.
- CPU inference: Stage 1 ~156 ms/image, Stage 2 ~238 ms/image.
- Tests take ~99 s on CPU because of real model forwards; `tests/` is ignored by git.
- `outputs/checkpoints_v3/stage1` was never committed (outputs are ignored).
- No secrets in the repo. Remote is public.
- `README.md` at HEAD is the only prose documentation; restore it before editing docs.
