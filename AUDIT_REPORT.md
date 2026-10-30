# DocShadow-Lite — Technical Audit

**Audit date:** 2026-10-28
**Scope:** full repository at `D:\Archive\Side Projects\Projects\DocShadow-Lite` — training/eval/inference code, datasets, models, losses, scripts, tests, config, notebook, checkpoints, repo hygiene.
**Method:** full source read, static analysis, dataset inspection, and empirical runs (CPU, `.venv`). No code was modified.
**Environment:** Python 3.14.5, torch 2.14.1+cpu, torchvision 0.29.1+cpu, CUDA unavailable.

---

## 1. Executive summary

The project implements a two-stage document shadow-removal pipeline (IOANet at 192×256, then a Laplacian refiner at 768×1024) and ships six trained checkpoints. It runs end-to-end, and the unit tests pass (61 passed). However, the audit found three classes of problems that undermine the reported results:

1. **Training never actually used augmentation.** Every dataloader factory disables augmentation on the shared dataset after creating the splits, and re-enabling it exposes a second latent bug (mask resized before augmentation) that crashes on the real dataset. The shipped checkpoints were trained on un-augmented, aspect-distorted, image-level-split data.
2. **The evaluation numbers are not trustworthy.** Region-based MAE is exactly 3× too high, region PSNR is ~4.77 dB too low, and global PSNR changes with batch size (26.71 dB at batch 8 vs 28.26 dB at batch 1 for the same checkpoint/data). Existing "best checkpoint" rankings are therefore not statistically or methodologically sound.
3. **Stage 2 cannot deliver what the documentation implies.** The refiner has only **1,879 trainable parameters**, the flow is hardcoded to 3 levels, and the notebook never generates the 768×1024 cache, so Stage 2 is effectively trained/evaluated on upsampled low-res images.

The code is generally readable and well-commented, and the model implementation is sound. The biggest wins are fixing the data pipeline and metric aggregation before any further training or publication.

---

## 2. Severity-ranked findings

| ID | Severity | Finding | Location |
|----|----------|---------|----------|
| C1 | **Critical** | Augmentation is silently disabled for train *and* val in all three dataloader factories | `src/data/dataset.py:381-382, 699-701`; `src/data/mixed_dataset.py:380-381` |
| C2 | **Critical** | Re-enabling augmentation crashes: mask is resized to 192×256 before `_augment` sees the full-resolution input | `src/data/dataset.py:185-196`; `mixed_dataset.py:147-158`; `dataset.py:551-562` |
| C3 | **Critical** | Region metrics are 3× inflated (MAE) and ~4.77 dB low (PSNR) | `src/utils/losses.py:236-251` |
| C4 | **Critical** | Global PSNR depends on batch size (per-batch averaging) | `train.py`, `evaluate.py`, `scripts/audit_checkpoints.py` |
| C5 | **High** | Train/val split is at image level, not document level → validation leakage | `src/data/dataset.py:353-382`; `dataset.py:672-701` |
| C6 | **High** | Stage 2 refiner has only 1,879 trainable params; `num_levels` is not honored | `src/models/laplacian_refiner.py:371` |
| C7 | **High** | int8 checkpoint cannot be used by documented inference (no dequantization) | `scripts/export_compact.py:99-104`; `inference.py:51-90` |
| C8 | **High** | `torch.load(..., weights_only=False)` on all checkpoint paths → pickle RCE risk | `evaluate.py:90/103/119`, `inference.py:51/69/90`, `scripts/*` |
| C9 | **High** | `--split test2` silently evaluates the `test` split; `test2` is dead data | `src/data/dataset.py:16-37`; `scripts/audit_checkpoints.py:302` |
| C10 | **High** | Stage 1 output is unbounded but documented/exported as `[0,1]`; LPIPS and metrics consume out-of-range values | `src/models/ioanet.py:322`; `scripts/export_onnx.py:139-140` |
| M1 | Medium | Full SSIM computed on every training batch, despite `_cheap_metrics` existing | `train.py:414-425` |
| M2 | Medium | `warnings.filterwarnings("ignore", UserWarning)` hides data-integrity and deprecation warnings | `train.py:6` |
| M3 | Medium | No global seeding; checkpoints omit RNG/config → runs not reproducible | `train.py` (`main`), checkpoint dicts |
| M4 | Medium | `test_loader` is created but never evaluated; no final test pass | `train.py:328-434, 309-546` |
| M5 | Medium | README/doc drift: 3.10M params, early-stop semantics, CLI flags, split behavior | `README.md` (at HEAD); `configs/config.yaml` |
| M6 | Medium | Mask resizing uses `INTER_LINEAR` and preserves `.jpg` → non-binary/compressed masks | `scripts/mixed_dataset.py`, `scripts/preprocess_dataset.py` |
| M7 | Medium | Mixed/preprocess merge subsets each folder independently → possible input/target/mask desync | `scripts/mixed_dataset.py`, `scripts/preprocess_dataset.py` |
| M8 | Medium | Checkpoint-unwrap + dequant logic duplicated 5×, already drifting | `train.py`, `evaluate.py`, `inference.py`, `scripts/audit_checkpoints.py`, `test_checkpoint.py` |
| M9 | Medium | `tests/` is git-ignored; data (`Data/`) is not; checkpoints/outputs ignored | `.gitignore` |
| L1 | Low | Deprecated `models.mobilenet_v2(pretrained=...)` | `src/models/ioanet.py:120` |
| L2 | Low | `output_attention` is run twice in debug path | `src/models/ioanet.py:325` |
| L3 | Low | Recursive `__getitem__` fallback can recurse deeply/forever on corrupt data | `dataset.py:162/173`, `mixed_dataset.py:131/137` |
| L4 | Low | Stage 2 silently ignores `--lr` and `--finetune` | `train.py:1033-1040` |
| L5 | Low | Unused config keys/dead code (`augmentation.enabled`, stage2 `l1_weight`, stage2 batch docstring, dead merge helper) | multiple |
| L6 | Low | `inference.py` glob misses `.bmp`/uppercase, caps at 50 images silently, split guessed by substring `"test"` | `inference.py:238-270` |
| L7 | Low | Audit composite score is arbitrary and ranks 192×256 vs 768×1024 on one table | `scripts/audit_checkpoints.py:228-241` |
| L8 | Low | ONNX scripts duplicate model forward and have no runtime dependency/verification | `scripts/export_onnx*.py` |
| L9 | Low | No LICENSE, no CI, no lint/format config, no `pyproject.toml`; `docs/` empty | repo root |
| L10 | Low | `inference.py` always calls `process_image_stage1`, even for Stage 2 | `inference.py:259` |

---

## 3. Detailed findings

### C1 — Augmentation is never applied (critical)

All three factories do the equivalent of:

```python
full_dataset = ShadowRemovalDataset(..., augment=True)
train_ds, val_ds = random_split(full_dataset, ...)
if hasattr(val_ds.dataset, "augment"):
    val_ds.dataset.augment = False   # val_ds.dataset IS full_dataset
```

`random_split` returns `Subset` views over the *same* dataset object, so this line turns augmentation off for the training split too. Verified empirically on the real `Data/Kligler`:

```
create_dataloaders(..., augment=True)
tr.dataset.dataset.augment -> False
val.dataset.dataset.augment -> False
```

Same result for `create_stage2_dataloaders` and for the mixed factory (`full_dataset.augment = False`, while the train loader uses `full_dataset` directly).

**Impact:** every shipped checkpoint was trained with zero flips, zero illumination variation, zero shadow-color shift, and no random crop. This alone invalidates claims that the model was trained with the augmentation described in `configs/config.yaml`.

**Fix:** make `augment` a per-sample/per-subset property. Simplest robust approach: split indices first, then build two dataset instances (or wrap each `Subset` in a view that overrides `augment`). Never mutate the shared dataset. Add a regression test that two epochs of the train loader are not identical given `augment=True`.

### C2 — Re-enabling augmentation crashes on the real dataset (critical)

In `ShadowRemovalDataset.__getitem__`, the mask is resized to `(192,256)` *before* `_augment` is called; input and target are still at original resolution (~1700×3300). `_augment` then does:

```python
color_shift = ... * shadow_mask[..., None]   # mask is 256×192
input_np = input_np * (1 + color_shift)      # input is full-res
```

Verified repro after manually forcing `augment=True`:

```
ValueError: operands could not be broadcast together with shapes (1017,1714,3) (256,192,3)
```

The same ordering exists in `HighResolutionDataset` and `MixedShadowRemovalDataset`. The mixed set would not crash (it is pre-resized flat data), but it is also disabled by C1.

**Impact:** the two bugs mask each other; fixing C1 alone makes training fail immediately. Random crop and rotation are also applied at full resolution and then followed by a resize, which is not the intended “crop to 256×192” behavior.

**Fix:** resize/normalize input, target, and mask to the working resolution *first*, then apply augmentation, then (optionally) random-crop. Unify the pipeline into one shared `_prepare` function used by all three datasets.

### C3 — Region metrics are 3× wrong (critical)

```python
shadow_pixels = binary_mask.sum()                      # counts pixels (1 channel)
shadow_mae = (torch.abs(output-target)*mask).sum() / shadow_pixels
```

The numerator sums over **3 color channels**, the denominator over **1**, so shadow MAE and shadow MSE are exactly 3× too large. Verified on one image: reported `shadow_mae = 0.060519`, correct per-channel-pixel MAE = `0.020173` (ratio 3.000). Shadow PSNR is consequently `10·log10(3) ≈ 4.77 dB` too low. Overall (non-masked) metrics are computed correctly.

**Impact:** all “shadow region” numbers in `outputs/audit/checkpoint_audit.md` and in `evaluate.py` output are wrong. The non-shadow branch has the same bug at `losses.py:247-251`.

**Fix:** divide by `mask.sum() * output.shape[1]`, or compute per-channel means and average. Add a numeric unit test that compares against a hand-computed mask case (the current test only checks key presence).

### C4 — PSNR depends on batch size (critical)

Metrics are accumulated as the mean of per-batch means, not from global squared error:

```
batch 8: MAE 0.0402, PSNR 26.71, SSIM 0.9369
batch 1: MAE 0.0402, PSNR 28.26, SSIM 0.9369
```

`outputs/audit/checkpoint_audit.md` reports yet another value (27.31) at its own batch size. PSNR is a log of a mean, so averaging per-batch PSNR is not the dataset PSNR.

**Impact:** reported PSNR is not comparable across runs, batch sizes, or scripts. The “best checkpoint” ranking (v3 67.43 vs v2 67.38) is within this noise and is meaningless.

**Fix:** accumulate `sum_sq_error`, `sum_abs_error`, and pixel count over the dataset; compute MAE/PSNR/RMSE globally. For SSIM, accumulate the per-image score weighted by image count (or report median + std). Add a test asserting identical PSNR for batch sizes 1 and 8.

### C5 — Validation leakage via image-level split (high)

`random_split(..., generator=manual_seed(42))` splits individual images. The dataset contains multiple pages from the same documents (train prefixes: `natiCHW04/05`, `CPO01-08`, `CPR01-05`, `CSY01-07`). A random split puts pages of the same document in both train and val, so `val_loss` is optimistic and early stopping/`best_model.pth` selection is biased.

The separate `test` split *is* document-disjoint (train prefixes vs `CHW01-03`), which is good — but the val split used for model selection is not.

**Fix:** split by document prefix (or provide explicit `train/val` folders) and verify zero prefix overlap. Consider a 3-way document-level split.

### C6 — Stage 2 refiner is tiny and not configurable (high)

Measured with `count_parameters(..., trainable_only=True)`:

- IOANet (Stage 1): **1,975,871** params (~1.98 M)
- LP-IOANet refiner (Stage 2 trainable): **1,879** params

The refiner is three depthwise-separable residual blocks at `base_channels=16` plus a mask head. With ~1.9 k trainable parameters it can only learn a very local correction, which matches the empirical result that Stage 2 is not better than Stage 1 (`v1_stage2` 64.35 vs `v3` 67.43 in the prior audit — different resolutions, so also inconclusive).

Additionally, `LaplacianRefiner.forward` hardcodes `mid_level = self.num_levels - 2` and always consumes `L0/L1/base`; `num_levels=2` double-refines L0, `num_levels=4` silently ignores a level. `RefinementBlock` is plural but instantiated once.

**Impact:** Stage 2 architecture does not match the “3-level Laplacian refiner” description in a generalizable way, and its capacity is likely too low for the intended 4× upscale correction.

**Fix:** either accept Stage 2 as a light refiner and document it honestly, or widen it (`base_channels` 32–64, more blocks) and implement a general N-level loop. Add tests for `num_levels` ∈ {2,3,4}.

### C7 — int8 artifact is unusable (high)

`scripts/export_compact.py` quantizes **every** fp32 tensor, including BatchNorm `running_mean`/`running_var`, and stores `<key>_scale` entries. `inference.py` has no dequantization and no dtype argument; loading the int8 file raises:

```
RuntimeError: Error(s) in loading state_dict for IOANet: Unexpected key(s) ... weight_scale ...
```

Only `evaluate.py` and `scripts/audit_checkpoints.py` dequantize. The prior audit also shows catastrophic quality: MAE 0.2251, PSNR 11.46 dB, SSIM 0.4381 (score 13.14) versus fp16 ≈ v2 (67.34).

**Impact:** the “deploy” int8 checkpoint cannot be used by the documented inference path and is not a viable deployment artifact.

**Fix:** quantize only Conv/Linear weights (per-channel), keep BN and biases in fp32, or export a real quantized model via `torch.ao.quantization`/ONNX QDQ. Centralize loading + dequantization in one utility. Alternatively delete the int8 artifact.

### C8 — Unsafe checkpoint loading (high)

`evaluate.py`, `inference.py`, and all scripts call `torch.load(..., weights_only=False)`. This executes arbitrary pickles. If checkpoints are ever downloaded from Kaggle/another user, this is a remote-code-execution vector. `train.py` omits the argument (on torch ≥ 2.6 the default is `weights_only=True`), so behavior is inconsistent across entry points.

**Fix:** use `weights_only=True` everywhere (the checkpoints are plain tensors/dicts and load fine), or migrate to `safetensors`. If a legacy payload truly needs `False`, isolate it behind an explicit `--trust-checkpoint` flag and document the trust boundary.

### C9 — `test2` is silently the `test` split (high)

`_resolve_data_folder` maps `train`/`val` to `root/train` and **everything else** to `root/test`. `scripts/audit_checkpoints.py` advertises `--split test2` but scores `test`; `evaluate.py --split test2` prints “Test samples: 28” and returns identical test metrics. `Data/Kligler/test2` (87 input/target pairs, no masks, `rect*` filenames) is never used except by `scripts/compare_stage1.py`, which hardcodes it.

**Fix:** accept explicit split directories (e.g. `--split-dir`), validate the split name against existing folders, or add a `test2` branch. Otherwise delete `test2` to avoid false confidence.

### C10 — Unbounded Stage 1 output vs documented `[0,1]` (high/medium)

`IOANet.forward` returns `output_attention(residual) + x_attended` with no clamp. Tests explicitly assert this is intentional (“unbounded residual”). But:

- `ShadowRemovalLoss` feeds `output*2-1` to LPIPS, which assumes `[0,1]` inputs.
- `evaluate.py` computes metrics on raw output, while only saved images are clipped.
- `scripts/export_onnx.py` claims the ONNX output is in `[0,1]` (false for Stage 1; true for Stage 2, which clamps).
- `inference.py` clips for saving, so the deployed behavior differs from evaluated behavior.

On one sample the raw range was `[0.094, 0.731]`, so it is not currently catastrophic, but there is no guarantee.

**Fix:** decide the contract. If `[0,1]` is intended, clamp in `IOANet.forward` (and update tests) or clamp before LPIPS/metrics. If unbounded is intentional, remove `[0,1]` claims from docs/ONNX and evaluate post-clamp metrics as the deployment metric.

### M1 — SSIM on every training batch (medium)

`train.py:414-425` calls `MetricsCalculator.compute_all` for every training batch, including a 6-conv Gaussian SSIM pass. `_cheap_metrics` exists specifically to avoid this but is only used in Stage 2. On CPU/GPU this is a large, avoidable slowdown (and the docstring says SSIM is intentionally excluded, contradicting the code).

**Fix:** use `_cheap_metrics` in the training loop; compute SSIM only on validation (or a sampled subset).

### M2 — Global warning suppression (medium)

`train.py:6` ignores all `UserWarning`s. This hides the dataset resolution banner, torchvision deprecation warnings, and any future library warnings. Dataset mismatch warnings are therefore invisible.

**Fix:** remove the global filter; silence only known third-party warnings by module, or make it opt-in via `--quiet`.

### M3 — Reproducibility (medium)

Only the split generator and mixed sampler are seeded (42). Python `random`, NumPy, and torch are not seeded; cuDNN is not deterministic; checkpoints store model/optimizer/epoch/val_loss/patience but no RNG state, config snapshot, or git SHA. Resuming is therefore not bit-reproducible and the exact training config is not recoverable from a checkpoint.

**Fix:** add `--seed`, seed all libraries, set `torch.use_deterministic_algorithms` where feasible, save RNG states + resolved config + git SHA in checkpoints.

### M4 — Test loader never used (medium)

Stage 1 creates a test loader (when a dedicated `test` split exists) but `Trainer.train` never evaluates it. Training ends without a test report; users must run `evaluate.py` separately (which is fine, but the unused loader suggests missing behavior). There is no final “best checkpoint” evaluation or artifact copy.

**Fix:** either remove the unused loader or run a final test evaluation and write a metrics JSON next to `best_model.pth`.

### M5 — Documentation drift (medium)

The tracked `README.md` is **deleted in the working tree** (`git status: D README.md`), recoverable via `git show HEAD:README.md`. Its claims vs reality:

| Claim | Reality |
|---|---|
| ~3.10 M parameters | Stage 1 is 1,975,871; Stage 2 trainable refiner is 1,879 |
| Early stop after 50/30 epochs | Patience increments per validation event → 250/150 epochs |
| Validation every 5/2 epochs | Config says 5/5 |
| `preprocess_dataset --input/--output` | Actual CLI is `--source/--dest` |
| Output `[0,1]` | Stage 1 unclamped |
| Single `root_dir` for both stages | Stage 2 has no separate root; notebook reuses 192×256 cache |

`docs/` is empty. All config comments reference a non-existent `Plan.md`. Config comment says LPIPS VGG-16; code uses AlexNet.

**Fix:** restore and correct README, document the actual split/metric semantics, and add a short `docs/ARCHITECTURE.md`. Remove `Plan.md` references or add the file.

### M6 — Mask handling in preprocessing (medium)

`mixed_dataset.py` / `preprocess_dataset.py` resize masks with `INTER_LINEAR` and preserve the source extension. A `.jpg` mask gets JPEG compression, producing soft/gray edges that are later binarized at 0.3. This introduces label noise and non-reproducible binarization.

**Fix:** resize masks with `INTER_NEAREST`, always write PNG, and binarize at write time (`>127`).

### M7 — Pairing desync risk in merge scripts (medium)

Fraction subsetting is computed independently per folder from each folder’s own sorted list. If any folder has an extra/missing file, `input[i]`, `target[i]`, `mask[i]` refer to different samples. Duplicate basenames across extensions can also collide/overwrite.

**Fix:** build pairs by stem, validate 1:1:1, and subset the paired list once. Abort on mismatches and print counts.

### M8 — Duplicated checkpoint logic (medium)

Checkpoint unwrap/load/dequant is implemented five times (train, evaluate, inference, audit script, test). They already differ (`weights_only`, scheduler handling, int8 support). This is a bug farm.

**Fix:** one `src/utils/checkpoint.py` with `load_checkpoint(path, map_location, trust=False)`, `extract_state_dict`, `dequantize`, `save_checkpoint(..., config, rng)`. Update all call sites and tests.

### M9 — Repo hygiene (medium)

`.gitignore` contains `tests/` (“Local tests (kept out of git)”), `test/`, `train/`, `notebook/`, `outputs/`, `*.pth`, but **not** `Data/` (removed in the working-tree diff). Only 18 files are tracked. The repository therefore publishes code without its tests, training artifacts, notebook, or audit scripts, and risks committing 276 MB of data.

**Fix:** stop ignoring `tests/`; ignore `Data/` and large artifacts; add a small sample fixture for tests. Commit the audit/export scripts if they are part of the workflow.

### Low-severity items

- **L1** `ioanet.py:120` uses deprecated `pretrained=`; use `weights=` (`MobileNet_V2_Weights.DEFAULT`).
- **L2** Debug forward runs `output_attention(residual)` twice (`ioanet.py:325`); cache the first result.
- **L3** Recursive `__getitem__` fallback on unreadable images; replace with bounded retry + `RuntimeError`.
- **L4** `main()` accepts `--lr`/`--finetune` for Stage 2 but ignores them; error out or implement.
- **L5** Dead/unused config: `augmentation.enabled` never read; stage2 `losses.l1_weight` unused (hardcoded L1); stage2 docstring says batch 4 while config says 16; `process_single_image` duplicated in merge scripts.
- **L6** `inference.py` glob `*.[jp][pn][g]` + `*.jpeg` misses `.bmp` and uppercase extensions, caps at 50 silently, and guesses split via `"test" in path.lower()` (any parent dir named `test` misclassifies).
- **L7** Audit composite score weights (0.5/0.3/0.2 with arbitrary ranges) are undocumented heuristics; ranking Stage 1 and Stage 2 together is invalid (different resolutions).
- **L8** `export_onnx_combined.py` reimplements `LPIOANet.forward`; drift risk. ONNX deps absent from requirements; no verification in `export_onnx.py`.
- **L9** No LICENSE, CI, lint/format config, or packaging metadata; `docs/` empty.
- **L10** `inference.py` uses `process_image_stage1` for both stages (works only because `LPIOANet` accepts the low-res tensor and internally upsamples); naming/semantics are misleading. Per-image timing excludes I/O and post-processing.
- **Additional:** `inference.py` `load_model` has no `dtype` argument; `--stage 2` uses the Stage 1 model path unless the caller passes the stage-2 checkpoint, and there is no automatic stage-1 dependency check in inference (unlike evaluate).
- **Additional:** `ShadowRemovalDataset.__getitem__` resize guard compares `(H,W)` to `(expected_w, expected_h)`, so for non-square inputs it always resizes (wasted CPU). HighResolution and mixed datasets use the correct comparison.
- **Additional:** validation DataLoader has no `drop_last`, and metrics are averaged per batch, so the last partial batch is over-weighted (small bias even after C4).
- **Additional:** Stage 2 `test` evaluation uses the low-resolution `Data/Kligler` test split; there is no high-res test set in the repo.
- **Additional:** `train.py` Stage 1 mixed mode sets `test_loader=None`; `--mixed` cannot be used with a test split. `Data/Mixed_Stage1` does not exist, so `--mixed` currently fails.

---

## 4. What works / verified behavior

- **Unit tests:** `.venv\Scripts\python.exe -m pytest tests -q` → **61 passed, 44 warnings in 98.63 s** (CPU). Warnings are the torchvision `pretrained=` deprecation and `pin_memory` without an accelerator.
- **Inference:** Stage 1 (v2) works, 156.1 ms/image CPU; Stage 2 (v1) works, 237.9 ms/image CPU; comparison + output PNGs written.
- **Evaluation:** `evaluate.py` runs and produces the expected keys; Stage 1 checkpoint load, dataset, and region metrics all execute (metrics correctness aside).
- **Model structure:** tests confirm coordinate attention, MobileNetV2 channel widths, decoder widths, LPIPS wiring, SSIM kernel, Laplacian decomposition, one refine block, clamp in Stage 2, and frozen IOANet in `LPIOANet`.
- **Checkpoints:** all six load without NaN/Inf; v1/v2/v3 Stage 1, v1 Stage 2, fp16, int8 (with dequant) are structurally valid. The fp16 file is numerically ≈ v2.
- **No secrets** found in code, config, or notebook. Git remote is public GitHub; no credentials committed.

---

## 5. Reported metric caveats (read before citing numbers)

From `outputs/audit/checkpoint_audit.md` (split `test`, 28 images, CPU):

| Checkpoint | MAE | PSNR | SSIM | Score |
|---|---|---|---|---|
| v3_stage1 | 0.0398 | 27.41 | 0.9305 | 67.43 |
| v2_stage1 | 0.0402 | 27.31 | 0.9369 | 67.38 |
| fp16 | 0.0402 | 27.30 | 0.9369 | 67.34 |
| v1_stage1 | 0.0446 | 26.32 | 0.9346 | 64.73 |
| v1_stage2 | 0.0452 | 26.62 | 0.9045 | 64.35 |
| int8 | 0.2251 | 11.46 | 0.4381 | 13.14 |

Caveats:
- PSNR is batch-size dependent (C4); the table’s batch size differs from `evaluate.py` defaults.
- Region/shadow columns are invalid (C3).
- Stage 1 (192×256) and Stage 2 (768×1024) are ranked together (L7); the comparison is not apples-to-apples.
- v3 vs v2 differ by 0.05 points — well inside noise.
- val-based checkpoint selection used a leaky image-level split (C5), so “best” is optimistic.
- The audit scores raw unclamped output; deployment clips (C10).

---

## 6. Prioritized remediation plan

### P0 — before any further training or reporting
1. **Fix augmentation propagation** (C1) and **reorder resize/augment** (C2) in all three datasets; add regression tests.
2. **Fix region metric normalization** (C3) and add numeric tests.
3. **Fix metric aggregation** (C4): global error sums, batch-invariant PSNR; assert batch 1 == batch 8.
4. **Re-run evaluation** and regenerate `outputs/audit/checkpoint_audit.md`.

### P1 — correctness and trust
5. **Document-level train/val split** (C5); verify prefix disjointness.
6. **Decide the `[0,1]` output contract** (C10) and make loss/metrics/export consistent.
7. **Fix or remove int8** (C7); add dequantization to `inference.py` if kept.
8. **`weights_only=True` everywhere** (C8).
9. **Seed everything and snapshot config/RNG in checkpoints** (M3).
10. **Fix `test2` handling or delete it** (C9).

### P2 — performance and maintainability
11. Use `_cheap_metrics` during training (M1); SSIM on validation only.
12. Centralize checkpoint I/O (M8).
13. Fix mask preprocessing (M6) and pair validation (M7) in merge scripts.
14. Restore/correct README and docs (M5); remove `Plan.md` references.
15. Un-ignore `tests/`, ignore `Data/` (M9).
16. Remove global warning suppression (M2); fix deprecated torchvision API (L1); clean dead code (L5).

### P3 — polish
17. Add LICENSE, CI (pytest + lint), pinned requirements, `pyproject.toml`.
18. Consolidate/fix ONNX export (L8), inference glob/split logic (L6), audit scoring (L7).
19. Add a high-resolution test set or clearly document that Stage 2 is only qualitatively evaluated.

---

## 7. Reproduction commands used

```powershell
# tests
.\.venv\Scripts\python.exe -m pytest tests -q

# evaluation (batch-size sensitivity)
.\.venv\Scripts\python.exe evaluate.py --model outputs/checkpoints_v2/stage1/best_model.pth --split test --batch-size 8
.\.venv\Scripts\python.exe evaluate.py --model outputs/checkpoints_v2/stage1/best_model.pth --split test --batch-size 1
.\.venv\Scripts\python.exe evaluate.py --model outputs/checkpoints_v2/stage1/best_model.pth --split test2

# inference smoke
.\.venv\Scripts\python.exe inference.py --stage 1 --model outputs/checkpoints_v2/stage1/best_model.pth --input <image> --dataset-root Data/Kligler --output $env:TEMP\dsl_infer_smoke
.\.venv\Scripts\python.exe inference.py --stage 2 --model outputs/checkpoints_v1/stage2/best_model.pth --input <image> --dataset-root Data/Kligler --output $env:TEMP\dsl_infer_smoke

# augmentation propagation check
.\.venv\Scripts\python.exe -c "from src.data.dataset import create_dataloaders; tr,va,te=create_dataloaders('Data/Kligler',(192,256),batch_size=2,augment=True,illumination_strength=.1,shadow_color_shift=.05,rotation_range=0); print(tr.dataset.dataset.augment, va.dataset.dataset.augment)"
```

---

## 8. Bottom line

The model code is competent; the **experiment pipeline is not**. Fix C1–C4 first, re-split by document, and re-evaluate. Until then, the reported metrics and checkpoint rankings should not be used to compare models or claim paper-level performance. Stage 2, as currently sized (1.9 k trainable params) and trained (no high-res cache in the notebook), is unlikely to add meaningful quality; either invest in its capacity/data or present it as a lightweight refiner.
