# NexaDoc retrain pipeline

This directory now contains a **new, self-contained training pipeline** for the
`nexadoc-50k` synthetic dataset. The legacy pipeline (`train.py`, `evaluate.py`,
`inference.py`, `src/data/dataset.py`, `src/models/laplacian_refiner.py`) is left
untouched for reference and still has the bugs documented in `AUDIT_REPORT.md`.

**Use the new files. Do not train the old `train.py` on this dataset.**

## Dataset facts (verified on the published archive)

`kaggle.com/datasets/walidmostafa77/nexadoc-50k` — 7.8 GB, 325,333 files, CC0.

| Task | Samples | Split (train/val/test) | Shapes |
|---|---|---|---|
| `deshadow` | 50,000 | 40,000 / 5,000 / 5,000 | input,target 256×256×3 uint8; mask 256×256×3 {0,255}; gain 256×256×3 float16 |
| `upsample` | 10,000 | 7,961 / 1,059 / 980 | input 256×256×3; target 768×1024 (W×H); mask all-zero 256×256 |

- `upsample/input` is **byte-identical** to `deshadow/target` for the same index.
- Therefore the task is exactly: **Stage 1** remove cast shadow at 256², **Stage 2**
  upsample the shadow-free 256² result to 768×1024.
- Splits are an 80/10/10 seeded shuffle of **sample indices**, not document-disjoint.
  Treat val/test as potentially document-leaky until the generator produces
  doc-disjoint splits. See “Known limits”.
- The Kaggle description says upsample input is 768×1024 and target 1024×1365.
  That is **wrong**; the actual files are 256² → 768×1024. The loader asserts this.

Expected mount inside Kaggle:

```
/kaggle/input/nexadoc-50k/
  deshadow/{train,val,test}/{input,target,mask,gain}/
  upsample/{train,val,test}/{input,target,mask}/
  manifest.json
```

## What changed

| New file | Purpose |
|---|---|
| `src/data/nexadoc_dataset.py` | Official-split datasets; no random split, no resize; 3-channel masks → 1-channel; paired flips. |
| `src/models/deshadow.py` | `DeshadowNet(IOANet)` with clamped `[0,1]` output. |
| `src/models/sr_refiner.py` | `SRUNet`: residual U-Net, bicubic base + zero-init residual head; replaces the 1,879-param LP-IOANet for the 256²→768×1024 task. |
| `src/utils/eval_metrics.py` | Global, batch-invariant MAE/MSE/PSNR/SSIM; correct per-pixel shadow/non-shadow metrics. |
| `src/utils/nexadoc_losses.py` | Stage 1 `L1 + 2·masked-L1 + 0.1·LPIPS(AlexNet)`; Stage 2 pure `L1` (LPIPS disabled for best PSNR; set `lpips_weight: 0.02` for perceptual sharpness). |
| `configs/nexadoc.yaml` | All hyperparameters for both stages. |
| `train_nexadoc.py` | DDP (`torchrun`), AMP, AdamW + cosine, EMA, resume, official splits, global metrics. |
| `evaluate_nexadoc.py` | Standalone test-set evaluation + comparison panels. |
| `tests/test_nexadoc_pipeline.py` | 7 tests for the new data/model/metric contracts. |

The Stage 1 backbone is still **MobileNetV2 + IOANet** (user decision). Only the
input resolution changes to 256×256. `configs/nexadoc.yaml` keeps a `backbone`
field so another encoder can be added later; only `mobilenet_v2` is implemented.

## Kaggle 2×T4: exact commands

Run from the repo root in a Kaggle notebook with **GPU T4 x2** and the dataset
attached. Use a fresh output directory per session.

```bash
pip install -q lpips tensorboard
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
# expect 2
```

### Stage 1 — shadow removal at 256²

```bash
torchrun --nproc_per_node=2 train_nexadoc.py \
  --stage 1 \
  --data-root /kaggle/input/nexadoc-50k \
  --output-dir /kaggle/working/outputs/stage1
```

- 40,000 train / 5,000 val; batch 24 per GPU (48 effective); 100 epochs.
- EMA 0.999, AMP, gradient clip 1.0, AdamW lr 2e-4, 5-epoch warmup.
- Writes `best.pt` (best of model/EMA by val PSNR), `last.pt`,
  `epoch_XXXX.pt` every 5 epochs, `metrics.jsonl`, `test_metrics.json`.

Resume after a Kaggle timeout:

```bash
torchrun --nproc_per_node=2 train_nexadoc.py \
  --stage 1 \
  --data-root /kaggle/input/nexadoc-50k \
  --output-dir /kaggle/working/outputs/stage1 \
  --resume /kaggle/working/outputs/stage1/last.pt
```

### Stage 2 — 256² → 768×1024 upsample

Stage 2 is independent of Stage 1 checkpoints: it learns the published
`upsample` mapping directly (the dataset already supplies its 256² inputs).
At inference, feed Stage 1's 256² output into Stage 2.

```bash
torchrun --nproc_per_node=2 train_nexadoc.py \
  --stage 2 \
  --data-root /kaggle/input/nexadoc-50k \
  --output-dir /kaggle/working/outputs/stage2
```

- 7,961 train / 1,059 val; batch 4 per GPU; 40 epochs; base 16 (~1.23M params).
- Full-resolution U-Net at 768×1024; if a T4 runs out of memory, use
  `--batch-size 2` (per GPU).
- Writes the same checkpoint/metrics layout.

### Evaluate

```bash
python evaluate_nexadoc.py --stage 1 \
  --data-root /kaggle/input/nexadoc-50k \
  --checkpoint /kaggle/working/outputs/stage1/best.pt \
  --save-samples 8

python evaluate_nexadoc.py --stage 2 \
  --data-root /kaggle/input/nexadoc-50k \
  --checkpoint /kaggle/working/outputs/stage2/best.pt \
  --save-samples 8
```

Stage 1 also reports shadow/non-shadow MAE and PSNR. Metrics are accumulated
globally (not per-batch means), so they do not change with batch size.

### CPU smoke test

```powershell
.\.venv\Scripts\python.exe train_nexadoc.py --stage 1 --smoke --no-lpips
.\.venv\Scripts\python.exe train_nexadoc.py --stage 2 --smoke --no-lpips
.\.venv\Scripts\python.exe -m pytest tests -q
```

The smoke flags use one epoch, two batches, no pretrained weights, and no AMP.

## Baselines to beat

Measured on one `upsample` pair (index 40000):

| Method | MAE | PSNR | SSIM |
|---|---|---|---|
| Bicubic 256²→768×1024 | 0.0377 | 20.01 | 0.835 |
| Lanczos | 0.0404 | 19.78 | 0.813 |
| Nearest | 0.0366 | 18.97 | 0.824 |

Stage 1 identity baseline (`input` vs `target`) on three deshadow samples:
PSNR 16.6–24.0. A trained model must be clearly above that; the old Kligler
checkpoint scores around 27–28 dB, but it was trained without augmentation and
on a different domain, so do not treat it as a target.

## Known limits / decisions

1. **Document leakage is unresolved.** The generator splits by sample index, not
   by document. The earlier 200-sample run had 3 val and 3 test documents also
   present in train. The 50k archive has no doc-path metadata. If you need
   publishable test numbers, regenerate or filter by document ID.
2. **Stage 2 is A4-only** in the published dataset (10k samples). It learns
   upsampling for A4 pages; other page types are out of distribution.
3. **Stage 2 is not conditioned on Stage 1 errors.** It is trained on clean
   256² targets. At inference it receives Stage 1 output; mild exposure mismatch
   is possible. If it matters, fine-tune Stage 2 on Stage 1 predictions.
4. **The dataset description is inaccurate** about upsample shapes; trust the
   loader's assertions, not the web page.
5. **The old pipeline is still present** and unsafe to mix: `train.py` disables
   augmentation, `evaluate.py` PSNR depends on batch size, `RegionBasedMetrics`
   is 3× wrong. Use the new modules.
6. **No document-level split in code.** The loaders honor the official split
   directories; fixing leakage needs a generator-side change or a filtered index
   list, not a different `random_split`.
7. **EMA is saved and restored** in `best.pt`/`last.pt`; `epoch_XXXX.pt` is
   inference-only and has no optimizer state.
8. **`weights_only=True`** is used for the new checkpoints. Old legacy
   checkpoints that contain arbitrary Python objects may need a separate
   conversion step.
9. **Compute estimates are rough.** Stage 1 is ~833 steps/epoch on 2×T4;
   Stage 2 is ~995 steps/epoch at 768×1024. Watch the first epoch and adjust
   `--batch-size`/epochs rather than assuming a fixed wall-clock budget.
10. **No `README.md` in the working tree** (tracked file was deleted locally).
    This file is the current pipeline guide; the legacy README is at
    `git show HEAD:README.md`.

## Files intentionally left alone

`train.py`, `evaluate.py`, `inference.py`, `src/data/dataset.py`,
`src/data/mixed_dataset.py`, `src/models/ioanet.py`,
`src/models/laplacian_refiner.py`, `src/utils/losses.py`, `scripts/*`, and the
Kaggle notebook. The audit report explains why.
