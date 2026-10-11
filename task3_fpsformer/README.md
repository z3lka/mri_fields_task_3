# Task 3 conditional FPS-Former

All commands below are run from the workspace root. The implementation lives
entirely in this package:

```text
task3_fpsformer/
├── common.py
├── data.py
├── model.py
├── losses.py
├── train.py
├── infer.py
├── task3_fpsformer_colab.ipynb
├── requirements-colab.txt
├── requirements-mac.txt
└── tests/
```

This experiment adapts the design in [the FPS-Former paper](https://arxiv.org/abs/2412.10776)
to MRIxFields2026 Task 3. An optional local `2412.10776v1.pdf` is ignored by Git.
The authors have now released their reconstruction source at
<https://github.com/XiaoMengLiLiLi/FPS-Former>. The comparison below is pinned
to commit `166d10d6e045b155b452fb26622b6b22063f82f5` (2025-04-07).

The released project solves undersampled k-space reconstruction; Task 3 solves
field-strength translation. Our implementation keeps the frequency, spatial,
multi-scale, expert-refinement, and hierarchical encoder/decoder ideas, removes
the reconstruction-specific complex/k-space path, and adds shared conditioning
for:

- modality: `T1W`, `T2W`, or `T2FLAIR`;
- source field: one of the five Task 3 fields; and
- target field: one of the other four fields.

There is one model and one checkpoint for all 60 modality/directed-pair strata.
No field-specific or modality-specific network is constructed.

## Released source compared with this adaptation

These are related architectures, not interchangeable implementations:

| Area | Authors' released reconstruction code | Our Task 3 adaptation |
| --- | --- | --- |
| Contract | HDF5 undersampled k-space; 2-channel complex image in/out | NIfTI image-space translation; 1-channel `[0,1]` image in/out |
| Control | One unconditional reconstruction network | One shared network conditioned on modality, source field, and target field |
| FMAM | Gaussian/Laplacian channel-attention maps, then channel-to-spatial recalibration | Pooled, multi-head token attention over learned Laplacian-band Q/K/V features |
| Spatial attention | Two fresh SALSH rounds inside non-overlapping `8x8` patches | One persistent random projection, global hash ordering, bounded token groups |
| Feed-forward | Released mixed-scale channel flow (`MSFN`) | A lighter two-round 3x3/5x5 depthwise variant |
| Expert refinement | Two `MEFC` subnets, each with four residual operation-mixture steps over eight operators | Configurable shared expert blocks before the encoder and after the decoder |
| Resizing | PixelUnshuffle/PixelShuffle with concatenated skips | Strided convolution/bilinear upsampling with fused skips; automatic padding to multiples of 8 |
| Output | Unbounded residual addition to the complex input | Near-identity residual in logit space, constrained to `[0,1]` |

The matching defaults are the four levels, 48 first-stage channels, depths
`(1,2,2,1)`, heads `(1,2,4,8)`, expansion 2, three frequency levels, L1
training, Adam/AdamW at `1e-4`, and an encoder-decoder residual topology. The
default `portable` model has 11,489,809 parameters and caps FMAM at 256 pooled
tokens and hash groups at 128 tokens for MPS and Colab GPUs.

`unbounded` removes those two portability caps but is still our adaptation.
The old preset name `paper_exact` remains only as a checkpoint-compatible alias
for `unbounded`; it is not source-exact. `tiny` is for plumbing tests, not a
quality result.

The upstream repository cannot be used directly for Task 3: it has no
modality/field conditioning or NIfTI translation path. At the pinned revision,
it also has an unterminated string in `utils/options.py`, imports a missing
`models_vit.dc_layer`, has no dependency lockfile or Colab notebook, and uses
device-bound Gaussian tensors that are not registered module buffers. We do
not vendor or silently patch that repository; its exact commit is recorded as
reference provenance in new training checkpoints and inference manifests.

## Environment

Apple Silicon (Python 3.11 or 3.12 recommended):

```bash
python3.12 -m venv .venv-fpsformer
source .venv-fpsformer/bin/activate
python -m pip install --upgrade pip
python -m pip install -r task3_fpsformer/requirements-mac.txt
```

Verify Metal acceleration before preparing data:

```bash
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

The last line must be `True`. On macOS use `--device mps --amp off --workers
0`; CUDA-style fp16 is not enabled on MPS in this runner.

### Terminal-only Apple Silicon workflow

Set machine-local paths in each new terminal. `DATA_DIR` is the parent dataset
root containing both the paired prospective training split and validation
split, not either split directory itself. Keep derived data and runs outside
the repository; an external SSD is also suitable.

```bash
cd /absolute/path/to/mri_competition
source .venv-fpsformer/bin/activate

export DATA_DIR="/absolute/path/to/mrixfields2026"
export FPS_RUN_ROOT="/absolute/path/to/mrixfields_task3_runs/fpsformer"
export FPS_CACHE_DIR="$FPS_RUN_ROOT/cache"
export FPS_OUTPUT_DIR="$FPS_RUN_ROOT/outputs"
export FPS_INFERENCE_DIR="$FPS_RUN_ROOT/inference"
export FPS_REPORT_DIR="$FPS_RUN_ROOT/reports"

mkdir -p "$FPS_CACHE_DIR" "$FPS_OUTPUT_DIR" \
  "$FPS_INFERENCE_DIR" "$FPS_REPORT_DIR"
```

Audit and cache the three paired prospective training subjects. These commands
read but do not modify the source NIfTI files:

```bash
python -m task3_fpsformer.data audit-raw \
  --data-dir "$DATA_DIR" \
  --check-values \
  --report "$FPS_REPORT_DIR/raw_audit.json"

python -m task3_fpsformer.data prepare-cache \
  --data-dir "$DATA_DIR" \
  --cache-dir "$FPS_CACHE_DIR"

python -m task3_fpsformer.data audit-cache \
  --cache-dir "$FPS_CACHE_DIR" \
  --report "$FPS_REPORT_DIR/cache_audit.json"
```

Run the two-update MPS smoke in a new output directory:

```bash
caffeinate -i python -m task3_fpsformer.train \
  --cache-dir "$FPS_CACHE_DIR" \
  --output-dir "$FPS_OUTPUT_DIR/tiny_smoke_01" \
  --preset tiny \
  --device mps --amp off --workers 0 \
  --crop-size 128 --eval-crop-size 128 \
  --gradient-accumulation 1 \
  --epochs 1 --updates-per-epoch 2 \
  --val-samples-per-pair 1 \
  --print-every 1 \
  --baseline-milestone-complete
```

After it writes `checkpoints/best.pt`, measure a deliberately short portable
probe at crop 128 before increasing either crop size or update count:

```bash
caffeinate -i python -m task3_fpsformer.train \
  --cache-dir "$FPS_CACHE_DIR" \
  --output-dir "$FPS_OUTPUT_DIR/portable_probe_01" \
  --preset portable \
  --device mps --amp off --workers 0 \
  --batch-size 1 --gradient-accumulation 1 \
  --crop-size 128 --eval-crop-size 128 \
  --epochs 1 --updates-per-epoch 10 \
  --val-samples-per-pair 1 \
  --print-every 1 \
  --baseline-milestone-complete
```

Check elapsed time, memory pressure, `metrics.jsonl`, and the holdout identity
baseline before starting a longer pilot. Checkpoints are saved after each
epoch. Resume an interrupted run with the same model/data settings, the same
output directory, a new total epoch count, and the last checkpoint:

```bash
caffeinate -i python -m task3_fpsformer.train \
  --cache-dir "$FPS_CACHE_DIR" \
  --output-dir "$FPS_OUTPUT_DIR/portable_probe_01" \
  --preset portable \
  --device mps --amp off --workers 0 \
  --batch-size 1 --gradient-accumulation 1 \
  --crop-size 128 --eval-crop-size 128 \
  --epochs 2 --updates-per-epoch 10 \
  --val-samples-per-pair 1 \
  --resume "$FPS_OUTPUT_DIR/portable_probe_01/checkpoints/last.pt" \
  --baseline-milestone-complete
```

Colab GPU: copy the `.py` files and requirements file to the runtime, then run:

```bash
python -m pip install -r task3_fpsformer/requirements-colab.txt
```

Do not install the Mac requirements in Colab because replacing Colab's matched
CUDA PyTorch build can break GPU support.

### Complete Colab notebook

[`task3_fpsformer_colab.ipynb`](task3_fpsformer_colab.ipynb) is the guided
Colab workflow. It keeps the model implementation in the Python package and
orchestrates package staging, GPU checks, raw/cache audits, tests, tiny and
portable training, checkpoint selection, validation inference, official
submission building, numerical validation, and explicit artifact export.

Create the source bundle on your Mac from the workspace root:

```bash
zip -FS -r task3_fpsformer_colab.zip task3_fpsformer \
  -x '*/__pycache__/*' '*.pyc' '*/.DS_Store' \
  '*/.ipynb_checkpoints/*' '*_colab-*.ipynb'
```

This packages the current working files, including uncommitted changes, and
excludes caches and saved notebook copies. Open
the notebook in Colab, select a GPU runtime, and use its default browser-upload
option to upload this ZIP. Alternatively, put it in Drive, set
`USE_BROWSER_UPLOAD_FOR_CODE = False`, and edit `DRIVE_CODE_ZIP_VALUE`.
`DRIVE_DATA_ROOT_VALUE` must name the parent dataset root containing the paired
prospective training split—not `validation_prospective` itself.

All data staging, audit, cache, training, inference, packaging, and export
switches default to off. Enable and run one stage at a time. The A100 training
recipe targets 100 epochs x 1,000 updates with batch 16, crop 256, bf16, paired
left-right augmentation, EMA weights, and a larger full-slice holdout sample.
It runs in at most twelve 45-minute chunks, saving a resumable checkpoint at an
epoch boundary after each chunk. Full validation inference, all-subject
continuation, submission building, and every external upload remain disabled
until explicitly selected.

On Colab, stage the cache and an optional resume/initialization checkpoint into
`/content`, unmount Drive, and train entirely in runtime storage. Between
chunks, after the trainer has stopped, the notebook mounts Drive and copies
`last.pt`, `best.pt`, provenance, and a metrics snapshot with size and SHA-256
verification, then unmounts Drive before the next chunk. Resource availability,
GPU type, and runtime duration are not guaranteed even on paid Colab, so treat
the chunk limit as a budget rather than a completion promise.

LPIPS is only instantiated when `--loss-lpips` is nonzero or `--eval-lpips` is
passed. Its AlexNet weights may be downloaded by the `lpips` package on first
use. The paper's exact training objective is L1, which remains the CLI default;
the competition recipe adds the organizer's `0.1 * LPIPS` fine-tuning term.
New checkpoints contain both online and EMA weights. Initialization and
inference prefer EMA weights, while older checkpoints remain supported.

## Data preparation

Audit the real paired prospective training split and build a read-only-derived
slice cache. Neither command changes the source NIfTI files.

```bash
python -m task3_fpsformer.data audit-raw \
  --data-dir "$DATA_DIR" \
  --check-values \
  --report "$OUTPUT_DIR/fpsformer_raw_audit.json"

python -m task3_fpsformer.data prepare-cache \
  --data-dir "$DATA_DIR" \
  --cache-dir "$PREPROCESSED_DIR/fpsformer"

python -m task3_fpsformer.data audit-cache \
  --cache-dir "$PREPROCESSED_DIR/fpsformer" \
  --report "$OUTPUT_DIR/fpsformer_cache_audit.json"
```

The cache contains axial indices `[72,292)`, matching the organizer's current
220-slice preprocessing. Pairing is by exact subject and slice index. The code
fails if a prospective subject/domain is missing and never fabricates pairs
from public validation IDs or unpaired retrospective data.

## First runs

Run a two-update plumbing test before spending GPU time:

```bash
python -m task3_fpsformer.train \
  --cache-dir "$PREPROCESSED_DIR/fpsformer" \
  --output-dir "$OUTPUT_DIR/fpsformer_smoke" \
  --preset tiny \
  --device auto \
  --crop-size 128 \
  --epochs 1 \
  --updates-per-epoch 2 \
  --val-samples-per-pair 1 \
  --baseline-milestone-complete
```

For one paid-Colab training chunk (the notebook automates repeated chunks and
verified Drive backups):

```bash
python -m task3_fpsformer.train \
  --cache-dir "$PREPROCESSED_DIR/fpsformer" \
  --output-dir "$OUTPUT_DIR/a100_holdout_ema_01" \
  --preset portable \
  --device cuda \
  --amp bf16 \
  --batch-size 16 --eval-batch-size 4 \
  --gradient-accumulation 1 \
  --crop-size 256 --eval-crop-size 0 \
  --epochs 100 \
  --updates-per-epoch 1000 \
  --max-runtime-minutes 45 \
  --val-samples-per-pair 30 \
  --val-slice-start 150 --val-slice-end 180 --eval-every 5 \
  --left-right-flip-probability 0.5 \
  --ema-beta 0.999 --loss-lpips 0.1 --eval-lpips \
  --workers 4 --progress \
  --baseline-milestone-complete
```

Resume with the same target epochs, update count, crop, and run directory by
adding `--resume "$OUTPUT_DIR/a100_holdout_ema_01/checkpoints/last.pt"`.
Use `--initialize` only once to start a new optimizer and learning-rate schedule
from the selected held-out model. The notebook checks that an
initialization or resume checkpoint uses the same held-out subject.

On an Apple Silicon machine, start with `--preset tiny --device mps
--amp off --crop-size 128`. Only move to `portable` after the smoke run fits.

The paper trains for 300,000 optimizer updates with AdamW, learning rate
`1e-4`, a flat first 92,000 updates, then cosine decay to `1e-6`. The matching
long-run command uses `--epochs 300 --updates-per-epoch 1000`; do not start it
until the paid held-out run beats an identity mapping on the held-out subject.
Metrics in
`metrics.jsonl` are explicitly slice proxies, not private validation scores.

After selecting settings on the untouched holdout, start a fresh output run
that initializes only the selected model weights and includes every prospective
training subject. Do not call the resulting run a holdout evaluation:

```bash
python -m task3_fpsformer.train \
  --cache-dir "$PREPROCESSED_DIR/fpsformer" \
  --output-dir "$OUTPUT_DIR/fpsformer_all_subjects" \
  --preset portable \
  --initialize "$OUTPUT_DIR/a100_holdout_ema_01/checkpoints/best.pt" \
  --holdout-subject none \
  --device cuda --amp bf16 \
  --batch-size 16 --crop-size 256 --workers 4 \
  --left-right-flip-probability 0.5 \
  --ema-beta 0.999 --loss-lpips 0.1 \
  --epochs 10 --updates-per-epoch 1000 \
  --baseline-milestone-complete
```

## Validation inference

First run exactly one full-volume translation:

```bash
python -m task3_fpsformer.infer \
  --checkpoint "$OUTPUT_DIR/a100_holdout_ema_01/checkpoints/best.pt" \
  --input-root "$DATA_DIR" \
  --output-dir "$INFERENCE_DIR/fpsformer_smoke" \
  --device auto \
  --smoke
```

Then run the complete resumable 180-volume sweep to a separate directory:

```bash
python -m task3_fpsformer.infer \
  --checkpoint "$OUTPUT_DIR/a100_holdout_ema_01/checkpoints/best.pt" \
  --input-root "$DATA_DIR" \
  --output-dir "$INFERENCE_DIR/fpsformer_full" \
  --device auto \
  --resume
```

The output layout is compatible with the official builder under method
`fpsformer`, mode `pro_scratch`, and epoch tag `best`. Use the current official
builder directly for this experimental checkpoint; the workspace's
`task3_build_submission.py` deliberately validates only the organizer-pretrained
baseline provenance.

```bash
python official/MRIxFields2026/Submission/build_submission/build_submission.py \
  --tasks task3 \
  --predictions-dir "$INFERENCE_DIR/fpsformer_full" \
  --predictions-seg-dir "$OUTPUT_DIR/unused_task3_placeholder" \
  --output-dir "$SUBMISSION_DIR/fpsformer" \
  --task3-method fpsformer \
  --task3-mode pro_scratch \
  --task3-epoch best
```

Re-check the live competition and submission rules immediately before a final
build. Do not upload anything without explicit confirmation.
