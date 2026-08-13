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

This experiment adapts the image-space backbone in `2412.10776v1.pdf` to
MRIxFields2026 Task 3. The paper solves undersampled k-space reconstruction;
Task 3 solves field-strength translation. The implementation therefore keeps
FMAM, SPAM, SDFN, HEFR, and the hierarchical encoder/decoder, removes the
inapplicable k-space data-consistency layer, and adds shared conditioning for:

- modality: `T1W`, `T2W`, or `T2FLAIR`;
- source field: one of the five Task 3 fields; and
- target field: one of the other four fields.

There is one model and one checkpoint for all 60 modality/directed-pair strata.
No field-specific or modality-specific network is constructed.

## Important implementation distinction

The paper does not publish code and leaves details such as channel widths and
the practical realization of full-resolution attention unspecified. Its
reported model costs 152 GFLOPs at the paper's image size on an RTX 3090.

The default `portable` preset uses the paper's reported stage depths
`(1,2,2,1)`, heads `(1,2,4,8)`, three frequency levels, four hash groups, SDFN
expansion 2, four HEFR blocks, and a 48-channel first stage. The resulting
shared model has 11,489,809 parameters, close to the paper's reported 12.51 M;
the remaining difference reflects details absent from the paper. It caps FMAM
at 256 pooled tokens and splits large hash groups into at most 128 tokens. This
keeps the method usable on MPS and typical Colab GPUs. `paper_exact` removes
both caps and is likely to run out of memory on full Task 3 slices. `tiny` is
for plumbing tests and Apple Silicon pilots, not a final result.

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
zip -r task3_fpsformer_colab.zip task3_fpsformer \
  -x '*/__pycache__/*' '*.pyc' '*/.DS_Store'
```

This packages the current working files, including changes that have not yet
been committed. Put the zip in Drive, open the notebook in Colab, select a GPU
runtime, and edit only the path values in step 1 first. The paper PDF is
optional; if you want it recorded with run provenance, put it in Drive and set
`DRIVE_PAPER_PDF_VALUE`. `DRIVE_DATA_ROOT_VALUE` must name the parent dataset
root containing the paired prospective training split—not
`validation_prospective` itself. The prepared notebook defaults run one
Colab-safe 10-epoch held-out pilot and export its selected checkpoint. Full
validation inference, all-subject continuation, submission building, and
external upload remain disabled.

On Colab, stage the cache and an optional resume/initialization checkpoint into
`/content`, unmount Drive, and train entirely in runtime storage. When the
process has stopped, copy only the selected checkpoint, metrics, and provenance
back to Drive. Do not train while reading slices or writing checkpoints on the
mounted Drive.

LPIPS is only instantiated when `--loss-lpips` is nonzero or `--eval-lpips` is
passed. Its AlexNet weights may be downloaded by the `lpips` package on first
use. The paper's exact training objective is L1, which is the default here.

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

For a Colab CUDA pilot:

```bash
python -m task3_fpsformer.train \
  --cache-dir "$PREPROCESSED_DIR/fpsformer" \
  --output-dir "$OUTPUT_DIR/fpsformer_portable_pilot" \
  --preset portable \
  --device cuda \
  --amp fp16 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --crop-size 256 \
  --epochs 10 \
  --updates-per-epoch 1000 \
  --baseline-milestone-complete
```

On an Apple Silicon machine, start with `--preset tiny --device mps
--amp off --crop-size 128`. Only move to `portable` after the smoke run fits.

The paper trains for 300,000 optimizer updates with AdamW, learning rate
`1e-4`, a flat first 92,000 updates, then cosine decay to `1e-6`. The matching
long-run command uses `--epochs 300 --updates-per-epoch 1000`; do not start it
until the pilot beats an identity mapping on the held-out subject. Metrics in
`metrics.jsonl` are explicitly slice proxies, not private validation scores.

After selecting settings on the untouched holdout, start a fresh output run
that initializes only the selected model weights and includes every prospective
training subject. Do not call the resulting run a holdout evaluation:

```bash
python -m task3_fpsformer.train \
  --cache-dir "$PREPROCESSED_DIR/fpsformer" \
  --output-dir "$OUTPUT_DIR/fpsformer_all_subjects" \
  --preset portable \
  --initialize "$OUTPUT_DIR/fpsformer_portable_pilot/checkpoints/best.pt" \
  --holdout-subject none \
  --device cuda --amp fp16 \
  --epochs 10 --updates-per-epoch 1000 \
  --baseline-milestone-complete
```

## Validation inference

First run exactly one full-volume translation:

```bash
python -m task3_fpsformer.infer \
  --checkpoint "$OUTPUT_DIR/fpsformer_portable_pilot/checkpoints/best.pt" \
  --input-root "$DATA_DIR" \
  --output-dir "$INFERENCE_DIR/fpsformer_smoke" \
  --device auto \
  --smoke
```

Then run the complete resumable 180-volume sweep to a separate directory:

```bash
python -m task3_fpsformer.infer \
  --checkpoint "$OUTPUT_DIR/fpsformer_portable_pilot/checkpoints/best.pt" \
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
