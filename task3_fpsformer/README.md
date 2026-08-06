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

Colab GPU: copy the `.py` files and requirements file to the runtime, then run:

```bash
python -m pip install -r task3_fpsformer/requirements-colab.txt
```

Do not install the Mac requirements in Colab because replacing Colab's matched
CUDA PyTorch build can break GPU support.

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
