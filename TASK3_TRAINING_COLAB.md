# Task 3 StarGAN v2 Colab training

The prepared workflow keeps the organizers' shared 15-domain architecture and
checkpoint layout, while adding Colab-oriented mixed precision, bounded
checkpoint retention, deterministic within-epoch resume, provenance, and a
subject-level holdout pilot. Heavy working files use the Colab runtime disk;
Drive is used only for the source dataset, small prepared scripts, and selected
final exports.

The workspace currently has no evidence that the pretrained validation
baseline milestone is complete: there is no validated 180-output manifest,
built `task3.zip`, or archive SHA-256. For that reason, every notebook training
switch defaults to off and the training CLI requires the explicit
`--baseline-milestone-complete` acknowledgement. Data audit and slice
preparation are not gated.

## Files to put in one private Drive source folder

- `task3_stargan_v2_colab.ipynb`
- `task3_colab_train.py`
- `task3_pipeline.py`
- `requirements-task3-training-colab.txt`

Open the notebook in Colab, select a GPU runtime, set `SOURCE_DIR` and
`DATA_DIR`, and run from the top. `PROJECT_DIR` defaults to the fast local path
`/content/mrixfields_task3`. The setup cell copies the three small prepared
files from `SOURCE_DIR` into that runtime directory. The dataset remains in
place on Drive. A temporary canonical-casing view uses symlinks and does not
rename, duplicate, or modify the source volumes.

## Reproducibility pins

These live revisions were rechecked on 2026-07-14:

- Organizer code commit:
  `f0f4ab2c54ab878b48ddfea8d13f4da512ca411e`
- Hugging Face revision:
  `f1a7d6b741b64c81871d0856fa2d7295d564c141`
- Shared initialization checkpoint:
  `task3_any_to_any_multimodal/stargan_v2/pro_pretrained/weights/checkpoint_epoch50.pth`
- Config:
  `Baseline/configs/task3/stargan/any_to_any_all_modalities.yaml`

The helper refuses a different organizer Git revision or a config that no
longer has the exact modality order, field order, and 15 joint domains. If the
organizers publish an update, review and update the code/config/weight pins as
one coherent set.

## Recommended experiment order

1. Finish and report the organizer-pretrained validation submission milestone.
2. Audit all 15 prospective training groups.
3. Extract and audit the organizer's fixed 220 axial slices.
4. Run the short `pilot_subject_holdout` experiment. It automatically withholds
   one complete prospective volunteer and reports per-modality proxies. For an
   honest holdout it must start from scratch or from your own retrospective-only
   checkpoint.
5. Compare the logged slice-level nRMSE, SSIM, and LPIPS proxies and inspect
   predicted images. These proxies are not public-validation scores.
6. Only after choosing a setting for a leakage-free pipeline, retrain with all
   prospective volunteers from the same kind of initialization. A separate
   `pro_pretrained_all_subjects` notebook cell instead continues the released
   baseline and must be judged through public validation.
7. Run the trained checkpoint through a full-volume held-out evaluation or the
   180-case validation/submission pipeline. Public validation performance still
   comes only from the leaderboard because its paired targets are private.

The full 50-epoch prospective schedule repeats every same-modality directed
field pair: 3 modalities × 20 directions. With three prospective volunteers
and 220 slices, that is roughly 39,600 training examples per epoch before
batching. Batch 1 with gradient accumulation 8 preserves the official
optimization batch of 8, but it does not reduce the number of forward/backward
passes. Expect the complete schedule to exceed a normal free-Colab session,
especially on a T4. The notebook therefore saves atomic checkpoints every 500
microbatches and resumes at the next deterministic microbatch.

Important: the released `pro_pretrained/checkpoint_epoch50.pth` was already
fine-tuned on the prospective split. It cannot initialize a truthful
prospective-subject holdout, because the putative holdout has already influenced
the weights. The helper enforces this: a holdout must use `--from-scratch`, or
an initialization checkpoint accompanied by
`--initialization-excludes-holdout`. The latter is an explicit attestation that
the checkpoint was trained only on retrospective data (or otherwise never saw
the held-out subject).

Continued fine-tuning from the released epoch-50 checkpoint is still a valid
all-subject experiment. It simply has no honest local prospective holdout.
Start with one additional epoch and compare its complete public-validation
submission against the untouched released baseline before spending compute on
longer continuation.

Retrospective pretraining is optional and deliberately last. Its official
schedule is 200,000 adversarial steps over the much larger retrospective
cache. It is generally a poor first use of Colab time when the official shared
checkpoint is already available.

## Outputs

Each run writes under:

```text
training_runs/
└── task3_any_to_any_multimodal/
    └── stargan_v2/
        └── <run-name>/
            ├── provenance.json
            ├── train.jsonl
            └── weights/checkpoint_*.pth
```

Checkpoints retain the organizer-compatible `nets` and `nets_ema` keys. The
default keeps only the newest three checkpoints in each run to bound local disk
usage. `provenance.json` records the code/config pins, initialization hash,
GPU, PyTorch/CUDA versions, paths, hyperparameters, domain coverage, subject
counts, and holdout choice.

The checkpoint download, Hugging Face and PyTorch caches, extracted slices,
training runs, reports, and validation predictions all default to
`/content/mrixfields_task3`. This avoids making thousands of small reads and
writes through the mounted Drive filesystem. These files disappear when Colab
resets. The notebook's final opt-in export cell copies only JSON reports and an
explicitly selected checkpoint and/or archive to `SAVE_DIR`; it does not copy
caches, extracted slices, prediction trees, or every checkpoint.

The checkpoint cell defaults to copying the previously saved official file
from
`/content/drive/MyDrive/mrixfields_weights/task3_any_to_any_multimodal/stargan_v2/pro_pretrained/weights/checkpoint_epoch50.pth`
into runtime storage. It verifies the pinned byte size and SHA-256 before the
checkpoint is used. Disable `USE_EXISTING_CHECKPOINT` and enable
`DOWNLOAD_OFFICIAL_CHECKPOINT` only when that saved copy is unavailable.

The workflow never packages or uploads anything. Training authorization does
not authorize external submission.
