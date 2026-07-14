# MRIxFields2026 Task 3 Working Instructions

## Scope: Task 3 only

These instructions apply to the entire workspace.

The user is participating only in **Task 3: controllable any-to-any,
multimodal field translation**. Work only on Task 3 unless the user explicitly
changes scope.

- Do not work on Task 1 or Task 2 models, configs, metrics, segmentation,
  archives, or submission paths.
- Do not run or introduce CUT, CycleGAN, SynthSeg, Dice, volume metrics, or
  `seg/` outputs; they are not part of Task 3.
- Do not use, inspect, install, run, train, maintain, or recommend JEPA,
  LEJEPA, SIGReg, their shards, their checkpoints, or the historical
  `external/lejepa` material. The user will not use JEPA for this project.
- Treat `AGENTS_2.md` as historical background only. Its Task 1/2 and JEPA
  plans are out of scope and must not drive work.

The immediate milestone is a correct validation submission made with the
organizers' pretrained Task 3 baseline. Do not start architectural experiments,
pretraining, fine-tuning, or expensive training before that milestone is
verified and reported to the user.

Success means a reproducible `task3.zip` that passes local structural and
numerical checks and is ready for the user to upload. Never upload or submit
externally without the user's explicit confirmation.

## Authoritative Task 3 resources

Use these before guessing or reimplementing:

- Competition: https://mrixfields.chihucloud.com/2026/
- Data page: https://mrixfields.chihucloud.com/2026/data.html
- Tasks page: https://mrixfields.chihucloud.com/2026/tasks.html
- Official repository: https://github.com/MRIxFields/MRIxFields2026
- Official baseline weights: https://huggingface.co/mrixfields/MRIxFields2026-Baseline/tree/main
- Task 3 shared weights: https://huggingface.co/mrixfields/MRIxFields2026-Baseline/tree/main/task3_any_to_any_multimodal
- Local organizer checkout: `official/MRIxFields2026/`
- Task 3 config: `official/MRIxFields2026/Baseline/configs/task3/stargan/any_to_any_all_modalities.yaml`
- Baseline instructions: `official/MRIxFields2026/Baseline/README.md`
- Submission contract: `official/MRIxFields2026/Submission/README.md`
- Submission builder: `official/MRIxFields2026/Submission/build_submission/`

The competition and submission documents are live and may change. Immediately
before a final build or upload, re-check the official repository, competition
page, and submission README. Record the upstream Git commit and Hugging Face
revision used. Prefer current official code over this file or old notes. If
official sources disagree, stop and report the discrepancy.

Keep organizer code distinguishable from local changes. Inspect and reuse the
official loader, model, inference, evaluation, and submission tooling before
writing replacements.

## Current workspace baseline

The workspace includes `task3_pipeline.py`, a resumable inference wrapper for
the organizers' one shared multimodal StarGAN v2 checkpoint. It can audit
inputs, plan cases, download the pinned checkpoint, run smoke/full inference,
and validate full-volume outputs. It does not train, package, upload, or compute
metrics without paired targets.

The runner currently records these preparation pins:

- Official code commit: `f0f4ab2c54ab878b48ddfea8d13f4da512ca411e`
- Hugging Face revision: `f1a7d6b741b64c81871d0856fa2d7295d564c141`
- Checkpoint: `task3_any_to_any_multimodal/stargan_v2/pro_pretrained/weights/checkpoint_epoch50.pth`

These pins make the prepared runner reproducible; they are not a claim that the
sources are still current. Re-check upstream before the final build and record
any update instead of silently mixing code, config, and weights from different
revisions.

## Non-negotiable Task 3 rules

- Use one jointly trained, parameter-sharing conditional model and one set of
  parameters for the entire Task 3 sweep.
- Explicitly account for source field, target field, and modality.
- Fields are `0.1T`, `1.5T`, `3T`, `5T`, and `7T`.
- Modalities are `T1W`, `T2W`, and `T2FLAIR`.
- Cover all 20 directed non-identity field pairs: every source field to each of
  the other four target fields, including down-field translations.
- Do not create field-specific, modality-specific, target-specific,
  source-target-specific, or routed dedicated models/subnetworks. Do not use an
  ensemble disguised as one checkpoint.
- Treat the official `task3_any_to_any_multimodal` checkpoint and matching YAML
  as the compliance reference. Do not substitute the older per-modality Task 3
  configs/checkpoints.
- Task 3 uses `nRMSE` (lower is better), `SSIM` (higher is better), and `LPIPS`
  (lower is better). It has no SynthSeg, Dice, volume, or segmentation output.

The official multimodal config has 15 joint target domains:

```text
domain = modality_index * 5 + field_index
modalities = [T1W, T2W, T2FLAIR]
fields     = [0.1T, 1.5T, 3T, 5T, 7T]
```

Do not use the organizer inference script's `--target_field` shortcut blindly
with this checkpoint: it maps only field names to indices `0..4`, which are the
T1W domains. For T2W and T2FLAIR, pass the correct 15-way target-domain index or
use the verified `task3_pipeline.py` mapping. Validate this with a smoke case
before the full sweep.

## Dataset location and mental model

The complete data is outside this repository on the user's mounted drive. Its
absolute path is not yet authoritative; discover and verify it without scanning
unrelated drive contents. Refer to the verified root as `DATA_DIR`. Do not
invent a path or hardcode a GPU/machine-specific location.

The user reported this lowercase local split layout:

```text
mrixfields2026/
├── training_prospective/{T1W,T2W,T2FLAIR}/{0.1T,1.5T,3T,5T,7T}/
├── training_retrospective/{T1W,T2W,T2FLAIR}/{0.1T,1.5T,3T,5T,7T}/
└── validation_prospective/{T1W,T2W,T2FLAIR}/{0.1T,1.5T,3T,5T,7T}/
```

Official code commonly expects `Training_retrospective`,
`Training_prospective`, and `Validating_prospective`, with aliases
`retro_train`, `pro_train`, and `pro_val`. Inspect the actual mounted casing
before configuring scripts. Adapt config paths or use a non-destructive
compatibility layer; do not casually rename, duplicate, or modify the source
dataset.

- Retrospective training data is unpaired: subjects differ across fields.
- Prospective training data is paired: the same volunteer has matching scans
  across fields.
- Public validation subjects are paired only in the organizers' private ground
  truth. Each released subject appears at one source field. Never compare
  different validation IDs as if they were paired targets.

Expected validation source IDs are:

- `0.1T`: `0001`, `0002`, `0003`
- `1.5T`: `0004`, `0005`, `0008`
- `3T`: `0010`, `0011`, `0012`
- `5T`: `0013`, `0014`, `0015`
- `7T`: `0016`, `0017`, `0018`

Official volumes are NIfTI `.nii.gz`, float32, normalized to `[0, 1]`, on a
`(364, 436, 364)` 0.5 mm isotropic MNI grid. Audit these properties on the real
files. Preserve affine/header spatial metadata. Do not re-register, bias
correct, resample, skull-strip, super-resolve, or persistently renormalize data
that the organizers have already processed unless a measured issue and explicit
plan justify it.

The official 2D baseline internally pads/crops axial slices to `(512, 512)` and
maps `[0, 1]` to `[-1, 1]` for its tanh generator. Map predictions back to
float32 `[0, 1]` when reconstructing NIfTI volumes.

## Baseline-first workflow

1. Inventory only the workspace, relevant mounted dataset root, file counts and
   IDs, GPU/CUDA, free disk, and Python/Conda. Save a small reproducible audit
   report without exposing unrelated or sensitive contents.
2. Verify the local organizer checkout and record its commit. Re-check current
   official Task 3 code, config, weights, and submission rules.
3. Create the official environment and a local, ignored `.env`. Configure at
   least `DATA_DIR`, `PREPROCESSED_DIR`, `OUTPUT_DIR`, `INFERENCE_DIR`,
   `SUBMISSION_DIR`, and `DEVICE`. A harmless `PREDICTIONS_SEG_DIR` placeholder
   may be needed by the generic official builder interface, but do not create or
   use Task 1/2 segmentations. Never commit `.env`, data, credentials, weights,
   predictions, or archives.
4. Download only the official shared Task 3 multimodal checkpoint. Record the
   Hugging Face revision, file size, and SHA-256. Load it strictly with the
   matching config/code before a long run.
5. Run one end-to-end full-volume smoke translation: one validation input and
   one non-identity target. Verify path/name, target-domain index, shape,
   affine/header, float32 dtype, finite values, and `[0, 1]` range.
6. Run the complete resumable sweep with the same loaded checkpoint: 3
   modalities x 5 source fields x 4 non-identity target fields = 180
   full-volume predictions. Never generate identity mappings.
7. Validate every full-volume output immediately. Existing outputs may be
   skipped only after they pass the same validation checks; write new outputs
   atomically and retain a run manifest/provenance log.
8. Use the current official `Submission/build_submission` tooling with
   `--tasks task3` to rename target fields, create the Task 3 tree, and crop the
   axial slab. Do not package `$INFERENCE_DIR` directly and do not hand-roll the
   final archive unless the official builder is demonstrably broken.
9. Run structural and numerical validation, build `task3.zip`, list its
   contents, and save its SHA-256 plus exact command/config/checkpoint
   provenance. Do not upload it without explicit user confirmation.

If private validation targets are unavailable, a validated prediction set is
not a metric result. Public validation metrics come from the leaderboard. Local
metric tests must use correctly paired `training_prospective` data or a
documented holdout; never fabricate pairs from public validation IDs.

## Current Task 3 validation submission contract

The current local official `Submission/README.md` says the archive must contain
one top-level `task3/` directory:

```text
task3.zip
└── task3/
    ├── T1W/<source>_to_<target>/pred/*.nii.gz
    ├── T2W/<source>_to_<target>/pred/*.nii.gz
    └── T2FLAIR/<source>_to_<target>/pred/*.nii.gz
```

This supersedes the older workspace note that put modality folders directly at
the archive root. Re-verify immediately before packaging because the official
format is explicitly subject to change.

Required contents for the official validation release:

- 20 directed pair directories per modality and 3 predictions per pair
- 60 `(modality, pair)` directories and 180 prediction files total
- no identity pair directories
- no `seg/` directories or `_seg.nii.gz` files
- no unrelated files, alternate checkpoints, or extra enclosing directories

Each output preserves the source subject ID and modality but uses the target
field in its filename:

```text
input:  P_T1W_0.1T_0001.nii.gz
output: task3/T1W/0.1T_to_7T/pred/P_T1W_7T_0001.nii.gz
```

Submission predictions must be `.nii.gz`, float32, finite, within `[0, 1]`,
and shaped `(364, 436, 30)`. The official builder obtains this slab from axial
indices `[150, 180)` of each full `(364, 436, 364)` prediction. Preserve the
correct cropped affine/header and verify the builder's current behavior rather
than slicing manually.

## Engineering and experiment discipline

- Preserve organizer code and make focused, reviewable local changes. Never
  overwrite the only checkpoint, smoke output, baseline run, or archive.
- Keep smoke, baseline, and any future experimental artifacts in separate paths
  outside source control.
- Make runs resumable and deterministic where practical. Log command, config,
  code commit, checkpoint hash, device, runtime, seed/style seed, and paths.
- Fail loudly on missing cases, duplicate IDs, wrong domain indices, filename
  collisions, wrong pair/file counts, NaN/Inf, incorrect dtype/range/shape, or
  mismatched spatial metadata.
- Do not silently train. First report baseline submission status, expected
  compute, and a measurable evaluation plan; train only after user approval.
- Treat all medical images and metadata as sensitive. Do not modify or commit
  them, redistribute them, or expose more identifiers/metadata than needed.
- Do not claim success because a command started. Report the verified artifact
  and the checks that prove completion.

## Definition of done for the baseline milestone

- Actual dataset root and split casing are recorded locally without committing
  machine-specific secrets.
- Official code revision, environment, Task 3 config, checkpoint revision, and
  checkpoint SHA-256 are recorded.
- One full-volume smoke inference passes all NIfTI, domain-index, and numerical
  checks.
- The full 20-pair x 3-modality sweep completes with one shared checkpoint and
  all 180 full-volume outputs validate.
- The official builder produces `task3/` with 60 `(modality, pair)` directories
  and 180 valid slab predictions, with correct names, no identity mappings, and
  no segmentation content.
- `task3.zip` has the required top-level `task3/` directory, passes archive and
  numerical checks, and has a recorded SHA-256.
- The user receives the archive path, size, provenance, validation report, and
  upload instructions. The user performs or explicitly authorizes the upload.
