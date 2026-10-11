# Task 3 training and training results on Colab

Use [the StarGAN training notebook](../notebooks/task3_stargan_v2_colab.ipynb)
only after [the pretrained baseline notebook](../notebooks/task3_pretrained_baseline_colab.ipynb)
has produced and validated the official
baseline submission. Every training switch defaults to off, and the Python CLI
also requires --baseline-milestone-complete.

The notebook stages inputs into its own runtime tree and guards every
processing cell against mounted Drive or paths outside that tree. It contains
training and training-result export only; validation inference and submission
packaging use the separate baseline workflow. Run its checks with
`python -m unittest -v tests.test_task3_notebooks`.

## Setup

Put `task3_colab_train.py` and `requirements-task3-training-colab.txt` together
in the private Drive source folder selected by `DRIVE_CODE_SOURCE_VALUE`.
Set `DRIVE_DATA_SOURCE_VALUE` to the parent dataset root. The existing helper
resolves organizer or lowercase split names without modifying source data.

Prospective staging is on by default. Enable `STAGE_RETROSPECTIVE_DATA` in the
initial staging cell if that split is needed. The subject-held-out pilot starts
from scratch, so official checkpoint staging is off by default. For
all-subject continuation, select either `USE_EXISTING_CHECKPOINT` with
`DRIVE_CHECKPOINT_SOURCE_VALUE`, or `DOWNLOAD_OFFICIAL_CHECKPOINT`. Saved
weights are copied before unmounting; downloads occur in runtime after the
dependencies are installed. Both routes verify the pinned size and SHA-256.

## Storage layout

The training notebook uses:

    /content/mrixfields_task3/training/
    ├── code/
    ├── data/
    │   ├── training_prospective/
    │   └── training_retrospective/       # optional
    ├── data_view/
    ├── preprocessed/
    ├── weights/
    ├── official/MRIxFields2026/
    └── results/
        ├── reports/
        └── training_runs/

The selected prospective data, optional retrospective data, and optional saved
initialization checkpoint are copied by the initial staging cell into this
runtime tree. It checks the NIfTI filename/size inventory and verifies copied
code and checkpoint hashes, then flushes Drive and
unmounts it. Audit, preprocessing, training, checkpointing, and metric
evaluation refuse to run while Drive is mounted and use only runtime paths.
The last cell optionally remounts Drive, exports one selected checkpoint with
its provenance.json and train.jsonl, verifies the checkpoint copy, and
unmounts Drive again.

Use distinct persistent export roots:

    mrixfields_task3_exports/
    ├── pretrained/
    └── training/

Do not export caches, extracted slices, raw data, every checkpoint, or
full-volume prediction trees.

## Experiment order

1. Complete and report the organizer-pretrained baseline milestone.
2. Select all needed inputs, stage them in runtime, and unmount Drive.
3. Audit the prospective split in runtime.
4. Extract and audit the organizer's 220 axial slices in runtime.
5. Run the short pilot_subject_holdout experiment from scratch.
6. Review per-modality slice-proxy nRMSE, SSIM, and LPIPS plus images.
7. Only with a measured reason, run a conservative all-subject continuation.
8. Optionally remount Drive and export one selected checkpoint plus its compact
   result/provenance files; the export cell unmounts Drive when finished.

The released pro_pretrained/checkpoint_epoch50.pth already saw the prospective
split. It cannot initialize an honest prospective-subject holdout. The default
pilot therefore starts from scratch. A future retrospective-only
initialization may be used only with the helper's explicit
--initialization-excludes-holdout attestation.

Continued fine-tuning from the released checkpoint is an all-subject
experiment and has no honest local prospective holdout. The public validation
targets remain private, so its real validation comparison requires a complete
candidate submission in a later, explicitly approved evaluation run.

## Result tree

    results/training_runs/
    └── task3_any_to_any_multimodal/
        └── stargan_v2/
            └── <run-name>/
                ├── provenance.json
                ├── train.jsonl
                └── weights/checkpoint_*.pth

Checkpoints retain organizer-compatible nets and nets_ema keys. Atomic
within-epoch checkpoints and bounded retention make a runtime session
resumable while it lasts, but runtime files vanish when Colab resets. Export
the one selected final checkpoint at the end of a successful run.

Set `SELECTED_CHECKPOINT_VALUE` to one checkpoint in this runtime's
`results/training_runs/.../weights/` directory, choose a new `EXPORT_NAME_VALUE`,
and enable `EXPORT_TRAINING_RESULT` after training stops. The export includes
that checkpoint, its `provenance.json` and `train.jsonl`, compact JSON reports,
and an export manifest with sizes and SHA-256 hashes. Drive is unmounted in a
`finally` block even if copying fails. Existing export folders are never
overwritten.

Training authorization never authorizes an external upload or submission.
