# Task 3 training and training results on Colab

Use task3_stargan_v2_colab.ipynb only after the separate
task3_pretrained_baseline_colab.ipynb has produced and validated the official
baseline submission. Every training switch defaults to off, and the Python CLI
also requires --baseline-milestone-complete.

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

The mounted dataset and optional initialization checkpoint are read once by
explicit staging cells and copied into this runtime tree. Audit,
preprocessing, training, checkpointing, and metric evaluation use only runtime
paths. The last cell is the only Drive write and exports one selected
checkpoint with its provenance.json and train.jsonl.

Use distinct persistent export roots:

    mrixfields_task3_exports/
    ├── pretrained/
    └── training/

Do not export caches, extracted slices, raw data, every checkpoint, or
full-volume prediction trees.

## Experiment order

1. Complete and report the organizer-pretrained baseline milestone.
2. Stage and audit the prospective split in runtime.
3. Extract and audit the organizer's 220 axial slices in runtime.
4. Run the short pilot_subject_holdout experiment from scratch.
5. Review per-modality slice-proxy nRMSE, SSIM, and LPIPS plus images.
6. Only with a measured reason, run a conservative all-subject continuation.
7. Export one selected checkpoint and its compact result/provenance files.

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

Training authorization never authorizes an external upload or submission.
