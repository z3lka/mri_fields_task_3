# MRIxFields2026 Task 3

This workspace is Task 3 only. The organizer-pretrained baseline and future
training are deliberately separate:

| Workflow | Entry point | Runtime results |
| --- | --- | --- |
| Official pretrained baseline, validation predictions, and task3.zip | task3_pretrained_baseline_colab.ipynb | /content/mrixfields_task3/pretrained/results/ |
| Future training and local holdout results | task3_stargan_v2_colab.ipynb | /content/mrixfields_task3/training/results/ |

The baseline milestone comes first. Do not enable any training switch until
the pretrained notebook has produced a structurally and numerically validated
180-file task3.zip.

## Runtime and Drive boundary

Both notebooks use mounted Drive only as persistent source/export storage:

1. Mount Drive.
2. Select every required input, then copy the scripts, NIfTI split(s), and
   checkpoint into that workflow's `/content` runtime tree.
3. Verify the runtime copies, flush Drive, and unmount it.
4. Audit, preprocess, infer, train, validate, and package entirely in runtime.
   Processing cells fail if Drive is still mounted.
5. At the end, explicitly enable the final export cell. It remounts Drive,
   copies only selected final artifacts to the workflow's separate export
   folder, verifies the copy, and unmounts Drive again.

There is no Drive I/O while a model or data-processing process is running.
Predictions, preprocessing caches, package caches, temporary checkpoints, and
submission build trees are never written directly to Drive. Runtime contents
disappear when Colab resets, so enable the final export only after the result
you want to keep has passed its checks.

## Pretrained baseline

Open task3_pretrained_baseline_colab.ipynb. It owns the complete baseline path:

- stage the 45 released validation NIfTI files into runtime;
- recheck the live organizer Git and Hugging Face revisions;
- stage and SHA-256-check the one shared multimodal checkpoint;
- audit all inputs and plan all 180 non-identity translations;
- run one separate full-volume smoke case;
- run and validate the complete 180-volume sweep;
- call the official Task 3 builder;
- validate the exact 60-pair-directory/180-file slab tree;
- create task3.zip with top-level task3/; and
- optionally export the archive and compact JSON reports.

No cell uploads the archive.

The underlying focused Python tools are:

- task3_pipeline.py: input audit, case plan, checkpoint download, inference,
  and full-volume validation;
- task3_build_submission.py: official builder invocation, submission-tree
  checks, numerical checks, archive creation, and archive SHA-256.

The final archive contract is:

    task3.zip
    └── task3/
        ├── T1W/<source>_to_<target>/pred/*.nii.gz
        ├── T2W/<source>_to_<target>/pred/*.nii.gz
        └── T2FLAIR/<source>_to_<target>/pred/*.nii.gz

It must contain 20 directed non-identity field pairs per modality, three files
per pair, no identity mappings, and no segmentation content.

## Training

Open task3_stargan_v2_colab.ipynb only after the pretrained milestone. It
contains:

- prospective/optional retrospective staging into its own runtime data tree;
- raw-data and extracted-slice audits;
- an explicit baseline-milestone gate;
- the subject-held-out pilot and its per-modality nRMSE/SSIM/LPIPS proxies;
- optional all-subject continuation; and
- one selected-checkpoint plus metrics/provenance export.

It contains no pretrained validation sweep, submission builder, or archive
export. See TASK3_TRAINING_COLAB.md for experiment constraints.

## Conditional FPS-Former experiment

The paper-based replacement model is implemented entirely in the
`task3_fpsformer/` Python package; it does not use a notebook:

- `task3_fpsformer/data.py`: paired prospective audit and slice cache;
- `task3_fpsformer/model.py`: FMAM, SPAM, SDFN, HEFR, and one shared
  modality/source-field/target-field conditioned encoder-decoder;
- `task3_fpsformer/train.py`: subject-held-out, resumable training on MPS or
  CUDA; and
- `task3_fpsformer/infer.py`: smoke or complete 180-volume validation sweep.

See `task3_fpsformer/README.md` for the paper-to-Task-3 adaptation, known
departures from the unpublished reference implementation, and exact Mac/Colab
commands.

## Reproducibility pins

Prepared pins:

- Official code commit:
  f0f4ab2c54ab878b48ddfea8d13f4da512ca411e
- Hugging Face revision:
  f1a7d6b741b64c81871d0856fa2d7295d564c141
- Shared checkpoint:
  task3_any_to_any_multimodal/stargan_v2/pro_pretrained/weights/checkpoint_epoch50.pth
- Checkpoint size: 631346455 bytes
- Checkpoint SHA-256:
  c2f8172c1fbdbf0426e22c7d3d2014632f175a38cf818f95e72b1917badabdeb

The pretrained notebook checks live default revisions before it runs. If a
revision differs, stop and review the official Task 3 config, weight, and
submission contract as one coherent update.

The final build is also locked behind LIVE_RULES_REVIEWED. The current GitHub
submission README specifies a validation task3.zip, while the challenge tasks
page also uses broader container-submission language. Confirm which mechanism
applies to the current validation phase before exporting or uploading. The
workspace never uploads either form.

## Local non-GPU checks

    python -m unittest discover -v
    python task3_pipeline.py audit --input-root T1W --modalities T1W
    python task3_pipeline.py plan --input-root T1W --modalities T1W

The released validation targets are private. A validated prediction set is not
a local metric result; public-validation metrics come from the leaderboard.
