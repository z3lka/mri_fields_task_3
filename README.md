# MRIxFields2026 Task 3 validation baseline

This workspace now contains a resumable Python runner for the organizers'
single shared Task 3 StarGAN v2 checkpoint. It is prepared for local use or a
Colab GPU. It does not train, package, upload, or pretend to compute public
validation scores without the organizers' private targets.

Pinned upstream versions used while preparing the runner on 2026-07-12:

- Official code: `MRIxFields/MRIxFields2026` commit
  `f0f4ab2c54ab878b48ddfea8d13f4da512ca411e`
- Official weights: `mrixfields/MRIxFields2026-Baseline` revision
  `f1a7d6b741b64c81871d0856fa2d7295d564c141`
- Checkpoint: `task3_any_to_any_multimodal/stargan_v2/pro_pretrained/weights/checkpoint_epoch50.pth`

The local `T1W/` sample was inspected: it contains the expected 15 validation
volumes and official IDs. All headers report float32, `(364, 436, 364)`, and
0.5 mm isotropic voxels.

## Local preparation (no inference)

Create an environment and install the small inference dependency set. A
CUDA-enabled PyTorch installation is also required for practical inference.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-colab.txt
```

The organizer repository has already been placed under
`official/MRIxFields2026/` in this workspace. To reproduce that checkout on a
new machine:

```bash
python task3_pipeline.py fetch-official
```

Audit the T1W inputs. Header-only audit is quick; `--check-values` reads all
871 MB and additionally checks finite values and the `[0,1]` range.

```bash
python task3_pipeline.py audit --input-root T1W --modalities T1W
python task3_pipeline.py audit --input-root T1W --modalities T1W --check-values
```

Preview every planned T1W output without loading a model:

```bash
python task3_pipeline.py plan --input-root T1W --modalities T1W
```

This should report 15 inputs and 60 translations: five source fields, four
non-identity targets, and three IDs per source field.

## Colab GPU workflow

Copy `task3_pipeline.py` and `requirements-colab.txt` to Colab or Google Drive,
select a GPU runtime, and run:

```bash
pip install -r requirements-colab.txt
python task3_pipeline.py fetch-official
python task3_pipeline.py download-checkpoint
```

Mount Drive and point `--input-root` and `--output-dir` at persistent Drive
paths. The input may be either the single `T1W/` directory shown in this
workspace or a validation root containing `T1W/`, `T2W/`, and `T2FLAIR/`.

Always start with exactly one full-volume smoke translation:

```bash
python task3_pipeline.py infer \
  --input-root /content/drive/MyDrive/mrixfields2026/validation_prospective/T1W \
  --output-dir /content/drive/MyDrive/mrixfields_runs/smoke \
  --modalities T1W \
  --device cuda \
  --batch-size 1 \
  --smoke

python task3_pipeline.py validate-outputs \
  --input-root /content/drive/MyDrive/mrixfields2026/validation_prospective/T1W \
  --output-dir /content/drive/MyDrive/mrixfields_runs/smoke \
  --modalities T1W \
  --smoke
```

If the smoke case passes and GPU memory is comfortable, run the T1W sweep:

```bash
python task3_pipeline.py infer \
  --input-root /content/drive/MyDrive/mrixfields2026/validation_prospective/T1W \
  --output-dir /content/drive/MyDrive/mrixfields_runs/baseline \
  --modalities T1W \
  --device cuda \
  --batch-size 1 \
  --resume

python task3_pipeline.py validate-outputs \
  --input-root /content/drive/MyDrive/mrixfields2026/validation_prospective/T1W \
  --output-dir /content/drive/MyDrive/mrixfields_runs/baseline \
  --modalities T1W
```

`--resume` validates each existing full-volume prediction before skipping it.
Outputs are written atomically, and the manifest is updated after every
completed volume. Start with batch size 1; only increase it after observing
the smoke run's peak GPU memory.

When all three modality folders are available, use their common parent and
request all modalities:

```bash
python task3_pipeline.py infer \
  --input-root /content/drive/MyDrive/mrixfields2026/validation_prospective \
  --output-dir /content/drive/MyDrive/mrixfields_runs/baseline \
  --modalities T1W T2W T2FLAIR \
  --device cuda \
  --batch-size 1 \
  --resume
```

That complete run plans 180 full-volume predictions. It loads one checkpoint
once and computes the correct 15-way joint target-domain index. This matters
for T2W and T2FLAIR: the organizer's current `--target_field` shortcut only
maps field names to indices 0-4, which are the T1W domains.

## What “validation results” means here

The released validation subjects do not include paired targets. Therefore the
local result is a validated set of predictions, not nRMSE/SSIM/LPIPS scores.
Those scores come from the public leaderboard after submission. Local metric
experiments require correctly paired `training_prospective` data or a documented
holdout; never pair different public validation IDs.

## Packaging discrepancy to resolve later

The current official repository says `task3.zip` must contain a top-level
`task3/` directory. The workspace instructions currently say the zip root must
instead directly contain `T1W/`, `T2W/`, and `T2FLAIR/`. Because these contracts
conflict, this pass intentionally does not create a zip. Re-check the official
repository and competition page immediately before packaging, then use the
current official builder after resolving the discrepancy.

