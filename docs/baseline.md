# Pretrained Task 3 baseline

Use the [Colab notebook](../notebooks/task3_pretrained_baseline_colab.ipynb) for
the guided workflow or the [Kaggle notebook](../notebooks/task3_pretrained_baseline_kaggle.ipynb)
for two T4 GPUs. Both use the official shared multimodal StarGAN v2 checkpoint.

## Before you run

You need the released validation split, a GPU runtime, and enough runtime disk
for the checkpoint and 180 full-volume predictions. Keep source data private
and unchanged. The validation split should look like this:

```text
validation_prospective/
├── T1W/{0.1T,1.5T,3T,5T,7T}/*.nii.gz
├── T2W/{0.1T,1.5T,3T,5T,7T}/*.nii.gz
└── T2FLAIR/{0.1T,1.5T,3T,5T,7T}/*.nii.gz
```

There are three source subjects per field and modality: 45 inputs total. Set
the input path to this split, using the actual casing on your drive.

For Colab, copy these three repository-root files into the private source folder
selected by `DRIVE_CODE_SOURCE_VALUE`:

- `task3_pipeline.py`
- `task3_build_submission.py`
- `requirements-colab.txt`

Set `DRIVE_VALIDATION_SOURCE_VALUE` to the validation split. Supply an existing
checkpoint through `DRIVE_CHECKPOINT_SOURCE_VALUE` or enable the pinned download
cell. Choose `DRIVE_EXPORT_ROOT_VALUE` for the final archive and reports.

The Colab notebook stages and verifies code, data, and weights under
`/content/mrixfields_task3/pretrained/`, then flushes and unmounts Drive.
Inference, validation, and packaging run on runtime storage. The final opt-in
export cell remounts Drive, verifies the copied archive, and unmounts it again.

For Kaggle, attach the same three code files and your private validation data
as inputs. Edit `PROJECT_SOURCE` and `VALIDATION_DIR`, enable internet for
source/checkpoint retrieval, and select two T4 GPUs. Its two workers share the
same checkpoint parameters and split the deterministic plan into 90 cases each.
Download the final `/kaggle/working/mrixfields-task3/task3.zip` and reports.

## Run order

1. Stage inputs and check the organizer code and checkpoint revisions.
2. Audit NIfTI files and plan all 180 translations.
3. Enable one full-volume smoke translation and inspect its validation report.
4. Enable the complete sweep and validate all 180 full-volume outputs.
5. Review the live official rules before enabling the submission builder.
6. Build and validate `task3.zip`, then export the archive and compact reports.

Each modality covers all 20 directed non-identity field pairs, including
translations to lower fields. The checkpoint uses 15 joint domains:

```text
domain = modality_index * 5 + field_index
modalities = [T1W, T2W, T2FLAIR]
fields = [0.1T, 1.5T, 3T, 5T, 7T]
```

The wrapper handles this mapping. A field-only index would select T1W domains
for T2W and T2FLAIR inputs, so retain the verified mapping.

## Terminal entry points

Run commands from the repository root. Use a Python environment with PyTorch
appropriate for your device, then install `requirements-colab.txt`. This file
deliberately leaves Colab's existing torch/torchvision installation alone.

Set your actual dataset root, then audit and inspect the complete case plan:

```bash
export DATA_DIR="/absolute/path/to/mrixfields2026"

python task3_pipeline.py audit \
  --input-root "$DATA_DIR/validation_prospective" \
  --modalities T1W T2W T2FLAIR --check-values

python task3_pipeline.py plan \
  --input-root "$DATA_DIR/validation_prospective" \
  --modalities T1W T2W T2FLAIR
```

`DATA_DIR` is an environment variable supplied by you; these tools take paths
through command-line arguments and do not automatically read a `.env` file.
Use `python task3_pipeline.py --help` and
`python task3_build_submission.py --help` for inference and packaging options.
The guided notebooks include the complete commands and prerequisite checks.

## Submission output

The prepared validation ZIP contract is:

```text
task3.zip
└── task3/
    ├── T1W/<source>_to_<target>/pred/*.nii.gz
    ├── T2W/<source>_to_<target>/pred/*.nii.gz
    └── T2FLAIR/<source>_to_<target>/pred/*.nii.gz
```

Expect 60 modality/pair directories and 180 files, with three predictions per
pair. Filenames retain the source subject ID and use the target field. The
official builder crops axial indices `[150, 180)` from `(364, 436, 364)` full
volumes to `(364, 436, 30)` slabs. Checks cover float32 dtype, finite values,
`[0, 1]` range, spatial metadata, names, counts, archive contents, and SHA-256.

Review the live [competition page](https://mrixfields.chihucloud.com/2026/),
[task rules](https://mrixfields.chihucloud.com/2026/tasks.html), and
[submission README](https://github.com/MRIxFields/MRIxFields2026/blob/main/Submission/README.md)
before the final build. Confirm the submission mechanism for the phase you are
entering; validation ZIP and test-phase container instructions are distinct.
The notebooks require an explicit live-rules acknowledgement. Uploads are
performed by the user.

## Reproducibility

These are preparation pins recorded in the code, not a statement of the latest
upstream revisions:

| Component | Prepared revision |
| --- | --- |
| Pipeline defaults and Colab organizer checkout | `f0f4ab2c54ab878b48ddfea8d13f4da512ca411e` |
| Kaggle organizer checkout override | `a22abccbbf61e2131d27331035ff5b9114565fbd` |
| Hugging Face weights | `f1a7d6b741b64c81871d0856fa2d7295d564c141` |

The checkpoint path is
`task3_any_to_any_multimodal/stargan_v2/pro_pretrained/weights/checkpoint_epoch50.pth`.
Its recorded size is 631,346,455 bytes and SHA-256 is:

```text
c2f8172c1fbdbf0426e22c7d3d2014632f175a38cf818f95e72b1917badabdeb
```

Keep the selected notebook's code/config/weight revisions together. If its live
revision check stops, review the upstream changes before updating pins. Save
the run manifest, validation report, archive hash, and exact revisions with
each exported submission.

## Tests and optional tools

Lightweight checks need no GPU:

```bash
python3 -m unittest -v tests.test_task3_pipeline tests.test_task3_colab_train \
  task3_fpsformer.tests.test_colab_notebook task3_3d_unet.tests.test_notebook
```

The full suite needs the packages' PyTorch/NIfTI dependencies. Use the Python
environment containing those packages, such as the environment from the
[FPS-Former guide](../task3_fpsformer/README.md):

```bash
.venv-fpsformer/bin/python -m unittest discover -v
```

Notebook checks cover source parsing, workflow separation, runtime-path guards,
input staging, verified export, and unmounting after a failed export. See the
[training guide](training.md) for the repaired StarGAN workflow.

`tools/task3_review.py` creates prediction montages and supports quality metrics
only with genuine paired targets; it needs nibabel, NumPy, and matplotlib.
`tools/task3_plot_metrics.py` creates an interactive dashboard from a training
`metrics.jsonl` file and needs Plotly. `tools/vram_probe.py` measures the CUDA
memory of a training step supplied by the caller. These are optional utilities.

Public validation targets are private. A validated prediction set is not a
local quality score. Task 3 quality metrics are nRMSE (lower), SSIM (higher),
and LPIPS (lower); local metric comparisons require paired prospective training
data or a documented holdout.
