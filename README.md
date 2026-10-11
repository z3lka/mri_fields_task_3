# MRIxFields2026 · Task 3

Translate MRI scans between five magnetic field strengths with one shared,
modality-conditioned model. This repository focuses on **Task 3**: T1W, T2W,
and T2FLAIR translation across 0.1T, 1.5T, 3T, 5T, and 7T.

Start with the organizers' pretrained baseline. It turns the 45 released
validation scans into 180 non-identity predictions and builds a checked
`task3.zip` for you to upload.

## Get started

1. Open the [pretrained Colab notebook](notebooks/task3_pretrained_baseline_colab.ipynb)
   in Google Colab and select a GPU runtime.
2. Put `task3_pipeline.py`, `task3_build_submission.py`, and
   `requirements-colab.txt` together in your private Drive source folder.
   Set the notebook's code, validation-data, checkpoint, and export paths.
   You can download the checkpoint through the notebook.
3. Run the setup and audit cells. Enable smoke inference first, then the full
   sweep after the smoke checks pass.
4. Review the current official submission rules, enable packaging, and export
   the validated archive and reports.

The Colab baseline copies inputs into runtime storage and unmounts Drive before
inference. Runtime files disappear when the session resets; export the final
result before leaving. The notebook creates the archive but never uploads it.

Using Kaggle instead? Open the [dual-T4 baseline notebook](notebooks/task3_pretrained_baseline_kaggle.ipynb).
See the [baseline guide](docs/baseline.md) for setup, terminal commands, and
reproducibility details.

## Find your workflow

| Workflow | Notebook | Guide |
| --- | --- | --- |
| Official pretrained baseline — start here | [Colab](notebooks/task3_pretrained_baseline_colab.ipynb) · [Kaggle](notebooks/task3_pretrained_baseline_kaggle.ipynb) | [Baseline guide](docs/baseline.md) |
| StarGAN v2 training | [Colab](notebooks/task3_stargan_v2_colab.ipynb) | [Training guide](docs/training.md) |
| Conditional FPS-Former experiment | [Colab](task3_fpsformer/task3_fpsformer_colab.ipynb) | [Package guide](task3_fpsformer/README.md) |
| Conditional 3D U-Net experiment | [Colab](notebooks/task3_3d_unet_colab.ipynb) | [Package guide](task3_3d_unet/README.md) |

Training and experimental workflows come after the pretrained submission
milestone is verified and reported. Review their run switches before use.

## Repository layout

```text
notebooks/                 Colab and Kaggle workflows
docs/                      Detailed baseline and training guides
tests/                     Baseline/helper checks
tools/                     Prediction review, metric plots, and VRAM probe
task3_fpsformer/            FPS-Former package, notebook, and tests
task3_3d_unet/              3D U-Net package and tests
task3_pipeline.py          Audit, plan, infer, and validate full volumes
task3_build_submission.py  Official builder wrapper and archive validation
task3_colab_train.py       StarGAN training helper
```

Data, weights, predictions, archives, generated reports, presentation material,
and saved notebook copies stay local and are ignored by Git. The organizer
checkout is fetched separately into `official/MRIxFields2026/`.

## Local checks

Run these from the repository root; they need Python 3.10+ and no GPU:

```bash
python3 -m unittest -v tests.test_task3_pipeline tests.test_task3_colab_train
```

The [baseline guide](docs/baseline.md) also covers dataset audits and the full
test suite. Public validation targets are private: local structural checks
confirm a valid artifact, while leaderboard results provide validation quality
metrics.

## Official resources

[Competition](https://mrixfields.chihucloud.com/2026/) ·
[Data](https://mrixfields.chihucloud.com/2026/data.html) ·
[Task rules](https://mrixfields.chihucloud.com/2026/tasks.html) ·
[Organizer code](https://github.com/MRIxFields/MRIxFields2026) ·
[Shared Task 3 weights](https://huggingface.co/mrixfields/MRIxFields2026-Baseline/tree/main/task3_any_to_any_multimodal) ·
[Submission format](https://github.com/MRIxFields/MRIxFields2026/blob/main/Submission/README.md)
