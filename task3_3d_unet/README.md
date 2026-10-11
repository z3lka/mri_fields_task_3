# Task 3 conditional 3D U-Net

This isolated experiment implements one shared 3D U-Net conditioned on modality,
source field, and target field. It does not modify or import the organizer model
or `task3_fpsformer`.

The guarded Colab workflow is [`task3_3d_unet_colab.ipynb`](../notebooks/task3_3d_unet_colab.ipynb).
The checked-in switches are configured for a fresh training run through full
validation inference. Over-budget acknowledgement, backup restoration, final
packaging, and archive export remain off until their prerequisites are met.

Core commands:

```bash
python -m task3_3d_unet.data audit --data-dir "$DATA_DIR" --report audit.json
python -m task3_3d_unet.data stage --data-dir "$DATA_DIR" --runtime-dir /content/task3_data
python -m task3_3d_unet.data prepare-cache --data-dir /content/task3_data --cache-dir /content/task3_cache
python -m task3_3d_unet.probe --cache-dir /content/task3_cache \
  --official-repo /content/MRIxFields2026 --report runtime_probe.json

python -m task3_3d_unet.train pretrain \
  --retrospective-manifest /content/task3_data/retrospective_subset.json \
  --output-dir /content/runs/pretrain --baseline-milestone-complete \
  --runtime-report runtime_probe.json

python -m task3_3d_unet.train finetune --arm paired_global \
  --cache-dir /content/task3_cache --official-repo /content/MRIxFields2026 \
  --output-dir /content/runs/ablations/paired_global \
  --baseline-milestone-complete --runtime-report runtime_probe.json
```

Run `paired_masked` identically. For `pretrained_masked`, also pass
`--pretrained /content/runs/pretrain/checkpoints/best.pt`. Select the winner:

```bash
python -m task3_3d_unet.compare \
  --ablation-root /content/runs/ablations --report /content/runs/winner.json
```

Retrain the selected recipe with `--all-subjects` from its original
initialization, not from its holdout-trained checkpoint. Then run smoke inference
before full inference:

```bash
python -m task3_3d_unet.infer --checkpoint FINAL.pt \
  --input-root /content/task3_data --output-dir /content/inference_smoke --smoke
python -m task3_3d_unet.infer --checkpoint FINAL.pt \
  --input-root /content/task3_data --output-dir /content/inference_full \
  --smoke-manifest /content/inference_smoke/task3_3d_unet_inference_manifest.json
```

Final packaging calls the current official builder with `--tasks task3` and is
gated by an explicit current commit and Submission README hash. It creates but
never uploads `task3.zip`.
