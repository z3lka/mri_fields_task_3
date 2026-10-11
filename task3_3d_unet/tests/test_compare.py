import json
import tempfile
import unittest
from pathlib import Path

import torch

from task3_3d_unet.common import HOLDOUT_SUBJECT, TRAIN_SUBJECTS, Task3Error
from task3_3d_unet.compare import select_winner
from task3_3d_unet.model import ARCHITECTURE_ID, CHECKPOINT_FORMAT, ModelConfig
from task3_3d_unet.train import ARMS


def write_arm(root: Path, arm: str, nrmse: float, ssim: float, lpips: float, identity: float):
    run = root / arm
    (run / "checkpoints").mkdir(parents=True)
    initialization = (
        {"path": "/tmp/pretrain.pt", "sha256": "pretrain-hash"}
        if arm == "pretrained_masked"
        else "scratch_seed_2026"
    )
    run_signature = {
        "kind": "finetune",
        "arm": arm,
        "all_subjects": False,
        "initialization": initialization,
        "total_updates": 10_000,
        "patch_size": 96,
        "batch_size": 2,
        "gradient_accumulation": 2,
        "lr": 1e-4,
        "min_lr": 1e-6,
        "warmup_updates": 500,
        "weight_decay": 1e-4,
        "ema_decay": 0.999,
        "seed": 2026,
        "tiny": False,
        "amp": "bf16",
        "clip_grad_norm": 1.0,
        "selection_report": None,
        "metric_provenance": {
            "official_git_revision": "test",
            "official_scorer_sha256": "same-scorer",
        },
        "runtime_gate": {"sha256": "same-runtime"},
        "eval_every": 1000,
        "holdout_samples_per_stratum": 1,
        "lpips_enabled": True,
    }
    split_manifest = {
        "train_subjects": list(TRAIN_SUBJECTS),
        "holdout_subject": HOLDOUT_SUBJECT,
        "cache_manifest_sha256": "same-data",
    }
    summary = {
        "status": "complete",
        "phase": "ablation",
        "arm": arm,
        "best_metrics": {
            "mean": {"nrmse": nrmse, "ssim": ssim, "lpips": lpips},
            "identity_mean_nrmse": identity,
        },
        "best_step": 1000,
        "initialization": initialization,
        "run_signature": run_signature,
        "split_manifest": split_manifest,
        "metric_provenance": {
            "official_git_revision": "test",
            "official_scorer_sha256": "same-scorer",
        },
    }
    (run / "summary.json").write_text(json.dumps(summary))
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "architecture": ARCHITECTURE_ID,
            "model_config": ModelConfig().to_dict(),
            "model": {},
            "model_ema": {},
            "optimizer": {},
            "scheduler": {},
            "scaler": {},
            "global_step": 10_000,
            "rng_state": {},
            "run_signature": run_signature,
            "split_manifest": split_manifest,
            "provenance": {},
            "best_model_ema": {},
            "best_step": 1000,
            "checkpoint_role": "lowest_holdout_nrmse",
            "selected_global_step": 1000,
        },
        run / "checkpoints/best.pt",
    )


class ComparisonTests(unittest.TestCase):
    def test_selects_by_nrmse_then_ssim_then_lpips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_arm(root, ARMS[0], 0.20, 0.80, 0.10, 0.30)
            write_arm(root, ARMS[1], 0.18, 0.75, 0.11, 0.30)
            write_arm(root, ARMS[2], 0.18, 0.82, 0.12, 0.30)
            report = select_winner(root)
            self.assertEqual(report["winner"]["arm"], ARMS[2])
            self.assertFalse(report["final_retrain"]["continue_holdout_checkpoint"])

    def test_blocks_winner_that_does_not_beat_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, arm in enumerate(ARMS):
                write_arm(root, arm, 0.20 + index * 0.01, 0.8, 0.1, 0.20)
            with self.assertRaisesRegex(Task3Error, "does not beat"):
                select_winner(root)


if __name__ == "__main__":
    unittest.main()
