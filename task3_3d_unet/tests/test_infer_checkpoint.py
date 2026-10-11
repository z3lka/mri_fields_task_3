import argparse
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from task3_3d_unet.common import Task3Error
from task3_3d_unet.infer import save_nifti_atomic, sliding_window_predict, validate_output
from task3_3d_unet.model import ARCHITECTURE_ID, CHECKPOINT_FORMAT, ModelConfig, build_model
from task3_3d_unet.submission import expected_files
from task3_3d_unet.train import (
    _build_state,
    _require_training_gate,
    _resolve_training_output,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    warmup_cosine_schedule,
)


class SlidingWindowTests(unittest.TestCase):
    def test_identity_reconstruction_on_irregular_shape(self):
        class Identity(torch.nn.Module):
            def forward(self, source, modality, source_field, target_field):
                return source

        volume = np.random.default_rng(2026).random((11, 13, 15), dtype=np.float32)
        prediction = sliding_window_predict(
            Identity(), volume, modality=0, source_field=0, target_field=1,
            window=8, overlap=0.25, batch_size=3,
        )
        np.testing.assert_allclose(prediction, volume, rtol=0, atol=4e-7)

    def test_identity_direction_is_rejected(self):
        with self.assertRaisesRegex(Task3Error, "identity"):
            sliding_window_predict(
                torch.nn.Identity(), np.zeros((4, 4, 4), np.float32),
                modality=0, source_field=1, target_field=1, window=4,
            )


class NiftiTests(unittest.TestCase):
    def test_atomic_output_preserves_orientation_affine_dtype_and_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            affine = np.array([[-0.5, 0, 0, 4], [0, 0.5, 0, -2], [0, 0, 0.5, 7], [0, 0, 0, 1]])
            reference = nib.Nifti1Image(np.zeros((5, 6, 7), np.float32), affine)
            reference.set_qform(affine, code=1)
            reference.set_sform(affine, code=2)
            source = root / "source.nii.gz"
            output = root / "output.nii.gz"
            nib.save(reference, source)
            values = np.linspace(0, 1, 5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
            save_nifti_atomic(values, nib.load(source), output)
            report = validate_output(output, source)
            self.assertEqual(report["failures"], [])
            image = nib.load(output)
            self.assertEqual(image.get_data_dtype(), np.dtype(np.float32))
            np.testing.assert_allclose(image.affine, affine)


class CheckpointTests(unittest.TestCase):
    def payload(self):
        model = build_model(ModelConfig.tiny())
        return {
            "format": CHECKPOINT_FORMAT,
            "architecture": ARCHITECTURE_ID,
            "model_config": model.config.to_dict(),
            "model": model.state_dict(),
            "model_ema": copy.deepcopy(model.state_dict()),
            "optimizer": {},
            "scheduler": {},
            "scaler": {},
            "global_step": 1,
            "rng_state": capture_rng_state(),
            "run_signature": {"kind": "test"},
            "split_manifest": {"phase": "test"},
            "provenance": {},
            "best_model_ema": None,
            "best_step": -1,
        }

    def test_strict_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            payload = self.payload()
            torch.save(payload, path)
            loaded = load_checkpoint(path)
            model = build_model(ModelConfig.from_dict(loaded["model_config"]))
            model.load_state_dict(loaded["model"], strict=True)
            payload["model_config"]["unexpected"] = True
            torch.save(payload, path)
            with self.assertRaisesRegex(Task3Error, "Strict model config"):
                load_checkpoint(path)
            payload = self.payload()
            payload["architecture"] = "wrong"
            torch.save(payload, path)
            with self.assertRaisesRegex(Task3Error, "architecture"):
                load_checkpoint(path)

    def test_rng_state_restores_deterministically(self):
        random.seed(5)
        np.random.seed(5)
        torch.manual_seed(5)
        state = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), torch.rand(3))
        restore_rng_state(state)
        actual = (random.random(), float(np.random.rand()), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2])

    def test_fresh_run_cannot_overwrite_and_resume_must_use_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            (output / "existing.txt").write_text("keep")
            args = argparse.Namespace(output_dir=output, resume=None)
            with self.assertRaisesRegex(Task3Error, "refuses to overwrite"):
                _resolve_training_output(args)
            latest = output / "checkpoints" / "last.pt"
            latest.parent.mkdir()
            latest.write_bytes(b"checkpoint")
            args.resume = output / "checkpoints" / "best.pt"
            with self.assertRaisesRegex(Task3Error, "latest checkpoint"):
                _resolve_training_output(args)
            args.resume = latest
            self.assertEqual(_resolve_training_output(args), output.resolve())

    def test_model_optimizer_scheduler_and_rng_resume_deterministically(self):
        torch.manual_seed(2026)
        model = build_model(ModelConfig.tiny())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = warmup_cosine_schedule(
            optimizer, total_updates=3, warmup_updates=1, min_lr=1e-6, lr=1e-4
        )

        def update(current_model, current_optimizer, current_scheduler):
            source, target = torch.rand(1, 1, 8, 8, 8), torch.rand(1, 1, 8, 8, 8)
            loss = (current_model(source, 0, 0, 1) - target).abs().mean()
            loss.backward()
            current_optimizer.step()
            current_optimizer.zero_grad(set_to_none=True)
            current_scheduler.step()

        update(model, optimizer, scheduler)
        run_signature = {"kind": "resume-test"}
        split_manifest = {"phase": "resume-test"}
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "checkpoint.pt"
            payload = self.payload()
            payload.update(
                model=model.state_dict(),
                model_ema=copy.deepcopy(model.state_dict()),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                global_step=1,
                rng_state=capture_rng_state(),
                run_signature=run_signature,
                split_manifest=split_manifest,
            )
            torch.save(payload, checkpoint_path)
            update(model, optimizer, scheduler)
            expected = copy.deepcopy(model.state_dict())

            args = argparse.Namespace(
                device="cpu",
                tiny=True,
                lr=1e-4,
                weight_decay=1e-4,
                total_updates=3,
                warmup_updates=1,
                min_lr=1e-6,
                amp="off",
                resume=checkpoint_path,
            )
            resumed, _, resumed_optimizer, resumed_scheduler, _, step, *_ = _build_state(
                args, run_signature, split_manifest
            )
            self.assertEqual(step, 1)
            update(resumed, resumed_optimizer, resumed_scheduler)
            for name, value in resumed.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0.0, atol=0.0)


class RuntimeGateTests(unittest.TestCase):
    def test_a100_report_is_bound_to_paid_training_signature(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "runtime.json"
            report_path.write_text(
                json.dumps(
                    {
                        "updates_measured": 100,
                        "patch_size": 96,
                        "batch_size": 2,
                        "gradient_accumulation": 2,
                        "complete_projection_hours": 9.5,
                        "checks": {
                            "a100_detected": True,
                            "vram_headroom_at_least_20_percent": True,
                            "projected_runtime_at_most_10_hours": True,
                        },
                    }
                )
            )
            args = argparse.Namespace(
                baseline_milestone_complete=True,
                total_updates=10_000,
                runtime_report=report_path,
                patch_size=96,
                batch_size=2,
                gradient_accumulation=2,
                acknowledge_over_budget=False,
            )
            record = _require_training_gate(args)
            self.assertEqual(record["complete_projection_hours"], 9.5)
            self.assertEqual(len(record["sha256"]), 64)


class SubmissionContractTests(unittest.TestCase):
    def test_exact_task3_layout_has_180_nonidentity_files(self):
        files = expected_files()
        self.assertEqual(len(files), 180)
        self.assertTrue(all(path.parts[0] == "task3" for path in files))
        self.assertTrue(all("/seg/" not in f"/{path.as_posix()}/" for path in files))
        self.assertTrue(all(pair.split("_to_")[0] != pair.split("_to_")[1] for _, _, pair, _, _ in (p.parts for p in files)))


if __name__ == "__main__":
    unittest.main()
