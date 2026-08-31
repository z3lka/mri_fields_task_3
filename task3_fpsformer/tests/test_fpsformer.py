from __future__ import annotations

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from task3_fpsformer.common import (
    FIELDS,
    MODALITIES,
    directed_pairs,
    joint_domain,
    parse_cache_name,
    prediction_path,
    split_domain,
)
from task3_fpsformer.data import (
    PairRecord,
    ProspectivePairDataset,
    SliceRecord,
    _paired_crop,
    build_cache_index,
    complete_subjects,
)


class CommonContractTests(unittest.TestCase):
    def test_joint_domains_cover_exactly_fifteen_indices(self) -> None:
        domains = {
            joint_domain(modality, field)
            for modality in MODALITIES
            for field in FIELDS
        }
        self.assertEqual(domains, set(range(15)))
        for domain in domains:
            modality, field = split_domain(domain)
            self.assertEqual(joint_domain(modality, field), domain)

    def test_directed_pairs_have_no_identity_and_cover_twenty(self) -> None:
        pairs = directed_pairs()
        self.assertEqual(len(pairs), 20)
        self.assertEqual(len(set(pairs)), 20)
        self.assertTrue(all(source != target for source, target in pairs))

    def test_prediction_path_matches_official_builder_source_layout(self) -> None:
        path = prediction_path(
            Path("outputs"),
            modality="T2W",
            source_field="0.1T",
            target_field="7T",
            source_name="P_T2W_0.1T_0001.nii.gz",
        )
        self.assertEqual(
            path,
            Path("outputs/task3_0.1T_to_7T_T2W/fpsformer/pro_scratch/best/")
            / "P_T2W_0.1T_0001.nii.gz",
        )

    def test_cache_name_round_trip(self) -> None:
        parsed = parse_cache_name("pro_train_T2FLAIR_1.5T_P_0012_s179.npz")
        self.assertEqual(parsed["modality"], "T2FLAIR")
        self.assertEqual(parsed["field"], "1.5T")
        self.assertEqual(parsed["subject_id"], "P_0012")
        self.assertEqual(parsed["slice"], 179)


class DataTests(unittest.TestCase):
    def test_paired_random_crop_uses_identical_coordinates(self) -> None:
        source = np.arange(100, dtype=np.float32).reshape(10, 10)
        target = source + 1000
        random.seed(17)
        source_crop, target_crop = _paired_crop(
            source, target, 6, random_crop=True
        )
        np.testing.assert_array_equal(target_crop - source_crop, np.full((6, 6), 1000))

    def test_paired_left_right_flip_keeps_source_and_target_aligned(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        target = source + 1000
        source_flip, target_flip = _paired_crop(
            source,
            target,
            None,
            random_crop=False,
            left_right_flip_probability=1.0,
        )
        np.testing.assert_array_equal(source_flip, source[::-1, :])
        np.testing.assert_array_equal(target_flip - source_flip, np.full((3, 4), 1000))

    def test_balanced_subset_can_match_submission_slab(self) -> None:
        dataset = object.__new__(ProspectivePairDataset)
        dataset.records = []
        for modality in MODALITIES:
            for source_field, target_field in directed_pairs():
                for slice_index in (149, 150, 179, 180):
                    source = SliceRecord(
                        Path("source"), modality, source_field, "P_0001", slice_index
                    )
                    target = SliceRecord(
                        Path("target"), modality, target_field, "P_0001", slice_index
                    )
                    dataset.records.append(PairRecord(source, target))
        subset = dataset.balanced_subset(
            30, seed=2026, slice_start=150, slice_end=180
        )
        self.assertEqual(len(subset), len(MODALITIES) * len(directed_pairs()) * 2)
        self.assertTrue(
            all(150 <= dataset.records[index].source.slice_index < 180 for index in subset.indices)
        )

    def test_complete_subject_requires_all_fifteen_domains(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for modality in MODALITIES:
                for field in FIELDS:
                    directory = root / "pro_train" / modality / field
                    directory.mkdir(parents=True)
                    path = directory / f"pro_train_{modality}_{field}_P_0001_s100.npz"
                    np.savez_compressed(path, image=np.zeros((4, 4), dtype=np.float32))
            index = build_cache_index(root)
            self.assertEqual(complete_subjects(index), ["P_0001"])
            self.assertEqual(len(index), 15)

    def test_misaligned_slice_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for modality in MODALITIES:
                for field in FIELDS:
                    directory = root / "pro_train" / modality / field
                    directory.mkdir(parents=True)
                    slice_index = 101 if (modality, field) == ("T2W", "3T") else 100
                    path = directory / (
                        f"pro_train_{modality}_{field}_P_0001_s{slice_index:03d}.npz"
                    )
                    np.savez_compressed(path, image=np.zeros((4, 4), dtype=np.float32))
            index = build_cache_index(root)
            with self.assertRaisesRegex(Exception, "not exactly aligned"):
                complete_subjects(index)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TrainingControlTests(unittest.TestCase):
    def test_runtime_budget_is_disabled_by_zero_and_trips_at_limit(self) -> None:
        from task3_fpsformer.train import runtime_budget_exhausted

        self.assertFalse(runtime_budget_exhausted(10_000, 0))
        self.assertFalse(runtime_budget_exhausted(599, 10))
        self.assertTrue(runtime_budget_exhausted(600, 10))

    def test_ema_update_averages_parameters_and_copies_buffers(self) -> None:
        import torch

        from task3_fpsformer.train import update_ema

        model = torch.nn.BatchNorm1d(2)
        ema_model = torch.nn.BatchNorm1d(2)
        with torch.no_grad():
            model.weight.fill_(3.0)
            model.running_mean.fill_(4.0)
            ema_model.weight.fill_(1.0)
            ema_model.running_mean.zero_()
        update_ema(ema_model, model, 0.5)
        torch.testing.assert_close(ema_model.weight, torch.full_like(model.weight, 2.0))
        torch.testing.assert_close(ema_model.running_mean, model.running_mean)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchModelTests(unittest.TestCase):
    def test_predict_volume_accepts_negative_numpy_strides(self) -> None:
        import torch

        from task3_fpsformer.infer import predict_volume

        class IdentityModel(torch.nn.Module):
            def forward(self, source, modality, source_field, target_field):
                return source

        volume = np.linspace(0.0, 1.0, 60, dtype=np.float32).reshape(4, 5, 3)[::-1]
        self.assertTrue(any(stride < 0 for stride in volume.strides))
        prediction = predict_volume(
            IdentityModel(),
            volume,
            modality=0,
            source_field=0,
            target_field=1,
            batch_size=2,
            device=torch.device("cpu"),
            amp="off",
        )
        np.testing.assert_array_equal(prediction, volume)

    def test_tiny_model_preserves_shape_and_initializes_near_identity(self) -> None:
        import torch

        from task3_fpsformer.model import ModelConfig, build_model

        model = build_model(ModelConfig.preset("tiny")).eval()
        source = torch.rand(1, 1, 17, 23)
        with torch.inference_mode():
            prediction = model(
                source,
                torch.tensor([2]),
                torch.tensor([0]),
                torch.tensor([4]),
            )
        self.assertEqual(prediction.shape, source.shape)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertGreaterEqual(float(prediction.min()), 0.0)
        self.assertLessEqual(float(prediction.max()), 1.0)
        self.assertTrue(torch.allclose(prediction, source, atol=2e-4, rtol=0.0))

    def test_spatial_attention_supports_mixed_precision_autocast(self) -> None:
        import torch

        from task3_fpsformer.model import SpatialPurificationAttention

        module = SpatialPurificationAttention(
            16, 1, groups=4, max_group_tokens=64
        ).eval()
        source = torch.rand(1, 16, 8, 8)
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            prediction = module(source)
        self.assertEqual(prediction.shape, source.shape)
        self.assertEqual(prediction.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(prediction).all())

    def test_losses_are_finite(self) -> None:
        import torch

        from task3_fpsformer.losses import LossWeights, TranslationLoss

        objective = TranslationLoss(LossWeights(l1=1.0, ssim=0.1, edge=0.1))
        prediction = torch.rand(2, 1, 16, 16)
        target = torch.rand(2, 1, 16, 16)
        total, terms = objective(prediction, target)
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(set(terms), {"l1", "ssim", "edge", "lpips"})

    def test_identity_mapping_is_rejected(self) -> None:
        import torch

        from task3_fpsformer.common import FPSFormerError
        from task3_fpsformer.model import ModelConfig, build_model

        model = build_model(ModelConfig.preset("tiny")).eval()
        with self.assertRaises(FPSFormerError):
            model(
                torch.rand(1, 1, 16, 16),
                torch.tensor([0]),
                torch.tensor([2]),
                torch.tensor([2]),
            )


if __name__ == "__main__":
    unittest.main()
