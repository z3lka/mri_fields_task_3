import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from task3_3d_unet.common import (
    ALL_PROSPECTIVE_SUBJECTS,
    FIELDS,
    HOLDOUT_SUBJECT,
    MODALITIES,
    TRAIN_SUBJECTS,
    Task3Error,
)
from task3_3d_unet.data import (
    ProspectivePatchDataset,
    RetrospectivePatchStream,
    VolumeRecord,
    paired_patch,
    select_retrospective,
    validate_subject_split,
)


def make_cache(root: Path, shape=(12, 12, 12)) -> Path:
    records = []
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / np.prod(shape)
    for modality in MODALITIES:
        for field_index, field in enumerate(FIELDS):
            for subject_index, subject in enumerate(ALL_PROSPECTIVE_SUBJECTS):
                path = root / f"pro_train_{modality}_{field}_{subject}.npy"
                np.save(path, np.clip(base + field_index * 0.01 + subject_index * 0.001, 0, 1))
                records.append(
                    {
                        "path": str(path),
                        "source": "test",
                        "modality": modality,
                        "field": field,
                        "subject_id": subject,
                        "shape": list(shape),
                        "dtype": "float32",
                        "affine": np.eye(4).tolist(),
                    }
                )
    (root / "manifest.json").write_text(json.dumps({"records": records}))
    return root


class PatchTests(unittest.TestCase):
    def test_paired_crop_and_flip_are_aligned(self):
        source = np.arange(16**3, dtype=np.float32).reshape(16, 16, 16)
        target = source + 1000
        source_patch, target_patch, starts = paired_patch(
            source, target, 8, np.random.default_rng(17), flip=True, foreground_bias=0
        )
        np.testing.assert_array_equal(target_patch - source_patch, np.full((8, 8, 8), 1000))
        self.assertEqual(len(starts), 3)
        self.assertTrue(source_patch.flags.c_contiguous)

    def test_holdout_dataset_has_all_sixty_strata_without_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = make_cache(Path(temporary))
            dataset = ProspectivePatchDataset(
                cache, (HOLDOUT_SUBJECT,), patch_size=8, samples_per_stratum=1,
                random_augmentation=False,
            )
            samples = [dataset[index] for index in range(len(dataset))]
            self.assertEqual(len(samples), 60)
            self.assertEqual({sample["subject_id"] for sample in samples}, {HOLDOUT_SUBJECT})
            self.assertEqual(len({sample["stratum"] for sample in samples}), 60)

    def test_paired_dataset_resume_offset_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = make_cache(Path(temporary))
            full = ProspectivePatchDataset(
                cache, TRAIN_SUBJECTS, patch_size=8, samples_per_stratum=2, seed=2026,
            )
            resumed = ProspectivePatchDataset(
                cache, TRAIN_SUBJECTS, patch_size=8, samples_per_stratum=2,
                seed=2026, start_index=7,
            )
            expected, actual = full[7], resumed[0]
            np.testing.assert_array_equal(expected["source"].numpy(), actual["source"].numpy())
            np.testing.assert_array_equal(expected["target"].numpy(), actual["target"].numpy())
            self.assertEqual(expected["stratum"], actual["stratum"])
            self.assertEqual(expected["subject_id"], actual["subject_id"])

    def test_fixed_subject_split_fails_instead_of_substituting(self):
        validate_subject_split(TRAIN_SUBJECTS, HOLDOUT_SUBJECT)
        validate_subject_split(ALL_PROSPECTIVE_SUBJECTS, None)
        with self.assertRaises(Task3Error):
            validate_subject_split(("P_0006", "P_0009"), "P_0007")


class RetrospectiveTests(unittest.TestCase):
    def test_selection_is_seeded_and_domain_balanced(self):
        inventory = {}
        for modality in MODALITIES:
            for field in FIELDS:
                inventory[(modality, field)] = [
                    VolumeRecord(
                        Path(f"R_{modality}_{field}_{index:04d}.nii.gz"),
                        "retro_train", modality, field, f"R_{index:04d}",
                    )
                    for index in range(50)
                ]
        first = select_retrospective(inventory, per_domain=40, seed=2026)
        second = select_retrospective(inventory, per_domain=40, seed=2026)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 600)
        self.assertEqual(set(Counter((r.modality, r.field) for r in first).values()), {40})

    def test_stream_rejects_non_retrospective_and_balances_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                {
                    "path": str(root / f"{modality}_{field}_{index}.nii.gz"),
                    "split": "retro_train",
                    "modality": modality,
                    "field": field,
                    "subject_id": f"R_{index:04d}",
                }
                for modality in MODALITIES
                for field in FIELDS
                for index in range(2)
            ]
            manifest = root / "retro.json"
            manifest.write_text(json.dumps({"records": records}))
            stream = RetrospectivePatchStream(manifest, patch_size=8)
            counts = Counter((r["modality"], r["field"]) for r in stream.episode_order(30))
            self.assertEqual(set(counts.values()), {2})
            records[0]["split"] = "pro_train"
            manifest.write_text(json.dumps({"records": records}))
            with self.assertRaisesRegex(Task3Error, "only retrospective"):
                RetrospectivePatchStream(manifest)

    def test_retrospective_stream_resume_offset_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume_path = root / "volume.nii.gz"
            values = np.random.default_rng(4).random((12, 12, 12), dtype=np.float32)
            nib.save(nib.Nifti1Image(values, np.eye(4)), volume_path)
            records = [
                {
                    "path": str(volume_path),
                    "split": "retro_train",
                    "modality": modality,
                    "field": field,
                    "subject_id": "R_0001",
                }
                for modality in MODALITIES
                for field in FIELDS
            ]
            manifest = root / "retro.json"
            manifest.write_text(json.dumps({"records": records}))
            with patch("task3_3d_unet.data.EXPECTED_VOLUME_SHAPE", (12, 12, 12)):
                full = iter(RetrospectivePatchStream(
                    manifest, patch_size=8, patches_per_volume=8, seed=2026,
                ))
                expected = None
                for _ in range(6):
                    expected = next(full)
                actual = next(iter(RetrospectivePatchStream(
                    manifest, patch_size=8, patches_per_volume=8, seed=2026, start_patch=5,
                )))
            np.testing.assert_array_equal(expected["source"].numpy(), actual["source"].numpy())
            self.assertEqual(expected["starts"], actual["starts"])


if __name__ == "__main__":
    unittest.main()
