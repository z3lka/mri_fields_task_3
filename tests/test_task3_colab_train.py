import tempfile
import unittest
import os
from pathlib import Path

import task3_colab_train as training


class Task3ColabTrainTests(unittest.TestCase):
    def test_joint_domain_contract_is_fifteen(self):
        self.assertEqual(training.NUM_DOMAINS, 15)
        mapping = {
            (modality, field): modality_index * 5 + field_index
            for modality_index, modality in enumerate(training.MODALITIES)
            for field_index, field in enumerate(training.FIELDS)
        }
        self.assertEqual(mapping[("T1W", "7T")], 4)
        self.assertEqual(mapping[("T2W", "0.1T")], 5)
        self.assertEqual(mapping[("T2FLAIR", "7T")], 14)

    def test_lowercase_split_creates_non_destructive_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            source = data / "training_prospective"
            source.mkdir(parents=True)
            view = root / "view"
            mapping = training.create_data_view(data, view, ["pro_train"])
            link = view / "Training_prospective"
            self.assertTrue(link.is_symlink())
            self.assertTrue(os.path.samefile(link, source))
            self.assertTrue(
                os.path.samefile(mapping["Training_prospective"], source)
            )

    def test_cached_subject_parser(self):
        path = Path("pro_train_T2FLAIR_0.1T_P_0009_s150.npz")
        self.assertEqual(training.subject_from_npz(path), "P_0009")

    def test_training_is_gated_by_default(self):
        parser = training.build_parser()
        args = parser.parse_args(
            [
                "pretrain",
                "--official-repo",
                "/tmp/official",
                "--preprocessed-dir",
                "/tmp/cache",
                "--output-dir",
                "/tmp/output",
                "--run-name",
                "test",
            ]
        )
        with self.assertRaises(training.TrainingError):
            training.require_training_acknowledgement(args)

    def test_checkpoint_pruning_keeps_newest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in (10, 20, 30, 40):
                (root / f"checkpoint_{step}.pth").touch()
            training.prune_checkpoints(
                root, r"checkpoint_([0-9]+)\.pth", keep_last=2
            )
            self.assertEqual(
                sorted(path.name for path in root.glob("*.pth")),
                ["checkpoint_30.pth", "checkpoint_40.pth"],
            )

    def test_mtime_pruning_never_removes_current_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protected = root / "checkpoint_step2.pth"
            old = root / "checkpoint_step10.pth"
            protected.touch()
            old.touch()
            training.prune_checkpoints_by_mtime(root, keep_last=1, protected=protected)
            self.assertTrue(protected.is_file())
            self.assertFalse(old.exists())


if __name__ == "__main__":
    unittest.main()
