import tempfile
import unittest
from pathlib import Path

import task3_pipeline as pipeline


class Task3PipelineTests(unittest.TestCase):
    def test_joint_domains_cover_all_fifteen_domains(self):
        found = {
            pipeline.joint_domain(modality, field)
            for modality in pipeline.MODALITIES
            for field in pipeline.FIELDS
        }
        self.assertEqual(found, set(range(15)))
        self.assertEqual(pipeline.joint_domain("T1W", "7T"), 4)
        self.assertEqual(pipeline.joint_domain("T2W", "0.1T"), 5)
        self.assertEqual(pipeline.joint_domain("T2FLAIR", "7T"), 14)

    def test_parse_expected_filename(self):
        item = pipeline.parse_input_filename(Path("P_T1W_1.5T_0005.nii.gz"))
        self.assertEqual(item.modality, "T1W")
        self.assertEqual(item.source_field, "1.5T")
        self.assertEqual(item.subject_id, "0005")

    def test_rejects_nonstandard_filename(self):
        with self.assertRaises(pipeline.PipelineError):
            pipeline.parse_input_filename(Path("P_T1W_1.5T_5.nii.gz"))

    def test_t1w_release_produces_sixty_non_identity_cases(self):
        volumes = []
        for field in pipeline.FIELDS:
            for subject_id in pipeline.EXPECTED_IDS[field]:
                path = Path("T1W") / field / f"P_T1W_{field}_{subject_id}.nii.gz"
                volumes.append(
                    pipeline.InputVolume(path, "T1W", field, subject_id)
                )
        cases = pipeline.make_cases(volumes, Path("runs/inference"))
        self.assertEqual(len(cases), 60)
        self.assertTrue(
            all(c.input_volume.source_field != c.target_field for c in cases)
        )
        case = next(
            c
            for c in cases
            if c.input_volume.source_field == "1.5T"
            and c.target_field == "7T"
            and c.input_volume.subject_id == "0005"
        )
        self.assertEqual(
            case.output_path,
            Path(
                "runs/inference/task3_1.5T_to_7T_T1W/"
                "stargan_v2/pro_pretrained/epoch50/P_T1W_1.5T_0005.nii.gz"
            ),
        )

    def test_complete_release_produces_one_hundred_eighty_cases(self):
        volumes = []
        for modality in pipeline.MODALITIES:
            for field in pipeline.FIELDS:
                for subject_id in pipeline.EXPECTED_IDS[field]:
                    path = Path(modality) / field / f"P_{modality}_{field}_{subject_id}.nii.gz"
                    volumes.append(
                        pipeline.InputVolume(path, modality, field, subject_id)
                    )
        cases = pipeline.make_cases(volumes, Path("runs/inference"))
        self.assertEqual(len(cases), 180)
        self.assertEqual({case.target_domain for case in cases}, set(range(15)))

    def test_two_shards_are_balanced_disjoint_and_complete(self):
        cases = list(range(180))
        first = pipeline.select_case_shard(cases, 2, 0)
        second = pipeline.select_case_shard(cases, 2, 1)
        self.assertEqual(len(first), 90)
        self.assertEqual(len(second), 90)
        self.assertFalse(set(first) & set(second))
        self.assertEqual(set(first) | set(second), set(cases))
        self.assertEqual(
            pipeline.shard_manifest_path(Path("runs"), 2, 1),
            Path("runs/task3_run_manifest.shard-01-of-02.json"),
        )

    def test_rejects_invalid_shard_coordinates(self):
        with self.assertRaises(pipeline.PipelineError):
            pipeline.select_case_shard([], 0, 0)
        with self.assertRaises(pipeline.PipelineError):
            pipeline.select_case_shard([], 2, 2)

    def test_custom_sweep_coordinates_keep_candidate_outputs_separate(self):
        volume = pipeline.InputVolume(
            Path("T2W/3T/P_T2W_3T_0010.nii.gz"), "T2W", "3T", "0010"
        )
        path = pipeline.output_path_for(
            Path("runs/inference"),
            volume,
            "7T",
            sweep_mode="continued_all_subjects",
            sweep_epoch="epoch1",
        )
        self.assertEqual(
            path,
            Path(
                "runs/inference/task3_3T_to_7T_T2W/stargan_v2/"
                "continued_all_subjects/epoch1/P_T2W_3T_0010.nii.gz"
            ),
        )

    def test_sweep_coordinate_rejects_path_traversal(self):
        with self.assertRaises(Exception):
            pipeline.sweep_component("../epoch1")

    def test_center_padding_matches_official_transform(self):
        source, target = pipeline._center_geometry((364, 436), (512, 512))
        self.assertEqual(source, (slice(None), slice(None)))
        self.assertEqual(target, (slice(74, 438), slice(38, 474)))

    def test_discovery_accepts_single_modality_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "T1W"
            for field in pipeline.FIELDS:
                directory = root / field
                directory.mkdir(parents=True)
                for subject_id in pipeline.EXPECTED_IDS[field]:
                    (directory / f"P_T1W_{field}_{subject_id}.nii.gz").touch()
            volumes = pipeline.discover_inputs(root, ["T1W"])
            self.assertEqual(len(volumes), 15)


if __name__ == "__main__":
    unittest.main()
