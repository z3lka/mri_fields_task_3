import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "task3_3d_unet_colab.ipynb"


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def source(cell):
    return "".join(cell.get("source", []))


class NotebookTests(unittest.TestCase):
    def test_syntax_metadata_and_empty_outputs(self):
        document = notebook()
        self.assertEqual(document["nbformat"], 4)
        self.assertEqual(document["metadata"]["kernelspec"]["language"], "python")
        for index, cell in enumerate(document["cells"]):
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse(source(cell), filename=f"cell-{index}")

    def test_package_commands_cover_complete_workflow(self):
        text = "\n".join(source(cell) for cell in notebook()["cells"])
        for command in (
            "task3_3d_unet.data', 'stage'",
            "task3_3d_unet.data', 'audit'",
            "task3_3d_unet.data', 'prepare-cache'",
            "task3_3d_unet.probe",
            "task3_3d_unet.train', 'pretrain'",
            "task3_3d_unet.train', 'finetune'",
            "task3_3d_unet.compare",
            "task3_3d_unet.infer",
            "task3_3d_unet.submission",
        ):
            self.assertIn(command, text)
        self.assertNotIn("class ConditionalUNet3D", text)

    def test_fresh_training_workflow_is_enabled_but_evidence_gates_stay_closed(self):
        text = "\n".join(source(cell) for cell in notebook()["cells"])
        enabled = (
            "BASELINE_MILESTONE_COMPLETE",
            "STAGE_RUNTIME",
            "RUN_PACKAGE_TESTS",
            "RUN_DATA_AUDIT",
            "PREPARE_PROSPECTIVE_CACHE",
            "RUN_TINY_SMOKE_UPDATES",
            "RUN_A100_PROBE",
            "RUN_RETROSPECTIVE_PRETRAINING",
            "RUN_THREE_ABLATIONS",
            "SELECT_WINNER",
            "RUN_FINAL_RETRAINING",
            "RUN_VALIDATION_SMOKE",
            "RUN_FULL_VALIDATION",
            "REFRESH_OFFICIAL_SOURCES",
            "EXPORT_SELECTED_ARTIFACTS",
        )
        for switch in enabled:
            self.assertIn(f"{switch} = True", text)
        closed = (
            "ACKNOWLEDGE_OVER_BUDGET",
            "STAGE_RUN_BACKUPS",
            "LIVE_RULES_REVIEWED",
            "BUILD_TASK3_ZIP",
            "EXPORT_ARCHIVE",
        )
        for switch in closed:
            self.assertIn(f"{switch} = False", text)

    def test_drive_is_unmounted_for_processing_and_chunk_backups_are_stopped(self):
        text = "\n".join(source(cell) for cell in notebook()["cells"])
        self.assertIn("assert_runtime_only()", text)
        self.assertIn("--max-runtime-minutes', '45'", text)
        self.assertIn("checkpoints/last.pt", text)
        self.assertNotIn("checkpoints/best.pt', 'provenance", text)
        self.assertEqual(text.count("drive.mount(str(DRIVE_MOUNT))"), 3)
        self.assertEqual(text.count("drive.flush_and_unmount()"), 3)

    def test_packaging_is_task3_only_and_has_no_submission_api(self):
        package = (ROOT / "task3_3d_unet" / "submission.py").read_text()
        text = NOTEBOOK.read_text()
        self.assertIn('"--tasks",\n            "task3"', package)
        self.assertIn("LIVE_RULES_REVIEWED = False", text)
        self.assertNotIn("requests.post(", text + package)
        self.assertNotIn("curl -F", text + package)
        self.assertNotIn("synapseclient", text + package)


if __name__ == "__main__":
    unittest.main()
