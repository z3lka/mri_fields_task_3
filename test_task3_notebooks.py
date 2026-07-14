import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PRETRAINED_NOTEBOOK = ROOT / "task3_pretrained_baseline_colab.ipynb"
TRAINING_NOTEBOOK = ROOT / "task3_stargan_v2_colab.ipynb"


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_text(notebook: dict) -> str:
    return "\n".join(str(cell["source"]) for cell in notebook["cells"])


class Task3NotebookSeparationTests(unittest.TestCase):
    def test_every_code_cell_is_valid_python(self):
        for path in (PRETRAINED_NOTEBOOK, TRAINING_NOTEBOOK):
            notebook = load_notebook(path)
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    ast.parse(cell["source"], filename=f"{path.name}:cell-{index}")

    def test_pretrained_notebook_owns_only_baseline_results(self):
        text = source_text(load_notebook(PRETRAINED_NOTEBOOK))
        self.assertIn("task3_pipeline.py", text)
        self.assertIn("task3_build_submission.py", text)
        self.assertIn("RUN_PRETRAINED_FULL_SWEEP", text)
        self.assertIn("BUILD_PRETRAINED_TASK3_ZIP", text)
        self.assertNotIn("task3_colab_train.py", text)
        self.assertNotIn("RUN_PILOT", text)
        self.assertNotIn("RUN_FULL_FINETUNE", text)

    def test_training_notebook_has_no_baseline_inference_or_builder(self):
        text = source_text(load_notebook(TRAINING_NOTEBOOK))
        self.assertIn("task3_colab_train.py", text)
        self.assertIn("RUN_PILOT", text)
        self.assertIn("RUN_FULL_FINETUNE", text)
        self.assertNotIn("task3_pipeline.py", text)
        self.assertNotIn("task3_build_submission.py", text)
        self.assertNotIn("RUN_PRETRAINED_FULL_SWEEP", text)
        self.assertNotIn("BUILD_PRETRAINED_TASK3_ZIP", text)

    def test_working_paths_are_separate_runtime_trees(self):
        pretrained = source_text(load_notebook(PRETRAINED_NOTEBOOK))
        training = source_text(load_notebook(TRAINING_NOTEBOOK))
        self.assertIn(
            "RUNTIME_ROOT_VALUE = '/content/mrixfields_task3/pretrained'",
            pretrained,
        )
        self.assertIn(
            "RUNTIME_ROOT_VALUE = '/content/mrixfields_task3/training'",
            training,
        )
        self.assertIn("VALIDATION_DIR = RUNTIME_ROOT / 'data'", pretrained)
        self.assertIn("DATA_DIR = RUNTIME_ROOT / 'data'", training)
        self.assertIn("require_runtime_path", pretrained)
        self.assertIn("require_runtime_path", training)

    def test_drive_writes_are_confined_to_final_export_cell(self):
        for path, export_flag in (
            (PRETRAINED_NOTEBOOK, "EXPORT_PRETRAINED_RESULT"),
            (TRAINING_NOTEBOOK, "EXPORT_TRAINING_RESULT"),
        ):
            notebook = load_notebook(path)
            code_cells = [
                str(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            ]
            export_cells = [cell for cell in code_cells if export_flag in cell]
            self.assertEqual(len(export_cells), 1)
            for cell in code_cells:
                if export_flag not in cell:
                    self.assertNotIn("export_dir.mkdir", cell)
                    self.assertNotIn("DRIVE_EXPORT_ROOT / EXPORT_NAME_VALUE", cell)

    def test_all_expensive_switches_default_to_off(self):
        pretrained = source_text(load_notebook(PRETRAINED_NOTEBOOK))
        training = source_text(load_notebook(TRAINING_NOTEBOOK))
        for assignment in (
            "RUN_PRETRAINED_SMOKE = False",
            "RUN_PRETRAINED_FULL_SWEEP = False",
            "BUILD_PRETRAINED_TASK3_ZIP = False",
            "EXPORT_PRETRAINED_RESULT = False",
            "LIVE_RULES_REVIEWED = False",
            "RUN_PILOT = False",
            "RUN_FULL_FINETUNE = False",
            "RUN_RETROSPECTIVE_PRETRAIN = False",
            "EXPORT_TRAINING_RESULT = False",
            "BASELINE_MILESTONE_COMPLETE = False",
        ):
            self.assertIn(assignment, pretrained + training)


if __name__ == "__main__":
    unittest.main()
