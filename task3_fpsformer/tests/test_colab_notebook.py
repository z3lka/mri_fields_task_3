import ast
import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "task3_fpsformer_colab.ipynb"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def code_cells(notebook: dict) -> list[str]:
    return [
        cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


class FPSFormerColabNotebookTests(unittest.TestCase):
    def test_notebook_metadata_and_every_code_cell(self):
        notebook = load_notebook()
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["kernelspec"]["language"], "python")
        for index, source in enumerate(code_cells(notebook)):
            ast.parse(source, filename=f"{NOTEBOOK.name}:code-cell-{index}")

    def test_notebook_orchestrates_python_modules_instead_of_reimplementing(self):
        text = "\n".join(cell_source(cell) for cell in load_notebook()["cells"])
        for command in (
            "task3_fpsformer.data', 'audit-raw",
            "task3_fpsformer.data', 'prepare-cache",
            "task3_fpsformer.data', 'audit-cache",
            "task3_fpsformer.train",
            "task3_fpsformer.infer",
            "build_submission.py",
        ):
            self.assertIn(command, text)
        for implementation in (
            "class ConditionalFPSFormer",
            "class FMAM",
            "class SPAM",
            "class SDFN",
            "class HEFR",
        ):
            self.assertNotIn(implementation, text)

    def test_colab_runs_functional_tests_without_self_testing_staged_notebook(self):
        cells = code_cells(load_notebook())
        test_cell = next(source for source in cells if "if RUN_PACKAGE_TESTS:" in source)
        self.assertIn("task3_fpsformer.tests.test_fpsformer", test_cell)
        self.assertNotIn("task3_fpsformer.tests.test_colab_notebook", test_cell)
        self.assertIn("stdout=subprocess.PIPE", test_cell)
        self.assertIn("print(test_result.stdout)", test_cell)

    def test_source_bundle_is_refreshed_when_staging_cell_is_rerun(self):
        cells = code_cells(load_notebook())
        staging = next(source for source in cells if "runtime_zip =" in source)
        self.assertIn("archive.extractall(PROJECT_DIR)", staging)
        self.assertNotIn("if not (PROJECT_DIR / 'task3_fpsformer').is_dir()", staging)

    def test_dataset_root_guidance_and_split_aliases_are_present(self):
        text = "\n".join(cell_source(cell) for cell in load_notebook()["cells"])
        self.assertIn("dataset root containing `training_prospective`", text)
        self.assertIn("not a split directory", text)
        self.assertIn("'Training_prospective', 'training_prospective'", text)
        self.assertIn("'Validating_prospective', 'validation_prospective'", text)

    def test_drive_is_unmounted_during_processing(self):
        cells = code_cells(load_notebook())
        staging = [
            index
            for index, source in enumerate(cells)
            if "Runtime staging complete. Drive is unmounted" in source
        ]
        exporting = [
            index
            for index, source in enumerate(cells)
            if "EXPORT_SELECTED_ARTIFACTS" in source
        ]
        self.assertEqual(staging, [2])
        self.assertEqual(exporting, [len(cells) - 1])
        self.assertEqual(sum(source.count("drive.mount(") for source in cells), 2)
        self.assertEqual(
            sum(source.count("drive.flush_and_unmount()") for source in cells), 2
        )
        self.assertIn("drive.mount(str(DRIVE_MOUNT))", cells[0])
        self.assertIn("drive.flush_and_unmount()", cells[staging[0]])
        self.assertIn("drive.mount(str(DRIVE_MOUNT))", cells[-1])
        self.assertIn("drive.flush_and_unmount()", cells[-1])

    def test_processing_cells_have_runtime_only_guard(self):
        cells = code_cells(load_notebook())
        for switch in (
            "if RUN_PACKAGE_TESTS:",
            "if RUN_RAW_AUDIT:",
            "if RUN_CACHE_PREPARATION:",
            "if RUN_CACHE_AUDIT:",
            "if RUN_TINY_SMOKE:",
            "if RUN_PORTABLE_PILOT:",
            "if RUN_ALL_SUBJECT_CONTINUATION:",
            "if RUN_VALIDATION_SMOKE:",
            "if RUN_FULL_VALIDATION:",
            "if PREPARE_OFFICIAL_REPO:",
            "if BUILD_TASK3_ZIP:",
        ):
            matches = [source for source in cells if switch in source]
            self.assertEqual(len(matches), 1, switch)
            self.assertIn("assert_runtime_only()", matches[0], switch)

    def test_tiny_smoke_has_colab_safe_preflight(self):
        cells = code_cells(load_notebook())
        matches = [source for source in cells if "if RUN_TINY_SMOKE:" in source]
        self.assertEqual(len(matches), 1)
        smoke = matches[0]
        self.assertIn("cache_audit.get('status') != 'passed'", smoke)
        self.assertIn("len(complete_subjects) < 2", smoke)
        self.assertIn("if not torch.cuda.is_available()", smoke)
        self.assertIn("existing_smoke_checkpoints", smoke)
        self.assertIn("'--workers', '0'", smoke)
        self.assertIn("'--progress'", smoke)

    def test_ten_epoch_training_is_colab_safe_and_has_progress(self):
        cells = code_cells(load_notebook())
        pilot = next(source for source in cells if "if RUN_PORTABLE_PILOT:" in source)
        continuation = next(
            source for source in cells if "if RUN_ALL_SUBJECT_CONTINUATION:" in source
        )
        self.assertIn("PILOT_EPOCHS = 10", pilot)
        self.assertIn("PILOT_UPDATES_PER_EPOCH = 100", pilot)
        self.assertIn("'--gradient-accumulation', '1'", pilot)
        self.assertIn("'--crop-size', '128'", pilot)
        self.assertIn("'--eval-crop-size', '128'", pilot)
        self.assertIn("'--val-samples-per-pair', '1'", pilot)
        self.assertIn("'--workers', '0'", pilot)
        self.assertIn("'--progress'", pilot)
        self.assertIn("ALL_SUBJECT_EPOCHS = 10", continuation)
        self.assertIn("ALL_SUBJECT_UPDATES_PER_EPOCH = 100", continuation)
        self.assertIn("'--gradient-accumulation', '1'", continuation)
        self.assertIn("'--crop-size', '128'", continuation)
        self.assertIn("'--workers', '0'", continuation)
        self.assertIn("'--progress'", continuation)

    def test_defaults_run_only_the_requested_ten_epoch_training_workflow(self):
        text = "\n".join(code_cells(load_notebook()))
        for assignment in (
            "STAGE_TRAINING_DATA = True",
            "RUN_PACKAGE_TESTS = True",
            "RUN_RAW_AUDIT = True",
            "RUN_CACHE_PREPARATION = True",
            "RUN_CACHE_AUDIT = True",
            "BASELINE_MILESTONE_COMPLETE = True",
            "RUN_TINY_SMOKE = True",
            "RUN_PORTABLE_PILOT = True",
            "EXPORT_SELECTED_ARTIFACTS = True",
            "EXPORT_CHECKPOINT = True",
        ):
            self.assertIn(assignment, text)
        for assignment in (
            "STAGE_VALIDATION_DATA = False",
            "STAGE_INITIAL_CHECKPOINT = False",
            "RESUME_PORTABLE_PILOT = False",
            "RUN_ALL_SUBJECT_CONTINUATION = False",
            "RUN_VALIDATION_SMOKE = False",
            "RUN_FULL_VALIDATION = False",
            "PREPARE_OFFICIAL_REPO = False",
            "LIVE_RULES_REVIEWED = False",
            "BUILD_TASK3_ZIP = False",
            "EXPORT_ARCHIVE = False",
        ):
            self.assertIn(assignment, text)

    def test_packaging_is_task3_only_and_has_no_submission_upload(self):
        text = "\n".join(cell_source(cell) for cell in load_notebook()["cells"])
        self.assertIn("'--tasks', 'task3'", text)
        self.assertIn("expected_files =", text)
        self.assertIn("expected_pred_dirs =", text)
        self.assertIn("actual_files != expected_files", text)
        self.assertIn("actual_pred_dirs != expected_pred_dirs", text)
        self.assertIn("Task 3 submission contains segmentation content", text)
        self.assertNotIn("requests.post(", text)
        self.assertNotIn("curl -F", text)
        self.assertNotIn("task1", text.lower())
        self.assertNotIn("task2", text.lower())


if __name__ == "__main__":
    unittest.main()
