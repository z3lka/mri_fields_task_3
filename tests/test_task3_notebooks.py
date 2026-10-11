import ast
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_NOTEBOOK = ROOT / "notebooks" / "task3_pretrained_baseline_colab.ipynb"
TRAINING_NOTEBOOK = ROOT / "notebooks" / "task3_stargan_v2_colab.ipynb"


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_source(cell: dict) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def source_text(notebook: dict) -> str:
    return "\n".join(cell_source(cell) for cell in notebook["cells"])


def training_functions(namespace: dict) -> None:
    definitions = []
    for cell in load_notebook(TRAINING_NOTEBOOK)["cells"]:
        if cell["cell_type"] == "code":
            definitions.extend(
                node for node in ast.parse(cell_source(cell)).body
                if isinstance(node, ast.FunctionDef)
            )
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(TRAINING_NOTEBOOK), "exec"), namespace)


class Task3NotebookSeparationTests(unittest.TestCase):
    def test_cell_source_accepts_strings_and_line_lists(self):
        for source in ("x = 1\n", ["x = ", "1\n"]):
            text = cell_source({"source": source})
            self.assertEqual(text, "x = 1\n")
            ast.parse(text)

    def test_every_code_cell_is_valid_python(self):
        for path in (PRETRAINED_NOTEBOOK, TRAINING_NOTEBOOK):
            notebook = load_notebook(path)
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    ast.parse(cell_source(cell), filename=f"{path.name}:cell-{index}")

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
                cell_source(cell)
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            ]
            export_cells = [cell for cell in code_cells if export_flag in cell]
            self.assertEqual(len(export_cells), 1)
            for cell in code_cells:
                if export_flag not in cell:
                    self.assertNotIn("export_dir.mkdir", cell)
                    self.assertNotIn("DRIVE_EXPORT_ROOT / EXPORT_NAME_VALUE", cell)

    def test_drive_is_unmounted_for_every_processing_cell(self):
        for path, export_flag in (
            (PRETRAINED_NOTEBOOK, "EXPORT_PRETRAINED_RESULT"),
            (TRAINING_NOTEBOOK, "EXPORT_TRAINING_RESULT"),
        ):
            notebook = load_notebook(path)
            code_cells = [
                cell_source(cell)
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            ]
            staging_indexes = [
                index
                for index, cell in enumerate(code_cells)
                if "Runtime staging complete. Drive is unmounted" in cell
            ]
            export_indexes = [
                index for index, cell in enumerate(code_cells) if export_flag in cell
            ]
            self.assertEqual(len(staging_indexes), 1)
            self.assertEqual(export_indexes, [len(code_cells) - 1])
            self.assertEqual(sum(cell.count("drive.mount(") for cell in code_cells), 2)
            self.assertEqual(
                sum(cell.count("drive.flush_and_unmount()") for cell in code_cells),
                2,
            )
            self.assertIn("drive.mount(str(DRIVE_MOUNT))", code_cells[0])
            self.assertIn("drive.flush_and_unmount()", code_cells[staging_indexes[0]])
            self.assertIn("drive.mount(str(DRIVE_MOUNT))", code_cells[-1])
            self.assertIn("drive.flush_and_unmount()", code_cells[-1])
            for cell in code_cells[staging_indexes[0] + 1 : export_indexes[0]]:
                self.assertNotIn("DRIVE_", cell)
                self.assertNotIn("drive.", cell)

    def test_expensive_work_has_runtime_only_guard(self):
        expected_cells = {
            PRETRAINED_NOTEBOOK: (
                "if RUN_PRETRAINED_SMOKE:",
                "if RUN_PRETRAINED_FULL_SWEEP:",
                "if BUILD_PRETRAINED_TASK3_ZIP:",
            ),
            TRAINING_NOTEBOOK: (
                "if RUN_PROSPECTIVE_PREPROCESS:",
                "if RUN_PILOT:",
                "if RUN_FULL_FINETUNE:",
                "if RUN_RETROSPECTIVE_PREPROCESS:",
                "if RUN_RETROSPECTIVE_PRETRAIN:",
            ),
        }
        for path, needles in expected_cells.items():
            notebook = load_notebook(path)
            code_cells = [
                cell_source(cell)
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            ]
            for needle in needles:
                matching = [cell for cell in code_cells if needle in cell]
                self.assertEqual(len(matching), 1, needle)
                self.assertIn("assert_runtime_only()", matching[0])

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

    def test_training_runtime_guard_and_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runtime = root / "runtime"
            runtime.mkdir()
            drive = root / "drive"
            source = drive / "source"
            source.mkdir(parents=True)
            namespace = {
                "Path": Path, "os": os, "shutil": shutil, "hashlib": hashlib,
                "RUNTIME_ROOT": runtime, "drive_root": drive,
                "DRIVE_MOUNT": drive, "RUNTIME_WORK_PATHS": (runtime,),
            }
            training_functions(namespace)
            require_path = namespace["require_runtime_path"]
            self.assertEqual(require_path(runtime / "data"), runtime / "data")
            for outside in (source, root / "outside"):
                with self.assertRaises(ValueError):
                    require_path(outside)
            (runtime / "linked").symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError):
                require_path(runtime / "linked" / "input.nii.gz")
            with patch.object(os.path, "ismount", return_value=True):
                with self.assertRaises(RuntimeError):
                    namespace["assert_runtime_only"]()
            with patch.object(os.path, "ismount", return_value=False):
                namespace["assert_runtime_only"]()

            volume = source / "input.nii.gz"
            volume.write_bytes(b"first")
            destination = runtime / "data"
            stage = namespace["stage_nifti_tree"]
            stage(source, destination)
            self.assertEqual((destination / volume.name).read_bytes(), b"first")
            volume.write_bytes(b"other")
            stage(source, destination)
            self.assertEqual((destination / volume.name).read_bytes(), b"other")
            (destination / "unexpected.nii.gz").write_bytes(b"extra")
            with self.assertRaises(RuntimeError):
                stage(source, destination)

    def test_training_export_verifies_copy_and_always_unmounts(self):
        export_cell = next(
            cell_source(cell) for cell in load_notebook(TRAINING_NOTEBOOK)["cells"]
            if cell["cell_type"] == "code" and "EXPORT_TRAINING_RESULT" in cell_source(cell)
        )
        for fail_copy in (False, True):
            with self.subTest(fail_copy=fail_copy), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                runtime = root / "runtime"
                output = runtime / "results/training_runs"
                run = output / "task3_any_to_any_multimodal/stargan_v2/pilot"
                checkpoint = run / "weights/checkpoint_epoch1.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"synthetic checkpoint")
                (run / "provenance.json").write_text("{}")
                (run / "train.jsonl").write_text("{}\n")
                drive_root = root / "drive"
                drive = Mock()
                namespace = {
                    "Path": Path, "os": os, "shutil": shutil,
                    "hashlib": hashlib, "json": json,
                    "RUNTIME_ROOT": runtime, "drive_root": drive_root,
                    "DRIVE_MOUNT": drive_root, "RUNTIME_WORK_PATHS": (runtime,),
                    "OUTPUT_DIR": output, "REPORT_DIR": runtime / "results/reports",
                    "DRIVE_EXPORT_ROOT": drive_root / "exports", "drive": drive,
                }
                training_functions(namespace)
                namespace["drive_is_mounted"] = lambda: False
                tree = ast.parse(export_cell)
                for node in tree.body:
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                        values = {
                            "EXPORT_TRAINING_RESULT": True,
                            "SELECTED_CHECKPOINT_VALUE": str(checkpoint),
                            "EXPORT_NAME_VALUE": "pilot_export",
                        }
                        if node.targets[0].id in values:
                            node.value = ast.Constant(values[node.targets[0].id])
                code = compile(ast.fix_missing_locations(tree), "training-export", "exec")
                if fail_copy:
                    with patch.object(shutil, "copy2", side_effect=OSError("copy interrupted")):
                        with self.assertRaises(OSError):
                            exec(code, namespace)
                else:
                    exec(code, namespace)
                    export = drive_root / "exports/pilot_export"
                    self.assertEqual((export / checkpoint.name).read_bytes(), checkpoint.read_bytes())
                    self.assertTrue((export / "provenance.json").is_file())
                    self.assertTrue((export / "train.jsonl").is_file())
                drive.mount.assert_called_once_with(str(drive_root))
                drive.flush_and_unmount.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
