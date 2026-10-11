"""Run the official Task 3 builder, validate its output, and create task3.zip."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np

from .common import (
    EXPECTED_VALIDATION_IDS,
    EXPECTED_VOLUME_SHAPE,
    FIELDS,
    MODALITIES,
    Task3Error,
    atomic_json,
    directed_pairs,
    git_revision,
    resolve_split,
    sha256_file,
    utc_now,
)


SLAB_SHAPE = (364, 436, 30)
SLAB_RANGE = (150, 180)


def expected_files() -> set[Path]:
    return {
        Path("task3")
        / modality
        / f"{source}_to_{target}"
        / "pred"
        / f"P_{modality}_{target}_{subject}.nii.gz"
        for modality in MODALITIES
        for source, target in directed_pairs()
        for subject in EXPECTED_VALIDATION_IDS[source]
    }


def expected_directories() -> set[Path]:
    directories = {Path("task3")}
    for modality in MODALITIES:
        directories.add(Path("task3") / modality)
        for source, target in directed_pairs():
            pair = Path("task3") / modality / f"{source}_to_{target}"
            directories.update((pair, pair / "pred"))
    return directories


def verify_live_contract(
    official_repo: Path,
    *,
    expected_commit: str,
    expected_readme_sha256: str,
    reviewed: bool,
) -> dict[str, str]:
    if not reviewed:
        raise Task3Error("Final packaging requires an explicit fresh live-rules review")
    revision = git_revision(official_repo)
    readme = official_repo / "Submission" / "README.md"
    builder = official_repo / "Submission" / "build_submission" / "build_submission.py"
    if revision != expected_commit:
        raise Task3Error(f"Official revision changed: {revision}; reviewed={expected_commit}")
    if not readme.is_file() or sha256_file(readme) != expected_readme_sha256:
        raise Task3Error("Submission README changed since the recorded live review")
    if not builder.is_file():
        raise Task3Error(f"Official builder not found: {builder}")
    text = readme.read_text(encoding="utf-8")
    required_contract = ("task3.zip", "task3/", "(364, 436, 30)", "[150, 180)")
    if any(marker not in text for marker in required_contract):
        raise Task3Error("The reviewed Submission README no longer states the validation ZIP contract")
    return {
        "official_git_revision": revision,
        "submission_readme_sha256": sha256_file(readme),
        "submission_builder_sha256": sha256_file(builder),
    }


def verify_inference_manifest(
    predictions_dir: Path, *, method: str, mode: str, epoch: str
) -> dict[str, Any]:
    path = predictions_dir / "task3_3d_unet_inference_manifest.json"
    if not path.is_file():
        raise Task3Error(f"Inference manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Task3Error("Inference manifest is not a JSON object")
    split_manifest = payload.get("split_manifest")
    training_signature = payload.get("training_run_signature")
    if not isinstance(split_manifest, dict) or not isinstance(training_signature, dict):
        raise Task3Error("Inference manifest lacks training/split provenance")
    checks = {
        "status": payload.get("status") == "complete",
        "not_smoke": payload.get("smoke") is False,
        "planned_cases": payload.get("planned_cases") == 180,
        "completed_cases": len(payload.get("completed_cases", [])) == 180,
        "method": payload.get("method") == method,
        "mode": payload.get("mode") == mode,
        "epoch": payload.get("epoch") == epoch,
        "window": payload.get("window") == 128,
        "overlap": payload.get("overlap") == 0.25,
        "all_subject_retrain": split_manifest.get("phase") == "all_subject_retrain",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise Task3Error(f"Inference manifest is not packaging-ready: {failed}")
    return payload


def run_official_builder(
    official_repo: Path,
    predictions_dir: Path,
    output_dir: Path,
    *,
    method: str,
    mode: str,
    epoch: str,
) -> Path:
    task_root = output_dir / "task3"
    if task_root.exists():
        raise Task3Error(f"Use a fresh submission output directory; found {task_root}")
    builder = official_repo / "Submission" / "build_submission" / "build_submission.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(official_repo / "Baseline") + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    subprocess.check_call(
        [
            sys.executable,
            str(builder),
            "--predictions-dir",
            str(predictions_dir),
            "--predictions-seg-dir",
            str(output_dir / "unused"),
            "--output-dir",
            str(output_dir),
            "--tasks",
            "task3",
            "--task3-method",
            method,
            "--task3-mode",
            mode,
            "--task3-epoch",
            epoch,
        ],
        env=environment,
    )
    return task_root


def validate_tree(
    task_root: Path,
    validation_root: Path,
    predictions_dir: Path,
    *,
    method: str,
    mode: str,
    epoch: str,
) -> list[dict[str, Any]]:
    expected = expected_files()
    base = task_root.parent
    actual = {path.relative_to(base) for path in task_root.rglob("*") if path.is_file()}
    actual_directories = {Path("task3")} | {
        path.relative_to(base) for path in task_root.rglob("*") if path.is_dir()
    }
    if actual != expected:
        raise Task3Error(
            f"Submission files differ: missing={sorted(expected - actual)[:10]}, "
            f"extra={sorted(actual - expected)[:10]}"
        )
    if actual_directories != expected_directories():
        raise Task3Error("Submission directory tree contains missing or unrelated directories")
    directories = [path for path in task_root.rglob("*") if path.is_dir()]
    pred_dirs = [path for path in directories if path.name == "pred"]
    if len(pred_dirs) != 60:
        raise Task3Error(f"Submission has {len(pred_dirs)} pred directories, expected 60")
    if any(path.name == "seg" for path in directories) or any("_seg.nii.gz" in path.name for path in actual):
        raise Task3Error("Task 3 submission contains segmentation content")
    reports: list[dict[str, Any]] = []
    z0, z1 = SLAB_RANGE
    for relative in sorted(expected):
        _, modality, pair, _, filename = relative.parts
        source_field, target_field = pair.split("_to_")
        if source_field == target_field:
            raise Task3Error(f"Identity pair found: {pair}")
        subject = filename.removesuffix(".nii.gz").rsplit("_", 1)[-1]
        source_path = (
            validation_root
            / modality
            / source_field
            / f"P_{modality}_{source_field}_{subject}.nii.gz"
        )
        full_prediction_path = (
            predictions_dir
            / f"task3_{pair}_{modality}"
            / method
            / mode
            / epoch
            / f"P_{modality}_{source_field}_{subject}.nii.gz"
        )
        output_path = base / relative
        if not source_path.is_file() or not full_prediction_path.is_file():
            raise Task3Error(f"Validation source or full prediction not found: {source_path}")
        source = nib.load(str(source_path))
        full_prediction = nib.load(str(full_prediction_path))
        output = nib.load(str(output_path))
        full_values = np.asanyarray(full_prediction.dataobj)
        values = np.asanyarray(output.dataobj)
        if (
            source.shape != EXPECTED_VOLUME_SHAPE
            or full_prediction.shape != EXPECTED_VOLUME_SHAPE
            or output.shape != SLAB_SHAPE
        ):
            raise Task3Error(f"Wrong source/full/slab shape: {source_path}/{output_path}")
        if np.dtype(full_prediction.get_data_dtype()) != np.dtype(np.float32):
            raise Task3Error(f"Full prediction is not float32: {full_prediction_path}")
        if np.dtype(output.get_data_dtype()) != np.dtype(np.float32):
            raise Task3Error(f"Prediction is not float32: {output_path}")
        if not np.allclose(full_prediction.affine, source.affine, rtol=0.0, atol=1e-5):
            raise Task3Error(f"Full prediction affine mismatch: {full_prediction_path}")
        if (
            not np.isfinite(full_values).all()
            or float(full_values.min()) < -1e-6
            or float(full_values.max()) > 1 + 1e-6
        ):
            raise Task3Error(f"Invalid full prediction values: {full_prediction_path}")
        if not np.allclose(output.affine, source.slicer[:, :, z0:z1].affine, rtol=0.0, atol=1e-5):
            raise Task3Error(f"Cropped affine mismatch: {output_path}")
        expected_crop = full_prediction.slicer[:, :, z0:z1]
        if not np.array_equal(values, np.asanyarray(expected_crop.dataobj)):
            raise Task3Error(f"Official builder did not preserve the exact prediction slab: {output_path}")
        for code in ("qform_code", "sform_code"):
            if int(output.header[code]) != int(expected_crop.header[code]):
                raise Task3Error(f"Cropped {code} mismatch: {output_path}")
        if not np.allclose(output.header.get_zooms()[:3], expected_crop.header.get_zooms()[:3]):
            raise Task3Error(f"Cropped voxel sizes mismatch: {output_path}")
        if not np.isfinite(values).all() or float(values.min()) < -1e-6 or float(values.max()) > 1 + 1e-6:
            raise Task3Error(f"Invalid prediction values: {output_path}")
        reports.append(
            {
                "path": str(relative),
                "source_field": source_field,
                "target_field": target_field,
                "shape": list(output.shape),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
    return reports


def create_archive(output_dir: Path, archive_path: Path) -> dict[str, Any]:
    if archive_path.exists():
        raise Task3Error(f"Archive already exists: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", dir=archive_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in sorted(expected_files()):
                archive.write(output_dir / relative, relative.as_posix())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or {Path(name) for name in archive.namelist()} != expected_files():
                raise Task3Error("Created archive failed integrity or exact-member validation")
            if any(not name.startswith("task3/") for name in archive.namelist()):
                raise Task3Error("Archive root is not task3/")
        os.replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(archive_path.resolve()),
        "size_bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "files": 180,
        "top_level": "task3/",
    }


def build_submission(args: argparse.Namespace) -> dict[str, Any]:
    official_repo = args.official_repo.expanduser().resolve()
    predictions_dir = args.predictions_dir.expanduser().resolve()
    validation_root = resolve_split(args.data_dir, "pro_val")
    output_dir = args.output_dir.expanduser().resolve()
    provenance = verify_live_contract(
        official_repo,
        expected_commit=args.expected_official_commit,
        expected_readme_sha256=args.expected_submission_readme_sha256,
        reviewed=args.live_rules_reviewed,
    )
    inference = verify_inference_manifest(
        predictions_dir, method=args.method, mode=args.mode, epoch=args.epoch
    )
    task_root = run_official_builder(
        official_repo,
        predictions_dir,
        output_dir,
        method=args.method,
        mode=args.mode,
        epoch=args.epoch,
    )
    files = validate_tree(
        task_root,
        validation_root,
        predictions_dir,
        method=args.method,
        mode=args.mode,
        epoch=args.epoch,
    )
    runtime_report = json.loads(args.runtime_report.read_text(encoding="utf-8"))
    selection_report = json.loads(args.selection_report.read_text(encoding="utf-8"))
    training_signature = inference["training_run_signature"]
    selection_record = training_signature.get("selection_report")
    runtime_record = training_signature.get("runtime_gate")
    if (
        selection_report.get("winner", {}).get("arm") != training_signature.get("arm")
        or not isinstance(selection_record, dict)
        or selection_record.get("sha256") != sha256_file(args.selection_report)
    ):
        raise Task3Error("Selection report does not match the final training provenance")
    if (
        not isinstance(runtime_record, dict)
        or runtime_record.get("sha256") != sha256_file(args.runtime_report)
    ):
        raise Task3Error("Runtime report does not match the final training provenance")
    runtime_checks = runtime_report.get("checks", {})
    if not runtime_checks.get("a100_detected") or not runtime_checks.get(
        "vram_headroom_at_least_20_percent"
    ):
        raise Task3Error("Recorded runtime report lacks the required A100 VRAM gate")
    if (
        not runtime_checks.get("projected_runtime_at_most_10_hours")
        and not runtime_record.get("over_budget_acknowledged")
    ):
        raise Task3Error("Over-budget runtime projection was not explicitly acknowledged")
    if (
        runtime_report.get("inference_window") != 128
        or runtime_report.get("inference_overlap") != 0.25
        or runtime_report.get("inference_batch_size")
        != inference.get("window_batch_size")
    ):
        raise Task3Error("Inference settings differ from the measured runtime projection")
    archive = create_archive(output_dir, args.archive.expanduser().resolve())
    return {
        "kind": "task3_3d_unet_submission",
        "created_at_utc": utc_now(),
        "status": "ready_for_user_upload",
        "task": "task3",
        "official": provenance,
        "checkpoint_sha256": inference["checkpoint_sha256"],
        "model_config": inference["model_config"],
        "training_split_manifest": inference["split_manifest"],
        "training_run_signature": training_signature,
        "selection_report": {
            "path": str(args.selection_report.resolve()),
            "sha256": sha256_file(args.selection_report),
            "winner": selection_report.get("winner"),
        },
        "runtime_report": {
            "path": str(args.runtime_report.resolve()),
            "sha256": sha256_file(args.runtime_report),
            "report": runtime_report,
        },
        "inference_manifest": str(
            (predictions_dir / "task3_3d_unet_inference_manifest.json").resolve()
        ),
        "validated_predictions": len(files),
        "archive": archive,
        "upload_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--method", default="unet3d")
    parser.add_argument("--mode", default="pro_final")
    parser.add_argument("--epoch", default="best")
    parser.add_argument("--expected-official-commit", required=True)
    parser.add_argument("--expected-submission-readme-sha256", required=True)
    parser.add_argument("--live-rules-reviewed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_submission(args)
        atomic_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        print("Upload was not performed.")
    except (Task3Error, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
