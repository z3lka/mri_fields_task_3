#!/usr/bin/env python3
"""Build and validate the MRIxFields2026 Task 3 validation submission.

This wrapper deliberately delegates tree construction, target-field renaming,
and axial cropping to the organizers' build_submission.py. It then validates
the complete Task 3 contract and creates task3.zip with task3/ at its root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
EXPECTED_IDS = {
    "0.1T": ("0001", "0002", "0003"),
    "1.5T": ("0004", "0005", "0008"),
    "3T": ("0010", "0011", "0012"),
    "5T": ("0013", "0014", "0015"),
    "7T": ("0016", "0017", "0018"),
}
EXPECTED_OFFICIAL_COMMIT = "f0f4ab2c54ab878b48ddfea8d13f4da512ca411e"
EXPECTED_CHECKPOINT_SHA256 = (
    "c2f8172c1fbdbf0426e22c7d3d2014632f175a38cf818f95e72b1917badabdeb"
)
EXPECTED_FULL_SHAPE = (364, 436, 364)
Z_CLIP_RANGE = (150, 180)
EXPECTED_SLAB_SHAPE = (364, 436, 30)


class SubmissionError(RuntimeError):
    """Raised when a submission prerequisite or validation check fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def git_revision(repository: Path) -> str:
    if not (repository / ".git").is_dir():
        raise SubmissionError(f"Not an organizer Git checkout: {repository}")
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def expected_layout() -> tuple[set[Path], set[Path]]:
    directories: set[Path] = {Path("task3")}
    files: set[Path] = set()
    for modality in MODALITIES:
        directories.add(Path("task3") / modality)
        for source_field in FIELDS:
            for target_field in FIELDS:
                if source_field == target_field:
                    continue
                pair_root = (
                    Path("task3")
                    / modality
                    / f"{source_field}_to_{target_field}"
                )
                pred_root = pair_root / "pred"
                directories.update((pair_root, pred_root))
                for subject_id in EXPECTED_IDS[source_field]:
                    files.add(
                        pred_root
                        / f"P_{modality}_{target_field}_{subject_id}.nii.gz"
                    )
    return directories, files


def validate_manifest(
    predictions_dir: Path,
    sweep_mode: str,
    sweep_epoch: str,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    manifest_path = predictions_dir / "task3_run_manifest.json"
    if not manifest_path.is_file():
        raise SubmissionError(f"Missing inference manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "status": manifest.get("status") == "complete",
        "not_smoke": manifest.get("smoke") is False,
        "planned_cases": manifest.get("planned_cases") == 180,
        "completed_cases": len(manifest.get("completed_cases", [])) == 180,
        "modalities": manifest.get("modalities") == list(MODALITIES),
        "sweep_mode": manifest.get("sweep_mode") == sweep_mode,
        "sweep_epoch": manifest.get("sweep_epoch") == sweep_epoch,
        "checkpoint_sha256": (
            manifest.get("checkpoint_sha256") == expected_checkpoint_sha256
        ),
        "checkpoint_origin": (
            manifest.get("checkpoint_origin") == "official_huggingface_baseline"
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise SubmissionError(
            "Inference manifest is not a complete organizer-pretrained run; failed: "
            + ", ".join(failures)
        )
    return manifest


def run_official_builder(
    official_repo: Path,
    predictions_dir: Path,
    output_dir: Path,
    sweep_mode: str,
    sweep_epoch: str,
) -> Path:
    builder = (
        official_repo
        / "Submission"
        / "build_submission"
        / "build_submission.py"
    )
    if not builder.is_file():
        raise SubmissionError(f"Official submission builder not found: {builder}")
    task_root = output_dir / "task3"
    if task_root.exists():
        raise SubmissionError(
            f"Submission output already exists: {task_root}. "
            "Use a fresh --output-dir so stale files cannot enter the archive."
        )

    environment = os.environ.copy()
    baseline_path = str(official_repo / "Baseline")
    environment["PYTHONPATH"] = baseline_path + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    subprocess.check_call(
        [
            sys.executable,
            str(builder),
            "--predictions-dir",
            str(predictions_dir),
            "--predictions-seg-dir",
            str(output_dir / "unused_task3_seg"),
            "--output-dir",
            str(output_dir),
            "--tasks",
            "task3",
            "--task3-method",
            "stargan_v2",
            "--task3-mode",
            sweep_mode,
            "--task3-epoch",
            sweep_epoch,
        ],
        env=environment,
    )
    return task_root


def validate_submission_tree(
    task_root: Path, validation_input_root: Path
) -> list[dict[str, Any]]:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise SubmissionError(
            "Submission validation requires nibabel and NumPy"
        ) from exc

    expected_directories, expected_files = expected_layout()
    base = task_root.parent
    actual_directories = {
        path.relative_to(base) for path in task_root.rglob("*") if path.is_dir()
    }
    actual_directories.add(Path("task3"))
    actual_files = {
        path.relative_to(base) for path in task_root.rglob("*") if path.is_file()
    }

    missing_directories = sorted(expected_directories - actual_directories)
    extra_directories = sorted(actual_directories - expected_directories)
    missing_files = sorted(expected_files - actual_files)
    extra_files = sorted(actual_files - expected_files)
    if missing_directories or extra_directories or missing_files or extra_files:
        details = []
        for label, items in (
            ("missing directories", missing_directories),
            ("extra directories", extra_directories),
            ("missing files", missing_files),
            ("extra files", extra_files),
        ):
            if items:
                details.append(f"{label}: {', '.join(map(str, items[:10]))}")
        raise SubmissionError("Submission tree mismatch; " + "; ".join(details))

    reports: list[dict[str, Any]] = []
    z0, z1 = Z_CLIP_RANGE
    for index, relative_path in enumerate(sorted(expected_files), start=1):
        output_path = base / relative_path
        _, modality, pair, _, filename = relative_path.parts
        source_field, target_field = pair.split("_to_")
        subject_id = filename.removesuffix(".nii.gz").rsplit("_", 1)[-1]
        source_path = (
            validation_input_root
            / modality
            / source_field
            / f"P_{modality}_{source_field}_{subject_id}.nii.gz"
        )
        if not source_path.is_file():
            raise SubmissionError(f"Validation source not found: {source_path}")

        output_image = nib.load(str(output_path))
        source_image = nib.load(str(source_path))
        if source_image.shape != EXPECTED_FULL_SHAPE:
            raise SubmissionError(
                f"Unexpected source shape for {source_path}: {source_image.shape}"
            )
        if output_image.shape != EXPECTED_SLAB_SHAPE:
            raise SubmissionError(
                f"Unexpected slab shape for {output_path}: {output_image.shape}"
            )
        if np.dtype(output_image.get_data_dtype()) != np.dtype(np.float32):
            raise SubmissionError(
                f"Prediction is not float32: {output_path}: "
                f"{output_image.get_data_dtype()}"
            )
        expected_affine = source_image.slicer[:, :, z0:z1].affine
        if not np.allclose(output_image.affine, expected_affine, rtol=0.0, atol=1e-5):
            raise SubmissionError(f"Cropped affine mismatch: {output_path}")

        values = np.asanyarray(output_image.dataobj)
        if not np.isfinite(values).all():
            raise SubmissionError(f"NaN or Inf found: {output_path}")
        minimum = float(values.min())
        maximum = float(values.max())
        if minimum < -1e-6 or maximum > 1.0 + 1e-6:
            raise SubmissionError(
                f"Values outside [0,1]: {output_path}: min={minimum}, max={maximum}"
            )
        reports.append(
            {
                "path": str(relative_path),
                "modality": modality,
                "source_field": source_field,
                "target_field": target_field,
                "subject_id": subject_id,
                "shape": list(output_image.shape),
                "dtype": str(output_image.get_data_dtype()),
                "minimum": minimum,
                "maximum": maximum,
            }
        )
        if index % 20 == 0 or index == len(expected_files):
            print(f"Validated {index}/{len(expected_files)} submission predictions")
    return reports


def create_and_validate_archive(
    output_dir: Path, archive_path: Path, expected_files: set[Path]
) -> dict[str, Any]:
    if shutil.which("zip") is None:
        raise SubmissionError("The system 'zip' command is not available")
    if archive_path.exists():
        raise SubmissionError(
            f"Archive already exists: {archive_path}. Rename it before rebuilding."
        )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["zip", "-q", "-r", str(archive_path.resolve()), "task3"],
        cwd=output_dir,
    )

    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise SubmissionError(f"Corrupt zip member: {bad_member}")
        names = archive.namelist()
        if not names or any(
            name != "task3/" and not name.startswith("task3/") for name in names
        ):
            raise SubmissionError("Archive root is not exactly task3/")
        archived_files = {
            Path(name) for name in names if name.endswith(".nii.gz")
        }
        if archived_files != expected_files:
            raise SubmissionError(
                f"Archive contains {len(archived_files)} NIfTI files; expected 180"
            )
        unexpected_regular_files = [
            name
            for name in names
            if not name.endswith("/") and not name.endswith(".nii.gz")
        ]
        if unexpected_regular_files:
            raise SubmissionError(
                "Archive contains unrelated files: "
                + ", ".join(unexpected_regular_files[:10])
            )

    return {
        "path": str(archive_path.resolve()),
        "size_bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "nifti_files": len(expected_files),
        "top_level": "task3/",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the official builder to create and fully validate Task 3 task3.zip"
        )
    )
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--validation-input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sweep-mode", default="pro_pretrained")
    parser.add_argument("--sweep-epoch", default="epoch50")
    parser.add_argument(
        "--expected-official-commit", default=EXPECTED_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    official_repo = args.official_repo.expanduser().resolve()
    predictions_dir = args.predictions_dir.expanduser().resolve()
    validation_input_root = args.validation_input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    archive_path = args.archive.expanduser().resolve()
    report_path = args.report.expanduser().resolve()

    if not predictions_dir.is_dir():
        raise SubmissionError(f"Predictions directory not found: {predictions_dir}")
    if not validation_input_root.is_dir():
        raise SubmissionError(
            f"Validation input root not found: {validation_input_root}"
        )
    revision = git_revision(official_repo)
    if revision != args.expected_official_commit:
        raise SubmissionError(
            f"Official revision mismatch: {revision}; "
            f"expected {args.expected_official_commit}"
        )

    manifest = validate_manifest(
        predictions_dir,
        args.sweep_mode,
        args.sweep_epoch,
        args.expected_checkpoint_sha256,
    )
    task_root = run_official_builder(
        official_repo,
        predictions_dir,
        output_dir,
        args.sweep_mode,
        args.sweep_epoch,
    )
    file_reports = validate_submission_tree(task_root, validation_input_root)
    _, expected_files = expected_layout()
    archive = create_and_validate_archive(output_dir, archive_path, expected_files)
    report = {
        "status": "ready_for_user_upload",
        "task": "task3",
        "official_git_revision": revision,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "huggingface_revision": manifest.get("huggingface_revision"),
        "sweep_mode": args.sweep_mode,
        "sweep_epoch": args.sweep_epoch,
        "inference_manifest": str(
            (predictions_dir / "task3_run_manifest.json").resolve()
        ),
        "submission_tree": str(task_root.resolve()),
        "validated_predictions": len(file_reports),
        "files": file_reports,
        "archive": archive,
        "upload_performed": False,
    }
    atomic_json(report_path, report)
    print(f"READY: {archive['path']}")
    print(f"Size: {archive['size_bytes']} bytes")
    print(f"SHA-256: {archive['sha256']}")
    print(f"Validation report: {report_path}")
    print("Upload was not performed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SubmissionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
