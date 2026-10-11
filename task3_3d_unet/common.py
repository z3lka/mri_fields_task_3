"""Task 3 constants and dependency-light helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
NUM_DOMAINS = 15
EXPECTED_VOLUME_SHAPE = (364, 436, 364)
EXPECTED_DTYPE = "float32"
EXPECTED_VOXEL_SIZE = (0.5, 0.5, 0.5)
TRAIN_SUBJECTS = ("P_0006", "P_0007")
HOLDOUT_SUBJECT = "P_0009"
ALL_PROSPECTIVE_SUBJECTS = TRAIN_SUBJECTS + (HOLDOUT_SUBJECT,)
EXPECTED_VALIDATION_IDS = {
    "0.1T": ("0001", "0002", "0003"),
    "1.5T": ("0004", "0005", "0008"),
    "3T": ("0010", "0011", "0012"),
    "5T": ("0013", "0014", "0015"),
    "7T": ("0016", "0017", "0018"),
}
SPLIT_ALIASES = {
    "retro_train": ("Training_retrospective", "training_retrospective", "retro_train"),
    "pro_train": ("Training_prospective", "training_prospective", "pro_train"),
    "pro_val": (
        "Validating_prospective",
        "validation_prospective",
        "validating_prospective",
        "pro_val",
    ),
}
NIFTI_RE = re.compile(
    r"^(?P<prefix>[PR])_(?P<modality>T1W|T2W|T2FLAIR)_"
    r"(?P<field>0\.1T|1\.5T|3T|5T|7T)_(?P<subject>\d{4})\.nii\.gz$"
)


class Task3Error(RuntimeError):
    """Raised when a Task 3 invariant is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def directed_pairs() -> tuple[tuple[str, str], ...]:
    return tuple((source, target) for source in FIELDS for target in FIELDS if source != target)


def directed_strata() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (modality, source, target)
        for modality in MODALITIES
        for source, target in directed_pairs()
    )


def joint_domain(modality: str, field: str) -> int:
    try:
        return MODALITIES.index(modality) * len(FIELDS) + FIELDS.index(field)
    except ValueError as exc:
        raise Task3Error(f"Unknown Task 3 modality/field: {modality!r}/{field!r}") from exc


def parse_nifti_name(path: str | Path) -> dict[str, str]:
    match = NIFTI_RE.fullmatch(Path(path).name)
    if match is None:
        raise Task3Error(f"Unexpected Task 3 NIfTI filename: {Path(path).name}")
    parsed = match.groupdict()
    parsed["subject_id"] = f"{parsed['prefix']}_{parsed['subject']}"
    return parsed


def resolve_split(root: Path, split: str) -> Path:
    if split not in SPLIT_ALIASES:
        raise Task3Error(f"Unknown split {split!r}")
    root = root.expanduser().resolve()
    matches = []
    for alias in SPLIT_ALIASES[split]:
        candidate = root / alias
        if candidate.is_dir() and candidate.resolve() not in matches:
            matches.append(candidate.resolve())
    if len(matches) != 1:
        raise Task3Error(
            f"Expected exactly one {split} directory below {root}; found {matches or 'none'}"
        )
    return matches[0]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repository: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise Task3Error(f"Not a readable Git checkout: {repository}") from exc


def atomic_json(path: Path, payload: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def ensure_exact(actual: Iterable[Any], expected: Iterable[Any], label: str) -> None:
    actual_set, expected_set = set(actual), set(expected)
    if actual_set != expected_set:
        raise Task3Error(
            f"{label} mismatch: missing={sorted(expected_set - actual_set)}, "
            f"extra={sorted(actual_set - expected_set)}"
        )
