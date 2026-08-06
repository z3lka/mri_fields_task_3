#!/usr/bin/env python3
"""Shared Task 3 constants and path helpers for the FPS-Former package.

This module intentionally has no PyTorch dependency.  It is safe to import from
data audits, packaging helpers, and unit tests on machines without a training
environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PAPER_ID = "arXiv:2412.10776v1"
MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
NUM_DOMAINS = len(MODALITIES) * len(FIELDS)
EXPECTED_VOLUME_SHAPE = (364, 436, 364)
EXPECTED_DTYPE = "float32"
EXPECTED_VOXEL_SIZE = (0.5, 0.5, 0.5)
DEFAULT_SLICE_RANGE = (72, 292)

SPLIT_ALIASES = {
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
    r"(?P<field>0\.1T|1\.5T|3T|5T|7T)_(?P<subject>\d+)\.nii\.gz$"
)
CACHE_RE = re.compile(
    r"^(?P<split>pro_train|retro_train)_(?P<modality>T1W|T2W|T2FLAIR)_"
    r"(?P<field>0\.1T|1\.5T|3T|5T|7T)_(?P<prefix>[PR])_"
    r"(?P<subject>\d+)_s(?P<slice>\d+)\.npz$"
)


class FPSFormerError(RuntimeError):
    """Raised when a Task 3 FPS-Former invariant is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def joint_domain(modality: str, field: str) -> int:
    """Return the official 15-way domain index."""
    try:
        return MODALITIES.index(modality) * len(FIELDS) + FIELDS.index(field)
    except ValueError as exc:
        raise FPSFormerError(
            f"Unknown Task 3 modality/field: {modality!r}/{field!r}"
        ) from exc


def split_domain(domain: int) -> tuple[str, str]:
    if not 0 <= domain < NUM_DOMAINS:
        raise FPSFormerError(f"Domain must be in [0, {NUM_DOMAINS}), found {domain}")
    modality_index, field_index = divmod(domain, len(FIELDS))
    return MODALITIES[modality_index], FIELDS[field_index]


def directed_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(
        (source, target)
        for source in FIELDS
        for target in FIELDS
        if source != target
    )


def parse_nifti_name(path: str | Path) -> dict[str, str]:
    name = Path(path).name
    match = NIFTI_RE.fullmatch(name)
    if match is None:
        raise FPSFormerError(f"Unexpected Task 3 NIfTI filename: {name}")
    result = match.groupdict()
    result["subject_id"] = f"{result['prefix']}_{result['subject']}"
    return result


def parse_cache_name(path: str | Path) -> dict[str, str | int]:
    name = Path(path).name
    match = CACHE_RE.fullmatch(name)
    if match is None:
        raise FPSFormerError(f"Unexpected Task 3 cache filename: {name}")
    result: dict[str, str | int] = match.groupdict()
    result["slice"] = int(str(result["slice"]))
    result["subject_id"] = f"{result['prefix']}_{result['subject']}"
    return result


def resolve_split(root: Path, split: str) -> Path:
    if split not in SPLIT_ALIASES:
        raise FPSFormerError(f"Unknown split {split!r}; choose from {tuple(SPLIT_ALIASES)}")
    root = root.expanduser().resolve()
    matches = [root / name for name in SPLIT_ALIASES[split] if (root / name).is_dir()]
    unique: list[Path] = []
    for match in matches:
        resolved = match.resolve()
        if resolved not in unique:
            unique.append(resolved)
    if len(unique) != 1:
        raise FPSFormerError(
            f"Expected exactly one {split} directory below {root}; found {unique or 'none'}"
        )
    return unique[0]


def prediction_path(
    output_root: Path,
    *,
    modality: str,
    source_field: str,
    target_field: str,
    source_name: str,
    method: str = "fpsformer",
    mode: str = "pro_scratch",
    epoch_tag: str = "best",
) -> Path:
    if source_field == target_field:
        raise FPSFormerError("Task 3 inference must not generate identity mappings")
    parsed = parse_nifti_name(source_name)
    if parsed["modality"] != modality or parsed["field"] != source_field:
        raise FPSFormerError(
            f"Filename metadata disagrees with requested case: {source_name}, "
            f"{modality}/{source_field}"
        )
    pair = f"{source_field}_to_{target_field}"
    return (
        output_root
        / f"task3_{pair}_{modality}"
        / method
        / mode
        / epoch_tag
        / source_name
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def ensure_exact_members(actual: Iterable[str], expected: Iterable[str], label: str) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise FPSFormerError(f"{label} mismatch: missing={missing}, extra={extra}")
