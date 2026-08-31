#!/usr/bin/env python3
"""Paired prospective data preparation for Task 3 FPS-Former.

The paper trains with supervised L1 reconstruction.  For Task 3, that maps to
the paired prospective split: source and target are the same subject, modality,
and axial slice, with different field strengths.  Retrospective data is not
silently treated as paired data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from task3_fpsformer.common import (
    DEFAULT_SLICE_RANGE,
    EXPECTED_DTYPE,
    EXPECTED_VOLUME_SHAPE,
    FIELDS,
    MODALITIES,
    FPSFormerError,
    atomic_json,
    directed_pairs,
    parse_cache_name,
    parse_nifti_name,
    resolve_split,
    utc_now,
)

try:
    import torch
    from torch.utils.data import Dataset, Subset
except ImportError:  # Data preparation and audits do not require PyTorch.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

    class Subset:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any):
            raise FPSFormerError("PyTorch is required to construct a training subset")


@dataclass(frozen=True)
class SliceRecord:
    path: Path
    modality: str
    field: str
    subject_id: str
    slice_index: int


@dataclass(frozen=True)
class PairRecord:
    source: SliceRecord
    target: SliceRecord

    @property
    def pair_key(self) -> tuple[str, str, str]:
        return self.source.modality, self.source.field, self.target.field


def _raw_inventory(split_dir: Path) -> dict[tuple[str, str], list[Path]]:
    inventory: dict[tuple[str, str], list[Path]] = {}
    failures: list[str] = []
    for modality in MODALITIES:
        for field in FIELDS:
            directory = split_dir / modality / field
            files = sorted(directory.glob("*.nii.gz")) if directory.is_dir() else []
            if not files:
                failures.append(f"No NIfTI files in {directory}")
            for path in files:
                parsed = parse_nifti_name(path)
                if parsed["prefix"] != "P":
                    failures.append(f"Prospective cache received non-P filename: {path.name}")
                if parsed["modality"] != modality or parsed["field"] != field:
                    failures.append(f"Path/filename metadata mismatch: {path}")
            inventory[(modality, field)] = files
    if failures:
        raise FPSFormerError("Raw prospective inventory failed:\n- " + "\n- ".join(failures))

    subject_sets = {
        key: {str(parse_nifti_name(path)["subject_id"]) for path in paths}
        for key, paths in inventory.items()
    }
    reference_key = (MODALITIES[0], FIELDS[0])
    reference = subject_sets[reference_key]
    mismatches = {
        f"{modality}/{field}": sorted(subjects)
        for (modality, field), subjects in subject_sets.items()
        if subjects != reference
    }
    if mismatches:
        raise FPSFormerError(
            "Prospective subjects are not complete across all 15 domains; "
            f"reference={sorted(reference)}, mismatches={mismatches}"
        )
    return inventory


def audit_raw(data_dir: Path, *, check_values: bool = False) -> dict[str, Any]:
    """Audit only the expected prospective Task 3 directories."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise FPSFormerError("nibabel is required for the raw-data audit") from exc

    split_dir = resolve_split(data_dir, "pro_train")
    inventory = _raw_inventory(split_dir)
    groups: list[dict[str, Any]] = []
    failures: list[str] = []
    for (modality, field), paths in inventory.items():
        shapes: set[tuple[int, ...]] = set()
        dtypes: set[str] = set()
        minimum = math.inf
        maximum = -math.inf
        finite = True
        for path in paths:
            image = nib.load(str(path))
            shapes.add(tuple(int(value) for value in image.shape))
            dtypes.add(str(image.get_data_dtype()))
            if check_values:
                array = image.get_fdata(dtype=np.float32)
                finite = finite and bool(np.isfinite(array).all())
                minimum = min(minimum, float(array.min()))
                maximum = max(maximum, float(array.max()))
        record: dict[str, Any] = {
            "modality": modality,
            "field": field,
            "files": len(paths),
            "shapes": [list(shape) for shape in sorted(shapes)],
            "dtypes": sorted(dtypes),
        }
        if check_values:
            record.update(finite=finite, minimum=minimum, maximum=maximum)
        groups.append(record)
        if shapes != {EXPECTED_VOLUME_SHAPE}:
            failures.append(f"{modality}/{field}: unexpected shapes {sorted(shapes)}")
        if dtypes != {EXPECTED_DTYPE}:
            failures.append(f"{modality}/{field}: unexpected dtypes {sorted(dtypes)}")
        if check_values and (not finite or minimum < -1e-6 or maximum > 1.0 + 1e-6):
            failures.append(
                f"{modality}/{field}: invalid values finite={finite}, "
                f"range=[{minimum}, {maximum}]"
            )
    report = {
        "kind": "task3_fpsformer_raw_audit",
        "created_at_utc": utc_now(),
        "split_dir": str(split_dir),
        "groups": groups,
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }
    if failures:
        raise FPSFormerError("Raw data audit failed:\n- " + "\n- ".join(failures))
    return report


def _save_npz_atomic(destination: Path, **arrays: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as stream:
        np.savez_compressed(stream, **arrays)
        temporary = Path(stream.name)
    os.replace(temporary, destination)


def prepare_cache(
    data_dir: Path,
    cache_dir: Path,
    *,
    slice_start: int = DEFAULT_SLICE_RANGE[0],
    slice_end: int = DEFAULT_SLICE_RANGE[1],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract aligned prospective axial slices without changing source data."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise FPSFormerError("nibabel is required to prepare the slice cache") from exc
    if not 0 <= slice_start < slice_end <= EXPECTED_VOLUME_SHAPE[2]:
        raise FPSFormerError(
            f"Invalid axial range [{slice_start}, {slice_end}) for {EXPECTED_VOLUME_SHAPE}"
        )

    split_dir = resolve_split(data_dir, "pro_train")
    inventory = _raw_inventory(split_dir)
    cache_dir = cache_dir.expanduser().resolve()
    written = 0
    skipped = 0
    for (modality, field), paths in inventory.items():
        destination_dir = cache_dir / "pro_train" / modality / field
        for path in paths:
            parsed = parse_nifti_name(path)
            image = nib.load(str(path))
            if tuple(image.shape) != EXPECTED_VOLUME_SHAPE:
                raise FPSFormerError(f"Unexpected input shape {image.shape}: {path}")
            volume = image.get_fdata(dtype=np.float32)
            if not np.isfinite(volume).all() or volume.min() < -1e-6 or volume.max() > 1 + 1e-6:
                raise FPSFormerError(f"Input must be finite and within [0,1]: {path}")
            subject_id = str(parsed["subject_id"])
            for slice_index in range(slice_start, slice_end):
                destination = destination_dir / (
                    f"pro_train_{modality}_{field}_{subject_id}_s{slice_index:03d}.npz"
                )
                if destination.exists() and not overwrite:
                    with np.load(destination) as cached_npz:
                        cached = cached_npz["image"]
                    if cached.shape != EXPECTED_VOLUME_SHAPE[:2] or cached.dtype != np.float32:
                        raise FPSFormerError(f"Existing cache entry is invalid: {destination}")
                    skipped += 1
                    continue
                _save_npz_atomic(
                    destination,
                    image=volume[:, :, slice_index].astype(np.float32, copy=False),
                    subject_id=subject_id,
                    split="pro_train",
                    modality=modality,
                    field_strength=field,
                    slice_idx=np.int32(slice_index),
                )
                written += 1

    index = build_cache_index(cache_dir)
    report = {
        "kind": "task3_fpsformer_cache_prepare",
        "created_at_utc": utc_now(),
        "source_split": str(split_dir),
        "cache_dir": str(cache_dir),
        "slice_range": [slice_start, slice_end],
        "written": written,
        "skipped": skipped,
        "cached_slices": len(index),
        "subjects": complete_subjects(index),
    }
    atomic_json(cache_dir / "task3_fpsformer_cache_manifest.json", report)
    return report


def build_cache_index(cache_dir: Path) -> dict[tuple[str, str, str, int], SliceRecord]:
    root = cache_dir.expanduser().resolve() / "pro_train"
    index: dict[tuple[str, str, str, int], SliceRecord] = {}
    failures: list[str] = []
    for modality in MODALITIES:
        for field in FIELDS:
            directory = root / modality / field
            files = sorted(directory.glob("*.npz")) if directory.is_dir() else []
            if not files:
                failures.append(f"No cached slices in {directory}")
                continue
            for path in files:
                try:
                    parsed = parse_cache_name(path)
                except FPSFormerError as exc:
                    failures.append(str(exc))
                    continue
                if parsed["modality"] != modality or parsed["field"] != field:
                    failures.append(f"Cache path/filename mismatch: {path}")
                    continue
                key = (modality, field, str(parsed["subject_id"]), int(parsed["slice"]))
                if key in index:
                    failures.append(f"Duplicate cache key {key}: {index[key].path} and {path}")
                    continue
                index[key] = SliceRecord(path, modality, field, key[2], key[3])
    if failures:
        raise FPSFormerError("Cache index failed:\n- " + "\n- ".join(failures))
    return index


def complete_subjects(index: dict[tuple[str, str, str, int], SliceRecord]) -> list[str]:
    subject_slices: dict[str, dict[tuple[str, str], set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for modality, field, subject, slice_index in index:
        subject_slices[subject][(modality, field)].add(slice_index)
    expected = {(modality, field) for modality in MODALITIES for field in FIELDS}
    failures: list[str] = []
    complete: list[str] = []
    for subject, by_domain in sorted(subject_slices.items()):
        if set(by_domain) != expected:
            missing = sorted(expected - set(by_domain))
            failures.append(f"{subject}: missing domains {missing}")
            continue
        reference_key = (MODALITIES[0], FIELDS[0])
        reference = by_domain[reference_key]
        mismatches = {
            f"{modality}/{field}": sorted(slices)
            for (modality, field), slices in by_domain.items()
            if slices != reference
        }
        if mismatches:
            failures.append(
                f"{subject}: axial indices differ from {reference_key[0]}/{reference_key[1]} "
                f"{sorted(reference)}; mismatches={mismatches}"
            )
            continue
        if not reference:
            failures.append(f"{subject}: contains no axial slices")
            continue
        complete.append(subject)
    if failures:
        raise FPSFormerError(
            "Prospective cache is not exactly aligned across all 15 domains:\n- "
            + "\n- ".join(failures)
        )
    return complete


def audit_cache(cache_dir: Path, *, check_values: bool = True) -> dict[str, Any]:
    index = build_cache_index(cache_dir)
    subjects = complete_subjects(index)
    if not subjects:
        raise FPSFormerError("No subject is complete across all 15 Task 3 domains")
    counts = Counter((record.modality, record.field) for record in index.values())
    failures: list[str] = []
    groups: list[dict[str, Any]] = []
    for modality in MODALITIES:
        for field in FIELDS:
            candidates = sorted(
                record.path
                for record in index.values()
                if record.modality == modality and record.field == field
            )
            minimum = math.inf
            maximum = -math.inf
            finite = True
            shapes: set[tuple[int, ...]] = set()
            dtypes: set[str] = set()
            for path in candidates:
                with np.load(path) as npz:
                    array = npz["image"]
                shapes.add(tuple(int(value) for value in array.shape))
                dtypes.add(str(array.dtype))
                if check_values:
                    finite = finite and bool(np.isfinite(array).all())
                    minimum = min(minimum, float(array.min()))
                    maximum = max(maximum, float(array.max()))
            record = {
                "modality": modality,
                "field": field,
                "count": counts[(modality, field)],
                "shapes": [list(shape) for shape in sorted(shapes)],
                "dtypes": sorted(dtypes),
            }
            if check_values:
                record.update(
                    finite=finite,
                    minimum=minimum,
                    maximum=maximum,
                )
            groups.append(record)
            if shapes != {EXPECTED_VOLUME_SHAPE[:2]} or dtypes != {"float32"}:
                failures.append(
                    f"Invalid cache geometry in {modality}/{field}: "
                    f"shapes={sorted(shapes)}, dtypes={sorted(dtypes)}"
                )
            if check_values and (
                not finite or minimum < -1e-6 or maximum > 1 + 1e-6
            ):
                failures.append(
                    f"Invalid cached values in {modality}/{field}: "
                    f"finite={finite}, range=[{minimum}, {maximum}]"
                )
    report = {
        "kind": "task3_fpsformer_cache_audit",
        "created_at_utc": utc_now(),
        "cache_dir": str(cache_dir.expanduser().resolve()),
        "total_cached_slices": len(index),
        "complete_subjects": subjects,
        "groups": groups,
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }
    if failures:
        raise FPSFormerError("Cache audit failed:\n- " + "\n- ".join(failures))
    return report


def _paired_crop(
    source: np.ndarray,
    target: np.ndarray,
    crop_size: int | None,
    *,
    random_crop: bool,
    left_right_flip_probability: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if source.shape != target.shape:
        raise FPSFormerError(f"Paired slices have different shapes: {source.shape}/{target.shape}")
    if not 0.0 <= left_right_flip_probability <= 1.0:
        raise FPSFormerError("Left-right flip probability must be in [0,1]")
    height, width = source.shape
    if crop_size is None:
        source_result, target_result = source, target
    else:
        if crop_size > min(height, width):
            raise FPSFormerError(
                f"Crop {crop_size} exceeds paired slice shape {source.shape}; use 0 for full slices"
            )
        if random_crop:
            top = random.randint(0, height - crop_size)
            left = random.randint(0, width - crop_size)
        else:
            top = (height - crop_size) // 2
            left = (width - crop_size) // 2
        selection = np.s_[top : top + crop_size, left : left + crop_size]
        source_result, target_result = source[selection], target[selection]
    if left_right_flip_probability and random.random() < left_right_flip_probability:
        source_result = np.flip(source_result, axis=0)
        target_result = np.flip(target_result, axis=0)
    return source_result, target_result


class ProspectivePairDataset(Dataset):
    """All 60 directed non-identity modality/field pair strata."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        include_subjects: set[str] | None = None,
        exclude_subjects: set[str] | None = None,
        crop_size: int | None = 256,
        random_crop: bool = True,
        left_right_flip_probability: float = 0.0,
    ):
        if torch is None:
            raise FPSFormerError("PyTorch is required for ProspectivePairDataset")
        if include_subjects and exclude_subjects:
            raise FPSFormerError("Use include_subjects or exclude_subjects, not both")
        index = build_cache_index(cache_dir)
        subjects = complete_subjects(index)
        if include_subjects is not None:
            subjects = [subject for subject in subjects if subject in include_subjects]
        if exclude_subjects is not None:
            subjects = [subject for subject in subjects if subject not in exclude_subjects]
        if not subjects:
            raise FPSFormerError("No complete prospective subjects remain for this dataset")

        by_domain_subject: dict[tuple[str, str, str], dict[int, SliceRecord]] = defaultdict(dict)
        for (modality, field, subject, slice_index), record in index.items():
            if subject in subjects:
                by_domain_subject[(modality, field, subject)][slice_index] = record

        self.records: list[PairRecord] = []
        pair_counts: Counter[tuple[str, str, str]] = Counter()
        for modality in MODALITIES:
            for source_field, target_field in directed_pairs():
                for subject in subjects:
                    source_slices = by_domain_subject[(modality, source_field, subject)]
                    target_slices = by_domain_subject[(modality, target_field, subject)]
                    common_slices = sorted(set(source_slices) & set(target_slices))
                    if not common_slices:
                        raise FPSFormerError(
                            f"No paired slices for {subject} {modality} "
                            f"{source_field}->{target_field}"
                        )
                    for slice_index in common_slices:
                        self.records.append(
                            PairRecord(source_slices[slice_index], target_slices[slice_index])
                        )
                        pair_counts[(modality, source_field, target_field)] += 1
        expected_pairs = len(MODALITIES) * len(directed_pairs())
        if len(pair_counts) != expected_pairs:
            raise FPSFormerError(
                f"Dataset contains {len(pair_counts)} pair strata, expected {expected_pairs}"
            )
        self.subjects = tuple(subjects)
        self.pair_counts = dict(pair_counts)
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.left_right_flip_probability = left_right_flip_probability

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with np.load(record.source.path) as source_npz:
            source = source_npz["image"].astype(np.float32, copy=False)
        with np.load(record.target.path) as target_npz:
            target = target_npz["image"].astype(np.float32, copy=False)
        source, target = _paired_crop(
            source,
            target,
            self.crop_size,
            random_crop=self.random_crop,
            left_right_flip_probability=self.left_right_flip_probability,
        )
        source = np.ascontiguousarray(source)
        target = np.ascontiguousarray(target)
        return {
            "source": torch.from_numpy(source).unsqueeze(0),
            "target": torch.from_numpy(target).unsqueeze(0),
            "modality": MODALITIES.index(record.source.modality),
            "source_field": FIELDS.index(record.source.field),
            "target_field": FIELDS.index(record.target.field),
            "modality_name": record.source.modality,
            "source_field_name": record.source.field,
            "target_field_name": record.target.field,
            "subject_id": record.source.subject_id,
            "slice_index": record.source.slice_index,
        }

    def balanced_subset(
        self,
        samples_per_pair: int,
        *,
        seed: int,
        slice_start: int | None = None,
        slice_end: int | None = None,
    ) -> Subset:
        if samples_per_pair < 1:
            raise FPSFormerError("samples_per_pair must be positive")
        if (slice_start is None) != (slice_end is None):
            raise FPSFormerError("Slice start and end must be provided together")
        if slice_start is not None and slice_start >= slice_end:
            raise FPSFormerError("Slice range must satisfy start < end")
        groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            if slice_start is not None and not (
                slice_start <= record.source.slice_index < slice_end
            ):
                continue
            groups[record.pair_key].append(index)
        if len(groups) != len(MODALITIES) * len(directed_pairs()):
            raise FPSFormerError("Requested validation slice range misses pair strata")
        rng = random.Random(seed)
        selected: list[int] = []
        for key in sorted(groups):
            candidates = groups[key]
            if len(candidates) <= samples_per_pair:
                selected.extend(candidates)
            else:
                selected.extend(rng.sample(candidates, samples_per_pair))
        return Subset(self, sorted(selected))


def _write_report(report: dict[str, Any], destination: Path | None) -> None:
    if destination is not None:
        atomic_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and audit paired prospective Task 3 data for FPS-Former"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw = subparsers.add_parser("audit-raw")
    raw.add_argument("--data-dir", type=Path, required=True)
    raw.add_argument("--check-values", action="store_true")
    raw.add_argument("--report", type=Path)

    prepare = subparsers.add_parser("prepare-cache")
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--cache-dir", type=Path, required=True)
    prepare.add_argument("--slice-start", type=int, default=DEFAULT_SLICE_RANGE[0])
    prepare.add_argument("--slice-end", type=int, default=DEFAULT_SLICE_RANGE[1])
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--report", type=Path)

    cached = subparsers.add_parser("audit-cache")
    cached.add_argument("--cache-dir", type=Path, required=True)
    cached.add_argument("--no-value-check", action="store_true")
    cached.add_argument("--report", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "audit-raw":
            report = audit_raw(args.data_dir, check_values=args.check_values)
        elif args.command == "prepare-cache":
            report = prepare_cache(
                args.data_dir,
                args.cache_dir,
                slice_start=args.slice_start,
                slice_end=args.slice_end,
                overwrite=args.overwrite,
            )
        else:
            report = audit_cache(args.cache_dir, check_values=not args.no_value_check)
        _write_report(report, args.report)
    except FPSFormerError as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
