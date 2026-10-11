"""Audit, stage, cache, and sample the Task 3 3D data."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .common import (
    ALL_PROSPECTIVE_SUBJECTS,
    EXPECTED_DTYPE,
    EXPECTED_VALIDATION_IDS,
    EXPECTED_VOLUME_SHAPE,
    EXPECTED_VOXEL_SIZE,
    FIELDS,
    HOLDOUT_SUBJECT,
    MODALITIES,
    TRAIN_SUBJECTS,
    Task3Error,
    atomic_json,
    directed_strata,
    parse_nifti_name,
    resolve_split,
    utc_now,
)

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset
except ImportError:  # Audits and staging do not require PyTorch.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

    class IterableDataset:  # type: ignore[no-redef]
        pass


@dataclass(frozen=True)
class VolumeRecord:
    path: Path
    split: str
    modality: str
    field: str
    subject_id: str


def inventory_split(data_dir: Path, split: str) -> dict[tuple[str, str], list[VolumeRecord]]:
    root = resolve_split(data_dir, split)
    expected_prefix = "R" if split == "retro_train" else "P"
    inventory: dict[tuple[str, str], list[VolumeRecord]] = {}
    failures: list[str] = []
    for modality in MODALITIES:
        for field in FIELDS:
            directory = root / modality / field
            paths = sorted(directory.glob("*.nii.gz")) if directory.is_dir() else []
            if not paths:
                failures.append(f"No NIfTI files in {directory}")
            records: list[VolumeRecord] = []
            for path in paths:
                try:
                    parsed = parse_nifti_name(path)
                except Task3Error as exc:
                    failures.append(str(exc))
                    continue
                if (
                    parsed["prefix"] != expected_prefix
                    or parsed["modality"] != modality
                    or parsed["field"] != field
                ):
                    failures.append(f"Split/path/filename mismatch: {path}")
                    continue
                records.append(
                    VolumeRecord(path, split, modality, field, parsed["subject_id"])
                )
            inventory[(modality, field)] = records
    if failures:
        raise Task3Error("Data inventory failed:\n- " + "\n- ".join(failures))
    return inventory


def _validate_subject_layout(
    pro: dict[tuple[str, str], list[VolumeRecord]],
    val: dict[tuple[str, str], list[VolumeRecord]],
) -> None:
    expected_pro = set(ALL_PROSPECTIVE_SUBJECTS)
    failures: list[str] = []
    for key, records in pro.items():
        found = {record.subject_id for record in records}
        if found != expected_pro or len(records) != len(expected_pro):
            failures.append(f"pro_train {key}: subjects={sorted(found)}, expected={sorted(expected_pro)}")
    for (modality, field), records in val.items():
        found = tuple(record.subject_id.removeprefix("P_") for record in records)
        if found != EXPECTED_VALIDATION_IDS[field]:
            failures.append(
                f"pro_val {(modality, field)}: IDs={found}, expected={EXPECTED_VALIDATION_IDS[field]}"
            )
    if failures:
        raise Task3Error("Fixed Task 3 subject split differs from the audited data:\n- " + "\n- ".join(failures))


def audit_data(data_dir: Path, *, check_values: bool = True) -> dict[str, Any]:
    """Audit only the three expected Task 3 splits and paired spatial metadata."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise Task3Error("nibabel is required for the raw-data audit") from exc

    inventories = {
        split: inventory_split(data_dir, split)
        for split in ("retro_train", "pro_train", "pro_val")
    }
    _validate_subject_layout(inventories["pro_train"], inventories["pro_val"])
    failures: list[str] = []
    groups: list[dict[str, Any]] = []
    prospective_geometry: dict[str, tuple[np.ndarray, tuple[float, ...]]] = {}
    for split, inventory in inventories.items():
        for (modality, field), records in inventory.items():
            if split == "retro_train" and len(records) < 40:
                failures.append(f"{split}/{modality}/{field}: {len(records)} volumes, need at least 40")
            minimum, maximum, finite = math.inf, -math.inf, True
            shapes: set[tuple[int, ...]] = set()
            dtypes: set[str] = set()
            zooms: set[tuple[float, ...]] = set()
            for record in records:
                image = nib.load(str(record.path))
                shape = tuple(int(value) for value in image.shape)
                dtype = str(image.get_data_dtype())
                voxel_size = tuple(float(value) for value in image.header.get_zooms()[:3])
                shapes.add(shape)
                dtypes.add(dtype)
                zooms.add(voxel_size)
                if split == "pro_train":
                    key = record.subject_id
                    geometry = (image.affine, voxel_size)
                    if key in prospective_geometry:
                        reference_affine, reference_zooms = prospective_geometry[key]
                        if not np.allclose(image.affine, reference_affine, rtol=0.0, atol=1e-5):
                            failures.append(f"Paired affine mismatch for {key}: {record.path}")
                        if voxel_size != reference_zooms:
                            failures.append(f"Paired voxel-size mismatch for {key}: {record.path}")
                    else:
                        prospective_geometry[key] = geometry
                if check_values:
                    values = np.asanyarray(image.dataobj, dtype=np.float32)
                    finite = finite and bool(np.isfinite(values).all())
                    minimum = min(minimum, float(values.min()))
                    maximum = max(maximum, float(values.max()))
            if shapes != {EXPECTED_VOLUME_SHAPE}:
                failures.append(f"{split}/{modality}/{field}: shapes={sorted(shapes)}")
            if dtypes != {EXPECTED_DTYPE}:
                failures.append(f"{split}/{modality}/{field}: dtypes={sorted(dtypes)}")
            if any(not np.allclose(zoom, EXPECTED_VOXEL_SIZE, atol=1e-6) for zoom in zooms):
                failures.append(f"{split}/{modality}/{field}: voxel sizes={sorted(zooms)}")
            if check_values and (not finite or minimum < -1e-6 or maximum > 1.0 + 1e-6):
                failures.append(
                    f"{split}/{modality}/{field}: finite={finite}, range=[{minimum},{maximum}]"
                )
            groups.append(
                {
                    "split": split,
                    "modality": modality,
                    "field": field,
                    "files": len(records),
                    "shapes": [list(value) for value in sorted(shapes)],
                    "dtypes": sorted(dtypes),
                    "voxel_sizes": [list(value) for value in sorted(zooms)],
                    **(
                        {"finite": finite, "minimum": minimum, "maximum": maximum}
                        if check_values
                        else {}
                    ),
                }
            )
    report = {
        "kind": "task3_3d_unet_data_audit",
        "created_at_utc": utc_now(),
        "data_dir": str(data_dir.expanduser().resolve()),
        "prospective_train_subjects": list(ALL_PROSPECTIVE_SUBJECTS),
        "train_subjects": list(TRAIN_SUBJECTS),
        "holdout_subject": HOLDOUT_SUBJECT,
        "groups": groups,
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }
    if failures:
        raise Task3Error("Raw data audit failed:\n- " + "\n- ".join(failures))
    return report


def select_retrospective(
    inventory: dict[tuple[str, str], list[VolumeRecord]], *, per_domain: int = 40, seed: int = 2026
) -> list[VolumeRecord]:
    if per_domain < 1:
        raise Task3Error("per_domain must be positive")
    selected: list[VolumeRecord] = []
    for domain_index, key in enumerate((m, f) for m in MODALITIES for f in FIELDS):
        records = inventory[key]
        if len(records) < per_domain:
            raise Task3Error(f"{key} has {len(records)} retrospective volumes; need {per_domain}")
        rng = np.random.default_rng(seed + domain_index)
        indices = sorted(int(value) for value in rng.choice(len(records), per_domain, replace=False))
        selected.extend(records[index] for index in indices)
    return selected


def write_retrospective_manifest(
    data_dir: Path, destination: Path, *, per_domain: int = 40, seed: int = 2026
) -> dict[str, Any]:
    selected = select_retrospective(
        inventory_split(data_dir, "retro_train"), per_domain=per_domain, seed=seed
    )
    payload = {
        "kind": "task3_3d_unet_retrospective_subset",
        "seed": seed,
        "per_domain": per_domain,
        "total": len(selected),
        "records": [
            {
                "path": str(record.path.resolve()),
                "split": record.split,
                "modality": record.modality,
                "field": record.field,
                "subject_id": record.subject_id,
                "size_bytes": record.path.stat().st_size,
            }
            for record in selected
        ],
    }
    atomic_json(destination, payload)
    return payload


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise Task3Error(f"Existing staged file has wrong size: {destination}")
        return
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def stage_data(data_dir: Path, runtime_dir: Path, *, per_domain: int = 40, seed: int = 2026) -> dict[str, Any]:
    """Copy exactly 600 retrospective plus 45 train and 45 validation volumes."""
    retro = select_retrospective(
        inventory_split(data_dir, "retro_train"), per_domain=per_domain, seed=seed
    )
    pro_inventory = inventory_split(data_dir, "pro_train")
    val_inventory = inventory_split(data_dir, "pro_val")
    _validate_subject_layout(pro_inventory, val_inventory)
    pro = [record for records in pro_inventory.values() for record in records]
    val = [record for records in val_inventory.values() for record in records]
    records = retro + pro + val
    compressed_bytes = sum(record.path.stat().st_size for record in records)
    cache_bytes = len(pro) * math.prod(EXPECTED_VOLUME_SHAPE) * 4
    full_inference_bytes = 180 * math.prod(EXPECTED_VOLUME_SHAPE) * 4
    required_bytes = math.ceil((compressed_bytes + cache_bytes + full_inference_bytes) * 1.2)
    runtime_dir = runtime_dir.expanduser().resolve()
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(runtime_dir.parent).free
    if free_bytes < required_bytes:
        raise Task3Error(
            f"Runtime disk preflight failed: free={free_bytes}, required_with_20pct_margin={required_bytes}"
        )
    split_names = {
        "retro_train": "training_retrospective",
        "pro_train": "training_prospective",
        "pro_val": "validation_prospective",
    }
    staged: list[dict[str, Any]] = []
    for record in records:
        destination = runtime_dir / split_names[record.split] / record.modality / record.field / record.path.name
        _atomic_copy(record.path, destination)
        staged.append(
            {
                "path": str(destination),
                "split": record.split,
                "modality": record.modality,
                "field": record.field,
                "subject_id": record.subject_id,
                "size_bytes": destination.stat().st_size,
            }
        )
    payload = {
        "kind": "task3_3d_unet_runtime_stage",
        "created_at_utc": utc_now(),
        "seed": seed,
        "per_domain": per_domain,
        "compressed_bytes": compressed_bytes,
        "prospective_cache_bytes": cache_bytes,
        "full_inference_uncompressed_budget_bytes": full_inference_bytes,
        "required_bytes_with_margin": required_bytes,
        "records": staged,
        "counts": dict(Counter(record["split"] for record in staged)),
    }
    atomic_json(runtime_dir / "stage_manifest.json", payload)
    write_retrospective_manifest(
        runtime_dir,
        runtime_dir / "retrospective_subset.json",
        per_domain=per_domain,
        seed=seed,
    )
    return payload


def _atomic_npy(destination: Path, values: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as stream:
        np.save(stream, values.astype(np.float32, copy=False), allow_pickle=False)
        temporary = Path(stream.name)
    os.replace(temporary, destination)


def prepare_prospective_cache(data_dir: Path, cache_dir: Path) -> dict[str, Any]:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise Task3Error("nibabel is required to prepare the prospective cache") from exc
    inventory = inventory_split(data_dir, "pro_train")
    _validate_subject_layout(inventory, inventory_split(data_dir, "pro_val"))
    cache_dir = cache_dir.expanduser().resolve()
    records: list[dict[str, Any]] = []
    for (modality, field), volumes in inventory.items():
        for record in volumes:
            image = nib.load(str(record.path))
            canonical = nib.as_closest_canonical(image)
            values = np.asanyarray(canonical.dataobj, dtype=np.float32)
            if values.shape != EXPECTED_VOLUME_SHAPE or not np.isfinite(values).all():
                raise Task3Error(f"Invalid prospective volume: {record.path}")
            if float(values.min()) < -1e-6 or float(values.max()) > 1.0 + 1e-6:
                raise Task3Error(f"Prospective volume outside [0,1]: {record.path}")
            destination = cache_dir / f"pro_train_{modality}_{field}_{record.subject_id}.npy"
            if destination.exists():
                cached = np.load(destination, mmap_mode="r", allow_pickle=False)
                if cached.shape != EXPECTED_VOLUME_SHAPE or cached.dtype != np.float32:
                    raise Task3Error(f"Existing prospective cache is invalid: {destination}")
            else:
                _atomic_npy(destination, values)
            records.append(
                {
                    "path": str(destination),
                    "source": str(record.path.resolve()),
                    "modality": modality,
                    "field": field,
                    "subject_id": record.subject_id,
                    "shape": list(values.shape),
                    "dtype": "float32",
                    "affine": canonical.affine.tolist(),
                    "source_size_bytes": record.path.stat().st_size,
                }
            )
    payload = {
        "kind": "task3_3d_unet_prospective_cache",
        "records": records,
        "subjects": list(ALL_PROSPECTIVE_SUBJECTS),
        "count": len(records),
    }
    atomic_json(cache_dir / "manifest.json", payload)
    return payload


def load_cache_index(cache_dir: Path) -> dict[tuple[str, str, str], Path]:
    manifest_path = cache_dir.expanduser().resolve() / "manifest.json"
    if not manifest_path.is_file():
        raise Task3Error(f"Prospective cache manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    index: dict[tuple[str, str, str], Path] = {}
    for record in payload.get("records", []):
        key = (record["modality"], record["field"], record["subject_id"])
        path = Path(record["path"])
        if key in index or not path.is_file():
            raise Task3Error(f"Invalid or duplicate prospective cache record: {key}")
        index[key] = path
    expected = {
        (modality, field, subject)
        for modality in MODALITIES
        for field in FIELDS
        for subject in ALL_PROSPECTIVE_SUBJECTS
    }
    if set(index) != expected:
        raise Task3Error("Prospective cache must contain exactly the fixed 45 volumes")
    return index


def paired_patch(
    source: np.ndarray,
    target: np.ndarray,
    patch_size: int,
    rng: np.random.Generator,
    *,
    flip: bool = False,
    foreground_bias: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    if source.shape != target.shape or source.ndim != 3:
        raise Task3Error(f"Paired volumes are not aligned: {source.shape}/{target.shape}")
    if patch_size < 1 or any(size < patch_size for size in source.shape):
        raise Task3Error(f"Patch {patch_size} does not fit volume {source.shape}")
    if not 0.0 <= foreground_bias <= 1.0:
        raise Task3Error("foreground_bias must be in [0,1]")
    center = None
    if rng.random() < foreground_bias:
        for _ in range(64):
            candidate = tuple(int(rng.integers(0, size)) for size in target.shape)
            if target[candidate] > 1e-6:
                center = candidate
                break
    if center is not None:
        starts = tuple(
            min(max(0, int(coordinate) - patch_size // 2), size - patch_size)
            for coordinate, size in zip(center, source.shape)
        )
    else:
        starts = tuple(int(rng.integers(0, size - patch_size + 1)) for size in source.shape)
    selection = tuple(slice(start, start + patch_size) for start in starts)
    source_patch, target_patch = source[selection], target[selection]
    if flip:
        source_patch, target_patch = np.flip(source_patch, 0), np.flip(target_patch, 0)
    return np.ascontiguousarray(source_patch), np.ascontiguousarray(target_patch), starts


class ProspectivePatchDataset(Dataset):
    """Deterministically balanced 60-stratum paired 3D patches."""

    def __init__(
        self,
        cache_dir: Path,
        subjects: Sequence[str],
        *,
        patch_size: int = 96,
        samples_per_stratum: int = 1,
        seed: int = 2026,
        random_augmentation: bool = True,
        start_index: int = 0,
    ):
        if torch is None:
            raise Task3Error("PyTorch is required for patch datasets")
        subjects = tuple(subjects)
        if not subjects or not set(subjects) <= set(ALL_PROSPECTIVE_SUBJECTS):
            raise Task3Error(f"Unexpected prospective subjects: {subjects}")
        if samples_per_stratum < 1:
            raise Task3Error("samples_per_stratum must be positive")
        self.index = load_cache_index(cache_dir)
        self.subjects = subjects
        self.patch_size = patch_size
        self.samples_per_stratum = samples_per_stratum
        self.seed = seed
        self.random_augmentation = random_augmentation
        self.strata = directed_strata()
        self.epoch = 0
        self.start_index = int(start_index)
        self.total_items = len(self.strata) * self.samples_per_stratum
        if not 0 <= self.start_index <= self.total_items:
            raise Task3Error("start_index lies outside the paired dataset")

    def __len__(self) -> int:
        return self.total_items - self.start_index

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, item: int) -> dict[str, Any]:
        item += self.start_index
        stratum_index = item % len(self.strata)
        sample_index = item // len(self.strata)
        modality, source_field, target_field = self.strata[stratum_index]
        subject = self.subjects[(sample_index + stratum_index) % len(self.subjects)]
        source = np.load(self.index[(modality, source_field, subject)], mmap_mode="r", allow_pickle=False)
        target = np.load(self.index[(modality, target_field, subject)], mmap_mode="r", allow_pickle=False)
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + item)
        flip = self.random_augmentation and bool(rng.random() < 0.5)
        source_patch, target_patch, starts = paired_patch(
            source, target, self.patch_size, rng, flip=flip
        )
        return {
            "source": torch.from_numpy(source_patch).unsqueeze(0),
            "target": torch.from_numpy(target_patch).unsqueeze(0),
            "modality": MODALITIES.index(modality),
            "source_field": FIELDS.index(source_field),
            "target_field": FIELDS.index(target_field),
            "subject_id": subject,
            "stratum": f"{modality}:{source_field}->{target_field}",
            "starts": starts,
        }


def validate_subject_split(train_subjects: Sequence[str], holdout_subject: str | None) -> None:
    train = tuple(train_subjects)
    if holdout_subject is None:
        if set(train) != set(ALL_PROSPECTIVE_SUBJECTS):
            raise Task3Error("All-subject retraining must use exactly P_0006, P_0007, and P_0009")
        return
    if set(train) != set(TRAIN_SUBJECTS) or holdout_subject != HOLDOUT_SUBJECT:
        raise Task3Error("Ablations must train on P_0006/P_0007 and hold out P_0009")
    if holdout_subject in train:
        raise Task3Error("Holdout subject leaked into training")


class RetrospectivePatchStream(IterableDataset):
    """Loads one compressed retrospective volume, then yields 64 patches from it."""

    def __init__(
        self,
        manifest: Path,
        *,
        patch_size: int = 96,
        patches_per_volume: int = 64,
        seed: int = 2026,
        start_patch: int = 0,
    ):
        if torch is None:
            raise Task3Error("PyTorch is required for retrospective pretraining")
        if patches_per_volume < 1:
            raise Task3Error("patches_per_volume must be positive")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.records = payload.get("records", [])
        if not self.records or any(record.get("split") != "retro_train" for record in self.records):
            raise Task3Error("Retrospective stream accepts only retrospective manifest records")
        counts = Counter((record["modality"], record["field"]) for record in self.records)
        if set(counts) != {(m, f) for m in MODALITIES for f in FIELDS} or len(set(counts.values())) != 1:
            raise Task3Error("Retrospective manifest is not balanced across all 15 domains")
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.seed = seed
        self.start_patch = int(start_patch)
        self.by_domain = {
            (modality, field): [
                record
                for record in self.records
                if record["modality"] == modality and record["field"] == field
            ]
            for modality in MODALITIES
            for field in FIELDS
        }

    def _episode(self, episode: int) -> dict[str, Any]:
        domain_index = episode % 15
        cycle = episode // 15
        domain = (MODALITIES[domain_index // 5], FIELDS[domain_index % 5])
        records = self.by_domain[domain]
        rng = np.random.default_rng(self.seed + domain_index + cycle * 15)
        return records[int(rng.integers(0, len(records)))]

    def episode_order(self, episode_count: int) -> list[dict[str, Any]]:
        return [self._episode(episode) for episode in range(episode_count)]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise Task3Error("nibabel is required for retrospective pretraining") from exc
        episode, first_patch = divmod(self.start_patch, self.patches_per_volume)
        while True:
            record = self._episode(episode)
            values = np.asanyarray(nib.load(record["path"]).dataobj, dtype=np.float32)
            if values.shape != EXPECTED_VOLUME_SHAPE:
                raise Task3Error(f"Unexpected retrospective shape: {record['path']}")
            for patch_index in range(first_patch, self.patches_per_volume):
                rng = np.random.default_rng(self.seed + episode * self.patches_per_volume + patch_index)
                patch, _, starts = paired_patch(
                    values, values, self.patch_size, rng, foreground_bias=0.8
                )
                yield {
                    "source": torch.from_numpy(patch).unsqueeze(0),
                    "modality": MODALITIES.index(record["modality"]),
                    "source_field": FIELDS.index(record["field"]),
                    "target_field": FIELDS.index(record["field"]),
                    "starts": starts,
                }
            episode += 1
            first_patch = 0


def _print_report(report: dict[str, Any], destination: Path | None) -> None:
    if destination is not None:
        atomic_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 3 3D U-Net data audit and preparation")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--skip-values", action="store_true")
    audit.add_argument("--report", type=Path)
    subset = commands.add_parser("select-retrospective")
    subset.add_argument("--data-dir", type=Path, required=True)
    subset.add_argument("--manifest", type=Path, required=True)
    subset.add_argument("--per-domain", type=int, default=40)
    subset.add_argument("--seed", type=int, default=2026)
    stage = commands.add_parser("stage")
    stage.add_argument("--data-dir", type=Path, required=True)
    stage.add_argument("--runtime-dir", type=Path, required=True)
    stage.add_argument("--per-domain", type=int, default=40)
    stage.add_argument("--seed", type=int, default=2026)
    cache = commands.add_parser("prepare-cache")
    cache.add_argument("--data-dir", type=Path, required=True)
    cache.add_argument("--cache-dir", type=Path, required=True)
    cache.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            report = audit_data(args.data_dir, check_values=not args.skip_values)
            _print_report(report, args.report)
        elif args.command == "select-retrospective":
            _print_report(
                write_retrospective_manifest(
                    args.data_dir, args.manifest, per_domain=args.per_domain, seed=args.seed
                ),
                None,
            )
        elif args.command == "stage":
            _print_report(
                stage_data(args.data_dir, args.runtime_dir, per_domain=args.per_domain, seed=args.seed),
                None,
            )
        else:
            _print_report(prepare_prospective_cache(args.data_dir, args.cache_dir), args.report)
    except (Task3Error, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
