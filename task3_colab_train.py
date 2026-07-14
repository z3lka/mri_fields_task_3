#!/usr/bin/env python3
"""Colab-oriented training for the official MRIxFields2026 Task 3 StarGAN v2.

This file is intentionally Task 3 only. It imports the organizers' model,
losses, and cached datasets from a pinned checkout, while keeping local
training changes (AMP, atomic checkpoints, holdout support, provenance, and
Colab runtime limits) outside organizer code.

Preparing data and auditing it are safe before the validation-baseline
milestone. Expensive ``pretrain`` and ``finetune`` commands require an explicit
``--baseline-milestone-complete`` acknowledgement.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


OFFICIAL_COMMIT = "f0f4ab2c54ab878b48ddfea8d13f4da512ca411e"
OFFICIAL_CONFIG = Path(
    "Baseline/configs/task3/stargan/any_to_any_all_modalities.yaml"
)
HF_REPO_ID = "mrixfields/MRIxFields2026-Baseline"
HF_REVISION = "f1a7d6b741b64c81871d0856fa2d7295d564c141"
HF_CHECKPOINT = (
    "task3_any_to_any_multimodal/stargan_v2/pro_pretrained/weights/"
    "checkpoint_epoch50.pth"
)

MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
NUM_DOMAINS = len(MODALITIES) * len(FIELDS)
CANONICAL_SPLITS = {
    "retro_train": "Training_retrospective",
    "pro_train": "Training_prospective",
}
SPLIT_ALIASES = {
    "Training_retrospective": (
        "Training_retrospective",
        "training_retrospective",
    ),
    "Training_prospective": (
        "Training_prospective",
        "training_prospective",
    ),
}
EXPECTED_SHAPE = (364, 436, 364)
EXPECTED_DTYPE = "float32"
EXPECTED_VOXEL_SIZE = (0.5, 0.5, 0.5)
SUBJECT_RE = re.compile(r"_(P|R)_([0-9]+)_s[0-9]+$")


class TrainingError(RuntimeError):
    """Raised for a failed invariant or unsafe training request."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def run_checked(command: Sequence[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def official_revision(repo: Path) -> str:
    if not (repo / ".git").exists():
        raise TrainingError(f"Official checkout is not a Git repository: {repo}")
    try:
        return run_checked(["git", "rev-parse", "HEAD"], cwd=repo)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise TrainingError(f"Could not read official Git revision: {exc}") from exc


def validate_official_checkout(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    revision = official_revision(repo)
    if revision != OFFICIAL_COMMIT:
        raise TrainingError(
            "Official checkout revision mismatch. "
            f"Expected {OFFICIAL_COMMIT}, found {revision}. Do not mix code/config/weights."
        )
    config_path = repo / OFFICIAL_CONFIG
    if not config_path.is_file():
        raise TrainingError(f"Official Task 3 config is missing: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise TrainingError("PyYAML is required; install the Colab requirements.") from exc

    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    expected = {
        "task": 3,
        "method": "stargan_v2",
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise TrainingError(
                f"Unexpected official config {key}: {cfg.get(key)!r}; expected {value!r}"
            )
    if int(cfg.get("model", {}).get("num_domains", -1)) != NUM_DOMAINS:
        raise TrainingError("Official config must define exactly 15 joint domains.")
    if tuple(cfg.get("data", {}).get("modalities", ())) != MODALITIES:
        raise TrainingError("Official config modality order changed; review domain mapping.")
    if tuple(cfg.get("domains", ())) != FIELDS:
        raise TrainingError("Official config field order changed; review domain mapping.")

    return {
        "repo": str(repo),
        "revision": revision,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "cfg": cfg,
    }


def resolve_split_dir(data_dir: Path, canonical: str) -> Path:
    aliases = SPLIT_ALIASES[canonical]
    matches = [data_dir / alias for alias in aliases if (data_dir / alias).is_dir()]
    unique: list[Path] = []
    for match in matches:
        if not any(os.path.samefile(match, existing) for existing in unique):
            unique.append(match)
    if not unique:
        raise TrainingError(
            f"Could not find {canonical} below {data_dir}; accepted names: {aliases}"
        )
    if len(unique) > 1:
        raise TrainingError(
            f"Multiple aliases exist for {canonical}: {unique}. Refusing an ambiguous data root."
        )
    return unique[0].resolve()


def create_data_view(data_dir: Path, view_dir: Path, splits: Iterable[str]) -> dict[str, str]:
    """Create non-destructive canonical split symlinks for organizer tooling."""
    data_dir = data_dir.resolve()
    view_dir = view_dir.resolve()
    view_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for split_abbr in splits:
        canonical = CANONICAL_SPLITS[split_abbr]
        source = resolve_split_dir(data_dir, canonical)
        destination = view_dir / canonical
        if destination.is_symlink():
            if destination.resolve() != source:
                raise TrainingError(
                    f"Existing data-view link points elsewhere: {destination} -> "
                    f"{destination.resolve()} (wanted {source})"
                )
        elif destination.exists():
            raise TrainingError(
                f"Data-view destination exists and is not a symlink: {destination}"
            )
        else:
            destination.symlink_to(source, target_is_directory=True)
        mapping[canonical] = str(source)
    atomic_json(
        view_dir / "task3_data_view.json",
        {"created_at_utc": utc_now(), "data_dir": str(data_dir), "splits": mapping},
    )
    return mapping


def subject_from_npz(path: Path) -> str:
    match = SUBJECT_RE.search(path.stem)
    if not match:
        raise TrainingError(f"Could not parse cached subject from {path.name}")
    return f"{match.group(1)}_{match.group(2)}"


def subject_from_nifti(path: Path) -> str:
    stem = path.name.removesuffix(".nii.gz")
    parts = stem.split("_")
    if len(parts) < 4 or parts[0] not in {"P", "R"}:
        raise TrainingError(f"Unexpected Task 3 filename: {path.name}")
    return f"{parts[0]}_{parts[-1]}"


def audit_raw_data(
    data_dir: Path,
    splits: Iterable[str],
    check_values: bool,
    report_path: Path | None,
) -> dict[str, Any]:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise TrainingError("nibabel and NumPy are required for the data audit.") from exc

    data_dir = data_dir.resolve()
    report: dict[str, Any] = {
        "kind": "task3_raw_training_audit",
        "created_at_utc": utc_now(),
        "data_dir": str(data_dir),
        "check_values": check_values,
        "groups": [],
    }
    failures: list[str] = []

    for split_abbr in splits:
        canonical = CANONICAL_SPLITS[split_abbr]
        split_dir = resolve_split_dir(data_dir, canonical)
        subject_sets: dict[str, dict[str, set[str]]] = {}
        for modality in MODALITIES:
            subject_sets[modality] = {}
            for field in FIELDS:
                group_dir = split_dir / modality / field
                files = sorted(group_dir.glob("*.nii.gz")) if group_dir.is_dir() else []
                group: dict[str, Any] = {
                    "split": split_abbr,
                    "resolved_split": str(split_dir),
                    "modality": modality,
                    "field": field,
                    "count": len(files),
                }
                report["groups"].append(group)
                if not files:
                    failures.append(f"No NIfTI files in {group_dir}")
                    continue
                try:
                    subject_sets[modality][field] = {subject_from_nifti(p) for p in files}
                except TrainingError as exc:
                    failures.append(str(exc))
                    continue

                # Header-audit every file. Value audit is deliberately opt-in.
                shapes: set[tuple[int, ...]] = set()
                dtypes: set[str] = set()
                voxel_sizes: set[tuple[float, ...]] = set()
                value_min = math.inf
                value_max = -math.inf
                all_finite = True
                for path in files:
                    image = nib.load(str(path))
                    shapes.add(tuple(int(v) for v in image.shape))
                    dtypes.add(str(image.get_data_dtype()))
                    voxel_sizes.add(tuple(round(float(v), 5) for v in image.header.get_zooms()[:3]))
                    if check_values:
                        array = image.get_fdata(dtype=np.float32)
                        all_finite = all_finite and bool(np.isfinite(array).all())
                        value_min = min(value_min, float(array.min()))
                        value_max = max(value_max, float(array.max()))
                group["shapes"] = [list(v) for v in sorted(shapes)]
                group["dtypes"] = sorted(dtypes)
                group["voxel_sizes"] = [list(v) for v in sorted(voxel_sizes)]
                if shapes != {EXPECTED_SHAPE}:
                    failures.append(f"Unexpected shapes in {group_dir}: {sorted(shapes)}")
                if dtypes != {EXPECTED_DTYPE}:
                    failures.append(f"Unexpected dtypes in {group_dir}: {sorted(dtypes)}")
                if voxel_sizes != {EXPECTED_VOXEL_SIZE}:
                    failures.append(
                        f"Unexpected voxel sizes in {group_dir}: {sorted(voxel_sizes)}"
                    )
                if check_values:
                    group.update(
                        {"finite": all_finite, "minimum": value_min, "maximum": value_max}
                    )
                    if not all_finite or value_min < -1e-6 or value_max > 1.0 + 1e-6:
                        failures.append(
                            f"Invalid values in {group_dir}: finite={all_finite}, "
                            f"range=[{value_min}, {value_max}]"
                        )

        if split_abbr == "pro_train":
            for modality, by_field in subject_sets.items():
                if set(by_field) != set(FIELDS):
                    continue
                reference = by_field[FIELDS[0]]
                for field in FIELDS[1:]:
                    if by_field[field] != reference:
                        failures.append(
                            f"Prospective subjects are not paired for {modality}: "
                            f"{FIELDS[0]}={sorted(reference)}, {field}={sorted(by_field[field])}"
                        )

    report["total_files"] = sum(int(group["count"]) for group in report["groups"])
    report["failures"] = failures
    report["status"] = "passed" if not failures else "failed"
    if report_path:
        atomic_json(report_path.resolve(), report)
    if failures:
        raise TrainingError("Raw-data audit failed:\n- " + "\n- ".join(failures))
    return report


def preprocess_slices(args: argparse.Namespace) -> None:
    provenance = validate_official_checkout(args.official_repo)
    mapping = create_data_view(args.data_dir, args.view_dir, args.splits)
    command = [
        sys.executable,
        str(args.official_repo.resolve() / "Baseline/scripts/preprocess.py"),
        "extract-slices",
        "--data_dir",
        str(args.view_dir.resolve()),
        "--output_dir",
        str(args.preprocessed_dir.resolve()),
        "--splits",
        *args.splits,
        "--modalities",
        *MODALITIES,
        "--fields",
        *FIELDS,
        "--skip_existing",
    ]
    print("Running organizer Task 3 slice extraction:")
    print(" ".join(command))
    subprocess.run(command, check=True)
    provenance_name = f"task3_preprocess_provenance_{'_'.join(args.splits)}.json"
    atomic_json(
        args.preprocessed_dir.resolve() / provenance_name,
        {
            "kind": "task3_preprocess",
            "created_at_utc": utc_now(),
            "official_revision": provenance["revision"],
            "official_config_sha256": provenance["config_sha256"],
            "source_mapping": mapping,
            "splits": args.splits,
            "modalities": MODALITIES,
            "fields": FIELDS,
            "command": command,
        },
    )


def cached_subjects(files: Iterable[Path]) -> set[str]:
    return {subject_from_npz(path) for path in files}


def audit_cached_data(
    preprocessed_dir: Path,
    splits: Iterable[str],
    report_path: Path | None,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise TrainingError("NumPy is required for the cached-data audit.") from exc

    root = preprocessed_dir.resolve()
    report: dict[str, Any] = {
        "kind": "task3_cached_training_audit",
        "created_at_utc": utc_now(),
        "preprocessed_dir": str(root),
        "groups": [],
    }
    failures: list[str] = []
    for split in splits:
        subjects_by_modality: dict[str, dict[str, set[str]]] = {}
        for modality in MODALITIES:
            subjects_by_modality[modality] = {}
            for field in FIELDS:
                group_dir = root / split / modality / field
                files = sorted(group_dir.glob("*.npz")) if group_dir.is_dir() else []
                record: dict[str, Any] = {
                    "split": split,
                    "modality": modality,
                    "field": field,
                    "count": len(files),
                }
                report["groups"].append(record)
                if not files:
                    failures.append(f"No cached slices in {group_dir}")
                    continue
                try:
                    subjects_by_modality[modality][field] = cached_subjects(files)
                    with np.load(files[0]) as sample:
                        image = sample["image"]
                    record.update(
                        {
                            "sample_shape": list(image.shape),
                            "sample_dtype": str(image.dtype),
                            "sample_minimum": float(image.min()),
                            "sample_maximum": float(image.max()),
                            "subjects": len(subjects_by_modality[modality][field]),
                        }
                    )
                    if image.shape != EXPECTED_SHAPE[:2]:
                        failures.append(f"Unexpected cached shape in {files[0]}: {image.shape}")
                    if image.dtype != np.float32:
                        failures.append(f"Unexpected cached dtype in {files[0]}: {image.dtype}")
                    if not np.isfinite(image).all() or image.min() < -1e-6 or image.max() > 1 + 1e-6:
                        failures.append(f"Invalid cached values in {files[0]}")
                except (KeyError, OSError, ValueError, TrainingError) as exc:
                    failures.append(f"Could not validate {files[0]}: {exc}")

        if split == "pro_train":
            for modality, by_field in subjects_by_modality.items():
                if set(by_field) != set(FIELDS):
                    continue
                reference = by_field[FIELDS[0]]
                for field in FIELDS[1:]:
                    if by_field[field] != reference:
                        failures.append(
                            f"Cached prospective subjects are not paired for {modality}: "
                            f"{FIELDS[0]}={sorted(reference)}, {field}={sorted(by_field[field])}"
                        )
    report["total_slices"] = sum(int(group["count"]) for group in report["groups"])
    report["failures"] = failures
    report["status"] = "passed" if not failures else "failed"
    if report_path:
        atomic_json(report_path.resolve(), report)
    if failures:
        raise TrainingError("Cached-data audit failed:\n- " + "\n- ".join(failures))
    return report


@dataclass(frozen=True)
class OfficialComponents:
    torch: Any
    DataLoader: Any
    ConcatDataset: Any
    CachedMultiDomainDataset: Any
    CachedPairedDataset: Any
    build_stargan_v2: Any
    adv_loss: Any
    r1_reg: Any
    compute_g_loss: Any
    moving_average: Any
    PerceptualLoss: Any
    SSIMLoss: Any
    get_joint_domain: Any


def load_official_components(repo: Path) -> OfficialComponents:
    validate_official_checkout(repo)
    baseline = str((repo.resolve() / "Baseline"))
    if baseline not in sys.path:
        sys.path.insert(0, baseline)
    try:
        import torch
        from torch.utils.data import ConcatDataset, DataLoader

        from mrixfields.data.cached_dataset import (
            CachedMultiDomainDataset,
            CachedPairedDataset,
        )
        from mrixfields.data.utils import get_joint_domain
        from mrixfields.losses.perceptual import PerceptualLoss
        from mrixfields.losses.structure import SSIMLoss
        from mrixfields.models.stargan_v2 import (
            adv_loss,
            build_stargan_v2,
            compute_g_loss,
            moving_average,
            r1_reg,
        )
    except ImportError as exc:
        raise TrainingError(
            "Training dependencies are incomplete. Run the Colab requirements cell first."
        ) from exc
    return OfficialComponents(
        torch=torch,
        DataLoader=DataLoader,
        ConcatDataset=ConcatDataset,
        CachedMultiDomainDataset=CachedMultiDomainDataset,
        CachedPairedDataset=CachedPairedDataset,
        build_stargan_v2=build_stargan_v2,
        adv_loss=adv_loss,
        r1_reg=r1_reg,
        compute_g_loss=compute_g_loss,
        moving_average=moving_average,
        PerceptualLoss=PerceptualLoss,
        SSIMLoss=SSIMLoss,
        get_joint_domain=get_joint_domain,
    )


def require_training_acknowledgement(args: argparse.Namespace) -> None:
    if not args.baseline_milestone_complete:
        raise TrainingError(
            "Training is gated because the pretrained-baseline validation submission milestone "
            "is not recorded as complete. Validate/build task3.zip first, then rerun with "
            "--baseline-milestone-complete."
        )


def seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True


def worker_seed(worker_id: int) -> None:
    del worker_id
    try:
        import numpy as np
        import torch

        seed = torch.initial_seed() % (2**32)
        random.seed(seed)
        np.random.seed(seed)
    except ImportError:
        return


@dataclass
class Amp:
    enabled: bool
    dtype_name: str
    dtype: Any
    scaler: Any
    torch: Any

    def context(self):
        if not self.enabled:
            return contextlib.nullcontext()
        return self.torch.autocast(device_type="cuda", dtype=self.dtype)


def configure_amp(torch: Any, choice: str) -> Amp:
    if choice == "off":
        return Amp(False, "float32", None, _make_scaler(torch, False), torch)
    if not torch.cuda.is_available():
        raise TrainingError("A CUDA GPU is required for practical 512x512 Task 3 training.")
    bf16_ok = bool(
        hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    )
    use_bf16 = choice == "bf16" or (choice == "auto" and bf16_ok)
    if choice == "bf16" and not bf16_ok:
        raise TrainingError("This GPU/PyTorch build does not support bfloat16 AMP.")
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    name = "bfloat16" if use_bf16 else "float16"
    return Amp(True, name, dtype, _make_scaler(torch, not use_bf16), torch)


def _make_scaler(torch: Any, enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def torch_load(torch: Any, path: Path, *, weights_only: bool) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=weights_only)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def atomic_torch_save(torch: Any, value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def set_requires_grad(module: Any, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def move_optimizer_state(torch: Any, optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def latest_checkpoint(weights_dir: Path, pattern: str) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    regex = re.compile(pattern)
    for path in weights_dir.glob("checkpoint_*.pth"):
        match = regex.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def latest_checkpoint_by_mtime(weights_dir: Path) -> Path | None:
    candidates = [path for path in weights_dir.glob("checkpoint_*.pth") if path.is_file()]
    return max(candidates, default=None, key=lambda path: path.stat().st_mtime_ns)


def latest_checkpoint_pointer(weights_dir: Path) -> Path | None:
    pointer = weights_dir / "latest_checkpoint.json"
    if not pointer.is_file():
        return None
    try:
        with pointer.open(encoding="utf-8") as handle:
            name = json.load(handle)["filename"]
        candidate = weights_dir / name
        return candidate if candidate.is_file() else None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def prune_checkpoints(weights_dir: Path, pattern: str, keep_last: int) -> None:
    """Bound Drive usage by pruning only checkpoints created in this run dir."""
    if keep_last == 0:
        return
    regex = re.compile(pattern)
    candidates: list[tuple[int, Path]] = []
    for path in weights_dir.glob("checkpoint_*.pth"):
        match = regex.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    candidates.sort()
    for _, path in candidates[:-keep_last]:
        path.unlink()


def prune_checkpoints_by_mtime(
    weights_dir: Path, keep_last: int, protected: Path
) -> None:
    if keep_last == 0:
        return
    candidates = sorted(
        (
            path
            for path in weights_dir.glob("checkpoint_*.pth")
            if path.is_file() and path != protected
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    keep_other = max(keep_last - 1, 0)
    for path in candidates[: len(candidates) - keep_other]:
        path.unlink()


def model_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    model = cfg["model"]
    return {
        "img_size": int(model["img_size"]),
        "style_dim": int(model["style_dim"]),
        "latent_dim": int(model["latent_dim"]),
        "num_domains": int(model["num_domains"]),
        "max_conv_dim": int(model["max_conv_dim"]),
        "input_nc": int(model["input_nc"]),
    }


def build_loader(
    components: OfficialComponents,
    dataset: Any,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> Any:
    generator = components.torch.Generator()
    generator.manual_seed(seed)
    return components.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=shuffle,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=worker_seed,
        generator=generator,
    )


class IndexSampler:
    def __init__(self, indices: Sequence[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def build_resumable_epoch_loader(
    components: OfficialComponents,
    dataset: Any,
    batch_size: int,
    workers: int,
    seed: int,
    start_microbatch: int,
    max_microbatches: int | None,
) -> tuple[Any, int]:
    """Build a deterministic shuffled epoch and start without re-reading skipped data."""
    torch = components.torch
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    total_microbatches = len(indices) // batch_size
    if max_microbatches is not None:
        total_microbatches = min(total_microbatches, max_microbatches)
    if start_microbatch > total_microbatches:
        raise TrainingError(
            f"Resume microbatch {start_microbatch} exceeds epoch length {total_microbatches}."
        )
    start_sample = start_microbatch * batch_size
    end_sample = total_microbatches * batch_size
    sampler = IndexSampler(indices[start_sample:end_sample])
    worker_generator = torch.Generator()
    worker_generator.manual_seed(seed)
    loader = components.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=worker_seed,
        generator=worker_generator,
    )
    return loader, total_microbatches


def build_balanced_proxy_loader(
    components: OfficialComponents,
    concat_dataset: Any,
    batch_size: int,
    workers: int,
    max_batches: int,
    seed: int,
) -> Any:
    """Select a repeatable, approximately even sample from every directed pair."""
    datasets = list(concat_dataset.datasets)
    target_samples = max_batches * batch_size
    if target_samples < len(datasets):
        raise TrainingError(
            f"Holdout proxy budget is {target_samples} samples but must cover all "
            f"{len(datasets)} modality/pair datasets. Increase --max-validation-batches."
        )
    base, remainder = divmod(target_samples, len(datasets))
    indices: list[int] = []
    offset = 0
    for dataset_index, dataset in enumerate(datasets):
        take = min(len(dataset), base + (1 if dataset_index < remainder else 0))
        if take:
            # Even spacing covers the axial range; the seed rotates selections while
            # keeping them identical across every evaluation in this run.
            rotation = seed % len(dataset)
            local = {
                (int((sample_index + 0.5) * len(dataset) / take) + rotation)
                % len(dataset)
                for sample_index in range(take)
            }
            indices.extend(offset + index for index in sorted(local))
        offset += len(dataset)
    sampler = IndexSampler(indices)
    generator = components.torch.Generator()
    generator.manual_seed(seed)
    return components.DataLoader(
        concat_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=worker_seed,
        generator=generator,
    )


def save_run_provenance(
    path: Path,
    args: argparse.Namespace,
    official: dict[str, Any],
    torch: Any,
    stage: str,
    extra: dict[str, Any],
) -> None:
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "func"
    }
    atomic_json(
        path,
        {
            "kind": "task3_stargan_v2_training",
            "stage": stage,
            "created_at_utc": utc_now(),
            "official_revision": official["revision"],
            "official_config": official["config"],
            "official_config_sha256": official["config_sha256"],
            "hf_reference_revision": HF_REVISION,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": gpu,
            "args": serializable_args,
            **extra,
        },
    )


def _mixed_d_loss(
    components: OfficialComponents,
    amp: Amp,
    nets: dict[str, Any],
    x_real: Any,
    y_org: Any,
    y_trg: Any,
    z_trg: Any | None,
    x_ref: Any | None,
    lambda_reg: float,
) -> tuple[Any, dict[str, float]]:
    torch = components.torch
    # R1 is kept in float32 for numerical stability; fake-image work uses AMP.
    x_real_r1 = x_real.detach().float().requires_grad_(True)
    with torch.autocast(device_type="cuda", enabled=False):
        out_real = nets["discriminator"](x_real_r1, y_org)
        loss_real = components.adv_loss(out_real, 1)
        loss_reg = components.r1_reg(out_real, x_real_r1)

    with torch.no_grad(), amp.context():
        if z_trg is not None:
            style = nets["mapping_network"](z_trg, y_trg)
        else:
            style = nets["style_encoder"](x_ref, y_trg)
        x_fake = nets["generator"](x_real, style)
    with amp.context():
        out_fake = nets["discriminator"](x_fake, y_trg)
        loss_fake = components.adv_loss(out_fake.float(), 0)
    loss = loss_real + loss_fake + lambda_reg * loss_reg
    return loss, {
        "real": float(loss_real.detach()),
        "fake": float(loss_fake.detach()),
        "reg": float(loss_reg.detach()),
    }


def command_pretrain(args: argparse.Namespace) -> None:
    require_training_acknowledgement(args)
    official = validate_official_checkout(args.official_repo)
    cfg = official["cfg"]
    components = load_official_components(args.official_repo)
    torch = components.torch
    if not torch.cuda.is_available():
        raise TrainingError("No CUDA GPU detected. Select a Colab GPU runtime.")
    seed_everything(torch, args.seed)
    device = torch.device("cuda:0")
    amp = configure_amp(torch, args.amp)

    train_cfg = cfg["pretrain"]
    total_steps = args.total_steps or int(train_cfg["total_iters"])
    batch_size = args.batch_size or min(int(train_cfg["batch_size"]), 1)
    workers = args.workers
    dataset = components.CachedMultiDomainDataset(
        preprocessed_dir=args.preprocessed_dir.resolve(),
        split="retro_train",
        modalities=list(MODALITIES),
        field_strengths=list(FIELDS),
        crop_size=tuple(cfg["data"]["crop_size"]),
    )
    if set(dataset.domain_files) != set(range(NUM_DOMAINS)):
        raise TrainingError(
            f"Cached pretraining data covers domains {sorted(dataset.domain_files)}, "
            f"expected {list(range(NUM_DOMAINS))}."
        )
    loader = build_loader(components, dataset, batch_size, workers, True, args.seed)
    if len(loader) == 0:
        raise TrainingError("Pretraining loader is empty; reduce batch size or audit the cache.")

    nets, nets_ema = components.build_stargan_v2(**model_kwargs(cfg))
    for module in nets.values():
        module.to(device)
    for module in nets_ema.values():
        module.to(device).eval()

    lr = float(train_cfg["lr"])
    f_lr = float(train_cfg["f_lr"])
    betas = (float(train_cfg["beta1"]), float(train_cfg["beta2"]))
    weight_decay = float(train_cfg["weight_decay"])
    optimizers = {
        "generator": torch.optim.Adam(
            nets["generator"].parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        ),
        "mapping_network": torch.optim.Adam(
            nets["mapping_network"].parameters(),
            lr=f_lr,
            betas=betas,
            weight_decay=weight_decay,
        ),
        "style_encoder": torch.optim.Adam(
            nets["style_encoder"].parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        ),
        "discriminator": torch.optim.Adam(
            nets["discriminator"].parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        ),
    }

    lr_decay_start = args.lr_decay_start
    if lr_decay_start is None:
        lr_decay_start = min(int(train_cfg["lr_decay_start"]), total_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < lr_decay_start:
            return 1.0
        return max(
            0.0,
            1.0 - (current_step - lr_decay_start) / (total_steps - lr_decay_start + 1),
        )

    schedulers = {
        key: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        for key, optimizer in optimizers.items()
    }

    run_dir = (
        args.output_dir.resolve()
        / "task3_any_to_any_multimodal"
        / "stargan_v2"
        / args.run_name
    )
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.jsonl"
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        resume_path = latest_checkpoint(weights_dir, r"checkpoint_([0-9]+)\.pth")
    step = 0
    epoch = 0
    if resume_path:
        resume_path = Path(resume_path).resolve()
        checkpoint = torch_load(torch, resume_path, weights_only=False)
        for key, module in nets.items():
            module.load_state_dict(checkpoint["nets"][key], strict=True)
        for key, module in nets_ema.items():
            module.load_state_dict(checkpoint["nets_ema"][key], strict=True)
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optims"][key])
            move_optimizer_state(torch, optimizer, device)
        if "schedulers" in checkpoint:
            for key, scheduler in schedulers.items():
                scheduler.load_state_dict(checkpoint["schedulers"][key])
        if "scaler" in checkpoint and amp.scaler.is_enabled():
            amp.scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["step"])
        epoch = int(checkpoint.get("epoch", 0))
        print(f"Resumed {resume_path} at step {step}")

    save_run_provenance(
        run_dir / "provenance.json",
        args,
        official,
        torch,
        "retro_pretrain",
        {
            "amp_dtype": amp.dtype_name,
            "dataset_slices": len(dataset),
            "domain_indices": sorted(dataset.domain_files),
            "effective_batch_size": batch_size,
            "total_steps": total_steps,
        },
    )

    latent_dim = int(cfg["model"]["latent_dim"])
    lambda_ds = float(train_cfg["lambda_ds"])
    ds_iter = int(train_cfg["ds_iter"])
    ema_beta = float(train_cfg["ema_beta"])
    started = time.monotonic()

    def checkpoint_now() -> Path:
        path = weights_dir / f"checkpoint_{step}.pth"
        atomic_torch_save(
            torch,
            {
                "step": step,
                "epoch": epoch,
                "nets": {key: module.state_dict() for key, module in nets.items()},
                "nets_ema": {key: module.state_dict() for key, module in nets_ema.items()},
                "optims": {key: optimizer.state_dict() for key, optimizer in optimizers.items()},
                "schedulers": {
                    key: scheduler.state_dict() for key, scheduler in schedulers.items()
                },
                "scaler": amp.scaler.state_dict(),
                "official_commit": OFFICIAL_COMMIT,
                "config_sha256": official["config_sha256"],
                "amp_dtype": amp.dtype_name,
            },
            path,
        )
        prune_checkpoints(
            weights_dir, r"checkpoint_([0-9]+)\.pth", args.keep_last
        )
        print(f"Saved: {path}")
        return path

    print(
        f"Task 3 pretraining: 15 domains, {len(dataset)} slices, batch={batch_size}, "
        f"AMP={amp.dtype_name}, steps={step}->{total_steps}"
    )
    while step < total_steps:
        epoch += 1
        for batch in loader:
            if step >= total_steps:
                break
            x_real = batch["image"].to(device, non_blocking=True)
            y_org = batch["domain"].to(device, non_blocking=True).long()
            x_ref = batch["ref_image"].to(device, non_blocking=True)
            y_trg = batch["ref_domain"].to(device, non_blocking=True).long()
            z1 = torch.randn(x_real.size(0), latent_dim, device=device)
            z2 = torch.randn(x_real.size(0), latent_dim, device=device)
            use_latent = step % 2 == 0
            current_lambda_ds = (
                lambda_ds
                if ds_iter <= 0
                else lambda_ds * max(0.0, 1.0 - step / ds_iter)
            )

            set_requires_grad(nets["discriminator"], True)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            d_loss, d_parts = _mixed_d_loss(
                components,
                amp,
                nets,
                x_real,
                y_org,
                y_trg,
                z1 if use_latent else None,
                None if use_latent else x_ref,
                float(train_cfg["lambda_reg"]),
            )
            amp.scaler.scale(d_loss).backward()
            amp.scaler.step(optimizers["discriminator"])

            set_requires_grad(nets["discriminator"], False)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            with amp.context():
                if use_latent:
                    g_loss, g_parts = components.compute_g_loss(
                        nets,
                        x_real.detach(),
                        y_org,
                        y_trg,
                        z_trgs=[z1, z2],
                        lambda_sty=float(train_cfg["lambda_sty"]),
                        lambda_ds=current_lambda_ds,
                        lambda_cyc=float(train_cfg["lambda_cyc"]),
                    )
                else:
                    permutation = torch.randperm(x_ref.size(0), device=device)
                    g_loss, g_parts = components.compute_g_loss(
                        nets,
                        x_real.detach(),
                        y_org,
                        y_trg,
                        x_refs=[x_ref, x_ref[permutation]],
                        lambda_sty=float(train_cfg["lambda_sty"]),
                        lambda_ds=current_lambda_ds,
                        lambda_cyc=float(train_cfg["lambda_cyc"]),
                    )
            amp.scaler.scale(g_loss).backward()
            for key in ("generator", "mapping_network", "style_encoder"):
                amp.scaler.step(optimizers[key])
            amp.scaler.update()
            set_requires_grad(nets["discriminator"], True)

            for key in ("generator", "mapping_network", "style_encoder"):
                components.moving_average(nets[key], nets_ema[key], ema_beta)
            for scheduler in schedulers.values():
                scheduler.step()
            step += 1

            if step == 1 or step % args.print_every == 0:
                record = {
                    "time_utc": utc_now(),
                    "step": step,
                    "epoch": epoch,
                    "d_loss": float(d_loss.detach()),
                    "g_loss": float(g_loss.detach()),
                    "lambda_ds": current_lambda_ds,
                    "lr_generator": optimizers["generator"].param_groups[0]["lr"],
                    "mode": "latent" if use_latent else "reference",
                    "d": d_parts,
                    "g": g_parts,
                }
                append_jsonl(log_path, record)
                print(
                    f"[{step}/{total_steps}] D={record['d_loss']:.4f} "
                    f"G={record['g_loss']:.4f} ds={current_lambda_ds:.3f}"
                )
            if step % args.save_every == 0:
                checkpoint_now()
            if args.max_runtime_minutes and (
                time.monotonic() - started >= args.max_runtime_minutes * 60
            ):
                checkpoint_now()
                print("Reached max runtime; exit cleanly and rerun to auto-resume.")
                return

    checkpoint_now()


class PairedWithDomains:
    def __init__(
        self,
        dataset: Any,
        src_domain: int,
        tgt_domain: int,
        modality: str,
        source_field: str,
        target_field: str,
    ):
        self.dataset = dataset
        self.src_domain = src_domain
        self.tgt_domain = tgt_domain
        self.modality = modality
        self.source_field = source_field
        self.target_field = target_field

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        item["src_domain"] = self.src_domain
        item["tgt_domain"] = self.tgt_domain
        item["modality"] = self.modality
        item["source_field"] = self.source_field
        item["target_field"] = self.target_field
        return item


def complete_prospective_subjects(preprocessed_dir: Path) -> list[str]:
    common: set[str] | None = None
    for modality in MODALITIES:
        for field in FIELDS:
            files = (preprocessed_dir / "pro_train" / modality / field).glob("*.npz")
            subjects = cached_subjects(files)
            common = subjects if common is None else common & subjects
    return sorted(common or set())


def build_paired_dataset(
    components: OfficialComponents,
    preprocessed_dir: Path,
    crop_size: tuple[int, int],
    include_subjects: set[str] | None,
    exclude_subjects: set[str] | None,
) -> tuple[Any, int]:
    wrapped: list[Any] = []
    pair_count = 0
    for modality in MODALITIES:
        for source_field in FIELDS:
            for target_field in FIELDS:
                if source_field == target_field:
                    continue
                dataset = components.CachedPairedDataset(
                    preprocessed_dir=preprocessed_dir,
                    split="pro_train",
                    modality=modality,
                    source_field=source_field,
                    target_field=target_field,
                    crop_size=crop_size,
                )
                filtered = []
                for source, target in dataset.pairs:
                    subject = subject_from_npz(source)
                    if include_subjects is not None and subject not in include_subjects:
                        continue
                    if exclude_subjects is not None and subject in exclude_subjects:
                        continue
                    filtered.append((source, target))
                dataset.pairs = filtered
                if not dataset.pairs:
                    continue
                wrapped.append(
                    PairedWithDomains(
                        dataset,
                        components.get_joint_domain(modality, source_field),
                        components.get_joint_domain(modality, target_field),
                        modality,
                        source_field,
                        target_field,
                    )
                )
                pair_count += 1
    if not wrapped:
        raise TrainingError("No prospective paired slices remain after subject filtering.")
    return components.ConcatDataset(wrapped), pair_count


def initialize_from_checkpoint(
    components: OfficialComponents,
    checkpoint_path: Path,
    nets: dict[str, Any],
    nets_ema: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = torch_load(components.torch, checkpoint_path, weights_only=True)
    if "nets_ema" in checkpoint:
        source = checkpoint["nets_ema"]
        for key, module in nets_ema.items():
            if key not in source:
                raise TrainingError(f"Initial checkpoint nets_ema is missing {key}")
            module.load_state_dict(source[key], strict=True)
            nets[key].load_state_dict(source[key], strict=True)
    elif "nets" in checkpoint:
        source = checkpoint["nets"]
        for key, module in nets.items():
            if key in source:
                module.load_state_dict(source[key], strict=True)
        for key, module in nets_ema.items():
            if key not in source:
                raise TrainingError(f"Initial checkpoint nets is missing {key}")
            module.load_state_dict(source[key], strict=True)
    else:
        raise TrainingError("Initial checkpoint contains neither nets_ema nor nets.")
    return checkpoint


def validate_holdout(
    components: OfficialComponents,
    amp: Amp,
    nets_ema: dict[str, Any],
    loader: Any,
    latent_dim: int,
    seed: int,
    max_batches: int,
    perceptual: Any,
    ssim_loss: Any,
    device: Any,
) -> dict[str, Any]:
    torch = components.torch
    generator = nets_ema["generator"]
    mapping = nets_ema["mapping_network"]
    generator.eval()
    mapping.eval()
    totals = {
        modality: {
            "sum_squared_error": 0.0,
            "sum_squared_target": 0.0,
            "ssim_total": 0.0,
            "lpips_total": 0.0,
            "samples": 0,
        }
        for modality in MODALITIES
    }
    batches = 0
    latent_generator = torch.Generator(device=device)
    latent_generator.manual_seed(seed)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            source = batch["source"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            target_domain = batch["tgt_domain"].to(device, non_blocking=True).long()
            z = torch.randn(
                source.size(0), latent_dim, generator=latent_generator, device=device
            )
            with amp.context():
                style = mapping(z, target_domain)
                prediction = generator(source, style)
            prediction = prediction.float()
            target = target.float()
            prediction_01 = (prediction + 1.0) / 2.0
            target_01 = (target + 1.0) / 2.0
            for sample_index, modality in enumerate(batch["modality"]):
                item = totals[modality]
                pred_item = prediction[sample_index : sample_index + 1]
                target_item = target[sample_index : sample_index + 1]
                pred_item_01 = prediction_01[sample_index : sample_index + 1]
                target_item_01 = target_01[sample_index : sample_index + 1]
                item["sum_squared_error"] += float(
                    ((pred_item_01 - target_item_01) ** 2).sum()
                )
                item["sum_squared_target"] += float((target_item_01**2).sum())
                item["ssim_total"] += float(
                    1.0 - ssim_loss(pred_item_01, target_item_01)
                )
                item["lpips_total"] += float(perceptual(pred_item, target_item))
                item["samples"] += 1
            batches += 1
    sample_count = sum(int(item["samples"]) for item in totals.values())
    if sample_count == 0:
        raise TrainingError("Holdout loader produced no validation batches.")

    def summarize(item: dict[str, Any]) -> dict[str, float | int]:
        samples = int(item["samples"])
        return {
            "slice_proxy_nrmse": math.sqrt(
                float(item["sum_squared_error"])
                / max(float(item["sum_squared_target"]), 1e-12)
            ),
            "slice_proxy_ssim": float(item["ssim_total"]) / samples,
            "slice_proxy_lpips": float(item["lpips_total"]) / samples,
            "samples": samples,
        }

    overall_totals = {
        key: sum(float(item[key]) for item in totals.values())
        for key in (
            "sum_squared_error",
            "sum_squared_target",
            "ssim_total",
            "lpips_total",
            "samples",
        )
    }
    return {
        "overall": summarize(overall_totals),
        "by_modality": {modality: summarize(totals[modality]) for modality in MODALITIES},
        "batches": batches,
    }


def command_finetune(args: argparse.Namespace) -> None:
    require_training_acknowledgement(args)
    official = validate_official_checkout(args.official_repo)
    cfg = official["cfg"]
    components = load_official_components(args.official_repo)
    torch = components.torch
    if not torch.cuda.is_available():
        raise TrainingError("No CUDA GPU detected. Select a Colab GPU runtime.")
    seed_everything(torch, args.seed)
    device = torch.device("cuda:0")
    amp = configure_amp(torch, args.amp)
    preprocessed_dir = args.preprocessed_dir.resolve()
    complete_subjects = complete_prospective_subjects(preprocessed_dir)
    if len(complete_subjects) < 2:
        raise TrainingError(
            f"Need at least two complete prospective subjects, found {complete_subjects}"
        )
    holdout: str | None
    if args.holdout_subject == "none":
        holdout = None
    elif args.holdout_subject == "auto":
        holdout = complete_subjects[-1]
    else:
        holdout = args.holdout_subject
        if holdout not in complete_subjects:
            raise TrainingError(
                f"Holdout subject {holdout!r} is not complete; choose from {complete_subjects}"
            )
    if (
        holdout
        and not args.from_scratch
        and not args.initialization_excludes_holdout
    ):
        raise TrainingError(
            "A subject holdout is invalid if its initialization may already have seen "
            "prospective training subjects. Use --from-scratch, or initialize from your own "
            "retrospective-only checkpoint and pass --initialization-excludes-holdout. "
            "Do not use the released pro_pretrained/epoch50 checkpoint for honest holdout metrics."
        )

    crop_size = tuple(int(v) for v in cfg["data"]["crop_size"])
    train_dataset, train_pair_count = build_paired_dataset(
        components,
        preprocessed_dir,
        crop_size,
        include_subjects=None,
        exclude_subjects={holdout} if holdout else None,
    )
    validation_dataset = None
    validation_pair_count = 0
    if holdout:
        validation_dataset, validation_pair_count = build_paired_dataset(
            components,
            preprocessed_dir,
            crop_size,
            include_subjects={holdout},
            exclude_subjects=None,
        )
    if train_pair_count != 60 or (holdout and validation_pair_count != 60):
        raise TrainingError(
            f"Expected all 60 Task 3 modality/pair datasets; got train={train_pair_count}, "
            f"validation={validation_pair_count}."
        )

    finetune_cfg = cfg["finetune"]
    batch_size = args.batch_size or min(int(finetune_cfg["batch_size"]), 1)
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = build_balanced_proxy_loader(
            components,
            validation_dataset,
            batch_size,
            args.workers,
            args.max_validation_batches,
            args.seed,
        )
    loader_for_size = build_loader(
        components, train_dataset, batch_size, 0, True, args.seed
    )
    train_loader_length = len(loader_for_size)
    del loader_for_size
    if train_loader_length == 0:
        raise TrainingError("Fine-tuning loader is empty.")

    nets, nets_ema = components.build_stargan_v2(**model_kwargs(cfg))
    init_sha = None
    if not args.from_scratch:
        if args.init_checkpoint is None or not args.init_checkpoint.is_file():
            raise TrainingError(
                "Fine-tuning needs --init-checkpoint, or pass --from-scratch deliberately."
            )
        initialize_from_checkpoint(
            components, args.init_checkpoint.resolve(), nets, nets_ema
        )
        init_sha = sha256_file(args.init_checkpoint.resolve())
    for module in nets.values():
        module.to(device)
    for module in nets_ema.values():
        module.to(device).eval()

    # Match the organizer fine-tune: only generator and mapping network update.
    for key in ("style_encoder", "discriminator"):
        set_requires_grad(nets[key], False)
        nets[key].eval()
    optimizer = torch.optim.Adam(
        list(nets["generator"].parameters()) + list(nets["mapping_network"].parameters()),
        lr=args.lr or float(finetune_cfg["lr"]),
        betas=(float(finetune_cfg["beta1"]), float(finetune_cfg["beta2"])),
    )
    l1_loss = torch.nn.L1Loss()
    perceptual = components.PerceptualLoss().to(device).eval()
    ssim_loss = components.SSIMLoss().to(device).eval()

    run_dir = (
        args.output_dir.resolve()
        / "task3_any_to_any_multimodal"
        / "stargan_v2"
        / args.run_name
    )
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.jsonl"
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        resume_path = latest_checkpoint_pointer(weights_dir)
        if resume_path is None:
            resume_path = latest_checkpoint_by_mtime(weights_dir)
    start_epoch = 0
    resume_micro_in_epoch = 0
    global_step = 0
    if resume_path:
        checkpoint = torch_load(torch, Path(resume_path).resolve(), weights_only=False)
        for key, module in nets.items():
            if key in checkpoint["nets"]:
                module.load_state_dict(checkpoint["nets"][key], strict=True)
        for key, module in nets_ema.items():
            module.load_state_dict(checkpoint["nets_ema"][key], strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            move_optimizer_state(torch, optimizer, device)
        if "scaler" in checkpoint and amp.scaler.is_enabled():
            amp.scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("next_epoch", checkpoint["epoch"]))
        resume_micro_in_epoch = int(checkpoint.get("next_micro_in_epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        print(
            f"Resumed {resume_path} at epoch index {start_epoch}, "
            f"microbatch {resume_micro_in_epoch}"
        )

    epochs = args.epochs or int(finetune_cfg["epochs"])
    save_run_provenance(
        run_dir / "provenance.json",
        args,
        official,
        torch,
        "prospective_finetune",
        {
            "amp_dtype": amp.dtype_name,
            "complete_prospective_subject_count": len(complete_subjects),
            "holdout_subject": holdout,
            "training_subject_count": len(complete_subjects) - (1 if holdout else 0),
            "train_slices_with_directed_pair_repetition": len(train_dataset),
            "train_modality_pair_datasets": train_pair_count,
            "validation_slices_with_directed_pair_repetition": (
                len(validation_dataset) if validation_dataset is not None else 0
            ),
            "initial_checkpoint": (
                str(args.init_checkpoint.resolve()) if args.init_checkpoint else None
            ),
            "initial_checkpoint_sha256": init_sha,
            "effective_batch_size": batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_optimization_batch_size": (
                batch_size * args.gradient_accumulation
            ),
            "epochs": epochs,
        },
    )

    lambda_l1 = float(finetune_cfg["loss_l1"])
    lambda_lpips = float(finetune_cfg["loss_lpips"])
    lambda_ssim = float(finetune_cfg["loss_ssim"])
    ema_beta = float(finetune_cfg.get("ema_beta", 0.999))
    latent_dim = int(cfg["model"]["latent_dim"])
    started = time.monotonic()

    def checkpoint_now(
        path: Path,
        *,
        next_epoch: int,
        next_micro_in_epoch: int,
    ) -> Path:
        atomic_torch_save(
            torch,
            {
                # epoch is retained for organizer compatibility. The next_* fields
                # provide exact local resume semantics for within-epoch saves.
                "epoch": next_epoch,
                "next_epoch": next_epoch,
                "next_micro_in_epoch": next_micro_in_epoch,
                "global_step": global_step,
                "nets": {key: module.state_dict() for key, module in nets.items()},
                "nets_ema": {key: module.state_dict() for key, module in nets_ema.items()},
                "optimizer": optimizer.state_dict(),
                "scaler": amp.scaler.state_dict(),
                "official_commit": OFFICIAL_COMMIT,
                "config_sha256": official["config_sha256"],
                "holdout_subject": holdout,
                "amp_dtype": amp.dtype_name,
            },
            path,
        )
        prune_checkpoints_by_mtime(weights_dir, args.keep_last, protected=path)
        atomic_json(
            weights_dir / "latest_checkpoint.json",
            {
                "filename": path.name,
                "saved_at_utc": utc_now(),
                "next_epoch": next_epoch,
                "next_micro_in_epoch": next_micro_in_epoch,
                "global_step": global_step,
            },
        )
        print(f"Saved: {path}")
        return path

    print(
        f"Task 3 fine-tuning: 60 same-modality directed pairs, "
        f"subjects={len(complete_subjects) - (1 if holdout else 0)}, holdout={holdout}, "
        f"batch={batch_size}, AMP={amp.dtype_name}, epochs={start_epoch}->{epochs}"
    )
    for epoch in range(start_epoch, epochs):
        epoch_resume_micro = resume_micro_in_epoch if epoch == start_epoch else 0
        train_loader, planned_micro_steps = build_resumable_epoch_loader(
            components,
            train_dataset,
            batch_size,
            args.workers,
            args.seed + epoch,
            epoch_resume_micro,
            args.max_steps_per_epoch,
        )
        nets["generator"].train()
        nets["mapping_network"].train()
        epoch_loss = 0.0
        epoch_steps = epoch_resume_micro
        processed_steps = 0
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            source = batch["source"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            target_domain = batch["tgt_domain"].to(device, non_blocking=True).long()
            z = torch.randn(source.size(0), latent_dim, device=device)
            with amp.context():
                style = nets["mapping_network"](z, target_domain)
                prediction = nets["generator"](source, style)
                loss = lambda_l1 * l1_loss(prediction, target)
                if lambda_lpips > 0:
                    loss = loss + lambda_lpips * perceptual(prediction, target)
                if lambda_ssim > 0:
                    loss = loss + lambda_ssim * ssim_loss(prediction, target)
            accumulation_group_start = (
                epoch_steps // args.gradient_accumulation
            ) * args.gradient_accumulation
            accumulation_group_size = min(
                args.gradient_accumulation,
                planned_micro_steps - accumulation_group_start,
            )
            scaled_loss = loss / accumulation_group_size
            amp.scaler.scale(scaled_loss).backward()
            completes_accumulation = (
                (epoch_steps + 1) % args.gradient_accumulation == 0
                or epoch_steps + 1 == planned_micro_steps
            )
            if completes_accumulation:
                amp.scaler.step(optimizer)
                amp.scaler.update()
                optimizer.zero_grad(set_to_none=True)
                components.moving_average(
                    nets["generator"], nets_ema["generator"], ema_beta
                )
                components.moving_average(
                    nets["mapping_network"], nets_ema["mapping_network"], ema_beta
                )
            epoch_loss += float(loss.detach())
            epoch_steps += 1
            processed_steps += 1
            global_step += 1
            if global_step == 1 or global_step % args.print_every == 0:
                print(
                    f"epoch={epoch + 1}/{epochs} step={epoch_steps} "
                    f"global={global_step} loss={float(loss.detach()):.5f}"
                )
            if (
                completes_accumulation
                and args.save_every_steps
                and global_step % args.save_every_steps == 0
            ):
                checkpoint_now(
                    weights_dir / f"checkpoint_step{global_step}.pth",
                    next_epoch=epoch,
                    next_micro_in_epoch=epoch_steps,
                )
            if completes_accumulation and args.max_runtime_minutes and (
                time.monotonic() - started >= args.max_runtime_minutes * 60
            ):
                partial = weights_dir / f"checkpoint_partial_step{global_step}.pth"
                checkpoint_now(
                    partial,
                    next_epoch=epoch,
                    next_micro_in_epoch=epoch_steps,
                )
                print(
                    f"Reached max runtime and saved {partial}. Rerun the same command; "
                    "auto-resume continues at the next deterministic microbatch."
                )
                return

        record: dict[str, Any] = {
            "time_utc": utc_now(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss_since_resume": epoch_loss / max(processed_steps, 1),
            "processed_microbatches_this_session": processed_steps,
            "completed_microbatches_in_epoch": epoch_steps,
        }
        if validation_loader is not None and (epoch + 1) % args.eval_every == 0:
            record["holdout"] = validate_holdout(
                components,
                amp,
                nets_ema,
                validation_loader,
                latent_dim,
                args.seed,
                args.max_validation_batches,
                perceptual,
                ssim_loss,
                device,
            )
            print(f"Holdout slice proxies: {record['holdout']}")
        append_jsonl(log_path, record)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == epochs:
            checkpoint_now(
                weights_dir / f"checkpoint_epoch{epoch + 1}.pth",
                next_epoch=epoch + 1,
                next_micro_in_epoch=0,
            )
        resume_micro_in_epoch = 0


def add_shared_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--amp", choices=("auto", "fp16", "bf16", "off"), default="auto")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--max-runtime-minutes", type=float, default=None)
    parser.add_argument(
        "--keep-last",
        type=int,
        default=3,
        help="Keep the newest N completed checkpoints in this run (0 keeps all).",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-auto-resume", dest="auto_resume", action="store_false")
    parser.set_defaults(auto_resume=True)
    parser.add_argument(
        "--baseline-milestone-complete",
        action="store_true",
        help="Explicit acknowledgement required before an expensive training command.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MRIxFields2026 Task 3-only Colab preparation and StarGAN v2 training"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-raw", help="Audit Task 3 training NIfTI data")
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument(
        "--splits", nargs="+", choices=tuple(CANONICAL_SPLITS), default=["pro_train"]
    )
    audit.add_argument("--check-values", action="store_true")
    audit.add_argument("--report", type=Path, default=None)
    audit.set_defaults(
        func=lambda args: print(
            json.dumps(
                audit_raw_data(args.data_dir, args.splits, args.check_values, args.report),
                indent=2,
            )
        )
    )

    view = subparsers.add_parser("make-data-view", help="Create canonical split symlinks")
    view.add_argument("--data-dir", type=Path, required=True)
    view.add_argument("--view-dir", type=Path, required=True)
    view.add_argument(
        "--splits", nargs="+", choices=tuple(CANONICAL_SPLITS), default=["pro_train"]
    )
    view.set_defaults(
        func=lambda args: print(
            json.dumps(create_data_view(args.data_dir, args.view_dir, args.splits), indent=2)
        )
    )

    preprocess = subparsers.add_parser(
        "preprocess", help="Run pinned organizer 220-slice extraction"
    )
    preprocess.add_argument("--official-repo", type=Path, required=True)
    preprocess.add_argument("--data-dir", type=Path, required=True)
    preprocess.add_argument("--view-dir", type=Path, required=True)
    preprocess.add_argument("--preprocessed-dir", type=Path, required=True)
    preprocess.add_argument(
        "--splits", nargs="+", choices=tuple(CANONICAL_SPLITS), default=["pro_train"]
    )
    preprocess.set_defaults(func=preprocess_slices)

    cached = subparsers.add_parser("audit-cache", help="Audit cached Task 3 NPZ slices")
    cached.add_argument("--preprocessed-dir", type=Path, required=True)
    cached.add_argument(
        "--splits", nargs="+", choices=tuple(CANONICAL_SPLITS), default=["pro_train"]
    )
    cached.add_argument("--report", type=Path, default=None)
    cached.set_defaults(
        func=lambda args: print(
            json.dumps(
                audit_cached_data(args.preprocessed_dir, args.splits, args.report), indent=2
            )
        )
    )

    pretrain = subparsers.add_parser(
        "pretrain", help="Unpaired retrospective training of one shared 15-domain model"
    )
    add_shared_training_args(pretrain)
    pretrain.add_argument("--total-steps", type=int, default=None)
    pretrain.add_argument("--lr-decay-start", type=int, default=None)
    pretrain.add_argument("--save-every", type=int, default=1000)
    pretrain.set_defaults(func=command_pretrain)

    finetune = subparsers.add_parser(
        "finetune", help="Paired prospective fine-tuning of one shared 15-domain model"
    )
    add_shared_training_args(finetune)
    finetune.add_argument("--init-checkpoint", type=Path, default=None)
    finetune.add_argument("--from-scratch", action="store_true")
    finetune.add_argument(
        "--initialization-excludes-holdout",
        action="store_true",
        help="Attest that --init-checkpoint never saw the prospective holdout subject.",
    )
    finetune.add_argument("--epochs", type=int, default=None)
    finetune.add_argument("--lr", type=float, default=None)
    finetune.add_argument(
        "--holdout-subject",
        default="auto",
        help="auto, none, or a complete prospective subject such as P_0009",
    )
    finetune.add_argument("--max-steps-per-epoch", type=int, default=None)
    finetune.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Accumulate microbatches; batch=1 and value=8 matches config batch 8.",
    )
    finetune.add_argument("--eval-every", type=int, default=1)
    finetune.add_argument("--max-validation-batches", type=int, default=120)
    finetune.add_argument("--save-every", type=int, default=1)
    finetune.add_argument(
        "--save-every-steps",
        type=int,
        default=1000,
        help="Save an auto-resumable within-epoch checkpoint every N microbatches (0 disables).",
    )
    finetune.set_defaults(func=command_finetune)
    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "workers",
        "print_every",
        "save_every",
        "gradient_accumulation",
        "save_every_steps",
    ):
        if hasattr(args, name):
            value = getattr(args, name)
            minimum = 0 if name in {"workers", "save_every_steps"} else 1
            if value is not None and value < minimum:
                raise TrainingError(f"--{name.replace('_', '-')} has an invalid value: {value}")
    if hasattr(args, "keep_last") and args.keep_last < 0:
        raise TrainingError("--keep-last must be zero or positive")
    if hasattr(args, "total_steps") and args.total_steps is not None and args.total_steps < 1:
        raise TrainingError("--total-steps must be positive")
    if hasattr(args, "epochs") and args.epochs is not None and args.epochs < 1:
        raise TrainingError("--epochs must be positive")
    if getattr(args, "from_scratch", False) and getattr(args, "init_checkpoint", None):
        raise TrainingError("Use either --from-scratch or --init-checkpoint, not both.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_cli_args(args)
        args.func(args)
    except (TrainingError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
