#!/usr/bin/env python3
"""Reproducible MRIxFields2026 Task 3 baseline validation inference.

This wrapper uses the organizers' StarGAN v2 model definitions and their
single shared multimodal checkpoint. It does not train, score public
validation data (the targets are private), package a submission, or upload.

Heavy dependencies are imported only by commands that need them so that
``plan`` and ``--help`` work in a plain Python installation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
MODALITIES = ("T1W", "T2W", "T2FLAIR")
EXPECTED_IDS = {
    "0.1T": ("0001", "0002", "0003"),
    "1.5T": ("0004", "0005", "0008"),
    "3T": ("0010", "0011", "0012"),
    "5T": ("0013", "0014", "0015"),
    "7T": ("0016", "0017", "0018"),
}

OFFICIAL_REPO_URL = "https://github.com/MRIxFields/MRIxFields2026.git"
OFFICIAL_GIT_REVISION = "f0f4ab2c54ab878b48ddfea8d13f4da512ca411e"
HF_REPO_ID = "mrixfields/MRIxFields2026-Baseline"
HF_REVISION = "f1a7d6b741b64c81871d0856fa2d7295d564c141"
CHECKPOINT_FILE = (
    "task3_any_to_any_multimodal/stargan_v2/pro_pretrained/weights/"
    "checkpoint_epoch50.pth"
)
CONFIG_RELATIVE_PATH = Path(
    "Baseline/configs/task3/stargan/any_to_any_all_modalities.yaml"
)

FILENAME_RE = re.compile(
    r"^P_(T1W|T2W|T2FLAIR)_(0\.1T|1\.5T|3T|5T|7T)_(\d{4})\.nii\.gz$"
)
SWEEP_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class PipelineError(RuntimeError):
    """A user-facing validation or configuration error."""


@dataclass(frozen=True)
class InputVolume:
    path: Path
    modality: str
    source_field: str
    subject_id: str


@dataclass(frozen=True)
class InferenceCase:
    input_volume: InputVolume
    target_field: str
    output_path: Path

    @property
    def target_domain(self) -> int:
        return joint_domain(self.input_volume.modality, self.target_field)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def joint_domain(modality: str, field: str) -> int:
    """Official 15-domain index: modality index * 5 + field index."""
    try:
        return MODALITIES.index(modality) * len(FIELDS) + FIELDS.index(field)
    except ValueError as exc:
        raise PipelineError(f"Unsupported modality/field: {modality}/{field}") from exc


def parse_input_filename(path: Path) -> InputVolume:
    match = FILENAME_RE.fullmatch(path.name)
    if not match:
        raise PipelineError(
            f"Invalid validation filename {path.name!r}; expected "
            "P_{MODALITY}_{FIELD}_{ID:04d}.nii.gz"
        )
    modality, field, subject_id = match.groups()
    return InputVolume(path=path, modality=modality, source_field=field, subject_id=subject_id)


def modality_directory(input_root: Path, modality: str) -> Path:
    """Accept either a validation split root or a single modality directory."""
    direct = input_root / modality
    if direct.is_dir():
        return direct
    if input_root.name == modality and all((input_root / f).is_dir() for f in FIELDS):
        return input_root
    raise PipelineError(
        f"Cannot find {modality}: expected {direct} or a {modality}/ directory "
        f"containing field folders under {input_root}"
    )


def discover_inputs(
    input_root: Path,
    modalities: Sequence[str],
    require_official_ids: bool = True,
) -> list[InputVolume]:
    volumes: list[InputVolume] = []
    problems: list[str] = []
    for modality in modalities:
        try:
            mod_dir = modality_directory(input_root, modality)
        except PipelineError as exc:
            problems.append(str(exc))
            continue
        for field in FIELDS:
            field_dir = mod_dir / field
            if not field_dir.is_dir():
                problems.append(f"Missing field directory: {field_dir}")
                continue
            paths = sorted(field_dir.glob("*.nii.gz"))
            parsed: list[InputVolume] = []
            for path in paths:
                try:
                    item = parse_input_filename(path)
                except PipelineError as exc:
                    problems.append(str(exc))
                    continue
                if item.modality != modality or item.source_field != field:
                    problems.append(
                        f"Folder/filename mismatch: {path} encodes "
                        f"{item.modality}/{item.source_field}"
                    )
                parsed.append(item)
            if require_official_ids:
                found_ids = tuple(item.subject_id for item in parsed)
                if found_ids != EXPECTED_IDS[field]:
                    problems.append(
                        f"{field_dir}: IDs {found_ids}, expected {EXPECTED_IDS[field]}"
                    )
            elif not parsed:
                problems.append(f"No NIfTI inputs found: {field_dir}")
            volumes.extend(parsed)
    if problems:
        raise PipelineError("Input audit failed:\n  - " + "\n  - ".join(problems))
    return volumes


def output_path_for(
    output_root: Path,
    volume: InputVolume,
    target_field: str,
    sweep_mode: str = "pro_pretrained",
    sweep_epoch: str = "epoch50",
) -> Path:
    pair = f"{volume.source_field}_to_{target_field}"
    return (
        output_root
        / f"task3_{pair}_{volume.modality}"
        / "stargan_v2"
        / sweep_mode
        / sweep_epoch
        / volume.path.name
    )


def make_cases(
    volumes: Iterable[InputVolume],
    output_root: Path,
    sweep_mode: str = "pro_pretrained",
    sweep_epoch: str = "epoch50",
) -> list[InferenceCase]:
    cases: list[InferenceCase] = []
    for volume in volumes:
        for target in FIELDS:
            if target == volume.source_field:
                continue
            cases.append(
                InferenceCase(
                    input_volume=volume,
                    target_field=target,
                    output_path=output_path_for(
                        output_root,
                        volume,
                        target,
                        sweep_mode=sweep_mode,
                        sweep_epoch=sweep_epoch,
                    ),
                )
            )
    return cases


def select_case_shard(
    cases: Sequence[InferenceCase], num_shards: int, shard_index: int
) -> list[InferenceCase]:
    if num_shards < 1:
        raise PipelineError("--num-shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise PipelineError(
            f"--shard-index must be in [0, {num_shards}); got {shard_index}"
        )
    return list(cases[shard_index::num_shards])


def shard_manifest_path(output_dir: Path, num_shards: int, shard_index: int) -> Path:
    if num_shards == 1:
        return output_dir / "task3_run_manifest.json"
    return output_dir / (
        f"task3_run_manifest.shard-{shard_index:02d}-of-{num_shards:02d}.json"
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def git_revision(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_official_repo(
    repo: Path, expected_revision: str = OFFICIAL_GIT_REVISION
) -> None:
    if not (repo / CONFIG_RELATIVE_PATH).is_file():
        raise PipelineError(f"Official repository/config not found under {repo}")
    try:
        revision = git_revision(repo)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PipelineError(f"Cannot read the official Git revision at {repo}") from exc
    if revision != expected_revision:
        raise PipelineError(
            f"Official repository is at {revision}, but this wrapper was checked against "
            f"{expected_revision}. Re-check upstream before changing the pin."
        )


def command_fetch_official(args: argparse.Namespace) -> None:
    destination = args.destination.resolve()
    if destination.exists():
        if not (destination / ".git").is_dir():
            raise PipelineError(f"Destination exists but is not a Git checkout: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", OFFICIAL_REPO_URL, str(destination)], check=True
        )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "origin", args.revision],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", args.revision],
        check=True,
    )
    verify_official_repo(destination, args.revision)
    print(f"Official repository ready: {destination}")
    print(f"Revision: {args.revision}")


def command_download_checkpoint(args: argparse.Namespace) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise PipelineError(
            "huggingface-hub is required; install requirements-colab.txt first"
        ) from exc

    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=CHECKPOINT_FILE,
            revision=HF_REVISION,
            local_dir=destination,
        )
    ).resolve()
    checksum = sha256_file(downloaded)
    provenance = {
        "downloaded_at_utc": utc_now(),
        "repository": HF_REPO_ID,
        "revision": HF_REVISION,
        "filename": CHECKPOINT_FILE,
        "path": str(downloaded),
        "size_bytes": downloaded.stat().st_size,
        "sha256": checksum,
    }
    atomic_json(destination / "task3_checkpoint_provenance.json", provenance)
    print(f"Checkpoint: {downloaded}")
    print(f"SHA-256:   {checksum}")


def _nifti_summary(path: Path, check_values: bool) -> dict[str, object]:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise PipelineError("nibabel and numpy are required for NIfTI auditing") from exc

    image = nib.load(str(path))
    summary: dict[str, object] = {
        "path": str(path),
        "shape": list(image.shape),
        "dtype": str(image.get_data_dtype()),
        "zooms": [float(x) for x in image.header.get_zooms()[:3]],
        "qform_code": int(image.header["qform_code"]),
        "sform_code": int(image.header["sform_code"]),
    }
    if check_values:
        data = image.get_fdata(dtype=np.float32)
        summary.update(
            finite=bool(np.isfinite(data).all()),
            minimum=float(data.min()),
            maximum=float(data.max()),
        )
    return summary


def _check_input_nifti(summary: dict[str, object], check_values: bool) -> list[str]:
    errors: list[str] = []
    if summary["shape"] != [364, 436, 364]:
        errors.append(f"wrong shape {summary['shape']}")
    if summary["dtype"] != "float32":
        errors.append(f"wrong dtype {summary['dtype']}")
    if any(abs(float(x) - 0.5) > 1e-5 for x in summary["zooms"]):
        errors.append(f"wrong voxel size {summary['zooms']}")
    if check_values:
        if not summary["finite"]:
            errors.append("contains NaN/Inf")
        if float(summary["minimum"]) < -1e-6 or float(summary["maximum"]) > 1.0 + 1e-6:
            errors.append(
                f"range [{summary['minimum']}, {summary['maximum']}] is outside [0,1]"
            )
    return errors


def command_audit(args: argparse.Namespace) -> None:
    volumes = discover_inputs(args.input_root.resolve(), args.modalities)
    summaries = []
    failures = []
    for index, volume in enumerate(volumes, start=1):
        summary = _nifti_summary(volume.path, args.check_values)
        errors = _check_input_nifti(summary, args.check_values)
        if errors:
            failures.append(f"{volume.path}: " + "; ".join(errors))
        summaries.append(summary)
        print(f"[{index:02d}/{len(volumes):02d}] {volume.path}: " + ("OK" if not errors else "FAIL"))
    report = {
        "created_at_utc": utc_now(),
        "input_root": str(args.input_root.resolve()),
        "modalities": list(args.modalities),
        "value_scan_performed": args.check_values,
        "file_count": len(volumes),
        "failures": failures,
        "files": summaries,
    }
    atomic_json(args.report.resolve(), report)
    if failures:
        raise PipelineError("NIfTI audit failed:\n  - " + "\n  - ".join(failures))
    print(f"Audit passed for {len(volumes)} files. Report: {args.report.resolve()}")


def command_plan(args: argparse.Namespace) -> None:
    volumes = discover_inputs(args.input_root.resolve(), args.modalities)
    cases = make_cases(
        volumes,
        args.output_dir.resolve(),
        sweep_mode=args.sweep_mode,
        sweep_epoch=args.sweep_epoch,
    )
    print(f"Input volumes:   {len(volumes)}")
    print(f"Modalities:      {', '.join(args.modalities)}")
    print(f"Inference cases: {len(cases)} (4 targets per input; no identities)")
    print(f"Full Task 3:     {'yes' if len(cases) == 180 else 'no (partial modality set)'}")
    for case in cases:
        item = case.input_volume
        print(
            f"{item.modality} {item.source_field}->{case.target_field} ID {item.subject_id} "
            f"domain={case.target_domain:02d} -> {case.output_path}"
        )


def resolve_device(requested: str):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PipelineError(f"CUDA was requested ({requested}) but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise PipelineError("MPS was requested but is unavailable")
    return device


def load_shared_model(official_repo: Path, checkpoint: Path, device):
    import torch
    import yaml

    baseline_dir = official_repo / "Baseline"
    model_path = baseline_dir / "mrixfields/models/stargan_v2.py"
    # Import this self-contained model file directly. Importing it through
    # ``mrixfields.models`` executes the organizer package's __init__.py, which
    # imports unrelated CUT training losses and makes LPIPS an accidental
    # inference dependency.
    spec = importlib.util.spec_from_file_location("_mrixfields_stargan_v2", model_path)
    if spec is None or spec.loader is None:
        raise PipelineError(f"Cannot import the official StarGAN module: {model_path}")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    MappingNetwork = model_module.MappingNetwork
    StarGANv2Generator = model_module.StarGANv2Generator

    config_path = official_repo / CONFIG_RELATIVE_PATH
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    model_config = config["model"]
    if model_config.get("num_domains") != 15:
        raise PipelineError(
            f"Expected the official multimodal config to have 15 domains; got "
            f"{model_config.get('num_domains')}"
        )

    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise PipelineError("This runner requires PyTorch with weights_only=True support") from exc
    if not isinstance(state, dict) or not isinstance(state.get("nets_ema"), dict):
        raise PipelineError("Checkpoint does not contain the required nets_ema state")
    ema = state["nets_ema"]
    missing = {"generator", "mapping_network"} - set(ema)
    if missing:
        raise PipelineError(f"Checkpoint nets_ema is missing: {sorted(missing)}")

    generator = StarGANv2Generator(
        img_size=model_config["img_size"],
        style_dim=model_config["style_dim"],
        max_conv_dim=model_config["max_conv_dim"],
        input_nc=model_config["input_nc"],
    )
    mapping = MappingNetwork(
        latent_dim=model_config["latent_dim"],
        style_dim=model_config["style_dim"],
        num_domains=model_config["num_domains"],
    )
    generator.load_state_dict(ema["generator"], strict=True)
    mapping.load_state_dict(ema["mapping_network"], strict=True)
    del state, ema
    generator.to(device).eval()
    mapping.to(device).eval()
    return generator, mapping, config


def _center_geometry(source_shape: Sequence[int], target_shape: Sequence[int]):
    source_slices = []
    target_slices = []
    for source, target in zip(source_shape, target_shape):
        if source > target:
            start = (source - target) // 2
            source_slices.append(slice(start, start + target))
            target_slices.append(slice(None))
        else:
            start = (target - source) // 2
            source_slices.append(slice(None))
            target_slices.append(slice(start, start + source))
    return tuple(source_slices), tuple(target_slices)


def deterministic_style(mapping, target_domain: int, latent_dim: int, seed_key: str, device):
    import torch

    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    derived_seed = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(derived_seed)
    latent = torch.randn((1, latent_dim), generator=cpu_generator).to(device)
    domain = torch.tensor([target_domain], dtype=torch.long, device=device)
    with torch.inference_mode():
        return mapping(latent, domain)


def predict_volume_batched(
    generator,
    volume,
    style,
    crop_size: tuple[int, int],
    slice_axis: int,
    batch_size: int,
    device,
):
    import numpy as np
    import torch

    moved = np.moveaxis(volume.astype(np.float32, copy=False), slice_axis, 0)
    output = np.empty_like(moved, dtype=np.float32)
    source_slices, crop_slices = _center_geometry(moved.shape[1:], crop_size)

    with torch.inference_mode():
        for start in range(0, moved.shape[0], batch_size):
            stop = min(start + batch_size, moved.shape[0])
            count = stop - start
            padded = np.zeros((count, *crop_size), dtype=np.float32)
            padded[(slice(None), *crop_slices)] = moved[(slice(start, stop), *source_slices)]
            tensor = torch.from_numpy(padded * 2.0 - 1.0).unsqueeze(1).to(device)
            prediction = generator(tensor, style.expand(count, -1))
            prediction = prediction.squeeze(1).float().cpu().numpy()
            prediction = np.clip(prediction, -1.0, 1.0) * 0.5 + 0.5
            restored = np.zeros((count, *moved.shape[1:]), dtype=np.float32)
            restored[(slice(None), *source_slices)] = prediction[(slice(None), *crop_slices)]
            output[start:stop] = restored
            del tensor, prediction, restored

    return np.moveaxis(output, 0, slice_axis)


def save_prediction_atomic(data, source_image, destination: Path) -> None:
    import nibabel as nib
    import numpy as np

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name.replace(".nii.gz", ".tmp.nii.gz"))
    header = source_image.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(data.astype(np.float32, copy=False), source_image.affine, header=header)
    nib.save(image, str(temporary))
    temporary.replace(destination)


def validate_prediction(
    path: Path,
    expected_shape: Sequence[int] = (364, 436, 364),
    reference_path: Path | None = None,
) -> dict[str, object]:
    import nibabel as nib
    import numpy as np

    image = nib.load(str(path))
    errors = []
    if tuple(image.shape) != tuple(expected_shape):
        errors.append(f"shape {image.shape}, expected {tuple(expected_shape)}")
    if str(image.get_data_dtype()) != "float32":
        errors.append(f"dtype {image.get_data_dtype()}, expected float32")
    if reference_path is not None:
        reference = nib.load(str(reference_path))
        if not np.allclose(image.affine, reference.affine, rtol=0.0, atol=1e-5):
            errors.append(f"affine differs from source {reference_path}")
        if image.header.get_zooms()[:3] != reference.header.get_zooms()[:3]:
            errors.append(f"voxel sizes differ from source {reference_path}")
        for code_name in ("qform_code", "sform_code"):
            if int(image.header[code_name]) != int(reference.header[code_name]):
                errors.append(f"{code_name} differs from source {reference_path}")
    data = image.get_fdata(dtype=np.float32)
    finite = bool(np.isfinite(data).all())
    minimum = float(data.min())
    maximum = float(data.max())
    if not finite:
        errors.append("contains NaN/Inf")
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        errors.append(f"range [{minimum}, {maximum}] outside [0,1]")
    return {
        "path": str(path),
        "shape": list(image.shape),
        "dtype": str(image.get_data_dtype()),
        "finite": finite,
        "minimum": minimum,
        "maximum": maximum,
        "affine_matches_source": not any("affine differs" in error for error in errors),
        "errors": errors,
    }


def run_one_case(case: InferenceCase, generator, mapping, config: dict, device, batch_size: int, seed: int):
    import nibabel as nib
    import numpy as np

    source = case.input_volume
    original_image = nib.load(str(source.path))
    canonical_image = nib.as_closest_canonical(original_image)
    canonical_data = canonical_image.get_fdata(dtype=np.float32)
    original_orientation = nib.io_orientation(original_image.affine)
    canonical_orientation = nib.io_orientation(canonical_image.affine)

    model_config = config["model"]
    data_config = config["data"]
    seed_key = (
        f"{seed}:{source.modality}:{source.source_field}:{case.target_field}:"
        f"{source.subject_id}"
    )
    style = deterministic_style(
        mapping,
        case.target_domain,
        model_config["latent_dim"],
        seed_key,
        device,
    )
    prediction = predict_volume_batched(
        generator,
        canonical_data,
        style,
        tuple(data_config["crop_size"]),
        int(data_config.get("slice_axis", 2)),
        batch_size,
        device,
    )
    transform = nib.orientations.ornt_transform(canonical_orientation, original_orientation)
    prediction = nib.orientations.apply_orientation(prediction, transform)
    source_data = original_image.get_fdata(dtype=np.float32)
    prediction *= (source_data > 1e-6).astype(np.float32)
    save_prediction_atomic(prediction, original_image, case.output_path)
    del canonical_data, prediction, source_data, style


def command_infer(args: argparse.Namespace) -> None:
    try:
        import nibabel  # noqa: F401
        import numpy  # noqa: F401
        import torch
        import yaml  # noqa: F401
    except ImportError as exc:
        raise PipelineError(
            "Inference dependencies are missing; install requirements-colab.txt and PyTorch"
        ) from exc

    official_repo = args.official_repo.resolve()
    checkpoint = args.checkpoint.resolve()
    verify_official_repo(official_repo, args.official_revision)
    if not checkpoint.is_file():
        raise PipelineError(f"Checkpoint not found: {checkpoint}")
    if args.batch_size < 1:
        raise PipelineError("--batch-size must be at least 1")

    volumes = discover_inputs(args.input_root.resolve(), args.modalities)
    all_cases = make_cases(
        volumes,
        args.output_dir.resolve(),
        sweep_mode=args.sweep_mode,
        sweep_epoch=args.sweep_epoch,
    )
    if args.smoke:
        if args.num_shards != 1 or args.shard_index != 0:
            raise PipelineError("--smoke cannot be combined with sharded inference")
        all_cases = [
            next(
                case
                for case in all_cases
                if case.input_volume.modality == args.modalities[0]
                and case.input_volume.source_field == "0.1T"
                and case.target_field == "1.5T"
                and case.input_volume.subject_id == "0001"
            )
        ]
    cases = select_case_shard(all_cases, args.num_shards, args.shard_index)

    device = resolve_device(args.device)
    if device.type == "cpu" and not args.allow_cpu:
        raise PipelineError(
            "Inference resolved to CPU and will be extremely slow. Use a GPU or pass "
            "--allow-cpu explicitly for a deliberate CPU smoke run."
        )
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    started = time.time()
    manifest_path = shard_manifest_path(
        args.output_dir.resolve(), args.num_shards, args.shard_index
    )
    manifest: dict[str, object] = {
        "status": "running",
        "started_at_utc": utc_now(),
        "official_repository": str(official_repo),
        "official_git_revision": args.official_revision,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_origin": (
            "official_huggingface_baseline"
            if checkpoint.as_posix().endswith(CHECKPOINT_FILE)
            else "local_or_user_supplied"
        ),
        "huggingface_repository": HF_REPO_ID,
        "huggingface_revision": HF_REVISION,
        "config": str(official_repo / CONFIG_RELATIVE_PATH),
        "input_root": str(args.input_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "sweep_mode": args.sweep_mode,
        "sweep_epoch": args.sweep_epoch,
        "modalities": list(args.modalities),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "smoke": args.smoke,
        "planned_cases": len(cases),
        "full_planned_cases": len(all_cases),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "completed_cases": [],
    }
    atomic_json(manifest_path, manifest)
    print(f"Loading one shared checkpoint on {device} ...")
    generator, mapping, config = load_shared_model(official_repo, checkpoint, device)
    print(f"Running {len(cases)} case(s); batch size {args.batch_size}")

    completed: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        if case.output_path.exists() and args.resume:
            validation = validate_prediction(
                case.output_path, reference_path=case.input_volume.path
            )
            if not validation["errors"]:
                print(f"[{index}/{len(cases)}] resume: {case.output_path}")
                completed.append({"output": str(case.output_path), "resumed": True})
                continue
            raise PipelineError(
                f"Existing resume output is invalid: {case.output_path}: "
                + "; ".join(validation["errors"])
            )
        if case.output_path.exists() and not args.overwrite:
            raise PipelineError(
                f"Output already exists: {case.output_path}; use --resume or --overwrite"
            )
        item = case.input_volume
        print(
            f"[{index}/{len(cases)}] {item.modality} {item.source_field}->{case.target_field} "
            f"ID {item.subject_id}, joint domain {case.target_domain}"
        )
        run_one_case(
            case,
            generator,
            mapping,
            config,
            device,
            args.batch_size,
            args.seed,
        )
        validation = validate_prediction(
            case.output_path, reference_path=case.input_volume.path
        )
        if validation["errors"]:
            raise PipelineError(
                f"New output failed validation: {case.output_path}: "
                + "; ".join(validation["errors"])
            )
        completed.append({"output": str(case.output_path), "resumed": False})
        manifest["completed_cases"] = completed
        atomic_json(manifest_path, manifest)

    manifest["status"] = "complete"
    manifest["completed_at_utc"] = utc_now()
    manifest["elapsed_seconds"] = round(time.time() - started, 3)
    manifest["completed_cases"] = completed
    atomic_json(manifest_path, manifest)
    print(f"Inference complete. Manifest: {manifest_path}")


def validate_cases(cases: Sequence[InferenceCase]) -> list[dict[str, object]]:
    missing = [str(case.output_path) for case in cases if not case.output_path.is_file()]
    if missing:
        raise PipelineError(
            f"Missing {len(missing)}/{len(cases)} expected outputs. First missing:\n  - "
            + "\n  - ".join(missing[:10])
        )
    reports = []
    failures = []
    for index, case in enumerate(cases, start=1):
        report = validate_prediction(
            case.output_path, reference_path=case.input_volume.path
        )
        reports.append(report)
        if report["errors"]:
            failures.append(f"{case.output_path}: " + "; ".join(report["errors"]))
        print(f"[{index:03d}/{len(cases):03d}] {case.output_path}: " + ("OK" if not report["errors"] else "FAIL"))
    if failures:
        raise PipelineError("Output validation failed:\n  - " + "\n  - ".join(failures[:20]))
    return reports


def command_validate_outputs(args: argparse.Namespace) -> None:
    volumes = discover_inputs(args.input_root.resolve(), args.modalities)
    cases = make_cases(
        volumes,
        args.output_dir.resolve(),
        sweep_mode=args.sweep_mode,
        sweep_epoch=args.sweep_epoch,
    )
    if args.smoke:
        cases = [
            next(
                case
                for case in cases
                if case.input_volume.modality == args.modalities[0]
                and case.input_volume.source_field == "0.1T"
                and case.target_field == "1.5T"
                and case.input_volume.subject_id == "0001"
            )
        ]
    reports = validate_cases(cases)
    result = {
        "created_at_utc": utc_now(),
        "kind": "smoke" if args.smoke else "full_volume_inference",
        "modalities": list(args.modalities),
        "expected_files": len(cases),
        "validated_files": len(reports),
        "failures": [],
        "files": reports,
    }
    atomic_json(args.report.resolve(), result)
    print(f"Validated {len(reports)} outputs. Report: {args.report.resolve()}")


def command_finalize_shards(args: argparse.Namespace) -> None:
    if args.num_shards < 2:
        raise PipelineError("finalize-shards requires --num-shards of at least 2")

    output_dir = args.output_dir.resolve()
    volumes = discover_inputs(args.input_root.resolve(), args.modalities)
    cases = make_cases(
        volumes,
        output_dir,
        sweep_mode=args.sweep_mode,
        sweep_epoch=args.sweep_epoch,
    )
    manifests: list[dict[str, object]] = []
    manifest_paths: list[Path] = []
    completed_by_output: dict[str, dict[str, object]] = {}
    provenance_keys = (
        "official_git_revision",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_origin",
        "huggingface_repository",
        "huggingface_revision",
        "config",
        "input_root",
        "output_dir",
        "sweep_mode",
        "sweep_epoch",
        "modalities",
        "torch_version",
        "cuda_version",
        "seed",
        "batch_size",
        "smoke",
    )

    for shard_index in range(args.num_shards):
        path = shard_manifest_path(output_dir, args.num_shards, shard_index)
        if not path.is_file():
            raise PipelineError(f"Missing shard manifest: {path}")
        manifest = json.loads(path.read_text())
        expected_cases = select_case_shard(cases, args.num_shards, shard_index)
        expected_outputs = [str(case.output_path) for case in expected_cases]
        completed = manifest.get("completed_cases")
        completed_outputs = (
            [str(item.get("output")) for item in completed]
            if isinstance(completed, list)
            and all(isinstance(item, dict) for item in completed)
            else []
        )
        checks = {
            "status": manifest.get("status") == "complete",
            "not_smoke": manifest.get("smoke") is False,
            "num_shards": manifest.get("num_shards") == args.num_shards,
            "shard_index": manifest.get("shard_index") == shard_index,
            "planned_cases": manifest.get("planned_cases") == len(expected_cases),
            "full_planned_cases": manifest.get("full_planned_cases") == len(cases),
            "modalities": manifest.get("modalities") == list(args.modalities),
            "sweep_mode": manifest.get("sweep_mode") == args.sweep_mode,
            "sweep_epoch": manifest.get("sweep_epoch") == args.sweep_epoch,
            "completed_outputs": completed_outputs == expected_outputs,
        }
        failures = [name for name, passed in checks.items() if not passed]
        if failures:
            raise PipelineError(
                f"Invalid shard manifest {path}; failed: " + ", ".join(failures)
            )
        if manifests:
            mismatched = [
                key
                for key in provenance_keys
                if manifest.get(key) != manifests[0].get(key)
            ]
            if mismatched:
                raise PipelineError(
                    f"Shard provenance differs in {path}: " + ", ".join(mismatched)
                )
        for item in completed:
            output = str(item["output"])
            if output in completed_by_output:
                raise PipelineError(f"Duplicate output across shard manifests: {output}")
            completed_by_output[output] = item
        manifests.append(manifest)
        manifest_paths.append(path)

    expected_outputs = [str(case.output_path) for case in cases]
    if set(completed_by_output) != set(expected_outputs):
        raise PipelineError("Shard manifests do not cover the complete planned sweep")

    reports = validate_cases(cases)
    report = {
        "created_at_utc": utc_now(),
        "kind": "full_volume_inference",
        "modalities": list(args.modalities),
        "expected_files": len(cases),
        "validated_files": len(reports),
        "failures": [],
        "files": reports,
    }
    atomic_json(args.report.resolve(), report)

    combined = dict(manifests[0])
    combined.update(
        status="complete",
        completed_at_utc=utc_now(),
        elapsed_seconds=max(float(item["elapsed_seconds"]) for item in manifests),
        planned_cases=len(cases),
        full_planned_cases=len(cases),
        num_shards=args.num_shards,
        completed_cases=[completed_by_output[output] for output in expected_outputs],
        devices=[item.get("device") for item in manifests],
        gpu_names=[item.get("gpu_name") for item in manifests],
        worker_elapsed_seconds=[item.get("elapsed_seconds") for item in manifests],
        worker_manifests=[str(path) for path in manifest_paths],
        finalized_at_utc=utc_now(),
    )
    combined.pop("shard_index", None)
    canonical_manifest = output_dir / "task3_run_manifest.json"
    atomic_json(canonical_manifest, combined)
    print(f"Validated {len(reports)} outputs. Report: {args.report.resolve()}")
    print(f"Finalized dual-GPU manifest: {canonical_manifest}")


def add_modalities(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=MODALITIES,
        default=["T1W"],
        help="Modalities to process (default: T1W)",
    )


def sweep_component(value: str) -> str:
    if not SWEEP_COMPONENT_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "sweep labels must contain only letters, digits, dot, underscore, or hyphen"
        )
    return value


def add_sweep_coordinates(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sweep-mode",
        type=sweep_component,
        default="pro_pretrained",
        help="output/builder mode label (default: pro_pretrained)",
    )
    parser.add_argument(
        "--sweep-epoch",
        type=sweep_component,
        default="epoch50",
        help="output/builder checkpoint label (default: epoch50)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch-official", help="clone and pin organizer code")
    fetch.add_argument("--destination", type=Path, default=Path("official/MRIxFields2026"))
    fetch.add_argument("--revision", default=OFFICIAL_GIT_REVISION)
    fetch.set_defaults(func=command_fetch_official)

    download = subparsers.add_parser("download-checkpoint", help="download only Task 3 weights")
    download.add_argument("--destination", type=Path, default=Path("weights"))
    download.set_defaults(func=command_download_checkpoint)

    audit = subparsers.add_parser("audit", help="audit validation inputs and NIfTI headers")
    audit.add_argument("--input-root", type=Path, required=True)
    add_modalities(audit)
    audit.add_argument(
        "--check-values",
        action="store_true",
        help="also load each full volume to check finite [0,1] values",
    )
    audit.add_argument("--report", type=Path, default=Path("runs/input_audit.json"))
    audit.set_defaults(func=command_audit)

    plan = subparsers.add_parser("plan", help="show cases/output paths without inference")
    plan.add_argument("--input-root", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, default=Path("runs/inference"))
    add_modalities(plan)
    add_sweep_coordinates(plan)
    plan.set_defaults(func=command_plan)

    infer = subparsers.add_parser("infer", help="run smoke or validation inference")
    infer.add_argument("--input-root", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, default=Path("runs/inference"))
    infer.add_argument("--official-repo", type=Path, default=Path("official/MRIxFields2026"))
    infer.add_argument("--official-revision", default=OFFICIAL_GIT_REVISION)
    infer.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("weights") / CHECKPOINT_FILE,
    )
    add_modalities(infer)
    add_sweep_coordinates(infer)
    infer.add_argument("--device", default="auto", help="auto, cuda, cuda:0, mps, or cpu")
    infer.add_argument("--batch-size", type=int, default=1)
    infer.add_argument("--seed", type=int, default=20260712)
    infer.add_argument("--num-shards", type=int, default=1)
    infer.add_argument("--shard-index", type=int, default=0)
    infer.add_argument(
        "--smoke",
        action="store_true",
        help="run only 0.1T->1.5T ID 0001 for the first requested modality",
    )
    infer.add_argument("--resume", action="store_true", help="validate and skip existing outputs")
    infer.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    infer.add_argument("--allow-cpu", action="store_true")
    infer.set_defaults(func=command_infer)

    validate = subparsers.add_parser("validate-outputs", help="validate full-volume predictions")
    validate.add_argument("--input-root", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, default=Path("runs/inference"))
    add_modalities(validate)
    add_sweep_coordinates(validate)
    validate.add_argument("--smoke", action="store_true")
    validate.add_argument(
        "--report", type=Path, default=Path("runs/output_validation.json")
    )
    validate.set_defaults(func=command_validate_outputs)

    finalize = subparsers.add_parser(
        "finalize-shards",
        help="verify worker manifests and all outputs, then write the canonical manifest",
    )
    finalize.add_argument("--input-root", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, default=Path("runs/inference"))
    finalize.add_argument("--num-shards", type=int, required=True)
    add_modalities(finalize)
    add_sweep_coordinates(finalize)
    finalize.add_argument(
        "--report", type=Path, default=Path("runs/output_validation.json")
    )
    finalize.set_defaults(func=command_finalize_shards)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except PipelineError as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
