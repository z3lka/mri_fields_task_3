#!/usr/bin/env python3
"""Full-volume Task 3 inference with one conditional FPS-Former checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import torch

from task3_fpsformer.common import (
    EXPECTED_VOLUME_SHAPE,
    FIELDS,
    MODALITIES,
    PAPER_ID,
    FPSFormerError,
    atomic_json,
    parse_nifti_name,
    prediction_path,
    resolve_split,
    sha256_file,
    utc_now,
)
from task3_fpsformer.model import (
    ARCHITECTURE_ID,
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
    ModelConfig,
    build_model,
)
from task3_fpsformer.train import load_checkpoint, select_device


EXPECTED_VALIDATION_IDS = {
    "0.1T": ("0001", "0002", "0003"),
    "1.5T": ("0004", "0005", "0008"),
    "3T": ("0010", "0011", "0012"),
    "5T": ("0013", "0014", "0015"),
    "7T": ("0016", "0017", "0018"),
}


@dataclass(frozen=True)
class InputVolume:
    path: Path
    modality: str
    source_field: str
    subject: str


@dataclass(frozen=True)
class InferenceCase:
    source: InputVolume
    target_field: str
    destination: Path


def resolve_validation_root(input_root: Path) -> Path:
    root = input_root.expanduser().resolve()
    if all((root / modality).is_dir() for modality in MODALITIES):
        return root
    return resolve_split(root, "pro_val")


def inventory_inputs(input_root: Path, *, require_official_ids: bool = True) -> list[InputVolume]:
    root = resolve_validation_root(input_root)
    volumes: list[InputVolume] = []
    failures: list[str] = []
    for modality in MODALITIES:
        for field in FIELDS:
            directory = root / modality / field
            paths = sorted(directory.glob("*.nii.gz")) if directory.is_dir() else []
            parsed_ids: list[str] = []
            for path in paths:
                try:
                    parsed = parse_nifti_name(path)
                except FPSFormerError as exc:
                    failures.append(str(exc))
                    continue
                if parsed["prefix"] != "P":
                    failures.append(f"Validation input is not prospective: {path.name}")
                    continue
                if parsed["modality"] != modality or parsed["field"] != field:
                    failures.append(f"Validation path/filename mismatch: {path}")
                    continue
                parsed_ids.append(str(parsed["subject"]))
                volumes.append(
                    InputVolume(
                        path=path,
                        modality=modality,
                        source_field=field,
                        subject=str(parsed["subject_id"]),
                    )
                )
            if require_official_ids and tuple(parsed_ids) != EXPECTED_VALIDATION_IDS[field]:
                failures.append(
                    f"{modality}/{field}: IDs {tuple(parsed_ids)}, "
                    f"expected {EXPECTED_VALIDATION_IDS[field]}"
                )
            elif not paths:
                failures.append(f"No inputs in {directory}")
    if failures:
        raise FPSFormerError("Validation input audit failed:\n- " + "\n- ".join(failures))
    return volumes


def plan_cases(
    volumes: Iterable[InputVolume],
    output_root: Path,
    *,
    method: str,
    mode: str,
    epoch_tag: str,
    target_fields: tuple[str, ...],
) -> list[InferenceCase]:
    cases: list[InferenceCase] = []
    for volume in volumes:
        for target_field in target_fields:
            if target_field == volume.source_field:
                continue
            cases.append(
                InferenceCase(
                    source=volume,
                    target_field=target_field,
                    destination=prediction_path(
                        output_root,
                        modality=volume.modality,
                        source_field=volume.source_field,
                        target_field=target_field,
                        source_name=volume.path.name,
                        method=method,
                        mode=mode,
                        epoch_tag=epoch_tag,
                    ),
                )
            )
    return cases


def inference_context(device: torch.device, amp: str) -> Any:
    if amp == "auto":
        amp = "fp16" if device.type == "cuda" else "off"
    if amp == "off":
        return contextlib.nullcontext()
    if device.type != "cuda":
        raise FPSFormerError(f"Inference AMP {amp} is only supported on CUDA")
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def predict_volume(
    model: torch.nn.Module,
    volume: np.ndarray,
    *,
    modality: int,
    source_field: int,
    target_field: int,
    batch_size: int,
    device: torch.device,
    amp: str,
) -> np.ndarray:
    moved = np.moveaxis(volume.astype(np.float32, copy=False), 2, 0)
    output = np.empty_like(moved, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, moved.shape[0], batch_size):
            stop = min(start + batch_size, moved.shape[0])
            count = stop - start
            source = torch.from_numpy(
                np.ascontiguousarray(moved[start:stop])
            ).unsqueeze(1).to(device)
            modality_tensor = torch.full((count,), modality, dtype=torch.long, device=device)
            source_tensor = torch.full(
                (count,), source_field, dtype=torch.long, device=device
            )
            target_tensor = torch.full(
                (count,), target_field, dtype=torch.long, device=device
            )
            with inference_context(device, amp):
                prediction = model(
                    source, modality_tensor, source_tensor, target_tensor
                )
            prediction_np = prediction.squeeze(1).float().cpu().numpy()
            if not np.isfinite(prediction_np).all():
                raise FPSFormerError(
                    f"Model produced NaN/Inf in slice batch [{start}, {stop})"
                )
            output[start:stop] = np.clip(prediction_np, 0.0, 1.0)
    return np.moveaxis(output, 0, 2)


def save_nifti_atomic(data: np.ndarray, reference: nib.spatialimages.SpatialImage, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    with tempfile.NamedTemporaryFile(
        suffix=".nii.gz", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        output = nib.Nifti1Image(
            data.astype(np.float32, copy=False), reference.affine, header=header
        )
        nib.save(output, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_output(path: Path, source_path: Path) -> dict[str, Any]:
    image = nib.load(str(path))
    source = nib.load(str(source_path))
    failures: list[str] = []
    if tuple(image.shape) != EXPECTED_VOLUME_SHAPE:
        failures.append(f"shape {image.shape}, expected {EXPECTED_VOLUME_SHAPE}")
    if str(image.get_data_dtype()) != "float32":
        failures.append(f"dtype {image.get_data_dtype()}, expected float32")
    if not np.allclose(image.affine, source.affine, rtol=0.0, atol=1e-5):
        failures.append("affine differs from source")
    for code in ("qform_code", "sform_code"):
        if int(image.header[code]) != int(source.header[code]):
            failures.append(f"{code} differs from source")
    data = image.get_fdata(dtype=np.float32)
    finite = bool(np.isfinite(data).all())
    minimum = float(data.min())
    maximum = float(data.max())
    if not finite:
        failures.append("contains NaN/Inf")
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        failures.append(f"range [{minimum}, {maximum}] is outside [0,1]")
    return {
        "path": str(path),
        "shape": list(image.shape),
        "dtype": str(image.get_data_dtype()),
        "minimum": minimum,
        "maximum": maximum,
        "finite": finite,
        "failures": failures,
    }


def run_case(
    case: InferenceCase,
    model: torch.nn.Module,
    *,
    batch_size: int,
    device: torch.device,
    amp: str,
    mask_background: bool,
) -> dict[str, Any]:
    original = nib.load(str(case.source.path))
    if tuple(original.shape) != EXPECTED_VOLUME_SHAPE:
        raise FPSFormerError(f"Unexpected source shape {original.shape}: {case.source.path}")
    canonical = nib.as_closest_canonical(original)
    canonical_data = canonical.get_fdata(dtype=np.float32)
    if not np.isfinite(canonical_data).all():
        raise FPSFormerError(f"Source contains NaN/Inf: {case.source.path}")
    prediction = predict_volume(
        model,
        canonical_data,
        modality=MODALITIES.index(case.source.modality),
        source_field=FIELDS.index(case.source.source_field),
        target_field=FIELDS.index(case.target_field),
        batch_size=batch_size,
        device=device,
        amp=amp,
    )
    original_orientation = nib.io_orientation(original.affine)
    canonical_orientation = nib.io_orientation(canonical.affine)
    transform = nib.orientations.ornt_transform(canonical_orientation, original_orientation)
    prediction = nib.orientations.apply_orientation(prediction, transform)
    if mask_background:
        source_data = original.get_fdata(dtype=np.float32)
        prediction *= (source_data > 1e-6).astype(np.float32)
    save_nifti_atomic(prediction, original, case.destination)
    report = validate_output(case.destination, case.source.path)
    if report["failures"]:
        raise FPSFormerError(
            f"New prediction failed validation: {case.destination}: {report['failures']}"
        )
    return report


def infer(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise FPSFormerError("--batch-size must be positive")
    device = select_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FPSFormerError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint.get("format") != "task3_conditional_fpsformer_v1":
        raise FPSFormerError(f"Unexpected checkpoint format: {checkpoint.get('format')!r}")
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    checkpoint_architecture = checkpoint.get("architecture", ARCHITECTURE_ID)
    upstream_reference = checkpoint.get(
        "upstream_reference",
        {"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION},
    )
    del checkpoint

    volumes = inventory_inputs(
        args.input_root, require_official_ids=not args.allow_nonstandard_ids
    )
    target_fields = tuple(args.target_fields or FIELDS)
    cases = plan_cases(
        volumes,
        args.output_dir.expanduser().resolve(),
        method=args.method,
        mode=args.mode,
        epoch_tag=args.epoch_tag,
        target_fields=target_fields,
    )
    if args.smoke:
        cases = cases[:1]
    expected = 1 if args.smoke else (
        len(MODALITIES)
        * 3
        * sum(
            1
            for source_field in FIELDS
            for target_field in target_fields
            if source_field != target_field
        )
    )
    if not args.allow_nonstandard_ids and len(cases) != expected:
        raise FPSFormerError(f"Planned {len(cases)} cases, expected {expected}")

    manifest_path = args.output_dir.expanduser().resolve() / "task3_fpsformer_manifest.json"
    manifest: dict[str, Any] = {
        "kind": "task3_fpsformer_inference",
        "created_at_utc": utc_now(),
        "status": "running",
        "paper": PAPER_ID,
        "architecture": checkpoint_architecture,
        "upstream_reference": upstream_reference,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "model_config": config.to_dict(),
        "device": str(device),
        "input_root": str(resolve_validation_root(args.input_root)),
        "output_root": str(args.output_dir.expanduser().resolve()),
        "method": args.method,
        "mode": args.mode,
        "epoch_tag": args.epoch_tag,
        "smoke": args.smoke,
        "planned_cases": len(cases),
        "completed_cases": [],
    }
    atomic_json(manifest_path, manifest)
    started = time.monotonic()
    for index, case in enumerate(cases, start=1):
        if case.destination.exists():
            if not args.resume:
                raise FPSFormerError(
                    f"Output exists: {case.destination}; use --resume after validating it"
                )
            report = validate_output(case.destination, case.source.path)
            if report["failures"]:
                raise FPSFormerError(
                    f"Existing resume output is invalid: {case.destination}: "
                    f"{report['failures']}"
                )
            resumed = True
        else:
            report = run_case(
                case,
                model,
                batch_size=args.batch_size,
                device=device,
                amp=args.amp,
                mask_background=not args.no_background_mask,
            )
            resumed = False
        print(
            f"[{index:03d}/{len(cases):03d}] {case.source.modality} "
            f"{case.source.source_field}->{case.target_field} {case.source.subject}: OK"
        )
        manifest["completed_cases"].append(
            {
                "source": str(case.source.path),
                "target_field": case.target_field,
                "output": str(case.destination),
                "resumed": resumed,
                "minimum": report["minimum"],
                "maximum": report["maximum"],
            }
        )
        atomic_json(manifest_path, manifest)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = utc_now()
    manifest["elapsed_seconds"] = time.monotonic() - started
    atomic_json(manifest_path, manifest)
    print(f"Inference complete: {len(cases)} validated volumes; manifest={manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one conditional FPS-Former across the Task 3 validation sweep"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--method", default="fpsformer")
    parser.add_argument("--mode", default="pro_scratch")
    parser.add_argument("--epoch-tag", default="best")
    parser.add_argument("--target-fields", nargs="+", choices=FIELDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-background-mask", action="store_true")
    parser.add_argument("--allow-nonstandard-ids", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        infer(args)
    except (FPSFormerError, RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
