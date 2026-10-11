"""Gaussian sliding-window inference for one final shared Task 3 checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import nibabel as nib
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .common import (
    EXPECTED_VALIDATION_IDS,
    EXPECTED_VOLUME_SHAPE,
    FIELDS,
    MODALITIES,
    Task3Error,
    atomic_json,
    parse_nifti_name,
    resolve_split,
    sha256_file,
    utc_now,
)
from .model import ModelConfig, build_model
from .train import load_checkpoint, select_device


@dataclass(frozen=True)
class InputVolume:
    path: Path
    modality: str
    source_field: str
    subject_id: str


@dataclass(frozen=True)
class InferenceCase:
    source: InputVolume
    target_field: str
    destination: Path


def sliding_starts(size: int, window: int, overlap: float) -> tuple[int, ...]:
    if size < 1 or window < 1 or not 0.0 <= overlap < 1.0:
        raise Task3Error("Invalid sliding-window geometry")
    if size <= window:
        return (0,)
    stride = max(1, int(round(window * (1.0 - overlap))))
    starts = list(range(0, size - window + 1, stride))
    if starts[-1] != size - window:
        starts.append(size - window)
    return tuple(starts)


def gaussian_weight(window: int, device: torch.device) -> Tensor:
    coordinate = torch.linspace(-1.0, 1.0, window, device=device, dtype=torch.float32)
    vector = torch.exp(-0.5 * (coordinate / 0.25).square()).clamp_min(1e-4)
    weight = vector[:, None, None] * vector[None, :, None] * vector[None, None, :]
    return weight.clamp_min(1e-3)[None, None]


def inference_context(device: torch.device, amp: str):
    if amp == "auto":
        amp = "bf16" if device.type == "cuda" else "off"
    if amp == "off":
        return contextlib.nullcontext()
    if device.type != "cuda":
        raise Task3Error("Inference mixed precision is supported only on CUDA")
    return torch.autocast(
        device_type="cuda", dtype=torch.float16 if amp == "fp16" else torch.bfloat16
    )


@torch.inference_mode()
def sliding_window_predict(
    model: torch.nn.Module,
    volume: np.ndarray | Tensor,
    *,
    modality: int,
    source_field: int,
    target_field: int,
    window: int = 128,
    overlap: float = 0.25,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
    amp: str = "off",
) -> np.ndarray:
    if source_field == target_field:
        raise Task3Error("Inference must reject identity mappings")
    device = torch.device(device)
    if isinstance(volume, np.ndarray):
        source = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))
    else:
        source = volume.detach().to(dtype=torch.float32, device="cpu")
    if source.ndim != 3:
        raise Task3Error(f"Expected a 3D volume, found {tuple(source.shape)}")
    original_shape = tuple(int(value) for value in source.shape)
    padding = tuple(max(0, window - size) for size in original_shape)
    source = source[None, None]
    if any(padding):
        source = F.pad(source, (0, padding[2], 0, padding[1], 0, padding[0]), mode="replicate")
    source = source.to(device)
    padded_shape = tuple(int(value) for value in source.shape[-3:])
    coordinates = [
        (z, y, x)
        for z in sliding_starts(padded_shape[0], window, overlap)
        for y in sliding_starts(padded_shape[1], window, overlap)
        for x in sliding_starts(padded_shape[2], window, overlap)
    ]
    output = torch.zeros_like(source, dtype=torch.float32)
    normalization = torch.zeros_like(source, dtype=torch.float32)
    weight = gaussian_weight(window, device)
    for offset in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[offset : offset + batch_size]
        patches = torch.cat(
            [source[:, :, z : z + window, y : y + window, x : x + window] for z, y, x in batch_coordinates],
            dim=0,
        )
        count = patches.shape[0]
        modalities = torch.full((count,), modality, dtype=torch.long, device=device)
        sources = torch.full((count,), source_field, dtype=torch.long, device=device)
        targets = torch.full((count,), target_field, dtype=torch.long, device=device)
        with inference_context(device, amp):
            predictions = model(patches, modalities, sources, targets)
        predictions = predictions.float()
        if not bool(torch.isfinite(predictions).all()):
            raise Task3Error("Model produced NaN or Inf during sliding-window inference")
        for prediction, (z, y, x) in zip(predictions, batch_coordinates):
            output[:, :, z : z + window, y : y + window, x : x + window] += prediction[None] * weight
            normalization[:, :, z : z + window, y : y + window, x : x + window] += weight
    output = output / normalization.clamp_min(1e-8)
    d, h, w = original_shape
    return output[0, 0, :d, :h, :w].clamp(0.0, 1.0).cpu().numpy().astype(np.float32)


def resolve_validation_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if all((root / modality).is_dir() for modality in MODALITIES):
        return root
    return resolve_split(root, "pro_val")


def inventory_inputs(root: Path, *, require_official_ids: bool = True) -> list[InputVolume]:
    root = resolve_validation_root(root)
    volumes: list[InputVolume] = []
    failures: list[str] = []
    for modality in MODALITIES:
        for field in FIELDS:
            directory = root / modality / field
            records: list[InputVolume] = []
            for path in sorted(directory.glob("*.nii.gz")) if directory.is_dir() else []:
                try:
                    parsed = parse_nifti_name(path)
                except Task3Error as exc:
                    failures.append(str(exc))
                    continue
                if (
                    parsed["prefix"] != "P"
                    or parsed["modality"] != modality
                    or parsed["field"] != field
                ):
                    failures.append(f"Validation path/filename mismatch: {path}")
                    continue
                records.append(InputVolume(path, modality, field, parsed["subject_id"]))
            found = tuple(record.subject_id.removeprefix("P_") for record in records)
            if require_official_ids and found != EXPECTED_VALIDATION_IDS[field]:
                failures.append(f"{modality}/{field}: IDs={found}, expected={EXPECTED_VALIDATION_IDS[field]}")
            elif not records:
                failures.append(f"No validation inputs in {directory}")
            volumes.extend(records)
    if failures:
        raise Task3Error("Validation input audit failed:\n- " + "\n- ".join(failures))
    return volumes


def prediction_path(
    output_root: Path, source: InputVolume, target_field: str, *, method: str, mode: str, epoch: str
) -> Path:
    if source.source_field == target_field:
        raise Task3Error("Inference must not create identity mappings")
    pair = f"{source.source_field}_to_{target_field}"
    return output_root / f"task3_{pair}_{source.modality}" / method / mode / epoch / source.path.name


def plan_cases(
    inputs: Iterable[InputVolume], output_root: Path, *, method: str, mode: str, epoch: str
) -> list[InferenceCase]:
    return [
        InferenceCase(
            source,
            target,
            prediction_path(output_root, source, target, method=method, mode=mode, epoch=epoch),
        )
        for source in inputs
        for target in FIELDS
        if target != source.source_field
    ]


def save_nifti_atomic(data: np.ndarray, reference: nib.spatialimages.SpatialImage, path: Path) -> None:
    if data.dtype != np.float32 or not np.isfinite(data).all():
        raise Task3Error("Atomic NIfTI output must be finite float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        nib.save(nib.Nifti1Image(data, reference.affine, header=header), str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_output(path: Path, reference_path: Path) -> dict[str, Any]:
    image, reference = nib.load(str(path)), nib.load(str(reference_path))
    values = np.asanyarray(image.dataobj)
    failures: list[str] = []
    if image.shape != reference.shape:
        failures.append(f"shape={image.shape}, reference={reference.shape}")
    if np.dtype(image.get_data_dtype()) != np.dtype(np.float32):
        failures.append(f"dtype={image.get_data_dtype()}, expected=float32")
    if not np.allclose(image.affine, reference.affine, rtol=0.0, atol=1e-5):
        failures.append("affine differs from source")
    for code in ("qform_code", "sform_code"):
        if int(image.header[code]) != int(reference.header[code]):
            failures.append(f"{code} differs from source")
    finite = bool(np.isfinite(values).all())
    minimum, maximum = float(values.min()), float(values.max())
    if not finite:
        failures.append("contains NaN/Inf")
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        failures.append(f"range=[{minimum},{maximum}] outside [0,1]")
    return {
        "path": str(path),
        "shape": list(image.shape),
        "dtype": str(image.get_data_dtype()),
        "finite": finite,
        "minimum": minimum,
        "maximum": maximum,
        "failures": failures,
    }


def run_case(
    case: InferenceCase,
    model: torch.nn.Module,
    *,
    device: torch.device,
    amp: str,
    window: int,
    overlap: float,
    batch_size: int,
) -> dict[str, Any]:
    original = nib.load(str(case.source.path))
    if original.shape != EXPECTED_VOLUME_SHAPE:
        raise Task3Error(f"Unexpected validation shape {original.shape}: {case.source.path}")
    canonical = nib.as_closest_canonical(original)
    canonical_values = canonical.get_fdata(dtype=np.float32)
    if not np.isfinite(canonical_values).all():
        raise Task3Error(f"Validation source contains NaN/Inf: {case.source.path}")
    prediction = sliding_window_predict(
        model,
        canonical_values,
        modality=MODALITIES.index(case.source.modality),
        source_field=FIELDS.index(case.source.source_field),
        target_field=FIELDS.index(case.target_field),
        window=window,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
        amp=amp,
    )
    transform = nib.orientations.ornt_transform(
        nib.io_orientation(canonical.affine), nib.io_orientation(original.affine)
    )
    restored = np.ascontiguousarray(
        nib.orientations.apply_orientation(prediction, transform), dtype=np.float32
    )
    save_nifti_atomic(restored, original, case.destination)
    report = validate_output(case.destination, case.source.path)
    if report["failures"]:
        raise Task3Error(f"New prediction failed validation: {report}")
    return report


def infer(args: argparse.Namespace) -> None:
    if not args.allow_nonstandard_ids and (args.window != 128 or args.overlap != 0.25):
        raise Task3Error("Official validation inference requires 128^3 windows with 25% overlap")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    split = checkpoint["split_manifest"]
    if (
        split.get("phase") != "all_subject_retrain"
        or set(split.get("train_subjects", [])) != {"P_0006", "P_0007", "P_0009"}
        or split.get("holdout_subject") is not None
    ):
        raise Task3Error("Validation inference requires the fresh all-subject retraining checkpoint")
    config = ModelConfig.from_dict(checkpoint["model_config"])
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if not args.smoke:
        if args.smoke_manifest is None or not args.smoke_manifest.is_file():
            raise Task3Error("Full inference requires the completed --smoke-manifest")
        smoke = json.loads(args.smoke_manifest.read_text(encoding="utf-8"))
        if (
            smoke.get("status") != "complete"
            or smoke.get("smoke") is not True
            or smoke.get("planned_cases") != 1
            or len(smoke.get("completed_cases", [])) != 1
            or smoke.get("checkpoint_sha256") != checkpoint_sha256
            or smoke.get("model_config") != config.to_dict()
        ):
            raise Task3Error("Smoke manifest does not prove a matching valid full-volume translation")
    model = build_model(config)
    model.load_state_dict(checkpoint["model_ema"], strict=True)
    device = select_device(args.device)
    model.to(device).eval()
    input_root = resolve_validation_root(args.input_root)
    inputs = inventory_inputs(input_root, require_official_ids=not args.allow_nonstandard_ids)
    output_root = args.output_dir.expanduser().resolve()
    cases = plan_cases(inputs, output_root, method=args.method, mode=args.mode, epoch=args.epoch)
    if args.smoke:
        cases = cases[:1]
    expected = 1 if args.smoke else 180
    if not args.allow_nonstandard_ids and len(cases) != expected:
        raise Task3Error(f"Planned {len(cases)} cases, expected {expected}")
    manifest_path = output_root / "task3_3d_unet_inference_manifest.json"
    manifest: dict[str, Any] = {
        "kind": "task3_3d_unet_inference",
        "created_at_utc": utc_now(),
        "status": "running",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_weights": "model_ema",
        "model_config": config.to_dict(),
        "split_manifest": checkpoint["split_manifest"],
        "training_run_signature": checkpoint["run_signature"],
        "input_root": str(input_root),
        "method": args.method,
        "mode": args.mode,
        "epoch": args.epoch,
        "window": args.window,
        "overlap": args.overlap,
        "window_batch_size": args.batch_size,
        "smoke": args.smoke,
        "planned_cases": len(cases),
        "completed_cases": [],
    }
    if manifest_path.is_file():
        if not args.resume:
            raise Task3Error(f"Inference manifest exists; use a verified resume: {manifest_path}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable = (
            "kind",
            "checkpoint_sha256",
            "model_config",
            "split_manifest",
            "training_run_signature",
            "input_root",
            "method",
            "mode",
            "epoch",
            "window",
            "overlap",
            "window_batch_size",
            "smoke",
            "planned_cases",
        )
        if not isinstance(existing, dict) or any(
            existing.get(key) != manifest.get(key) for key in immutable
        ):
            raise Task3Error("Existing inference manifest belongs to a different run")
        manifest["created_at_utc"] = existing.get("created_at_utc", manifest["created_at_utc"])
        manifest["resumed_at_utc"] = utc_now()
    elif any(case.destination.exists() for case in cases):
        raise Task3Error("Prediction files exist without a matching resume manifest")
    atomic_json(manifest_path, manifest)
    started = time.monotonic()
    for index, case in enumerate(cases, 1):
        if case.destination.exists():
            if not args.resume:
                raise Task3Error(f"Output exists; use --resume only after validation: {case.destination}")
            report = validate_output(case.destination, case.source.path)
            if report["failures"]:
                raise Task3Error(f"Existing output failed resume validation: {report}")
            resumed = True
        else:
            report = run_case(
                case,
                model,
                device=device,
                amp=args.amp,
                window=args.window,
                overlap=args.overlap,
                batch_size=args.batch_size,
            )
            resumed = False
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
        print(f"[{index:03d}/{len(cases):03d}] {case.source.modality} {case.source.source_field}->{case.target_field} OK")
    manifest.update(
        status="complete",
        completed_at_utc=utc_now(),
        elapsed_seconds=time.monotonic() - started,
    )
    atomic_json(manifest_path, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--method", default="unet3d")
    parser.add_argument("--mode", default="pro_final")
    parser.add_argument("--epoch", default="best")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-nonstandard-ids", action="store_true", help="test-only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.batch_size < 1:
            raise Task3Error("batch-size must be positive")
        infer(args)
    except (Task3Error, RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
