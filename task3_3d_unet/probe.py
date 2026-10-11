"""A100 VRAM/timing gate for the complete 3D U-Net experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .common import TRAIN_SUBJECTS, Task3Error, atomic_json, utc_now
from .data import ProspectivePatchDataset
from .losses import fine_tuning_loss
from .metrics import load_official_scorer, score_arrays
from .model import ModelConfig, build_model
from .train import Amp


def window_count(shape: tuple[int, int, int], window: int, overlap: float) -> int:
    stride = max(1, int(round(window * (1.0 - overlap))))
    return math.prod(max(1, math.ceil(max(0, size - window) / stride) + 1) for size in shape)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.updates < 1 or args.inference_repeats < 1:
        raise Task3Error("Probe update and inference repeat counts must be positive")
    if not torch.cuda.is_available():
        raise Task3Error("The paid-runtime probe requires CUDA")
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    properties = torch.cuda.get_device_properties(device)
    total_bytes = int(properties.total_memory)
    dataset = ProspectivePatchDataset(
        args.cache_dir,
        TRAIN_SUBJECTS,
        patch_size=args.patch_size,
        samples_per_stratum=math.ceil(args.updates * args.batch_size * args.gradient_accumulation / 60),
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    iterator = iter(loader)
    model = build_model(ModelConfig()).to(device).train()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    amp = Amp(device, "bf16")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.gradient_accumulation):
            batch = next(iterator)
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            modality = batch["modality"].to(device)
            source_field = batch["source_field"].to(device)
            target_field = batch["target_field"].to(device)
            with amp.context():
                prediction = model(source, modality, source_field, target_field)
                loss, _ = fine_tuning_loss(prediction, target, masked=True)
            (loss / args.gradient_accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    training_seconds = time.monotonic() - started
    model.eval()
    inference_source = torch.rand(
        args.inference_batch_size, 1, args.inference_window, args.inference_window, args.inference_window,
        device=device,
    )
    labels = torch.zeros(args.inference_batch_size, dtype=torch.long, device=device)
    targets = torch.ones_like(labels)
    with torch.inference_mode(), amp.context():
        model(inference_source, labels, labels, targets)
    torch.cuda.synchronize()
    inference_started = time.monotonic()
    with torch.inference_mode():
        for _ in range(args.inference_repeats):
            with amp.context():
                model(inference_source, labels, labels, targets)
    torch.cuda.synchronize()
    inference_seconds_per_batch = (time.monotonic() - inference_started) / args.inference_repeats

    with torch.inference_mode(), amp.context():
        evaluation_prediction = model(source[:1], modality[:1], source_field[:1], target_field[:1])
    scorer, metric_provenance = load_official_scorer(args.official_repo)
    prediction_array = evaluation_prediction[0, 0].float().cpu().numpy()
    target_array = target[0, 0].float().cpu().numpy()
    score_arrays(
        scorer,
        prediction_array,
        target_array,
        lpips_device="cuda",
        include_lpips=True,
    )
    torch.cuda.synchronize()
    evaluation_started = time.monotonic()
    with torch.inference_mode(), amp.context():
        evaluation_prediction = model(source[:1], modality[:1], source_field[:1], target_field[:1])
    score_arrays(
        scorer,
        evaluation_prediction[0, 0].float().cpu().numpy(),
        target_array,
        lpips_device="cuda",
        include_lpips=True,
    )
    torch.cuda.synchronize()
    evaluation_seconds_per_patch = time.monotonic() - evaluation_started
    peak_bytes = int(torch.cuda.max_memory_reserved(device))

    seconds_per_update = training_seconds / args.updates
    train_updates = 20_000 + 3 * 10_000 + 10_000
    windows_per_volume = window_count((364, 436, 364), args.inference_window, 0.25)
    inference_batches = math.ceil(180 * windows_per_volume / args.inference_batch_size)
    training_hours = seconds_per_update * train_updates / 3600
    evaluation_patches = 3 * 10 * 60 * args.holdout_samples_per_stratum
    evaluation_hours = evaluation_patches * evaluation_seconds_per_patch / 3600
    inference_hours = inference_batches * inference_seconds_per_batch / 3600
    projected_hours = training_hours + evaluation_hours + inference_hours
    headroom_fraction = 1.0 - peak_bytes / total_bytes
    checks = {
        "a100_detected": "A100" in name.upper(),
        "vram_headroom_at_least_20_percent": headroom_fraction >= 0.20,
        "projected_runtime_at_most_10_hours": projected_hours <= 10.0,
    }
    return {
        "kind": "task3_3d_unet_a100_probe",
        "created_at_utc": utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "gpu": name,
        "total_vram_bytes": total_bytes,
        "peak_reserved_bytes": peak_bytes,
        "headroom_fraction": headroom_fraction,
        "updates_measured": args.updates,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "seconds_per_optimizer_update": seconds_per_update,
        "training_projection_hours": training_hours,
        "holdout_evaluation_patches": evaluation_patches,
        "evaluation_seconds_per_patch": evaluation_seconds_per_patch,
        "evaluation_projection_hours": evaluation_hours,
        "metric_provenance": metric_provenance,
        "inference_window": args.inference_window,
        "inference_overlap": 0.25,
        "inference_batch_size": args.inference_batch_size,
        "windows_per_volume": windows_per_volume,
        "inference_seconds_per_batch": inference_seconds_per_batch,
        "inference_projection_hours": inference_hours,
        "complete_projection_hours": projected_hours,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--inference-window", type=int, default=128)
    parser.add_argument("--inference-batch-size", type=int, default=2)
    parser.add_argument("--inference-repeats", type=int, default=3)
    parser.add_argument("--holdout-samples-per-stratum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_probe(args)
        atomic_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "passed" else 2
    except (Task3Error, RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
