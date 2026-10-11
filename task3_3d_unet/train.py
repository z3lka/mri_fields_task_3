"""Resumable retrospective pretraining and prospective ablation training."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .common import (
    ALL_PROSPECTIVE_SUBJECTS,
    HOLDOUT_SUBJECT,
    TRAIN_SUBJECTS,
    Task3Error,
    atomic_json,
    sha256_file,
    utc_now,
)
from .data import ProspectivePatchDataset, RetrospectivePatchStream, validate_subject_split
from .losses import block_mask, fine_tuning_loss, masked_reconstruction_loss
from .metrics import load_official_scorer, score_arrays
from .model import ARCHITECTURE_ID, CHECKPOINT_FORMAT, ModelConfig, build_model


ARMS = ("paired_global", "paired_masked", "pretrained_masked")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Task3Error("CUDA was requested but is unavailable")
    return device


class Amp:
    def __init__(self, device: torch.device, mode: str):
        if mode == "auto":
            mode = "bf16" if device.type == "cuda" else "off"
        if mode != "off" and device.type != "cuda":
            raise Task3Error("Mixed precision is supported only on CUDA")
        self.device = device
        self.mode = mode
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and mode == "fp16"
        )

    def context(self):
        if self.mode == "off":
            return contextlib.nullcontext()
        dtype = torch.float16 if self.mode == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)


def warmup_cosine_schedule(
    optimizer: AdamW, *, total_updates: int, warmup_updates: int, min_lr: float, lr: float
) -> LambdaLR:
    if not 0 <= warmup_updates < total_updates or not 0 < min_lr <= lr:
        raise Task3Error("Invalid warmup/cosine schedule")
    minimum = min_lr / lr

    def multiplier(step: int) -> float:
        if step < warmup_updates:
            return max(minimum, (step + 1) / max(1, warmup_updates))
        progress = min(1.0, (step - warmup_updates) / max(1, total_updates - warmup_updates))
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    if not 0.0 <= decay < 1.0:
        raise Task3Error("EMA decay must be in [0,1)")
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
        ema_buffer.copy_(buffer.detach())


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise Task3Error(f"Not a {CHECKPOINT_FORMAT} checkpoint: {path}")
    if payload.get("architecture") != ARCHITECTURE_ID:
        raise Task3Error(f"Unexpected checkpoint architecture: {payload.get('architecture')!r}")
    required = {
        "architecture",
        "model_config",
        "model",
        "model_ema",
        "optimizer",
        "scheduler",
        "scaler",
        "global_step",
        "rng_state",
        "run_signature",
        "split_manifest",
        "provenance",
        "best_model_ema",
        "best_step",
    }
    if not required <= set(payload):
        raise Task3Error(f"Checkpoint is incomplete: missing={sorted(required - set(payload))}")
    ModelConfig.from_dict(payload["model_config"])
    return payload


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


def _move(batch: dict[str, Any], device: torch.device, keys: Sequence[str]) -> dict[str, Tensor]:
    return {
        key: torch.as_tensor(batch[key]).to(device, non_blocking=device.type == "cuda")
        for key in keys
    }


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    amp: Amp,
    global_step: int,
    run_signature: dict[str, Any],
    split_manifest: dict[str, Any],
    provenance: dict[str, Any],
    best_nrmse: float,
    best_metrics: dict[str, Any] | None,
    best_model_ema: dict[str, Tensor] | None,
    best_step: int,
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "architecture": ARCHITECTURE_ID,
        "model_config": model.config.to_dict(),
        "model": model.state_dict(),
        "model_ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": amp.scaler.state_dict(),
        "global_step": global_step,
        "rng_state": capture_rng_state(),
        "run_signature": run_signature,
        "split_manifest": split_manifest,
        "provenance": provenance,
        "best_nrmse": best_nrmse,
        "best_metrics": best_metrics,
        "best_model_ema": best_model_ema,
        "best_step": best_step,
    }


def _require_training_gate(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.baseline_milestone_complete:
        raise Task3Error(
            "Training is gated by the verified Task 3 baseline milestone; pass "
            "--baseline-milestone-complete only after checking its report"
        )
    if args.total_updates > 100:
        if args.runtime_report is None or not args.runtime_report.is_file():
            raise Task3Error("Paid training requires the A100 timing/VRAM --runtime-report")
        report = json.loads(args.runtime_report.read_text(encoding="utf-8"))
        compatible = (
            report.get("updates_measured", 0) >= 100
            and report.get("patch_size") == args.patch_size
            and report.get("batch_size") == args.batch_size
            and report.get("gradient_accumulation") == args.gradient_accumulation
        )
        if not compatible:
            raise Task3Error("Runtime report does not match this patch/batch/accumulation recipe")
        checks = report.get("checks", {})
        if not checks.get("a100_detected") or not checks.get("vram_headroom_at_least_20_percent"):
            raise Task3Error("Paid training requires an A100 with at least 20% measured VRAM headroom")
        if not checks.get("projected_runtime_at_most_10_hours") and not args.acknowledge_over_budget:
            raise Task3Error(
                "Runtime report projects more than 10 hours; use "
                "--acknowledge-over-budget only after accepting the measured cost"
            )
        return {
            "path": str(args.runtime_report.expanduser().resolve()),
            "sha256": sha256_file(args.runtime_report),
            "checks": checks,
            "complete_projection_hours": report.get("complete_projection_hours"),
            "over_budget_acknowledged": bool(args.acknowledge_over_budget),
        }
    return None


def _build_state(
    args: argparse.Namespace,
    run_signature: dict[str, Any],
    split_manifest: dict[str, Any],
    *,
    initialize: Path | None = None,
) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    AdamW,
    LambdaLR,
    Amp,
    int,
    float,
    dict[str, Any] | None,
    dict[str, Tensor] | None,
    int,
]:
    device = select_device(args.device)
    config = ModelConfig.tiny() if args.tiny else ModelConfig()
    model = build_model(config).to(device)
    if initialize is not None:
        initial = load_checkpoint(initialize)
        if (
            initial["run_signature"].get("kind") != "pretrain"
            or int(initial["global_step"])
            != int(initial["run_signature"].get("total_updates", -1))
        ):
            raise Task3Error("Initialization must be a completed retrospective pretraining checkpoint")
        if initial["model_config"] != config.to_dict():
            raise Task3Error("Pretraining checkpoint model configuration does not match")
        fixed_pretraining = {
            "total_updates": 20_000,
            "patch_size": 96,
            "block_size": 12,
            "mask_fraction": 0.5,
            "patches_per_volume": 64,
            "batch_size": 2,
            "gradient_accumulation": 2,
            "lr": 1e-4,
            "min_lr": 1e-6,
            "warmup_updates": 500,
            "weight_decay": 1e-4,
            "ema_decay": 0.999,
            "seed": 2026,
            "tiny": False,
            "amp": "bf16",
            "clip_grad_norm": 1.0,
        }
        if not args.tiny and any(
            initial["run_signature"].get(key) != value
            for key, value in fixed_pretraining.items()
        ):
            raise Task3Error("Initialization checkpoint does not use the fixed v1 pretraining recipe")
        model.load_state_dict(initial["model_ema"], strict=True)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = warmup_cosine_schedule(
        optimizer,
        total_updates=args.total_updates,
        warmup_updates=min(args.warmup_updates, args.total_updates - 1),
        min_lr=args.min_lr,
        lr=args.lr,
    )
    amp = Amp(device, args.amp)
    global_step, best_nrmse, best_metrics = 0, math.inf, None
    best_model_ema, best_step = None, -1
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint["run_signature"] != run_signature:
            raise Task3Error("Resume run signature differs from this command")
        if checkpoint["split_manifest"] != split_manifest:
            raise Task3Error("Resume split/data manifest differs from this command")
        if checkpoint["model_config"] != config.to_dict():
            raise Task3Error("Resume model configuration differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        ema.load_state_dict(checkpoint["model_ema"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        amp.scaler.load_state_dict(checkpoint["scaler"])
        global_step = int(checkpoint["global_step"])
        best_nrmse = float(checkpoint.get("best_nrmse", math.inf))
        best_metrics = checkpoint.get("best_metrics")
        best_model_ema = checkpoint["best_model_ema"]
        best_step = int(checkpoint["best_step"])
        restore_rng_state(checkpoint["rng_state"])
    return (
        model,
        ema,
        optimizer,
        scheduler,
        amp,
        global_step,
        best_nrmse,
        best_metrics,
        best_model_ema,
        best_step,
    )


@torch.inference_mode()
def evaluate_holdout(
    model: torch.nn.Module,
    dataset: ProspectivePatchDataset,
    *,
    device: torch.device,
    amp: Amp,
    official_repo: Path,
    include_lpips: bool,
) -> dict[str, Any]:
    scorer, provenance = load_official_scorer(official_repo)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(dataset.seed),
    )
    totals: dict[str, list[dict[str, float]]] = defaultdict(list)
    identity_totals: list[float] = []
    model.eval()
    for batch in loader:
        moved = _move(
            batch,
            device,
            ("source", "target", "modality", "source_field", "target_field"),
        )
        with amp.context():
            prediction = model(
                moved["source"], moved["modality"], moved["source_field"], moved["target_field"]
            )
        pred = prediction[0, 0].float().cpu().numpy()
        source = moved["source"][0, 0].float().cpu().numpy()
        target = moved["target"][0, 0].float().cpu().numpy()
        stratum = batch["stratum"][0]
        totals[stratum].append(
            score_arrays(
                scorer,
                pred,
                target,
                lpips_device="cuda" if device.type == "cuda" else "cpu",
                include_lpips=include_lpips,
            )
        )
        identity_totals.append(float(scorer.compute_nrmse(source, target)))
    flat = [score for scores in totals.values() for score in scores]
    keys = ("nrmse", "ssim", "lpips") if include_lpips else ("nrmse", "ssim")
    result: dict[str, Any] = {
        "mean": {key: float(np.mean([score[key] for score in flat])) for key in keys},
        "identity_mean_nrmse": float(np.mean(identity_totals)),
        "strata": {
            stratum: {
                key: float(np.mean([score[key] for score in scores])) for key in keys
            }
            for stratum, scores in sorted(totals.items())
        },
        "samples": len(flat),
        "metric_provenance": provenance,
    }
    model.train()
    return result


def _optimizer_step(
    *,
    loss: Tensor,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    amp: Amp,
    accumulation: int,
    final_micro_step: bool,
    clip_grad_norm: float,
) -> None:
    amp.scaler.scale(loss / accumulation).backward()
    if not final_micro_step:
        return
    amp.scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    amp.scaler.step(optimizer)
    amp.scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()


def _save_training_state(
    output_dir: Path,
    payload: dict[str, Any],
    *,
    is_best: bool = False,
) -> None:
    atomic_torch_save(payload, output_dir / "checkpoints" / "last.pt")
    if is_best:
        atomic_torch_save(payload, output_dir / "checkpoints" / "best.pt")


def _resolve_training_output(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.expanduser().resolve()
    if args.resume is not None:
        expected_resume = output_dir / "checkpoints" / "last.pt"
        if args.resume.expanduser().resolve() != expected_resume or not expected_resume.is_file():
            raise Task3Error(f"Resume must use this run's latest checkpoint: {expected_resume}")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise Task3Error(f"Fresh training refuses to overwrite non-empty output: {output_dir}")
    return output_dir


def pretrain(args: argparse.Namespace) -> None:
    output_dir = _resolve_training_output(args)
    runtime_gate = _require_training_gate(args)
    seed_everything(args.seed)
    manifest_path = args.retrospective_manifest.expanduser().resolve()
    split_manifest = {
        "retrospective_manifest": str(manifest_path),
        "retrospective_manifest_sha256": sha256_file(manifest_path),
        "paired_subjects": None,
    }
    run_signature = {
        "kind": "pretrain",
        "total_updates": args.total_updates,
        "patch_size": args.patch_size,
        "block_size": args.block_size,
        "mask_fraction": args.mask_fraction,
        "patches_per_volume": args.patches_per_volume,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_updates": args.warmup_updates,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "seed": args.seed,
        "tiny": args.tiny,
        "amp": args.amp,
        "clip_grad_norm": args.clip_grad_norm,
        "runtime_gate": runtime_gate,
    }
    state = _build_state(args, run_signature, split_manifest)
    (
        model,
        ema,
        optimizer,
        scheduler,
        amp,
        global_step,
        best_nrmse,
        best_metrics,
        best_model_ema,
        best_step,
    ) = state
    device = next(model.parameters()).device
    stream = RetrospectivePatchStream(
        manifest_path,
        patch_size=args.patch_size,
        patches_per_volume=args.patches_per_volume,
        seed=args.seed,
        start_patch=global_step * args.batch_size * args.gradient_accumulation,
    )
    loader = DataLoader(
        stream,
        batch_size=args.batch_size,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    iterator: Iterator[dict[str, Any]] = iter(loader)
    provenance = {
        "created_at_utc": utc_now(),
        "status": "running",
        "architecture": ARCHITECTURE_ID,
        "run_signature": run_signature,
        "split_manifest": split_manifest,
    }
    atomic_json(output_dir / "provenance.json", provenance)
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    while global_step < args.total_updates:
        running = 0.0
        for micro_step in range(args.gradient_accumulation):
            batch = next(iterator)
            moved = _move(batch, device, ("source", "modality", "source_field", "target_field"))
            mask = block_mask(
                tuple(moved["source"].shape),
                block_size=args.block_size,
                fraction=args.mask_fraction,
                device=device,
            )
            corrupted = model.apply_mask(moved["source"], mask)
            with amp.context():
                prediction = model(
                    corrupted,
                    moved["modality"],
                    moved["source_field"],
                    moved["target_field"],
                    allow_identity=True,
                )
                loss = masked_reconstruction_loss(prediction, moved["source"], mask)
            _optimizer_step(
                loss=loss,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                amp=amp,
                accumulation=args.gradient_accumulation,
                final_micro_step=micro_step + 1 == args.gradient_accumulation,
                clip_grad_norm=args.clip_grad_norm,
            )
            running += float(loss.detach())
        global_step += 1
        update_ema(ema, model, args.ema_decay)
        if global_step % args.log_every == 0 or global_step == args.total_updates:
            record = {
                "kind": "pretrain",
                "global_step": global_step,
                "loss": running / args.gradient_accumulation,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.monotonic() - started,
            }
            _append_jsonl(output_dir / "metrics.jsonl", record)
            print(json.dumps(record, sort_keys=True))
        checkpoint_due = global_step % args.checkpoint_every == 0 or global_step == args.total_updates
        time_due = args.max_runtime_minutes > 0 and time.monotonic() - started >= args.max_runtime_minutes * 60
        if checkpoint_due or time_due:
            if global_step == args.total_updates:
                best_model_ema = {
                    name: value.detach().cpu().clone() for name, value in ema.state_dict().items()
                }
                best_step = global_step
            checkpoint_provenance = dict(provenance)
            if global_step == args.total_updates or time_due:
                checkpoint_provenance.update(
                    status=(
                        "complete"
                        if global_step == args.total_updates
                        else "paused_runtime_budget"
                    ),
                    global_step=global_step,
                    elapsed_seconds=time.monotonic() - started,
                )
            payload = _checkpoint_payload(
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                amp=amp,
                global_step=global_step,
                run_signature=run_signature,
                split_manifest=split_manifest,
                provenance=checkpoint_provenance,
                best_nrmse=best_nrmse,
                best_metrics=best_metrics,
                best_model_ema=best_model_ema,
                best_step=best_step,
            )
            _save_training_state(output_dir, payload, is_best=global_step == args.total_updates)
        if time_due:
            break
    provenance.update(
        status="complete" if global_step == args.total_updates else "paused_runtime_budget",
        global_step=global_step,
        elapsed_seconds=time.monotonic() - started,
        finished_at_utc=utc_now(),
    )
    atomic_json(output_dir / "provenance.json", provenance)


def finetune(args: argparse.Namespace) -> None:
    output_dir = _resolve_training_output(args)
    runtime_gate = _require_training_gate(args)
    seed_everything(args.seed)
    if args.arm not in ARMS:
        raise Task3Error(f"Unknown arm {args.arm!r}")
    if (args.arm == "pretrained_masked") != (args.pretrained is not None):
        raise Task3Error("pretrained_masked requires --pretrained; scratch arms forbid it")
    train_subjects = ALL_PROSPECTIVE_SUBJECTS if args.all_subjects else TRAIN_SUBJECTS
    holdout_subject = None if args.all_subjects else HOLDOUT_SUBJECT
    validate_subject_split(train_subjects, holdout_subject)
    selection_record = None
    selection = None
    if args.all_subjects:
        if args.selection_report is None or not args.selection_report.is_file():
            raise Task3Error("Fresh all-subject retraining requires --selection-report")
        selection = json.loads(args.selection_report.read_text(encoding="utf-8"))
        winner = selection.get("winner") if isinstance(selection, dict) else None
        if not isinstance(winner, dict) or winner.get("arm") != args.arm:
            raise Task3Error("Final arm does not match the recorded ablation winner")
        selection_record = {
            "path": str(args.selection_report.resolve()),
            "sha256": sha256_file(args.selection_report),
        }
        fixed_final = {
            "total_updates": 10_000,
            "patch_size": 96,
            "batch_size": 2,
            "gradient_accumulation": 2,
            "lr": 1e-4,
            "min_lr": 1e-6,
            "warmup_updates": 500,
            "weight_decay": 1e-4,
            "ema_decay": 0.999,
            "seed": 2026,
            "tiny": False,
            "amp": "bf16",
            "clip_grad_norm": 1.0,
        }
        if any(getattr(args, key) != value for key, value in fixed_final.items()):
            raise Task3Error("Final all-subject retraining must use the fixed winning v1 recipe")
    cache_manifest = args.cache_dir.expanduser().resolve() / "manifest.json"
    split_manifest = {
        "cache_manifest": str(cache_manifest),
        "cache_manifest_sha256": sha256_file(cache_manifest),
        "train_subjects": list(train_subjects),
        "holdout_subject": holdout_subject,
        "phase": "all_subject_retrain" if args.all_subjects else "ablation",
    }
    metric_provenance = None
    if holdout_subject is not None:
        _, metric_provenance = load_official_scorer(args.official_repo)
    initialization = (
        {"path": str(args.pretrained.resolve()), "sha256": sha256_file(args.pretrained)}
        if args.pretrained
        else "scratch_seed_2026"
    )
    if args.all_subjects:
        expected_initialization = winner.get("initialization")
        same_initialization = (
            isinstance(initialization, dict)
            and isinstance(expected_initialization, dict)
            and initialization.get("sha256") == expected_initialization.get("sha256")
        ) or initialization == expected_initialization
        if not same_initialization:
            raise Task3Error("Final retraining initialization differs from the selected recipe")
    run_signature = {
        "kind": "finetune",
        "arm": args.arm,
        "all_subjects": args.all_subjects,
        "initialization": initialization,
        "total_updates": args.total_updates,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_updates": args.warmup_updates,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "seed": args.seed,
        "tiny": args.tiny,
        "amp": args.amp,
        "clip_grad_norm": args.clip_grad_norm,
        "selection_report": selection_record,
        "metric_provenance": metric_provenance,
        "runtime_gate": runtime_gate,
        "eval_every": args.eval_every,
        "holdout_samples_per_stratum": args.holdout_samples_per_stratum,
        "lpips_enabled": not args.skip_lpips,
    }
    state = _build_state(
        args,
        run_signature,
        split_manifest,
        initialize=args.pretrained if args.resume is None else None,
    )
    (
        model,
        ema,
        optimizer,
        scheduler,
        amp,
        global_step,
        best_nrmse,
        best_metrics,
        best_model_ema,
        best_step,
    ) = state
    device = next(model.parameters()).device
    total_samples = args.total_updates * args.batch_size * args.gradient_accumulation
    start_sample = global_step * args.batch_size * args.gradient_accumulation
    samples_per_stratum = math.ceil(total_samples / 60)
    train_dataset = ProspectivePatchDataset(
        args.cache_dir,
        train_subjects,
        patch_size=args.patch_size,
        samples_per_stratum=samples_per_stratum,
        seed=args.seed,
        random_augmentation=True,
        start_index=start_sample,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    iterator = iter(loader)
    holdout = None
    if holdout_subject is not None:
        holdout = ProspectivePatchDataset(
            args.cache_dir,
            (holdout_subject,),
            patch_size=args.patch_size,
            samples_per_stratum=args.holdout_samples_per_stratum,
            seed=args.seed,
            random_augmentation=False,
        )
    provenance = {
        "created_at_utc": utc_now(),
        "status": "running",
        "architecture": ARCHITECTURE_ID,
        "run_signature": run_signature,
        "split_manifest": split_manifest,
        "metric_provenance": metric_provenance,
    }
    atomic_json(output_dir / "provenance.json", provenance)
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    while global_step < args.total_updates:
        running = defaultdict(float)
        for micro_step in range(args.gradient_accumulation):
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise Task3Error("Paired dataset exhausted before the requested updates") from exc
            moved = _move(
                batch,
                device,
                ("source", "target", "modality", "source_field", "target_field"),
            )
            with amp.context():
                prediction = model(
                    moved["source"],
                    moved["modality"],
                    moved["source_field"],
                    moved["target_field"],
                )
                loss, terms = fine_tuning_loss(
                    prediction, moved["target"], masked=args.arm != "paired_global"
                )
            _optimizer_step(
                loss=loss,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                amp=amp,
                accumulation=args.gradient_accumulation,
                final_micro_step=micro_step + 1 == args.gradient_accumulation,
                clip_grad_norm=args.clip_grad_norm,
            )
            running["loss"] += float(loss.detach())
            for name, value in terms.items():
                running[name] += float(value.detach())
        global_step += 1
        update_ema(ema, model, args.ema_decay)
        record: dict[str, Any] | None = None
        improved = False
        if holdout is not None and (global_step % args.eval_every == 0 or global_step == args.total_updates):
            metrics = evaluate_holdout(
                ema,
                holdout,
                device=device,
                amp=amp,
                official_repo=args.official_repo,
                include_lpips=not args.skip_lpips,
            )
            current = float(metrics["mean"]["nrmse"])
            improved = current < best_nrmse
            if improved:
                best_nrmse, best_metrics = current, metrics
                best_model_ema = {
                    name: value.detach().cpu().clone() for name, value in ema.state_dict().items()
                }
                best_step = global_step
            record = {
                "kind": "finetune_eval",
                "arm": args.arm,
                "global_step": global_step,
                "train": {
                    name: value / args.gradient_accumulation for name, value in running.items()
                },
                "holdout": metrics,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.monotonic() - started,
            }
        elif global_step % args.log_every == 0 or global_step == args.total_updates:
            record = {
                "kind": "finetune",
                "arm": args.arm,
                "global_step": global_step,
                "train": {
                    name: value / args.gradient_accumulation for name, value in running.items()
                },
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.monotonic() - started,
            }
        if record is not None:
            _append_jsonl(output_dir / "metrics.jsonl", record)
            print(json.dumps(record, sort_keys=True))
        checkpoint_due = (
            improved
            or global_step % args.checkpoint_every == 0
            or global_step == args.total_updates
        )
        time_due = args.max_runtime_minutes > 0 and time.monotonic() - started >= args.max_runtime_minutes * 60
        if checkpoint_due or time_due:
            checkpoint_provenance = dict(provenance)
            if global_step == args.total_updates or time_due:
                checkpoint_provenance.update(
                    status=(
                        "complete"
                        if global_step == args.total_updates
                        else "paused_runtime_budget"
                    ),
                    global_step=global_step,
                    elapsed_seconds=time.monotonic() - started,
                )
            payload = _checkpoint_payload(
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                amp=amp,
                global_step=global_step,
                run_signature=run_signature,
                split_manifest=split_manifest,
                provenance=checkpoint_provenance,
                best_nrmse=best_nrmse,
                best_metrics=best_metrics,
                best_model_ema=best_model_ema,
                best_step=best_step,
            )
            _save_training_state(output_dir, payload, is_best=improved or (holdout is None and global_step == args.total_updates))
        if time_due:
            break
    complete = global_step == args.total_updates
    if complete and holdout is not None:
        if best_model_ema is None:
            raise Task3Error("Completed ablation has no selected holdout weights")
        selected = dict(payload)
        selected["model_ema"] = best_model_ema
        selected["checkpoint_role"] = "lowest_holdout_nrmse"
        selected["selected_global_step"] = best_step
        atomic_torch_save(selected, output_dir / "checkpoints" / "best.pt")
    summary = {
        "status": "complete" if complete else "paused_runtime_budget",
        "arm": args.arm,
        "phase": split_manifest["phase"],
        "global_step": global_step,
        "best_nrmse": None if math.isinf(best_nrmse) else best_nrmse,
        "best_metrics": best_metrics,
        "best_step": best_step,
        "initialization": run_signature["initialization"],
        "split_manifest": split_manifest,
        "metric_provenance": metric_provenance,
        "run_signature": run_signature,
    }
    atomic_json(output_dir / "summary.json", summary)
    provenance.update(
        status=summary["status"],
        global_step=global_step,
        elapsed_seconds=time.monotonic() - started,
        finished_at_utc=utc_now(),
    )
    atomic_json(output_dir / "provenance.json", provenance)


def _shared_arguments(parser: argparse.ArgumentParser, *, default_updates: int) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-updates", type=int, default=default_updates)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-updates", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-runtime-minutes", type=float, default=45.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--runtime-report", type=Path)
    parser.add_argument("--baseline-milestone-complete", action="store_true")
    parser.add_argument("--acknowledge-over-budget", action="store_true")
    parser.add_argument("--tiny", action="store_true", help="test-only small model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("pretrain")
    _shared_arguments(pre, default_updates=20_000)
    pre.add_argument("--retrospective-manifest", type=Path, required=True)
    pre.add_argument("--block-size", type=int, default=12)
    pre.add_argument("--mask-fraction", type=float, default=0.5)
    pre.add_argument("--patches-per-volume", type=int, default=64)
    fine = commands.add_parser("finetune")
    _shared_arguments(fine, default_updates=10_000)
    fine.add_argument("--cache-dir", type=Path, required=True)
    fine.add_argument("--arm", choices=ARMS, required=True)
    fine.add_argument("--pretrained", type=Path)
    fine.add_argument("--all-subjects", action="store_true")
    fine.add_argument("--selection-report", type=Path)
    fine.add_argument("--official-repo", type=Path, required=True)
    fine.add_argument("--eval-every", type=int, default=1000)
    fine.add_argument("--holdout-samples-per-stratum", type=int, default=1)
    fine.add_argument("--skip-lpips", action="store_true", help="test-only metric shortcut")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.total_updates < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
            raise Task3Error("Updates, batch size, and gradient accumulation must be positive")
        if args.patch_size < 8 or args.checkpoint_every < 1 or args.log_every < 1:
            raise Task3Error("Patch size must be at least 8 and checkpoint/log intervals positive")
        if args.lr <= 0 or args.min_lr <= 0 or args.min_lr > args.lr:
            raise Task3Error("Learning rates must satisfy 0 < min_lr <= lr")
        if args.weight_decay < 0 or args.clip_grad_norm <= 0 or not 0 <= args.ema_decay < 1:
            raise Task3Error("Invalid optimizer, clipping, or EMA setting")
        if args.max_runtime_minutes < 0:
            raise Task3Error("max-runtime-minutes cannot be negative")
        if args.command == "pretrain" and args.patches_per_volume < 1:
            raise Task3Error("patches-per-volume must be positive")
        if args.command == "finetune" and (
            args.eval_every < 1 or args.holdout_samples_per_stratum < 1
        ):
            raise Task3Error("Evaluation interval and holdout samples must be positive")
        if args.command == "pretrain":
            pretrain(args)
        else:
            finetune(args)
    except (Task3Error, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
