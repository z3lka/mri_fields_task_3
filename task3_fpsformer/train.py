#!/usr/bin/env python3
"""Train one conditional FPS-Former for MRIxFields2026 Task 3.

This is a supervised, paired-prospective experiment derived from
arXiv:2412.10776v1.  It never creates a model per field, modality, or directed
pair: all 60 (modality, source, target) strata update the same parameters.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from task3_fpsformer.common import (
    PAPER_ID,
    FPSFormerError,
    atomic_json,
    sha256_file,
    utc_now,
)
from task3_fpsformer.data import (
    ProspectivePairDataset,
    build_cache_index,
    complete_subjects,
)
from task3_fpsformer.losses import (
    LossWeights,
    MetricAccumulator,
    OptionalLPIPS,
    TranslationLoss,
)
from task3_fpsformer.model import (
    ARCHITECTURE_ID,
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
    ModelConfig,
    build_model,
    count_parameters,
)


class AmpController:
    def __init__(self, device: torch.device, choice: str):
        if choice == "auto":
            choice = "fp16" if device.type == "cuda" else "off"
        if choice in {"fp16", "bf16"} and device.type != "cuda":
            raise FPSFormerError(
                f"--amp {choice} is only enabled for CUDA in this portable runner; "
                f"use --amp off on {device.type}"
            )
        self.device = device
        self.enabled = choice != "off"
        self.dtype = torch.float16 if choice == "fp16" else torch.bfloat16
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled)
        except (AttributeError, TypeError):  # PyTorch < 2.4
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)

    def context(self) -> Any:
        if not self.enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FPSFormerError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise FPSFormerError("MPS was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise FPSFormerError(f"Not a Task 3 FPS-Former checkpoint: {path}")
    return checkpoint


def cosine_schedule(
    optimizer: AdamW,
    *,
    total_updates: int,
    flat_fraction: float,
    minimum_factor: float,
) -> LambdaLR:
    flat_updates = int(total_updates * flat_fraction)

    def multiplier(step: int) -> float:
        if step <= flat_updates:
            return 1.0
        progress = (step - flat_updates) / max(1, total_updates - flat_updates)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    return LambdaLR(optimizer, multiplier)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Tensor]:
    return {
        key: batch[key].to(device, non_blocking=device.type == "cuda")
        for key in ("source", "target", "modality", "source_field", "target_field")
    }


def build_train_loader(
    dataset: ProspectivePairDataset,
    *,
    batch_size: int,
    workers: int,
    updates_per_epoch: int | None,
    gradient_accumulation: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    shuffle = True
    if updates_per_epoch is not None:
        sample_count = updates_per_epoch * gradient_accumulation * batch_size
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=sample_count,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
    )


def build_validation_loader(
    dataset: ProspectivePairDataset,
    *,
    samples_per_pair: int,
    slice_start: int | None,
    slice_end: int | None,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    subset = dataset.balanced_subset(
        samples_per_pair,
        seed=seed,
        slice_start=slice_start,
        slice_end=slice_end,
    )
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: AmpController,
    perceptual: OptionalLPIPS,
    *,
    progress: bool = False,
    description: str = "validation",
) -> dict[str, Any]:
    model.eval()
    metrics = MetricAccumulator()
    batches = tqdm(
        loader,
        total=len(loader),
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        mininterval=2.0,
        disable=not progress,
    )
    for batch in batches:
        moved = move_batch(batch, device)
        with amp.context():
            prediction = model(
                moved["source"],
                moved["modality"],
                moved["source_field"],
                moved["target_field"],
            )
        metrics.update(
            prediction,
            moved["target"],
            moved["modality"],
            perceptual if perceptual.enabled else None,
        )
    return metrics.summarize(include_lpips=perceptual.enabled)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    model_ema: torch.nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    model_config: ModelConfig,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_nrmse: float,
    holdout_subject: str | None,
) -> dict[str, Any]:
    return {
        "format": "task3_conditional_fpsformer_v1",
        "architecture": ARCHITECTURE_ID,
        "paper": PAPER_ID,
        "upstream_reference": {
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
        },
        "model_config": model_config.to_dict(),
        "model": model.state_dict(),
        "model_ema": model_ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_nrmse": best_nrmse,
        "holdout_subject": holdout_subject,
        "loss_weights": asdict(
            LossWeights(args.loss_l1, args.loss_ssim, args.loss_edge, args.loss_lpips)
        ),
    }


def model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    config = ModelConfig.preset(args.preset).to_dict()
    overrides = {
        "base_channels": args.base_channels,
        "max_frequency_tokens": args.max_frequency_tokens,
        "max_spam_group_tokens": args.max_spam_group_tokens,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    return ModelConfig.from_dict(config)


def require_acknowledgement(args: argparse.Namespace) -> None:
    if not args.baseline_milestone_complete:
        raise FPSFormerError(
            "Training is gated by the Task 3 baseline milestone. Re-run with "
            "--baseline-milestone-complete only after confirming that milestone."
        )


def runtime_budget_exhausted(elapsed_seconds: float, max_runtime_minutes: float) -> bool:
    return max_runtime_minutes > 0 and elapsed_seconds >= max_runtime_minutes * 60


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, beta: float) -> None:
    if not 0.0 <= beta < 1.0:
        raise FPSFormerError("--ema-beta must be in [0,1)")
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter.detach(), 1.0 - beta)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer.detach())


def train(args: argparse.Namespace) -> None:
    require_acknowledgement(args)
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        raise FPSFormerError("Epochs, batch size, and gradient accumulation must be positive")
    if not 0 <= args.flat_fraction < 1:
        raise FPSFormerError("--flat-fraction must be in [0,1)")
    if args.min_lr <= 0 or args.min_lr > args.lr:
        raise FPSFormerError("--min-lr must be positive and no greater than --lr")
    if args.max_runtime_minutes < 0:
        raise FPSFormerError("--max-runtime-minutes cannot be negative")
    if (args.val_slice_start is None) != (args.val_slice_end is None):
        raise FPSFormerError("--val-slice-start and --val-slice-end must be used together")
    if args.val_slice_start is not None and args.val_slice_start >= args.val_slice_end:
        raise FPSFormerError("Validation slice range must satisfy start < end")
    if not 0.0 <= args.left_right_flip_probability <= 1.0:
        raise FPSFormerError("--left-right-flip-probability must be in [0,1]")
    if not 0.0 <= args.ema_beta < 1.0:
        raise FPSFormerError("--ema-beta must be in [0,1)")

    seed_everything(args.seed)
    device = select_device(args.device)
    amp = AmpController(device, args.amp)
    model_config = model_config_from_args(args)
    cache_index = build_cache_index(args.cache_dir)
    subjects = complete_subjects(cache_index)
    if not subjects:
        raise FPSFormerError("No complete prospective subjects were found")
    if args.holdout_subject == "none":
        holdout_subject = None
    elif args.holdout_subject == "auto":
        if len(subjects) < 2:
            raise FPSFormerError("At least two complete subjects are needed for an honest holdout")
        holdout_subject = subjects[-1]
    else:
        holdout_subject = args.holdout_subject
        if holdout_subject not in subjects:
            raise FPSFormerError(
                f"Unknown/incomplete holdout {holdout_subject!r}; choose from {subjects}"
            )

    crop_size = None if args.crop_size == 0 else args.crop_size
    train_dataset = ProspectivePairDataset(
        args.cache_dir,
        exclude_subjects={holdout_subject} if holdout_subject else None,
        crop_size=crop_size,
        random_crop=True,
        left_right_flip_probability=args.left_right_flip_probability,
    )
    train_loader = build_train_loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        updates_per_epoch=args.updates_per_epoch,
        gradient_accumulation=args.gradient_accumulation,
        seed=args.seed,
        device=device,
    )
    updates_per_epoch = len(train_loader) // args.gradient_accumulation
    if updates_per_epoch < 1:
        raise FPSFormerError("Training loader is shorter than one optimizer update")
    total_updates = updates_per_epoch * args.epochs

    validation_loader: DataLoader | None = None
    if holdout_subject:
        validation_dataset = ProspectivePairDataset(
            args.cache_dir,
            include_subjects={holdout_subject},
            crop_size=None if args.eval_crop_size == 0 else args.eval_crop_size,
            random_crop=False,
        )
        validation_loader = build_validation_loader(
            validation_dataset,
            samples_per_pair=args.val_samples_per_pair,
            slice_start=args.val_slice_start,
            slice_end=args.val_slice_end,
            batch_size=args.eval_batch_size,
            workers=args.workers,
            seed=args.seed,
            device=device,
        )

    model = build_model(model_config).to(device)
    initialization: dict[str, Any] | None = None
    if args.initialize is not None:
        initialization_path = args.initialize.expanduser().resolve()
        initial_checkpoint = load_checkpoint(initialization_path, device)
        initial_config = ModelConfig.from_dict(initial_checkpoint["model_config"])
        if initial_config != model_config:
            raise FPSFormerError("Initialization checkpoint model configuration differs from CLI")
        initialization_weights = (
            "model_ema" if "model_ema" in initial_checkpoint else "model"
        )
        model.load_state_dict(initial_checkpoint[initialization_weights], strict=True)
        initialization = {
            "path": str(initialization_path),
            "sha256": sha256_file(initialization_path),
            "source_epoch": int(initial_checkpoint.get("epoch", -1)),
            "weights": initialization_weights,
            "weights_only": True,
        }
        del initial_checkpoint
    model_ema = copy.deepcopy(model).eval()
    model_ema.requires_grad_(False)
    objective = TranslationLoss(
        LossWeights(args.loss_l1, args.loss_ssim, args.loss_edge, args.loss_lpips)
    ).to(device)
    validation_lpips = OptionalLPIPS(args.eval_lpips).to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = cosine_schedule(
        optimizer,
        total_updates=total_updates,
        flat_fraction=args.flat_fraction,
        minimum_factor=args.min_lr / args.lr,
    )

    output_dir = args.output_dir.expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    existing = list(checkpoints_dir.glob("*.pt")) if checkpoints_dir.is_dir() else []
    if existing and args.resume is None:
        raise FPSFormerError(
            f"Checkpoint files already exist in {checkpoints_dir}; pass --resume explicitly "
            "or choose a new output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    global_step = 0
    best_nrmse = math.inf
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume.expanduser().resolve(), device)
        saved_config = ModelConfig.from_dict(checkpoint["model_config"])
        if saved_config != model_config:
            raise FPSFormerError("Resume checkpoint model configuration differs from CLI")
        model.load_state_dict(checkpoint["model"], strict=True)
        model_ema.load_state_dict(
            checkpoint.get("model_ema", checkpoint["model"]), strict=True
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_nrmse = float(checkpoint.get("best_nrmse", math.inf))
        if checkpoint.get("holdout_subject") != holdout_subject:
            raise FPSFormerError("Resume checkpoint holdout differs from this run")

    paper_path = Path(__file__).resolve().parent.parent / "2412.10776v1.pdf"
    provenance = {
        "kind": "task3_conditional_fpsformer_training",
        "created_at_utc": utc_now(),
        "status": "running",
        "target_epochs": args.epochs,
        "paper": PAPER_ID,
        "architecture": ARCHITECTURE_ID,
        "upstream_reference": {
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
        },
        "paper_pdf": str(paper_path) if paper_path.is_file() else None,
        "paper_sha256": sha256_file(paper_path) if paper_path.is_file() else None,
        "workspace_revision": git_revision(Path(__file__).resolve().parent),
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
        ),
        "model_config": model_config.to_dict(),
        "parameters": count_parameters(model),
        "subjects": subjects,
        "train_subjects": list(train_dataset.subjects),
        "holdout_subject": holdout_subject,
        "pair_strata": len(train_dataset.pair_counts),
        "train_records": len(train_dataset),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "initialization": initialization,
        "attention_note": (
            "Quadratic attention is bounded for portability."
            if model_config.max_frequency_tokens or model_config.max_spam_group_tokens
            else "Unbounded Task 3 adaptation requested."
        ),
        "metric_note": "Holdout values are slice proxies, not private validation scores.",
    }
    atomic_json(output_dir / "provenance.json", provenance)
    print(json.dumps(provenance, indent=2, sort_keys=True))

    metrics_path = output_dir / "metrics.jsonl"
    started = time.monotonic()
    last_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {
            "total": 0.0,
            "l1": 0.0,
            "ssim": 0.0,
            "edge": 0.0,
            "lpips": 0.0,
        }
        micro_count = 0
        epoch_progress = tqdm(
            total=updates_per_epoch,
            desc=f"train epoch {epoch + 1}/{args.epochs}",
            unit="update",
            dynamic_ncols=True,
            mininterval=2.0,
            disable=not args.progress,
        )
        try:
            for micro_step, batch in enumerate(train_loader):
                moved = move_batch(batch, device)
                with amp.context():
                    prediction = model(
                        moved["source"],
                        moved["modality"],
                        moved["source_field"],
                        moved["target_field"],
                    )
                    loss, terms = objective(prediction, moved["target"])
                    scaled_loss = loss / args.gradient_accumulation
                amp.scaler.scale(scaled_loss).backward()
                running["total"] += float(loss.detach())
                for name, value in terms.items():
                    running[name] += float(value.detach())
                micro_count += 1

                if (micro_step + 1) % args.gradient_accumulation:
                    continue
                if args.clip_grad_norm > 0:
                    amp.scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                amp.scaler.step(optimizer)
                amp.scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_ema(model_ema, model, args.ema_beta)
                global_step += 1
                divisor = max(1, micro_count)
                epoch_progress.set_postfix(
                    loss=f"{running['total'] / divisor:.5f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    refresh=False,
                )
                epoch_progress.update(1)
                if not args.progress and global_step % args.print_every == 0:
                    print(
                        f"epoch={epoch + 1}/{args.epochs} step={global_step} "
                        f"loss={running['total'] / divisor:.6f} "
                        f"l1={running['l1'] / divisor:.6f} "
                        f"lr={optimizer.param_groups[0]['lr']:.3e}"
                    )
        finally:
            epoch_progress.close()

        record: dict[str, Any] = {
            "kind": "epoch",
            "created_at_utc": utc_now(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "train": {
                name: value / max(1, micro_count) for name, value in running.items()
            },
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.monotonic() - started,
        }
        improved = False
        if validation_loader is not None and (epoch + 1) % args.eval_every == 0:
            record["holdout"] = validate(
                model_ema,
                validation_loader,
                device,
                amp,
                validation_lpips,
                progress=args.progress,
                description=f"validate epoch {epoch + 1}/{args.epochs}",
            )
            current_nrmse = float(record["holdout"]["overall"]["slice_proxy_nrmse"])
            improved = current_nrmse < best_nrmse
            if improved:
                best_nrmse = current_nrmse
            print(json.dumps(record["holdout"], indent=2, sort_keys=True))
        append_jsonl(metrics_path, record)

        payload = checkpoint_payload(
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            args=args,
            epoch=epoch + 1,
            global_step=global_step,
            best_nrmse=best_nrmse,
            holdout_subject=holdout_subject,
        )
        atomic_torch_save(payload, checkpoints_dir / "last.pt")
        if improved or validation_loader is None:
            atomic_torch_save(payload, checkpoints_dir / "best.pt")
        last_epoch = epoch + 1
        elapsed_seconds = time.monotonic() - started
        if last_epoch < args.epochs and runtime_budget_exhausted(
            elapsed_seconds, args.max_runtime_minutes
        ):
            print(
                f"Runtime budget reached after epoch {last_epoch}; "
                f"resume from {checkpoints_dir / 'last.pt'}"
            )
            break

    complete = last_epoch >= args.epochs
    provenance.update(
        {
            "status": "complete" if complete else "paused_runtime_budget",
            "finished_at_utc": utc_now(),
            "last_epoch": last_epoch,
            "global_step": global_step,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    atomic_json(output_dir / "provenance.json", provenance)
    state = "complete" if complete else "paused safely"
    print(f"Training {state}. Checkpoints: {checkpoints_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one paired conditional FPS-Former for Task 3"
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preset",
        choices=("portable", "tiny", "unbounded", "paper_exact"),
        default="portable",
        help="paper_exact is a legacy alias for unbounded; neither is source-exact",
    )
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--max-frequency-tokens", type=int)
    parser.add_argument("--max-spam-group-tokens", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256, help="0 uses full 364x436 slices")
    parser.add_argument("--eval-crop-size", type=int, default=0, help="0 evaluates full slices")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=0,
        help="Pause safely at an epoch boundary after this many minutes; 0 disables",
    )
    parser.add_argument(
        "--updates-per-epoch",
        type=int,
        default=1000,
        help="Balanced replacement samples per epoch; 0 uses the complete dataset",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--flat-fraction", type=float, default=92_000 / 300_000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-beta", type=float, default=0.999)
    parser.add_argument("--left-right-flip-probability", type=float, default=0.0)
    parser.add_argument("--loss-l1", type=float, default=1.0)
    parser.add_argument("--loss-ssim", type=float, default=0.0)
    parser.add_argument("--loss-edge", type=float, default=0.0)
    parser.add_argument("--loss-lpips", type=float, default=0.0)
    parser.add_argument("--holdout-subject", default="auto")
    parser.add_argument("--val-samples-per-pair", type=int, default=2)
    parser.add_argument("--val-slice-start", type=int)
    parser.add_argument("--val-slice-end", type=int)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-lpips", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show rate-limited tqdm bars for training and holdout validation",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize",
        type=Path,
        help="Load model weights only into a fresh run (for all-subject continuation)",
    )
    parser.add_argument("--baseline-milestone-complete", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.updates_per_epoch == 0:
        args.updates_per_epoch = None
    try:
        if args.resume is not None and args.initialize is not None:
            raise FPSFormerError("Use --resume or --initialize, not both")
        train(args)
    except (FPSFormerError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
