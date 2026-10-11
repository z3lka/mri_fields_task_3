#!/usr/bin/env python3
"""Small, model-agnostic helpers for measuring PyTorch CUDA training VRAM.

This module intentionally does not construct a model or load project data. Import
``measure_training_vram`` from a training script and pass a callable that performs
one *complete* training iteration: zero gradients, forward pass, loss, backward
pass, and optimizer step.

Example::

    from tools.vram_probe import measure_training_vram

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(inputs)
            loss = criterion(prediction, targets)
        loss.backward()
        optimizer.step()

    result = measure_training_vram(train_step, device="cuda:0")
    print(result.format())

Run this in an otherwise idle GPU process. The model, optimizer, representative
batch, and any AMP GradScaler must already exist before calling the function.
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Callable


GIB = 1024**3


def _gib(byte_count: int) -> float:
    return byte_count / GIB


@dataclass(frozen=True)
class VramMeasurement:
    """Peak memory observed during representative training iterations."""

    device_name: str
    measured_steps: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    device_used_before_bytes: int
    device_used_after_bytes: int
    device_total_bytes: int
    estimated_untracked_bytes: int
    recommended_capacity_bytes: int
    headroom_fraction: float

    @property
    def peak_allocated_gib(self) -> float:
        return _gib(self.peak_allocated_bytes)

    @property
    def peak_reserved_gib(self) -> float:
        return _gib(self.peak_reserved_bytes)

    @property
    def recommended_capacity_gib(self) -> float:
        return _gib(self.recommended_capacity_bytes)

    def format(self) -> str:
        """Return a compact human-readable report."""
        used_before = _gib(self.device_used_before_bytes)
        used_after = _gib(self.device_used_after_bytes)
        total = _gib(self.device_total_bytes)
        untracked = _gib(self.estimated_untracked_bytes)
        return "\n".join(
            (
                f"device: {self.device_name} ({total:.2f} GiB)",
                f"measured training steps: {self.measured_steps}",
                f"peak PyTorch allocated: {self.peak_allocated_gib:.2f} GiB",
                f"peak PyTorch reserved:  {self.peak_reserved_gib:.2f} GiB",
                f"whole-device used before/after: {used_before:.2f}/{used_after:.2f} GiB",
                f"estimated CUDA/non-PyTorch overhead: {untracked:.2f} GiB",
                f"recommended capacity ({self.headroom_fraction:.0%} headroom): "
                f"{self.recommended_capacity_gib:.2f} GiB",
            )
        )


def measure_training_vram(
    train_step: Callable[[], object],
    *,
    device: str = "cuda:0",
    warmup_steps: int = 2,
    measured_steps: int = 3,
    headroom_fraction: float = 0.20,
) -> VramMeasurement:
    """Measure peak CUDA memory for a complete, representative training step.

    Args:
        train_step: Zero-argument callable which executes one entire training
            iteration. It must not retain computation graphs between calls.
        device: CUDA device accepted by ``torch.device``.
        warmup_steps: Steps run before measurement so lazy optimizer state,
            kernels, and CUDA libraries have initialized.
        measured_steps: Number of subsequent steps included in the peak.
        headroom_fraction: Extra capacity recommended above the observed peak
            and estimated non-PyTorch CUDA overhead.

    Returns:
        A :class:`VramMeasurement` containing byte and GiB values.

    Notes:
        Epoch count normally changes runtime, not peak VRAM. Measure separately
        for training, validation, and unusually shaped batches, then keep the
        largest result. Distributed training must be measured per rank/device.
    """
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be at least 1")
    if measured_steps < 1:
        raise ValueError("measured_steps must be at least 1")
    if not 0.0 <= headroom_fraction <= 1.0:
        raise ValueError("headroom_fraction must be between 0 and 1")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on caller environment
        raise RuntimeError("PyTorch is required to measure CUDA memory") from exc

    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        raise ValueError(f"device must be CUDA, got {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch environment")

    torch.cuda.set_device(cuda_device)

    for _ in range(warmup_steps):
        train_step()
    torch.cuda.synchronize(cuda_device)

    # Release only unused cached blocks. Live model, optimizer, batch, and
    # gradient tensors remain allocated and are correctly included in the peak.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(cuda_device)

    free_before, total = torch.cuda.mem_get_info(cuda_device)
    used_before = total - free_before
    torch.cuda.reset_peak_memory_stats(cuda_device)

    for _ in range(measured_steps):
        train_step()
    torch.cuda.synchronize(cuda_device)

    peak_allocated = torch.cuda.max_memory_allocated(cuda_device)
    peak_reserved = torch.cuda.max_memory_reserved(cuda_device)
    current_reserved = torch.cuda.memory_reserved(cuda_device)
    free_after, total_after = torch.cuda.mem_get_info(cuda_device)
    used_after = total_after - free_after

    # CUDA contexts/libraries and other processes are outside PyTorch's caching
    # allocator. On an idle device, the positive difference is a useful overhead
    # estimate. If another process is active, the report is deliberately
    # conservative and that process is included here.
    untracked = max(0, max(used_before, used_after) - current_reserved)
    observed_capacity = max(peak_allocated, peak_reserved) + untracked
    recommended = math.ceil(observed_capacity * (1.0 + headroom_fraction))

    return VramMeasurement(
        device_name=torch.cuda.get_device_name(cuda_device),
        measured_steps=measured_steps,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        device_used_before_bytes=used_before,
        device_used_after_bytes=used_after,
        device_total_bytes=total_after,
        estimated_untracked_bytes=untracked,
        recommended_capacity_bytes=recommended,
        headroom_fraction=headroom_fraction,
    )


def estimate_adam_model_state_gib(
    trainable_parameters: int,
    *,
    parameter_bytes: int = 4,
    gradient_bytes: int = 4,
    master_parameter_bytes: int = 0,
    moment_bytes: int = 4,
) -> float:
    """Estimate persistent parameter/gradient/Adam state, excluding activations.

    Adam keeps two moments. For ordinary FP32 Adam, defaults give 16 bytes per
    trainable parameter. A common mixed-precision layout can be represented with
    ``parameter_bytes=2, gradient_bytes=2, master_parameter_bytes=4``; optimizer
    implementations vary, so measurement is still authoritative.
    """
    if trainable_parameters < 0:
        raise ValueError("trainable_parameters cannot be negative")
    widths = (
        parameter_bytes,
        gradient_bytes,
        master_parameter_bytes,
        moment_bytes,
    )
    if any(width < 0 for width in widths):
        raise ValueError("byte widths cannot be negative")

    bytes_per_parameter = (
        parameter_bytes
        + gradient_bytes
        + master_parameter_bytes
        + 2 * moment_bytes
    )
    return trainable_parameters * bytes_per_parameter / GIB
