"""Masked-reconstruction and paired fine-tuning objectives."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .common import Task3Error


def block_mask(
    shape: tuple[int, ...],
    *,
    block_size: int = 12,
    fraction: float = 0.5,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a deterministic-with-generator block mask for BxCxDxHxW tensors."""
    if len(shape) != 5 or shape[1] != 1:
        raise Task3Error(f"Expected Bx1xDxHxW shape, found {shape}")
    if block_size < 1 or not 0.0 < fraction <= 1.0:
        raise Task3Error("block_size must be positive and fraction must be in (0,1]")
    batch, _, depth, height, width = shape
    grid = tuple((size + block_size - 1) // block_size for size in (depth, height, width))
    scores = torch.rand((batch, 1, *grid), generator=generator, device=device)
    block_count = grid[0] * grid[1] * grid[2]
    selected = max(1, round(block_count * fraction))
    threshold = scores.flatten(2).kthvalue(selected, dim=2).values[:, :, None, None, None]
    coarse = scores <= threshold
    mask = coarse.repeat_interleave(block_size, 2).repeat_interleave(block_size, 3)
    mask = mask.repeat_interleave(block_size, 4)
    return mask[:, :, :depth, :height, :width]


def normalized_l1(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise Task3Error("Prediction, target, and mask shapes must match")
    weights = mask.to(dtype=prediction.dtype)
    denominator = weights.sum()
    if float(denominator.detach()) == 0.0:
        return prediction.sum() * 0.0
    return ((prediction - target).abs() * weights).sum() / denominator


def masked_reconstruction_loss(prediction: Tensor, target: Tensor, corruption_mask: Tensor) -> Tensor:
    foreground = target > 1e-6
    return normalized_l1(prediction, target, corruption_mask & foreground)


def dilated_foreground(target: Tensor, kernel_size: int = 5) -> Tensor:
    if target.ndim != 5 or kernel_size < 1 or kernel_size % 2 == 0:
        raise Task3Error("Foreground dilation needs BxCxDxHxW and an odd positive kernel")
    return F.max_pool3d(
        (target > 1e-6).to(dtype=target.dtype),
        kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).bool()


def fine_tuning_loss(
    prediction: Tensor, target: Tensor, *, masked: bool
) -> tuple[Tensor, dict[str, Tensor]]:
    global_l1 = F.l1_loss(prediction, target)
    if not masked:
        return global_l1, {"global_l1": global_l1}
    foreground_l1 = normalized_l1(prediction, target, dilated_foreground(target))
    total = 0.9 * foreground_l1 + 0.1 * global_l1
    return total, {"global_l1": global_l1, "foreground_l1": foreground_l1}
