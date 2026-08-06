#!/usr/bin/env python3
"""Training losses and validation proxies for Task 3 FPS-Former."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from task3_fpsformer.common import MODALITIES


def _gaussian_window(size: int, sigma: float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    coordinates = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    window = torch.exp(-(coordinates**2) / (2 * sigma**2))
    window /= window.sum()
    kernel = window[:, None] * window[None, :]
    return kernel.to(dtype=dtype)[None, None]


def ssim_per_image(prediction: Tensor, target: Tensor, window_size: int = 11) -> Tensor:
    """Differentiable SSIM for single-channel images in [0, 1]."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(f"SSIM shapes differ or are not BCHW: {prediction.shape}/{target.shape}")
    effective = min(window_size, prediction.shape[-2], prediction.shape[-1])
    if effective % 2 == 0:
        effective -= 1
    effective = max(effective, 1)
    sigma = max(0.5, 1.5 * effective / 11.0)
    kernel = _gaussian_window(
        effective, sigma, device=prediction.device, dtype=prediction.dtype
    )
    padding = effective // 2
    mean_prediction = F.conv2d(prediction, kernel, padding=padding)
    mean_target = F.conv2d(target, kernel, padding=padding)
    prediction_sq = mean_prediction.square()
    target_sq = mean_target.square()
    cross = mean_prediction * mean_target
    variance_prediction = F.conv2d(prediction.square(), kernel, padding=padding) - prediction_sq
    variance_target = F.conv2d(target.square(), kernel, padding=padding) - target_sq
    covariance = F.conv2d(prediction * target, kernel, padding=padding) - cross
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * cross + c1) * (2 * covariance + c2)) / (
        (prediction_sq + target_sq + c1)
        * (variance_prediction + variance_target + c2)
    )
    return score.flatten(1).mean(dim=1)


def gradient_l1(prediction: Tensor, target: Tensor) -> Tensor:
    pred_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


class OptionalLPIPS(nn.Module):
    """Lazy LPIPS wrapper; no network or weight download occurs unless requested."""

    def __init__(self, enabled: bool):
        super().__init__()
        self.enabled = enabled
        self.model: nn.Module | None = None
        if enabled:
            try:
                import lpips
            except ImportError as exc:
                raise RuntimeError(
                    "LPIPS was requested but is not installed; install lpips==0.1.4"
                ) from exc
            self.model = lpips.LPIPS(net="alex").eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if not self.enabled or self.model is None:
            return prediction.new_zeros(())
        prediction_3 = prediction.repeat(1, 3, 1, 1) * 2.0 - 1.0
        target_3 = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
        return self.model(prediction_3, target_3).mean()


@dataclass(frozen=True)
class LossWeights:
    l1: float = 1.0
    ssim: float = 0.0
    edge: float = 0.0
    lpips: float = 0.0

    def __post_init__(self) -> None:
        if min(self.l1, self.ssim, self.edge, self.lpips) < 0:
            raise ValueError("Loss weights cannot be negative")
        if self.l1 + self.ssim + self.edge + self.lpips <= 0:
            raise ValueError("At least one training loss must be enabled")


class TranslationLoss(nn.Module):
    def __init__(self, weights: LossWeights):
        super().__init__()
        self.weights = weights
        self.perceptual = OptionalLPIPS(weights.lpips > 0)

    def forward(self, prediction: Tensor, target: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        terms = {
            "l1": F.l1_loss(prediction, target),
            "ssim": 1.0 - ssim_per_image(prediction, target).mean(),
            "edge": gradient_l1(prediction, target),
        }
        if self.weights.lpips:
            terms["lpips"] = self.perceptual(prediction, target)
        else:
            terms["lpips"] = prediction.new_zeros(())
        total = (
            terms["l1"] * self.weights.l1
            + terms["ssim"] * self.weights.ssim
            + terms["edge"] * self.weights.edge
            + terms["lpips"] * self.weights.lpips
        )
        return total, terms


class MetricAccumulator:
    """Aggregate slice proxies without averaging nRMSE ratios per slice."""

    def __init__(self):
        self.totals: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "squared_error": 0.0,
                "squared_target": 0.0,
                "ssim": 0.0,
                "lpips": 0.0,
                "samples": 0.0,
            }
        )

    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        modality_indices: Tensor,
        lpips_model: OptionalLPIPS | None = None,
    ) -> None:
        with torch.no_grad():
            prediction = prediction.float()
            target = target.float()
            ssim = ssim_per_image(prediction, target)
            for index in range(prediction.shape[0]):
                modality = MODALITIES[int(modality_indices[index])]
                item = self.totals[modality]
                item["squared_error"] += float(
                    (prediction[index] - target[index]).square().sum()
                )
                item["squared_target"] += float(target[index].square().sum())
                item["ssim"] += float(ssim[index])
                if lpips_model is not None and lpips_model.enabled:
                    item["lpips"] += float(
                        lpips_model(
                            prediction[index : index + 1], target[index : index + 1]
                        )
                    )
                item["samples"] += 1.0

    @staticmethod
    def _summarize(item: dict[str, float], include_lpips: bool) -> dict[str, float | int]:
        samples = max(1, int(item["samples"]))
        summary: dict[str, float | int] = {
            "slice_proxy_nrmse": (
                item["squared_error"] / max(item["squared_target"], 1e-12)
            )
            ** 0.5,
            "slice_proxy_ssim": item["ssim"] / samples,
            "samples": int(item["samples"]),
        }
        if include_lpips:
            summary["slice_proxy_lpips"] = item["lpips"] / samples
        return summary

    def summarize(self, *, include_lpips: bool) -> dict[str, Any]:
        combined = {
            key: sum(item[key] for item in self.totals.values())
            for key in ("squared_error", "squared_target", "ssim", "lpips", "samples")
        }
        return {
            "overall": self._summarize(combined, include_lpips),
            "by_modality": {
                modality: self._summarize(self.totals[modality], include_lpips)
                for modality in MODALITIES
                if self.totals[modality]["samples"]
            },
        }
