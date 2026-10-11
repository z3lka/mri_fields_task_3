"""One FiLM-conditioned residual 3D U-Net shared by every Task 3 direction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import FIELDS, MODALITIES, Task3Error


CHECKPOINT_FORMAT = "task3_conditional_3d_unet_v1"
ARCHITECTURE_ID = "conditional_residual_3d_unet_v1"


def _groups(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


@dataclass(frozen=True)
class ModelConfig:
    widths: tuple[int, ...] = (24, 48, 96, 192)
    blocks_per_level: int = 2
    embedding_dim: int = 16
    condition_dim: int = 128
    input_channels: int = 1

    def __post_init__(self) -> None:
        if len(self.widths) != 4 or min(self.widths) < 1:
            raise Task3Error("Model widths must contain four positive values")
        if self.blocks_per_level < 1 or self.embedding_dim < 1 or self.condition_dim < 1:
            raise Task3Error("Model dimensions and block count must be positive")
        if self.input_channels != 1:
            raise Task3Error("Task 3 volumes must be single-channel")

    @classmethod
    def tiny(cls) -> "ModelConfig":
        return cls(widths=(4, 8, 16, 32), blocks_per_level=1, embedding_dim=4, condition_dim=16)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["widths"] = list(self.widths)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise Task3Error(
                f"Strict model config keys differ: missing={sorted(expected - set(payload))}, "
                f"extra={sorted(set(payload) - expected)}"
            )
        values = dict(payload)
        values["widths"] = tuple(int(value) for value in values["widths"])
        return cls(**values)


class FiLMResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.film = nn.Linear(condition_dim, 2 * out_channels)
        self.skip = (
            nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)
        )

    def forward(self, source: Tensor, condition: Tensor) -> Tensor:
        residual = self.skip(source)
        hidden = self.conv1(F.silu(self.norm1(source)))
        scale, shift = self.film(condition).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None, None])
        hidden = hidden + shift[:, :, None, None, None]
        return residual + self.conv2(F.silu(hidden))


class ConditionalUNet3D(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        widths = config.widths
        self.modality_embedding = nn.Embedding(len(MODALITIES), config.embedding_dim)
        self.source_embedding = nn.Embedding(len(FIELDS), config.embedding_dim)
        self.target_embedding = nn.Embedding(len(FIELDS), config.embedding_dim)
        self.condition = nn.Sequential(
            nn.Linear(3 * config.embedding_dim, config.condition_dim), nn.SiLU()
        )
        self.mask_value = nn.Parameter(torch.zeros(()))
        self.stem = nn.Conv3d(config.input_channels, widths[0], 3, padding=1)

        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for level, width in enumerate(widths):
            self.encoder.append(
                nn.ModuleList(
                    FiLMResidualBlock(width, width, config.condition_dim)
                    for _ in range(config.blocks_per_level)
                )
            )
            if level < len(widths) - 1:
                self.downsample.append(nn.Conv3d(width, widths[level + 1], 3, stride=2, padding=1))

        self.decoder = nn.ModuleList()
        for level in range(len(widths) - 2, -1, -1):
            blocks = [
                FiLMResidualBlock(widths[level + 1] + widths[level], widths[level], config.condition_dim)
            ]
            blocks.extend(
                FiLMResidualBlock(widths[level], widths[level], config.condition_dim)
                for _ in range(config.blocks_per_level - 1)
            )
            self.decoder.append(nn.ModuleList(blocks))
        self.output = nn.Conv3d(widths[0], 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _labels(labels: Tensor | int, batch: int, upper: int, device: torch.device, name: str) -> Tensor:
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        if labels.ndim == 0:
            labels = labels.repeat(batch)
        if labels.shape != (batch,) or bool(((labels < 0) | (labels >= upper)).any()):
            raise Task3Error(f"{name} must have shape ({batch},) with values in [0,{upper})")
        return labels

    def apply_mask(self, source: Tensor, mask: Tensor) -> Tensor:
        if mask.shape != source.shape or mask.dtype != torch.bool:
            raise Task3Error("Mask must be boolean and match the source tensor")
        return torch.where(mask, self.mask_value.to(dtype=source.dtype), source)

    def forward(
        self,
        source: Tensor,
        modality: Tensor | int,
        source_field: Tensor | int,
        target_field: Tensor | int,
        *,
        allow_identity: bool = False,
    ) -> Tensor:
        if source.ndim != 5 or source.shape[1] != 1:
            raise Task3Error(f"Expected source shape Bx1xDxHxW, found {tuple(source.shape)}")
        batch = source.shape[0]
        modality = self._labels(modality, batch, len(MODALITIES), source.device, "modality")
        source_field = self._labels(source_field, batch, len(FIELDS), source.device, "source_field")
        target_field = self._labels(target_field, batch, len(FIELDS), source.device, "target_field")
        if not allow_identity and bool((source_field == target_field).any()):
            raise Task3Error("Task 3 inference/fine-tuning must not use identity mappings")
        condition = self.condition(
            torch.cat(
                (
                    self.modality_embedding(modality),
                    self.source_embedding(source_field),
                    self.target_embedding(target_field),
                ),
                dim=1,
            )
        )

        hidden = self.stem(source)
        skips: list[Tensor] = []
        for level, blocks in enumerate(self.encoder):
            for block in blocks:
                hidden = block(hidden, condition)
            skips.append(hidden)
            if level < len(self.downsample):
                hidden = self.downsample[level](hidden)
        for blocks, skip in zip(self.decoder, reversed(skips[:-1])):
            hidden = F.interpolate(hidden, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            hidden = torch.cat((hidden, skip), dim=1)
            for block in blocks:
                hidden = block(hidden, condition)
        delta = self.output(hidden)
        source_logit = torch.logit(source.clamp(1e-6, 1.0 - 1e-6))
        return torch.sigmoid(source_logit + delta)


def build_model(config: ModelConfig | dict[str, Any] | None = None) -> ConditionalUNet3D:
    if config is None:
        config = ModelConfig()
    if isinstance(config, dict):
        config = ModelConfig.from_dict(config)
    return ConditionalUNet3D(config)
