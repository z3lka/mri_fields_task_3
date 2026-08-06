#!/usr/bin/env python3
"""Conditional FPS-Former backbone adapted to MRIxFields2026 Task 3.

The implementation follows arXiv:2412.10776v1:

* a four-level hierarchical encoder/decoder;
* frequency modulation attention (FMAM) from a Gaussian/Laplacian pyramid;
* hash-sorted spatial purification attention (SPAM);
* the multi-scale depthwise feed-forward network (SDFN); and
* hybrid-expert feature refinement (HEFR).

Task 3 is image-to-image field translation, not undersampled k-space
reconstruction.  Consequently, the paper's k-space data-consistency layer is
not applicable.  A single shared network is instead conditioned on modality,
source field, and target field.  For practical full-resolution training,
``portable`` bounds the quadratic attention token counts while retaining the
paper's content grouping.  ``paper_exact`` removes those bounds and can require
substantially more than Colab-class memory.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from task3_fpsformer.common import FIELDS, MODALITIES, NUM_DOMAINS, FPSFormerError


def _valid_heads(channels: int, heads: int) -> None:
    if heads < 1 or channels % heads:
        raise FPSFormerError(f"channels={channels} must be divisible by heads={heads}")


def _pooled_shape(height: int, width: int, max_tokens: int) -> tuple[int, int]:
    if max_tokens <= 0 or height * width <= max_tokens:
        return height, width
    scale = math.sqrt(max_tokens / float(height * width))
    pooled_height = max(1, int(height * scale))
    pooled_width = max(1, int(width * scale))
    while pooled_height * pooled_width > max_tokens:
        if pooled_width >= pooled_height:
            pooled_width -= 1
        else:
            pooled_height -= 1
    return pooled_height, pooled_width


def _attention(query: Tensor, key: Tensor, value: Tensor, key_valid: Tensor | None = None) -> Tensor:
    """MPS-safe scaled dot-product attention.

    Inputs are ``(batch, heads, tokens, head_dim)``.  Attention logits are
    accumulated in float32 for stable mixed-precision training.
    """
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    if key_valid is not None:
        mask = key_valid[:, None, None, :]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = scores.softmax(dim=-1).to(value.dtype)
    return torch.matmul(weights, value)


class ConditionalLayerNorm2d(nn.Module):
    """Channels-last layer norm with shared conditional affine modulation."""

    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.affine = nn.Linear(condition_dim, channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, image: Tensor, condition: Tensor) -> Tensor:
        normalized = self.norm(image.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        scale, shift = self.affine(condition).chunk(2, dim=1)
        return normalized * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class ConditionEncoder(nn.Module):
    """One parameter-sharing encoding of modality and translation direction."""

    def __init__(self, condition_dim: int):
        super().__init__()
        embedding_dim = max(16, condition_dim // 4)
        self.modality = nn.Embedding(len(MODALITIES), embedding_dim)
        self.source_field = nn.Embedding(len(FIELDS), embedding_dim)
        self.target_field = nn.Embedding(len(FIELDS), embedding_dim)
        self.joint_target = nn.Embedding(NUM_DOMAINS, embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * 4, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
        )

    def forward(self, modality: Tensor, source_field: Tensor, target_field: Tensor) -> Tensor:
        if modality.ndim != 1 or source_field.ndim != 1 or target_field.ndim != 1:
            raise FPSFormerError("Condition indices must be one-dimensional batch tensors")
        if not (modality.shape == source_field.shape == target_field.shape):
            raise FPSFormerError("Condition index batches must have identical shapes")
        joint_target = modality * len(FIELDS) + target_field
        encoded = torch.cat(
            (
                self.modality(modality),
                self.source_field(source_field),
                self.target_field(target_field),
                self.joint_target(joint_target),
            ),
            dim=1,
        )
        return self.projection(encoded)


def _gaussian_kernel(sigma: float) -> Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d[None, None]


class FrequencyModulationAttention(nn.Module):
    """FMAM using M Laplacian bands and within-frequency attention."""

    def __init__(
        self,
        channels: int,
        heads: int,
        levels: int = 3,
        max_tokens: int = 256,
    ):
        super().__init__()
        _valid_heads(channels, heads)
        if levels < 1:
            raise FPSFormerError("FMAM needs at least one frequency level")
        self.channels = channels
        self.heads = heads
        self.levels = levels
        self.max_tokens = max_tokens
        for level in range(levels + 1):
            self.register_buffer(f"gaussian_{level}", _gaussian_kernel(float(level + 1)))
        self.query = nn.ModuleList(nn.Conv2d(channels, channels, 1) for _ in range(levels))
        self.key = nn.ModuleList(nn.Conv2d(channels, channels, 1) for _ in range(levels))
        self.value = nn.Conv2d(channels, channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)

    def _blur(self, image: Tensor, kernel: Tensor) -> Tensor:
        radius = kernel.shape[-1] // 2
        padding_mode = "reflect" if min(image.shape[-2:]) > radius else "replicate"
        padded = F.pad(image, (radius, radius, radius, radius), mode=padding_mode)
        expanded = kernel.to(dtype=image.dtype).expand(self.channels, 1, -1, -1)
        return F.conv2d(padded, expanded, groups=self.channels)

    def _tokens(self, image: Tensor) -> Tensor:
        batch, channels, height, width = image.shape
        head_dim = channels // self.heads
        return image.reshape(batch, self.heads, head_dim, height * width).transpose(-2, -1)

    def forward(self, image: Tensor) -> Tensor:
        height, width = image.shape[-2:]
        pooled_height, pooled_width = _pooled_shape(height, width, self.max_tokens)
        if (pooled_height, pooled_width) == (height, width):
            working = image
        else:
            working = F.adaptive_avg_pool2d(image, (pooled_height, pooled_width))

        value = self._tokens(self.value(working))
        previous = self._blur(working, getattr(self, "gaussian_0"))
        attended: Tensor | None = None
        for level in range(self.levels):
            current = self._blur(working, getattr(self, f"gaussian_{level + 1}"))
            band = current - previous
            query = self._tokens(self.query[level](band))
            key = self._tokens(self.key[level](band))
            level_output = _attention(query, key, value)
            attended = level_output if attended is None else attended + level_output
            previous = current
        if attended is None:  # pragma: no cover - guarded in __init__
            raise AssertionError("FMAM produced no levels")
        attended = attended / float(self.levels)
        batch = image.shape[0]
        output = attended.transpose(-2, -1).reshape(
            batch, self.channels, pooled_height, pooled_width
        )
        if output.shape[-2:] != (height, width):
            output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
        return self.output(output)


class SpatialPurificationAttention(nn.Module):
    """SPAM with random-projection hashes and within-group MSA."""

    def __init__(
        self,
        channels: int,
        heads: int,
        groups: int = 4,
        max_group_tokens: int = 128,
        hash_width: float = 1.0,
    ):
        super().__init__()
        _valid_heads(channels, heads)
        if groups < 1 or max_group_tokens < 0 or hash_width <= 0:
            raise FPSFormerError("Invalid SPAM grouping configuration")
        self.channels = channels
        self.heads = heads
        self.groups = groups
        self.max_group_tokens = max_group_tokens
        self.hash_width = hash_width
        self.register_buffer("hash_vector", torch.randn(channels))
        self.register_buffer("hash_bias", torch.rand(()) * hash_width)
        self.qkv = nn.Linear(channels, channels * 3)
        self.output = nn.Linear(channels, channels)

    def forward(self, image: Tensor) -> Tensor:
        batch, channels, height, width = image.shape
        tokens = image.flatten(2).transpose(1, 2)
        token_count = tokens.shape[1]
        hashes = torch.floor(
            (torch.matmul(tokens.float(), self.hash_vector.float()) + self.hash_bias)
            / self.hash_width
        )
        order = hashes.argsort(dim=1, stable=True)
        sorted_tokens = tokens.gather(1, order[:, :, None].expand(-1, -1, channels))

        group_count = self.groups
        if self.max_group_tokens:
            group_count = max(group_count, math.ceil(token_count / self.max_group_tokens))
        group_size = math.ceil(token_count / group_count)
        padded_count = group_count * group_size
        padding = padded_count - token_count
        if padding:
            sorted_tokens = F.pad(sorted_tokens, (0, 0, 0, padding))
        valid = torch.arange(padded_count, device=image.device)[None, :] < token_count
        valid = valid.expand(batch, -1).reshape(batch * group_count, group_size)

        grouped = sorted_tokens.reshape(batch * group_count, group_size, channels)
        head_dim = channels // self.heads
        qkv = self.qkv(grouped).reshape(
            batch * group_count, group_size, 3, self.heads, head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = _attention(query, key, value, valid)
        attended = attended.transpose(1, 2).reshape(
            batch * group_count, group_size, channels
        )
        attended = self.output(attended).reshape(batch, padded_count, channels)
        attended = attended[:, :token_count]

        restored = torch.zeros_like(tokens)
        restored.scatter_(1, order[:, :, None].expand(-1, -1, channels), attended)
        return restored.transpose(1, 2).reshape(batch, channels, height, width)


class ScaleDiversificationFeedForward(nn.Module):
    """SDFN with two rounds of 3x3/5x5 depthwise feature mixing."""

    def __init__(self, channels: int, expansion: float = 2.0):
        super().__init__()
        hidden = max(channels, int(round(channels * expansion)))
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.first_3 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.first_5 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        mixed = hidden * 2
        self.second_3 = nn.Conv2d(mixed, mixed, 3, padding=1, groups=mixed)
        self.second_5 = nn.Conv2d(mixed, mixed, 5, padding=2, groups=mixed)
        self.project = nn.Conv2d(mixed * 2, channels, 1)

    def forward(self, image: Tensor) -> Tensor:
        expanded = self.expand(image)
        path_3 = F.relu(self.first_3(expanded), inplace=False)
        path_5 = F.relu(self.first_5(expanded), inplace=False)
        mixed = torch.cat((path_3, path_5), dim=1)
        path_3 = F.relu(self.second_3(mixed), inplace=False)
        path_5 = F.relu(self.second_5(mixed), inplace=False)
        return self.project(torch.cat((path_3, path_5), dim=1))


class SeparableConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.layers = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
            ),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.layers(image)


class AveragePoolExpert(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, image: Tensor) -> Tensor:
        return F.avg_pool2d(
            image,
            self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            count_include_pad=False,
        )


class HybridExpertsFeatureRefinement(nn.Module):
    """HEFR with eight lightweight convolution/pooling experts."""

    def __init__(self, channels: int, experts: int = 8, gate_dim: int = 32):
        super().__init__()
        candidates: list[nn.Module] = [
            AveragePoolExpert(3),
            AveragePoolExpert(5),
            AveragePoolExpert(7),
            SeparableConv(channels, 3),
            SeparableConv(channels, 5),
            SeparableConv(channels, 3, dilation=2),
            SeparableConv(channels, 3, dilation=3),
            SeparableConv(channels, 3, dilation=4),
        ]
        if not 1 <= experts <= len(candidates):
            raise FPSFormerError(f"HEFR experts must be in [1, {len(candidates)}]")
        self.experts = nn.ModuleList(candidates[:experts])
        self.switcher = nn.Sequential(
            nn.Linear(channels, gate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_dim, experts),
        )
        self.fuse = nn.Conv2d(channels * experts, channels, 1)

    def forward(self, image: Tensor) -> Tensor:
        descriptor = image.mean(dim=(2, 3))
        weights = self.switcher(descriptor).softmax(dim=1)
        outputs = [
            expert(image) * weights[:, index, None, None, None]
            for index, expert in enumerate(self.experts)
        ]
        return image + self.fuse(torch.cat(outputs, dim=1))


class FPSBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        condition_dim: int,
        *,
        frequency_levels: int,
        spam_groups: int,
        sdfn_expansion: float,
        max_frequency_tokens: int,
        max_spam_group_tokens: int,
    ):
        super().__init__()
        self.norm_attention = ConditionalLayerNorm2d(channels, condition_dim)
        self.frequency = FrequencyModulationAttention(
            channels, heads, frequency_levels, max_frequency_tokens
        )
        self.spatial = SpatialPurificationAttention(
            channels, heads, spam_groups, max_spam_group_tokens
        )
        self.aggregate = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
            nn.Conv2d(channels * 2, channels, 1),
        )
        self.norm_feedforward = ConditionalLayerNorm2d(channels, condition_dim)
        self.feedforward = ScaleDiversificationFeedForward(channels, sdfn_expansion)

    def forward(self, image: Tensor, condition: Tensor) -> Tensor:
        normalized = self.norm_attention(image, condition)
        mixed = self.aggregate(
            torch.cat((self.frequency(normalized), self.spatial(normalized)), dim=1)
        )
        image = image + mixed
        return image + self.feedforward(self.norm_feedforward(image, condition))


class FPSStage(nn.Module):
    def __init__(self, blocks: Sequence[FPSBlock]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, image: Tensor, condition: Tensor) -> Tensor:
        for block in self.blocks:
            image = block(image, condition)
        return image


@dataclass(frozen=True)
class ModelConfig:
    base_channels: int = 48
    depths: tuple[int, int, int, int] = (1, 2, 2, 1)
    heads: tuple[int, int, int, int] = (1, 2, 4, 8)
    hefr_blocks: int = 4
    hefr_experts: int = 8
    hefr_gate_dim: int = 32
    frequency_levels: int = 3
    spam_groups: int = 4
    sdfn_expansion: float = 2.0
    condition_dim: int = 128
    max_frequency_tokens: int = 256
    max_spam_group_tokens: int = 128
    residual_logit_scale: float = 1.0

    def __post_init__(self) -> None:
        if len(self.depths) != 4 or len(self.heads) != 4:
            raise FPSFormerError("FPS-Former requires exactly four stages")
        if self.base_channels < 4 or self.hefr_blocks < 0:
            raise FPSFormerError("Invalid FPS-Former width/depth")
        for stage, (depth, heads) in enumerate(zip(self.depths, self.heads)):
            if depth < 1:
                raise FPSFormerError(f"Stage {stage} depth must be positive")
            _valid_heads(self.base_channels * (2**stage), heads)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise FPSFormerError(f"Unknown model configuration keys: {sorted(unknown)}")
        converted = dict(values)
        for key in ("depths", "heads"):
            if key in converted:
                converted[key] = tuple(int(value) for value in converted[key])
        return cls(**converted)

    @classmethod
    def preset(cls, name: str) -> "ModelConfig":
        if name == "portable":
            return cls()
        if name == "paper_exact":
            return cls(max_frequency_tokens=0, max_spam_group_tokens=0)
        if name == "tiny":
            return cls(
                base_channels=16,
                depths=(1, 1, 1, 1),
                heads=(1, 2, 4, 8),
                hefr_blocks=1,
                hefr_experts=4,
                condition_dim=64,
                max_frequency_tokens=64,
                max_spam_group_tokens=64,
            )
        raise FPSFormerError(f"Unknown model preset {name!r}")


def _make_stage(config: ModelConfig, stage: int) -> FPSStage:
    channels = config.base_channels * (2**stage)
    return FPSStage(
        [
            FPSBlock(
                channels,
                config.heads[stage],
                config.condition_dim,
                frequency_levels=config.frequency_levels,
                spam_groups=config.spam_groups,
                sdfn_expansion=config.sdfn_expansion,
                max_frequency_tokens=config.max_frequency_tokens,
                max_spam_group_tokens=config.max_spam_group_tokens,
            )
            for _ in range(config.depths[stage])
        ]
    )


class ConditionalFPSFormer(nn.Module):
    """One jointly conditioned model for all 3 modalities and 20 field pairs."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        widths = [config.base_channels * (2**stage) for stage in range(4)]
        self.condition = ConditionEncoder(config.condition_dim)
        self.stem = nn.Conv2d(1, widths[0], 3, padding=1)
        self.early_refinement = nn.Sequential(
            *[
                HybridExpertsFeatureRefinement(
                    widths[0], config.hefr_experts, config.hefr_gate_dim
                )
                for _ in range(config.hefr_blocks)
            ]
        )
        self.encoder = nn.ModuleList(_make_stage(config, stage) for stage in range(4))
        self.downsample = nn.ModuleList(
            nn.Conv2d(widths[stage], widths[stage + 1], 3, stride=2, padding=1)
            for stage in range(3)
        )
        self.upsample = nn.ModuleList(
            nn.Conv2d(widths[stage + 1], widths[stage], 3, padding=1)
            for stage in range(3)
        )
        self.skip_fuse = nn.ModuleList(
            nn.Conv2d(widths[stage] * 2, widths[stage], 1) for stage in range(3)
        )
        self.decoder = nn.ModuleList(_make_stage(config, stage) for stage in range(3))
        self.final_refinement = nn.Sequential(
            *[
                HybridExpertsFeatureRefinement(
                    widths[0], config.hefr_experts, config.hefr_gate_dim
                )
                for _ in range(config.hefr_blocks)
            ]
        )
        self.output = nn.Conv2d(widths[0], 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _pad(image: Tensor) -> tuple[Tensor, tuple[int, int]]:
        height, width = image.shape[-2:]
        pad_height = (-height) % 8
        pad_width = (-width) % 8
        if pad_height or pad_width:
            mode = "reflect" if min(height, width) > max(pad_height, pad_width) else "replicate"
            image = F.pad(image, (0, pad_width, 0, pad_height), mode=mode)
        return image, (height, width)

    def forward(
        self,
        source: Tensor,
        modality: Tensor,
        source_field: Tensor,
        target_field: Tensor,
    ) -> Tensor:
        if source.ndim != 4 or source.shape[1] != 1:
            raise FPSFormerError(f"Expected source shape (B,1,H,W), found {tuple(source.shape)}")
        if source.shape[0] != modality.shape[0]:
            raise FPSFormerError("Image and condition batch sizes differ")
        if bool(torch.any(source_field == target_field)):
            raise FPSFormerError("Task 3 must not generate identity field mappings")
        condition = self.condition(
            modality.long(), source_field.long(), target_field.long()
        )
        padded, original_shape = self._pad(source)
        image = self.early_refinement(self.stem(padded))
        skips: list[Tensor] = []
        for stage in range(4):
            image = self.encoder[stage](image, condition)
            skips.append(image)
            if stage < 3:
                image = self.downsample[stage](image)

        for stage in reversed(range(3)):
            image = F.interpolate(
                image, size=skips[stage].shape[-2:], mode="bilinear", align_corners=False
            )
            image = self.upsample[stage](image)
            image = self.skip_fuse[stage](torch.cat((image, skips[stage]), dim=1))
            image = self.decoder[stage](image, condition)
        image = self.final_refinement(image)
        delta = self.output(image) * self.config.residual_logit_scale

        epsilon = 1e-4
        base = padded.clamp(epsilon, 1.0 - epsilon)
        prediction = torch.sigmoid(torch.logit(base) + delta)
        return prediction[:, :, : original_shape[0], : original_shape[1]]


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_model(config: ModelConfig | dict[str, Any]) -> ConditionalFPSFormer:
    if isinstance(config, dict):
        config = ModelConfig.from_dict(config)
    return ConditionalFPSFormer(config)
