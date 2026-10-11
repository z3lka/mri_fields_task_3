"""Load the organizers' current Task 3 scorer instead of reimplementing metrics."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .common import Task3Error, git_revision, sha256_file


@lru_cache(maxsize=2)
def load_official_scorer(official_repo: Path) -> tuple[ModuleType, dict[str, str]]:
    official_repo = official_repo.expanduser().resolve()
    score_path = official_repo / "Submission" / "evaluation-2026" / "score.py"
    if not score_path.is_file():
        raise Task3Error(f"Official scorer not found: {score_path}")
    spec = importlib.util.spec_from_file_location("mrixfields_task3_official_score", score_path)
    if spec is None or spec.loader is None:
        raise Task3Error(f"Could not load official scorer: {score_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for function in ("compute_nrmse", "compute_ssim", "compute_lpips"):
        if not callable(getattr(module, function, None)):
            raise Task3Error(f"Official scorer lacks {function}: {score_path}")
    return module, {
        "official_git_revision": git_revision(official_repo),
        "official_scorer": str(score_path),
        "official_scorer_sha256": sha256_file(score_path),
    }


def score_arrays(
    scorer: ModuleType,
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    lpips_device: str,
    include_lpips: bool = True,
) -> dict[str, float]:
    result = {
        "nrmse": float(scorer.compute_nrmse(prediction, target)),
        "ssim": float(scorer.compute_ssim(prediction, target)),
    }
    if include_lpips:
        result["lpips"] = float(
            scorer.compute_lpips(prediction, target, device=lpips_device)
        )
    return result
