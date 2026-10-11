"""Compare the three fixed paired-training ablations and select one winner."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

from .common import HOLDOUT_SUBJECT, TRAIN_SUBJECTS, Task3Error, atomic_json, sha256_file, utc_now
from .train import ARMS, load_checkpoint


def _load_arm(root: Path, arm: str) -> dict[str, Any]:
    run_dir = root / arm
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise Task3Error(f"Incomplete ablation arm {arm}: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(checkpoint_path)
    split = summary.get("split_manifest", {})
    if (
        summary.get("status") != "complete"
        or summary.get("phase") != "ablation"
        or summary.get("arm") != arm
        or split.get("train_subjects") != list(TRAIN_SUBJECTS)
        or split.get("holdout_subject") != HOLDOUT_SUBJECT
    ):
        raise Task3Error(f"Ablation arm has wrong status or subject split: {arm}")
    metrics = summary.get("best_metrics")
    if not metrics or set(metrics.get("mean", {})) != {"nrmse", "ssim", "lpips"}:
        raise Task3Error(f"Ablation arm lacks complete official metrics: {arm}")
    if (
        checkpoint.get("checkpoint_role") != "lowest_holdout_nrmse"
        or checkpoint.get("run_signature") != summary.get("run_signature")
        or checkpoint.get("split_manifest") != split
        or checkpoint.get("selected_global_step") != summary.get("best_step")
    ):
        raise Task3Error(f"Selected checkpoint and summary disagree for arm {arm}")
    return {
        "arm": arm,
        "nrmse": float(metrics["mean"]["nrmse"]),
        "ssim": float(metrics["mean"]["ssim"]),
        "lpips": float(metrics["mean"]["lpips"]),
        "identity_nrmse": float(metrics["identity_mean_nrmse"]),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "initialization": summary.get("initialization"),
        "run_signature": summary.get("run_signature"),
        "cache_manifest_sha256": split.get("cache_manifest_sha256"),
        "metric_provenance": summary.get("metric_provenance"),
    }


def select_winner(ablation_root: Path) -> dict[str, Any]:
    results = [_load_arm(ablation_root, arm) for arm in ARMS]
    cache_hashes = {result["cache_manifest_sha256"] for result in results}
    if any(not isinstance(result["metric_provenance"], dict) for result in results):
        raise Task3Error("An ablation arm lacks official scorer provenance")
    scorer_revisions = {
        (
            result["metric_provenance"].get("official_git_revision"),
            result["metric_provenance"].get("official_scorer_sha256"),
        )
        for result in results
    }
    if len(cache_hashes) != 1 or None in cache_hashes or len(scorer_revisions) != 1:
        raise Task3Error("Ablation arms do not share one data manifest and official scorer revision")
    expected_recipe = {
        "kind": "finetune",
        "all_subjects": False,
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
        "eval_every": 1000,
        "lpips_enabled": True,
    }
    for result in results:
        signature = result["run_signature"]
        if not isinstance(signature, dict) or any(
            signature.get(key) != value for key, value in expected_recipe.items()
        ):
            raise Task3Error(f"Ablation arm does not use the fixed v1 recipe: {result['arm']}")
        if signature.get("arm") != result["arm"]:
            raise Task3Error(f"Ablation signature has the wrong arm: {result['arm']}")
        if not isinstance(signature.get("runtime_gate"), dict):
            raise Task3Error(f"Ablation arm lacks a measured runtime gate: {result['arm']}")
        if not all(
            math.isfinite(result[key])
            for key in ("nrmse", "ssim", "lpips", "identity_nrmse")
        ):
            raise Task3Error(f"Ablation arm has non-finite metrics: {result['arm']}")
    scratch_initializations = {
        result["initialization"] for result in results if result["arm"] != "pretrained_masked"
    }
    runtime_hashes = {
        result["run_signature"].get("runtime_gate", {}).get("sha256") for result in results
    }
    holdout_counts = {
        result["run_signature"].get("holdout_samples_per_stratum") for result in results
    }
    if (
        scratch_initializations != {"scratch_seed_2026"}
        or len(runtime_hashes) != 1
        or None in runtime_hashes
        or len(holdout_counts) != 1
        or next(iter(holdout_counts), 0) < 1
    ):
        raise Task3Error("Ablation arms do not share one initialization/runtime/holdout recipe")
    pretrained_initialization = next(
        result["initialization"]
        for result in results
        if result["arm"] == "pretrained_masked"
    )
    if not isinstance(pretrained_initialization, dict) or not {
        "path",
        "sha256",
    } <= set(pretrained_initialization):
        raise Task3Error("pretrained_masked lacks its completed pretraining provenance")
    if len({result["identity_nrmse"] for result in results}) != 1:
        raise Task3Error("Ablation identity baselines differ; holdout patches were not identical")
    results.sort(key=lambda item: (item["nrmse"], -item["ssim"], item["lpips"]))
    winner = results[0]
    if winner["nrmse"] >= winner["identity_nrmse"]:
        raise Task3Error(
            f"Best arm {winner['arm']} nRMSE={winner['nrmse']:.6g} does not beat "
            f"source identity={winner['identity_nrmse']:.6g}; final retraining is blocked"
        )
    return {
        "kind": "task3_3d_unet_ablation_selection",
        "created_at_utc": utc_now(),
        "selection_order": ["lowest_nrmse", "highest_ssim", "lowest_lpips"],
        "results": results,
        "winner": winner,
        "final_retrain": {
            "arm": winner["arm"],
            "updates": 10_000,
            "subjects": ["P_0006", "P_0007", "P_0009"],
            "continue_holdout_checkpoint": False,
            "initialization": winner["initialization"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = select_winner(args.ablation_root.expanduser().resolve())
        atomic_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    except (Task3Error, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
