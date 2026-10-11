#!/usr/bin/env python3
"""Visual review and optional paired metrics for MRIxFields2026 Task 3.

This script understands the raw output layout produced by task3_pipeline.py.
Visual review and numerical output diagnostics do not require ground truth.
Official quality metrics require genuine, subject-matched target volumes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np


FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
MODALITIES = ("T1W", "T2W", "T2FLAIR")
RAW_PAIR_RE = re.compile(
    r"^task3_(0\.1T|1\.5T|3T|5T|7T)_to_"
    r"(0\.1T|1\.5T|3T|5T|7T)_(T1W|T2W|T2FLAIR)$"
)


class ReviewError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    path: Path
    modality: str
    source: str
    target: str
    subject: str


def subject_id(path: Path) -> str:
    name = path.name.removesuffix(".nii.gz")
    parts = name.split("_")
    if len(parts) < 4:
        raise ReviewError(f"Cannot parse subject ID: {path}")
    return parts[-1]


def parse_prediction(path: Path) -> Prediction | None:
    # Raw task3_pipeline.py layout.
    for parent in path.parents:
        match = RAW_PAIR_RE.fullmatch(parent.name)
        if match:
            source, target, modality = match.groups()
            return Prediction(path, modality, source, target, subject_id(path))

    # Built submission layout: MODALITY/source_to_target/pred/file.nii.gz.
    if path.parent.name == "pred" and len(path.parents) >= 3:
        pair = path.parent.parent.name
        modality = path.parent.parent.parent.name
        if modality in MODALITIES and "_to_" in pair:
            source, target = pair.split("_to_", 1)
            if source in FIELDS and target in FIELDS:
                return Prediction(path, modality, source, target, subject_id(path))
    return None


def discover_predictions(root: Path, modalities: Sequence[str]) -> list[Prediction]:
    found: dict[tuple[str, str, str, str], Prediction] = {}
    for path in sorted(root.rglob("*.nii.gz")):
        item = parse_prediction(path)
        if item is None or item.modality not in modalities:
            continue
        key = (item.modality, item.source, item.target, item.subject)
        if key in found:
            raise ReviewError(f"Duplicate prediction for {key}: {found[key].path} and {path}")
        found[key] = item
    if not found:
        raise ReviewError(f"No Task 3 predictions found under {root}")
    return list(found.values())


def field_directory(root: Path, modality: str, field: str) -> Path:
    choices = (root / modality / field, root / field)
    matches = [p for p in choices if p.is_dir()]
    if len(matches) != 1:
        raise ReviewError(
            f"Cannot uniquely locate {modality}/{field} under {root}; checked: "
            + ", ".join(str(p) for p in choices)
        )
    return matches[0]


def find_volume(root: Path, modality: str, field: str, subject: str) -> Path:
    directory = field_directory(root, modality, field)
    matches = sorted(directory.glob(f"*_{modality}_{field}_{subject}.nii.gz"))
    if len(matches) != 1:
        raise ReviewError(
            f"Expected one matched volume for {modality}/{field}/{subject}, "
            f"found {len(matches)} in {directory}"
        )
    return matches[0]


def load_array(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = image.get_fdata(dtype=np.float32)
    return image, data


def diagnostic_row(item: Prediction, source_path: Path) -> dict[str, object]:
    pred_image, pred = load_array(item.path)
    source_image, source = load_array(source_path)
    if pred.shape != source.shape:
        raise ReviewError(f"Shape mismatch: {item.path} {pred.shape} vs {source_path} {source.shape}")
    finite = np.isfinite(pred)
    slab = pred[:, :, 150:180] if pred.shape[2] >= 180 else pred
    source_slab = source[:, :, 150:180] if source.shape[2] >= 180 else source
    finite_slab = np.isfinite(slab)
    row = {
        "modality": item.modality,
        "source_field": item.source,
        "target_field": item.target,
        "subject": item.subject,
        "prediction": str(item.path),
        "shape": "x".join(str(v) for v in pred.shape),
        "dtype": str(pred_image.get_data_dtype()),
        "affine_matches_source": bool(
            np.allclose(pred_image.affine, source_image.affine, rtol=0.0, atol=1e-5)
        ),
        "nonfinite_voxels": int(pred.size - finite.sum()),
        "out_of_range_voxels": int(np.count_nonzero(finite & ((pred < 0.0) | (pred > 1.0)))),
        "slab_min": float(slab[finite_slab].min()) if finite_slab.any() else float("nan"),
        "slab_max": float(slab[finite_slab].max()) if finite_slab.any() else float("nan"),
        "slab_mean": float(slab[finite_slab].mean()) if finite_slab.any() else float("nan"),
        # This measures how much the translation changed the input. It is not an error metric.
        "slab_mean_abs_change_from_source": float(
            np.mean(np.abs(slab.astype(np.float64) - source_slab.astype(np.float64)))
        ),
    }
    del pred, source
    return row


def view_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    index = min(max(index, 0), volume.shape[axis] - 1)
    result = np.take(volume, index, axis=axis)
    return result.T


def make_montages(
    predictions: Iterable[Prediction], input_root: Path, report_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    groups: dict[tuple[str, str, str], list[Prediction]] = defaultdict(list)
    for item in predictions:
        groups[(item.modality, item.source, item.subject)].append(item)

    montage_dir = report_dir / "montages"
    montage_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for (modality, source_field, subject), items in sorted(groups.items()):
        source_path = find_volume(input_root, modality, source_field, subject)
        _, source = load_array(source_path)
        target_map = {item.target: item for item in items}
        targets = [field for field in FIELDS if field != source_field]
        volumes: list[tuple[str, np.ndarray]] = [(f"Source {source_field}", source)]
        for target in targets:
            item = target_map.get(target)
            if item is None:
                continue
            _, prediction = load_array(item.path)
            volumes.append((f"Generated {target}", prediction))

        midpoints = (source.shape[0] // 2, source.shape[1] // 2, source.shape[2] // 2)
        row_names = ("Sagittal", "Coronal", "Axial")
        fig, axes = plt.subplots(3, len(volumes), figsize=(3.2 * len(volumes), 10))
        axes = np.asarray(axes).reshape(3, len(volumes))
        for row, (axis, index) in enumerate(zip((0, 1, 2), midpoints)):
            for col, (title, volume) in enumerate(volumes):
                axes[row, col].imshow(
                    view_slice(volume, axis, index), cmap="gray", vmin=0.0, vmax=1.0,
                    origin="lower"
                )
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(title)
                if col == 0:
                    axes[row, col].set_ylabel(row_names[row])
        fig.suptitle(f"{modality} | source {source_field} | subject {subject}")
        fig.tight_layout()
        output = montage_dir / f"{modality}_{source_field}_{subject}.png"
        fig.savefig(output, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved.append(output)
        del source, volumes
    return saved


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command_inspect(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    predictions = discover_predictions(output_root, args.modalities)
    rows = []
    for index, item in enumerate(predictions, 1):
        source = find_volume(input_root, item.modality, item.source, item.subject)
        rows.append(diagnostic_row(item, source))
        print(f"[{index:03d}/{len(predictions):03d}] checked {item.modality} "
              f"{item.source}->{item.target} {item.subject}")
    csv_path = report_dir / "output_diagnostics.csv"
    write_csv(csv_path, rows)
    montages = make_montages(predictions, input_root, report_dir)
    summary = {
        "prediction_count": len(predictions),
        "pair_count": len({(p.modality, p.source, p.target) for p in predictions}),
        "montage_count": len(montages),
        "failed_numeric_rows": sum(
            bool(r["nonfinite_voxels"] or r["out_of_range_voxels"]) for r in rows
        ),
        "diagnostics_csv": str(csv_path),
        "montage_directory": str(report_dir / "montages"),
        "quality_metrics": "not computed; genuine matched targets are required",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "review_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def paired_metrics(
    predictions: list[Prediction], ground_truth_root: Path, device: str,
    lpips_batch_size: int,
) -> list[dict[str, object]]:
    import lpips
    import torch
    from skimage.metrics import structural_similarity

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    rows: list[dict[str, object]] = []
    for index, item in enumerate(predictions, 1):
        target_path = find_volume(
            ground_truth_root, item.modality, item.target, item.subject
        )
        _, pred = load_array(item.path)
        _, target = load_array(target_path)
        if pred.shape != target.shape:
            raise ReviewError(
                f"Metric shape mismatch: {item.path} {pred.shape} vs "
                f"{target_path} {target.shape}"
            )
        pred64, target64 = pred.astype(np.float64), target.astype(np.float64)
        target_norm = np.linalg.norm(target64)
        nrmse = (
            float(np.linalg.norm(pred64 - target64) / target_norm)
            if target_norm > 1e-10 else 0.0
        )
        data_range = float(target64.max() - target64.min())
        ssim_values = []
        lpips_values = []
        for z in range(pred.shape[2]):
            ps, ts = pred64[:, :, z], target64[:, :, z]
            if data_range >= 1e-10 and ts.max() - ts.min() >= 1e-10:
                ssim_values.append(structural_similarity(ps, ts, data_range=data_range))
        for start in range(0, pred.shape[2], lpips_batch_size):
            stop = min(start + lpips_batch_size, pred.shape[2])
            pred_batch = np.moveaxis(pred[:, :, start:stop], 2, 0)
            target_batch = np.moveaxis(target[:, :, start:stop], 2, 0)
            keep = np.max(np.abs(target_batch * 2.0 - 1.0), axis=(1, 2)) >= 1e-10
            if not np.any(keep):
                continue
            pred_batch = pred_batch[keep]
            target_batch = target_batch[keep]
            pn = torch.from_numpy(pred_batch * 2.0 - 1.0)[:, None]
            tn = torch.from_numpy(target_batch * 2.0 - 1.0)[:, None]
            pn = pn.repeat(1, 3, 1, 1).to(device)
            tn = tn.repeat(1, 3, 1, 1).to(device)
            with torch.inference_mode():
                values = loss_fn(pn, tn).detach().cpu().numpy().reshape(-1)
            lpips_values.extend(float(value) for value in values)
        rows.append({
            "modality": item.modality,
            "source_field": item.source,
            "target_field": item.target,
            "subject": item.subject,
            "nrmse": nrmse,
            "ssim": float(np.mean(ssim_values)) if ssim_values else 1.0,
            "lpips": float(np.mean(lpips_values)) if lpips_values else 0.0,
            "prediction": str(item.path),
            "target": str(target_path),
        })
        print(f"[{index:03d}/{len(predictions):03d}] metrics {item.modality} "
              f"{item.source}->{item.target} {item.subject}")
        del pred, target, pred64, target64
    return rows


def command_metrics(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    gt_root = Path(args.ground_truth_root).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    if output_root == gt_root:
        raise ReviewError("Prediction and ground-truth roots must be different")
    if args.lpips_batch_size < 1:
        raise ReviewError("--lpips-batch-size must be at least 1")
    if not args.confirm_matched_subjects:
        raise ReviewError(
            "Metrics require genuine paired targets from the same subjects. "
            "After verifying that condition, add --confirm-matched-subjects."
        )
    predictions = discover_predictions(output_root, args.modalities)
    # find_volume enforces exact modality, target field, and subject matching.
    rows = paired_metrics(
        predictions, gt_root, args.device, args.lpips_batch_size
    )
    output_csv = report_dir / "official_metrics.csv"
    write_csv(output_csv, rows)
    summary_rows = []
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["modality"]), str(row["source_field"]), str(row["target_field"]))].append(row)
    for (modality, source, target), group in sorted(groups.items()):
        summary_rows.append({
            "modality": modality,
            "source_field": source,
            "target_field": target,
            "cases": len(group),
            "nrmse_mean": float(np.mean([float(r["nrmse"]) for r in group])),
            "ssim_mean": float(np.mean([float(r["ssim"]) for r in group])),
            "lpips_mean": float(np.mean([float(r["lpips"]) for r in group])),
        })
    summary_csv = report_dir / "official_metrics_by_pair.csv"
    write_csv(summary_csv, summary_rows)
    print(f"Saved {output_csv}")
    print(f"Saved {summary_csv}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--input-root", required=True)
    inspect_parser.add_argument("--output-root", required=True)
    inspect_parser.add_argument("--report-dir", required=True)
    inspect_parser.add_argument("--modalities", nargs="+", choices=MODALITIES, required=True)
    inspect_parser.set_defaults(func=command_inspect)
    metrics_parser = subparsers.add_parser("metrics")
    metrics_parser.add_argument("--output-root", required=True)
    metrics_parser.add_argument("--ground-truth-root", required=True)
    metrics_parser.add_argument("--report-dir", required=True)
    metrics_parser.add_argument("--modalities", nargs="+", choices=MODALITIES, required=True)
    metrics_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    metrics_parser.add_argument("--lpips-batch-size", type=int, default=8)
    metrics_parser.add_argument("--confirm-matched-subjects", action="store_true")
    metrics_parser.set_defaults(func=command_metrics)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        args.func(args)
    except ReviewError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
