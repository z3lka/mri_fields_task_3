#!/usr/bin/env python3
"""Build an interactive Task 3 training dashboard from metrics.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODALITIES = ("T1W", "T2W", "T2FLAIR")
COLORS = {
    "Overall": "#111827",
    "T1W": "#2563eb",
    "T2W": "#f97316",
    "T2FLAIR": "#16a34a",
}
HOLDOUT_METRICS = (
    ("slice_proxy_nrmse", "Holdout nRMSE (lower is better)", min),
    ("slice_proxy_ssim", "Holdout SSIM (higher is better)", max),
    ("slice_proxy_lpips", "Holdout LPIPS (lower is better)", min),
)


def load_records(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if record.get("kind") == "epoch" and record.get("train") and record.get("holdout"):
            records.append(record)
    if not records:
        raise ValueError(f"No epoch records with train and holdout data in {path}")
    records.sort(key=lambda record: record["epoch"])
    epochs = [record["epoch"] for record in records]
    if len(epochs) != len(set(epochs)):
        raise ValueError("Duplicate epoch records found")
    return records


def add_holdout_plot(fig: go.Figure, records: list[dict], metric: str, row: int, col: int) -> None:
    epochs = [record["epoch"] for record in records]
    series = [("Overall", [record["holdout"]["overall"][metric] for record in records])]
    series.extend(
        (
            modality,
            [record["holdout"]["by_modality"][modality][metric] for record in records],
        )
        for modality in MODALITIES
    )
    for name, values in series:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=values,
                mode="lines",
                name=name,
                legendgroup="holdout",
                legendgrouptitle_text="Holdout" if name == "Overall" else None,
                line={
                    "color": COLORS[name],
                    "width": 3 if name == "Overall" else 2,
                    "dash": "dash" if name == "Overall" else "solid",
                },
                showlegend=row == 1 and col == 2,
                hovertemplate=f"%{{y:.6f}}<extra>{name}</extra>",
            ),
            row=row,
            col=col,
        )


def build_dashboard(records: list[dict], source: Path) -> go.Figure:
    epochs = [record["epoch"] for record in records]
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{}, {}, {}], [{}, {}, {"type": "table"}]],
        subplot_titles=[
            "Training loss terms",
            HOLDOUT_METRICS[0][1],
            HOLDOUT_METRICS[1][1],
            HOLDOUT_METRICS[2][1],
            "Learning rate",
            "Overall holdout summary",
        ],
        horizontal_spacing=0.07,
        vertical_spacing=0.14,
    )

    train_styles = {
        "total": ("#7c3aed", 3, "solid", True),
        "l1": ("#2563eb", 2, "dot", "legendonly"),
        "edge": ("#f97316", 2, "solid", True),
        "ssim": ("#16a34a", 2, "solid", True),
        "lpips": ("#dc2626", 2, "solid", "legendonly"),
    }
    for name, (color, width, dash, visible) in train_styles.items():
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[record["train"][name] for record in records],
                mode="lines",
                name=name,
                legendgroup="train",
                legendgrouptitle_text="Training terms" if name == "total" else None,
                line={"color": color, "width": width, "dash": dash},
                visible=visible,
                hovertemplate=f"%{{y:.6f}}<extra>{name}</extra>",
            ),
            row=1,
            col=1,
        )

    add_holdout_plot(fig, records, HOLDOUT_METRICS[0][0], 1, 2)
    add_holdout_plot(fig, records, HOLDOUT_METRICS[1][0], 1, 3)
    add_holdout_plot(fig, records, HOLDOUT_METRICS[2][0], 2, 1)

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=[record["lr"] for record in records],
            mode="lines",
            name="learning rate",
            line={"color": "#0891b2", "width": 3},
            showlegend=False,
            hovertemplate="%{y:.3e}<extra>learning rate</extra>",
        ),
        row=2,
        col=2,
    )

    summary_rows = []
    for metric, label, choose in HOLDOUT_METRICS:
        values = [record["holdout"]["overall"][metric] for record in records]
        best_value = choose(values)
        best_index = values.index(best_value)
        summary_rows.append(
            (label.split()[1], epochs[best_index], best_value, values[-1], values[-1] - values[0])
        )
    fig.add_trace(
        go.Table(
            header={
                "values": ["Metric", "Best epoch", "Best", "Latest", "Latest − first"],
                "fill_color": "#e5e7eb",
                "align": ["left", "right", "right", "right", "right"],
                "font": {"size": 12},
            },
            cells={
                "values": [
                    [row[0] for row in summary_rows],
                    [row[1] for row in summary_rows],
                    [f"{row[2]:.6f}" for row in summary_rows],
                    [f"{row[3]:.6f}" for row in summary_rows],
                    [f"{row[4]:+.6f}" for row in summary_rows],
                ],
                "align": ["left", "right", "right", "right", "right"],
                "height": 28,
            },
        ),
        row=2,
        col=3,
    )

    for row, col in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2)):
        fig.update_xaxes(title_text="Epoch", gridcolor="#e5e7eb", row=row, col=col)
        fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#d1d5db", row=row, col=col)
    fig.update_yaxes(type="log", row=2, col=2)
    fig.update_layout(
        title={
            "text": (
                f"Task 3 FPS-Former metrics — {source.name}<br>"
                "<sup>Holdout values are slice proxies, not private validation scores. "
                f"Epochs {epochs[0]}–{epochs[-1]} ({len(records)} records).</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        width=1800,
        height=1000,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.13, "x": 0.0},
        margin={"l": 70, "r": 40, "t": 125, "b": 130},
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="?", type=Path, default=Path("metrics.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("metrics_dashboard.html"))
    args = parser.parse_args()

    records = load_records(args.metrics)
    figure = build_dashboard(records, args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Wrote {args.output} from {len(records)} records")


if __name__ == "__main__":
    main()
