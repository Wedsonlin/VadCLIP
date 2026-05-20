from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from common import (
    add_xd_runtime_args,
    build_xd_dataloader,
    build_xd_model,
    collect_xd_predictions,
    load_config,
    load_model_weights,
    resolve_path,
    write_json,
    xd_prompt_text,
    xd_settings,
)


def parse_indices(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def safe_name(value: str, fallback: str) -> str:
    value = Path(value).stem if value else fallback
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value[:120] or fallback


def video_ground_truth(length: int, gtsegments: np.ndarray, index: int) -> np.ndarray:
    gt = np.zeros(length, dtype=np.float32)
    if index >= len(gtsegments):
        return gt
    for segment in gtsegments[index]:
        if len(segment) < 2:
            continue
        start = max(0, min(length, int(segment[0])))
        end = max(0, min(length, int(segment[1])))
        if end > start:
            gt[start:end] = 1.0
    return gt


def write_trace_csv(
    path: Path,
    branch1: np.ndarray,
    branch2: np.ndarray,
    gt: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "branch1_score", "branch2_score", "ground_truth"])
        writer.writeheader()
        for frame in range(len(gt)):
            writer.writerow(
                {
                    "frame": frame,
                    "branch1_score": float(branch1[frame]),
                    "branch2_score": float(branch2[frame]),
                    "ground_truth": int(gt[frame]),
                }
            )


def points_for_line(values: np.ndarray, width: int, height: int, pad: int) -> str:
    if len(values) <= 1:
        return ""
    x_scale = (width - 2 * pad) / (len(values) - 1)
    y_scale = height - 2 * pad
    points = []
    for i, value in enumerate(values):
        x = pad + i * x_scale
        y = height - pad - float(np.clip(value, 0.0, 1.0)) * y_scale
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def gt_rectangles(gt: np.ndarray, width: int, height: int, pad: int) -> str:
    if len(gt) <= 1:
        return ""
    x_scale = (width - 2 * pad) / (len(gt) - 1)
    rectangles = []
    active = np.where(gt > 0)[0]
    if active.size == 0:
        return ""

    splits = np.where(np.diff(active) > 1)[0] + 1
    for group in np.split(active, splits):
        start = int(group[0])
        end = int(group[-1])
        x = pad + start * x_scale
        rect_width = max(1.0, (end - start + 1) * x_scale)
        rectangles.append(
            f'<rect x="{x:.2f}" y="{pad}" width="{rect_width:.2f}" '
            f'height="{height - 2 * pad}" fill="#ef4444" opacity="0.16" />'
        )
    return "\n".join(rectangles)


def write_svg(path: Path, title: str, branch1: np.ndarray, branch2: np.ndarray, gt: np.ndarray) -> None:
    width, height, pad = 960, 360, 44
    branch1_points = points_for_line(branch1, width, height, pad)
    branch2_points = points_for_line(branch2, width, height, pad)
    gt_shapes = gt_rectangles(gt, width, height, pad)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{pad}" y="24" font-family="Arial" font-size="16" fill="#111827">{title}</text>
  {gt_shapes}
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#9ca3af" />
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#9ca3af" />
  <text x="{pad}" y="{height - 12}" font-family="Arial" font-size="12" fill="#6b7280">frame</text>
  <text x="8" y="{pad}" font-family="Arial" font-size="12" fill="#6b7280">score</text>
  <polyline points="{branch1_points}" fill="none" stroke="#2563eb" stroke-width="2" />
  <polyline points="{branch2_points}" fill="none" stroke="#16a34a" stroke-width="2" />
  <rect x="{width - 240}" y="32" width="190" height="70" fill="white" opacity="0.85" />
  <line x1="{width - 225}" y1="52" x2="{width - 190}" y2="52" stroke="#2563eb" stroke-width="3" />
  <text x="{width - 180}" y="56" font-family="Arial" font-size="13" fill="#111827">Branch 1</text>
  <line x1="{width - 225}" y1="76" x2="{width - 190}" y2="76" stroke="#16a34a" stroke-width="3" />
  <text x="{width - 180}" y="80" font-family="Arial" font-size="13" fill="#111827">Branch 2</text>
  <rect x="{width - 225}" y="88" width="35" height="10" fill="#ef4444" opacity="0.16" />
  <text x="{width - 180}" y="99" font-family="Arial" font-size="13" fill="#111827">GT anomaly</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export XD-Violence qualitative score curves.")
    add_xd_runtime_args(parser)
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated test-video indices.")
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="experiment/results/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = xd_settings(args, config)

    model = build_xd_model(settings)
    load_model_weights(model, settings["model_path"], settings["device"], strict=bool(settings["strict_load"]))

    dataloader = build_xd_dataloader(settings)
    prompt_text = xd_prompt_text(settings.get("prompt_template"))
    predictions = collect_xd_predictions(
        model,
        dataloader,
        int(settings["visual_length"]),
        prompt_text,
        settings["device"],
    )
    gtsegments = np.load(resolve_path(settings["gt_segment_path"]), allow_pickle=True)

    indices = parse_indices(args.indices)
    if indices is None:
        indices = list(range(min(args.num_cases, len(predictions["branch1_segments"]))))

    output_dir = resolve_path(args.output_dir)
    summary = []
    for index in indices:
        if index < 0 or index >= len(predictions["branch1_segments"]):
            print(f"Skipping out-of-range index: {index}")
            continue

        branch1 = np.repeat(predictions["branch1_segments"][index], 16)
        branch2 = np.repeat(predictions["branch2_segments"][index], 16)
        length = min(len(branch1), len(branch2))
        gt = video_ground_truth(length, gtsegments, index)
        branch1 = branch1[:length]
        branch2 = branch2[:length]

        meta = predictions["video_meta"][index]
        name = safe_name(meta.get("path", ""), f"case_{index}")
        csv_path = output_dir / f"{index:04d}_{name}.csv"
        svg_path = output_dir / f"{index:04d}_{name}.svg"

        write_trace_csv(csv_path, branch1, branch2, gt)
        write_svg(svg_path, f"XD case {index}: {meta.get('label', '')}", branch1, branch2, gt)
        summary.append(
            {
                "index": index,
                "label": meta.get("label", ""),
                "source": meta.get("path", ""),
                "csv": str(csv_path.relative_to(resolve_path("."))),
                "svg": str(svg_path.relative_to(resolve_path("."))),
            }
        )

    write_json(output_dir / "visualization_summary.json", summary)
    for item in summary:
        print(f"case={item['index']} csv={item['csv']} svg={item['svg']}")


if __name__ == "__main__":
    main()
