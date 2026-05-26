from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

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


def parse_extensions(value: str) -> tuple[str, ...]:
    extensions = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        extensions.append(item if item.startswith(".") else f".{item}")
    return tuple(extensions)


def normalize_video_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def candidate_video_keys(path: str | Path) -> list[str]:
    stem = Path(path).stem
    candidates = [stem, stem.replace("__#", "__")]
    if "_label_" in stem:
        candidates.append(stem.split("_label_", 1)[0])
    if "__#" in stem:
        movie_name, clip_part = stem.split("__#", 1)
        candidates.extend([movie_name, clip_part])

    keys = []
    for candidate in candidates:
        key = normalize_video_key(candidate)
        if key and key not in keys:
            keys.append(key)
    return keys


def build_video_index(video_root: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
    if not video_root.exists():
        raise FileNotFoundError(f"Video root does not exist: {video_root}")

    index: dict[str, Path] = {}
    for path in video_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        for key in candidate_video_keys(path):
            index.setdefault(key, path)
    return index


def find_video_path(feature_path: str, video_index: dict[str, Path]) -> Path | None:
    for key in candidate_video_keys(feature_path):
        if key in video_index:
            return video_index[key]
    return None


def timecode_to_seconds(value: str) -> int:
    hours, minutes, seconds = [int(part) for part in value.split("-")]
    return hours * 3600 + minutes * 60 + seconds


def parse_clip_time_range(feature_path: str) -> tuple[int, int] | None:
    match = re.search(r"__#(\d{2}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", Path(feature_path).stem)
    if match is None:
        return None
    start = timecode_to_seconds(match.group(1))
    end = timecode_to_seconds(match.group(2))
    return (start, end) if end > start else None


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


def default_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def resize_filter():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def score_to_video_frame(score_index: int, score_length: int, video_frame_count: int) -> int:
    if score_length <= 1 or video_frame_count <= 1:
        return 0
    return int(round(score_index / (score_length - 1) * (video_frame_count - 1)))


def score_to_clip_frame(
    score_index: int,
    score_length: int,
    frame_count: int,
    fps: float,
    clip_time_range: tuple[int, int] | None,
) -> int:
    if score_length <= 1:
        return 0
    if clip_time_range is None or fps <= 0:
        return score_to_video_frame(score_index, score_length, frame_count)

    start_seconds, end_seconds = clip_time_range
    clip_duration = end_seconds - start_seconds
    video_duration = frame_count / fps if frame_count > 0 else 0
    if video_duration <= clip_duration * 1.5:
        return score_to_video_frame(score_index, score_length, frame_count)

    seconds = start_seconds + score_index / (score_length - 1) * clip_duration
    return max(0, min(frame_count - 1, int(round(seconds * fps))))


def read_video_frame(cap: cv2.VideoCapture, frame_index: int) -> Image.Image | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def sample_video_thumbnails(
    video_path: Path,
    score_length: int,
    num_thumbnails: int,
    clip_time_range: tuple[int, int] | None = None,
) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f"Unable to read frame count for video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        count = max(1, min(num_thumbnails, score_length))
        score_indices = np.linspace(0, max(0, score_length - 1), count, dtype=int)
        thumbnails = []
        for score_index in score_indices:
            video_frame_index = score_to_clip_frame(
                int(score_index),
                score_length,
                frame_count,
                fps,
                clip_time_range,
            )
            image = read_video_frame(cap, video_frame_index)
            if image is None:
                print(f"Warning: failed to read frame {video_frame_index} from {video_path}")
                continue
            thumbnails.append(
                {
                    "score_index": int(score_index),
                    "video_frame_index": video_frame_index,
                    "image": image,
                }
            )
        if not thumbnails:
            raise RuntimeError(f"No thumbnails could be decoded from video: {video_path}")
        return thumbnails
    finally:
        cap.release()


def contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    active = np.where(mask > 0)[0]
    if active.size == 0:
        return []
    splits = np.where(np.diff(active) > 1)[0] + 1
    return [(int(group[0]), int(group[-1])) for group in np.split(active, splits)]


def score_point(
    index: int,
    value: float,
    length: int,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> tuple[int, int]:
    x = left if length <= 1 else left + round(index / (length - 1) * (right - left))
    y = bottom - round(float(np.clip(value, 0.0, 1.0)) * (bottom - top))
    return int(x), int(y)


def curve_points(values: np.ndarray, left: int, right: int, top: int, bottom: int) -> list[tuple[int, int]]:
    if len(values) == 0:
        return []
    return [score_point(index, float(value), len(values), left, right, top, bottom) for index, value in enumerate(values)]


def draw_text_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((xy[0] - width // 2, xy[1] - height // 2), text, font=font, fill=fill)


def draw_thumbnail_strip(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    thumbnails: list[dict],
    gt: np.ndarray,
    left: int,
    right: int,
    top: int,
    thumb_width: int,
    thumb_height: int,
) -> None:
    if not thumbnails:
        return

    count = len(thumbnails)
    available = right - left
    gap = max(8, (available - count * thumb_width) // max(1, count - 1)) if count > 1 else 0
    total_width = count * thumb_width + (count - 1) * gap
    x = left + max(0, (available - total_width) // 2)

    for thumb in thumbnails:
        image = ImageOps.fit(thumb["image"], (thumb_width, thumb_height), method=resize_filter())
        score_index = min(max(int(thumb["score_index"]), 0), len(gt) - 1)
        anomalous = bool(len(gt) and gt[score_index] > 0)

        canvas.paste(image, (x, top))
        border = "#f4a6b8" if anomalous else "#9ca3af"
        draw.rectangle((x - 2, top - 2, x + thumb_width + 1, top + thumb_height + 1), outline=border, width=3)
        if anomalous:
            draw.rectangle((x - 5, top - 5, x + thumb_width + 4, top + thumb_height + 4), outline="#f8b4c4", width=2)
        x += thumb_width + gap


def draw_gt_regions(
    draw: ImageDraw.ImageDraw,
    gt: np.ndarray,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> None:
    overlay_color = (244, 114, 182, 82)
    for start, end in contiguous_regions(gt):
        x1, _ = score_point(start, 0.0, len(gt), left, right, top, bottom)
        x2, _ = score_point(end, 0.0, len(gt), left, right, top, bottom)
        draw.rectangle((x1, top, max(x1 + 1, x2), bottom), fill=overlay_color)


def write_coarse_png(
    path: Path,
    title: str,
    branch1: np.ndarray,
    branch2: np.ndarray,
    gt: np.ndarray,
    thumbnails: list[dict],
    width: int,
    height: int,
) -> None:
    canvas = Image.new("RGB", (width, height), "white")
    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    base_draw = ImageDraw.Draw(canvas)

    title_font = default_font(24)
    label_font = default_font(18)
    small_font = default_font(13)

    margin_x = 44
    title_y = 18
    thumb_top = 52
    thumb_height = max(56, min(96, int(height * 0.18)))
    thumb_width = int(thumb_height * 1.45)
    chart_top = thumb_top + thumb_height + 26
    chart_bottom = height - 54
    chart_left = 58
    chart_right = width - 38

    base_draw.text((margin_x, title_y), title, font=title_font, fill="#111827")
    draw_thumbnail_strip(canvas, base_draw, thumbnails, gt, margin_x, width - margin_x, thumb_top, thumb_width, thumb_height)

    draw.rounded_rectangle((chart_left, chart_top, chart_right, chart_bottom), radius=4, fill=(255, 255, 255, 255), outline=(209, 213, 219, 255), width=1)
    draw_gt_regions(draw, gt, chart_left, chart_right, chart_top, chart_bottom)

    for fraction, label in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
        y = chart_bottom - round(fraction * (chart_bottom - chart_top))
        draw.line((chart_left, y, chart_right, y), fill=(229, 231, 235, 255), width=1)
        base_draw.text((14, y - 8), label, font=small_font, fill="#6b7280")

    branch1_points = curve_points(branch1, chart_left, chart_right, chart_top, chart_bottom)
    branch2_points = curve_points(branch2, chart_left, chart_right, chart_top, chart_bottom)
    if len(branch1_points) > 1:
        draw.line(branch1_points, fill=(37, 99, 235, 255), width=3, joint="curve")
    if len(branch2_points) > 1:
        draw.line(branch2_points, fill=(22, 163, 74, 255), width=3, joint="curve")

    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=(107, 114, 128, 255), width=1)
    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=(107, 114, 128, 255), width=1)

    label = title.split("label=", 1)[1].split(" | ", 1)[0] if "label=" in title else "Anomaly score"
    draw_text_centered(draw, ((chart_left + chart_right) // 2, chart_top + 32), label, label_font, "#315f9f")

    legend_x = chart_right - 260
    legend_y = chart_top + 10
    draw.rounded_rectangle((legend_x, legend_y, chart_right - 10, legend_y + 74), radius=4, fill=(255, 255, 255, 218), outline=(229, 231, 235, 255))
    draw.line((legend_x + 14, legend_y + 18, legend_x + 54, legend_y + 18), fill=(37, 99, 235, 255), width=4)
    draw.text((legend_x + 64, legend_y + 10), "Branch 1", font=small_font, fill="#111827")
    draw.line((legend_x + 14, legend_y + 42, legend_x + 54, legend_y + 42), fill=(22, 163, 74, 255), width=4)
    draw.text((legend_x + 64, legend_y + 34), "Branch 2", font=small_font, fill="#111827")
    draw.rectangle((legend_x + 14, legend_y + 58, legend_x + 54, legend_y + 68), fill=(244, 114, 182, 82))
    draw.text((legend_x + 64, legend_y + 54), "GT anomaly", font=small_font, fill="#111827")

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


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
    parser.add_argument("--video-root", type=str, required=True, help="Directory containing original XD-Violence test videos.")
    parser.add_argument("--num-thumbnails", type=int, default=5)
    parser.add_argument("--figure-width", type=int, default=960)
    parser.add_argument("--figure-height", type=int, default=360)
    parser.add_argument("--video-extensions", type=str, default=".mp4,.avi,.mkv,.mov")
    parser.add_argument("--output-dir", type=str, default="experiment/results/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = xd_settings(args, config)
    video_root = resolve_path(args.video_root)
    video_extensions = parse_extensions(args.video_extensions)
    video_index = build_video_index(video_root, video_extensions)
    if not video_index:
        raise FileNotFoundError(f"No videos with extensions {video_extensions} found under {video_root}")
    print(f"Indexed {len(set(video_index.values()))} videos from {video_root}")

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
        png_path = output_dir / f"{index:04d}_{name}.png"
        video_path = find_video_path(meta.get("path", ""), video_index)
        if video_path is None:
            print(f"Skipping case {index}: unable to match feature path to video: {meta.get('path', '')}")
            continue

        try:
            thumbnails = sample_video_thumbnails(
                video_path,
                length,
                int(args.num_thumbnails),
                parse_clip_time_range(meta.get("path", "")),
            )
        except RuntimeError as error:
            print(f"Skipping case {index}: {error}")
            continue

        write_trace_csv(csv_path, branch1, branch2, gt)
        write_coarse_png(
            png_path,
            f"XD case {index} | label={meta.get('label', '')} | {Path(str(meta.get('path', ''))).stem}",
            branch1,
            branch2,
            gt,
            thumbnails,
            int(args.figure_width),
            int(args.figure_height),
        )
        summary.append(
            {
                "index": index,
                "label": meta.get("label", ""),
                "source": meta.get("path", ""),
                "video": str(video_path),
                "csv": str(csv_path.relative_to(resolve_path("."))),
                "png": str(png_path.relative_to(resolve_path("."))),
            }
        )

    write_json(output_dir / "visualization_summary.json", summary)
    for item in summary:
        print(f"case={item['index']} csv={item['csv']} png={item['png']}")


if __name__ == "__main__":
    main()
