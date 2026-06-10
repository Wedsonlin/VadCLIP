from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "experiment" / "results"

for path in (str(SRC_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from model import CLIPVAD  # noqa: E402
from utils.dataset import XDDataset  # noqa: E402
from utils.tools import get_batch_mask  # noqa: E402
from utils.xd_detectionMAP import getDetectionMAP as xd_detection_map  # noqa: E402


XD_LABEL_MAP = {
    "A": "normal",
    "B1": "fighting",
    "B2": "shooting",
    "B4": "riot",
    "B5": "abuse",
    "B6": "car accident",
    "G": "explosion",
}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip("'\"")


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))

    config: dict[str, Any] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = parse_scalar(value)
    return config


def config_get(config: dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def add_xd_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="Optional simple YAML/JSON config.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--test-list", type=str, default=None)
    parser.add_argument("--gt-path", type=str, default=None)
    parser.add_argument("--gt-segment-path", type=str, default=None)
    parser.add_argument("--gt-label-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--visual-length", type=int, default=None)
    parser.add_argument("--visual-width", type=int, default=None)
    parser.add_argument("--visual-head", type=int, default=None)
    parser.add_argument("--visual-layers", type=int, default=None)
    parser.add_argument("--attn-window", type=int, default=None)
    parser.add_argument("--prompt-prefix", type=int, default=None)
    parser.add_argument("--prompt-postfix", type=int, default=None)
    parser.add_argument("--classes-num", type=int, default=None)
    parser.add_argument("--prompt-template", type=str, default=None)
    parser.add_argument("--metrics-csv", type=str, default=None)
    parser.add_argument("--metrics-json", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--strict-load", action=argparse.BooleanOptionalAction, default=None)


def xd_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "model_path": "model/model_xd.pth",
        "test_list": "list/xd_CLIP_rgbtest.csv",
        "gt_path": "list/gt.npy",
        "gt_segment_path": "list/gt_segment.npy",
        "gt_label_path": "list/gt_label.npy",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 1,
        "num_workers": 0,
        "embed_dim": 512,
        "visual_length": 256,
        "visual_width": 512,
        "visual_head": 1,
        "visual_layers": 1,
        "attn_window": 64,
        "prompt_prefix": 10,
        "prompt_postfix": 10,
        "classes_num": 7,
        "prompt_template": None,
        "metrics_csv": "experiment/results/metrics.csv",
        "metrics_json": None,
        "experiment_name": "xd_eval",
        "strict_load": True,
    }

    settings: dict[str, Any] = {}
    for key, default in defaults.items():
        arg_value = getattr(args, key, None)
        settings[key] = arg_value if arg_value is not None else config_get(config, key, default)
    if str(settings["device"]).lower() == "auto":
        settings["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return settings


def xd_prompt_text(prompt_template: str | None = None) -> list[str]:
    class_names = list(XD_LABEL_MAP.values())
    if not prompt_template:
        return class_names
    return [prompt_template.format(label=label) for label in class_names]


def build_xd_dataloader(settings: dict[str, Any]) -> DataLoader:
    dataset = XDDataset(
        int(settings["visual_length"]),
        str(resolve_path(settings["test_list"])),
        True,
        XD_LABEL_MAP,
    )
    return DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        num_workers=int(settings["num_workers"]),
    )


def build_xd_model(
    settings: dict[str, Any],
    model_cls: type[CLIPVAD] = CLIPVAD, # model_cls is class CLIPVAD or its subclass
    **model_kwargs: Any,
) -> CLIPVAD:
    return model_cls(
        int(settings["classes_num"]),
        int(settings["embed_dim"]),
        int(settings["visual_length"]),
        int(settings["visual_width"]),
        int(settings["visual_head"]),
        int(settings["visual_layers"]),
        int(settings["attn_window"]),
        int(settings["prompt_prefix"]),
        int(settings["prompt_postfix"]),
        settings["device"],
        **model_kwargs,
    )


def load_model_weights(
    model: torch.nn.Module,
    model_path: str | Path,
    device: str | torch.device,
    strict: bool = True,
) -> None:
    state = torch.load(resolve_path(model_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if not strict and (missing or unexpected):
        print(f"Non-strict load: missing={len(missing)} unexpected={len(unexpected)}")


def split_lengths(total_length: int, max_length: int) -> torch.Tensor:
    if total_length <= 0:
        return torch.zeros(1, dtype=torch.int)
    split_num = math.ceil(total_length / max_length)
    values = [min(max_length, total_length - i * max_length) for i in range(split_num)]
    return torch.tensor(values, dtype=torch.int)


def collect_xd_predictions(
    model: CLIPVAD,
    dataloader: DataLoader,
    max_length: int,
    prompt_text: list[str],
    device: str | torch.device,
) -> dict[str, Any]:
    model.to(device)
    model.eval()

    branch1_segments: list[np.ndarray] = []
    branch2_segments: list[np.ndarray] = []
    alignment_stack: list[np.ndarray] = []
    video_meta: list[dict[str, Any]] = []

    dataset = getattr(dataloader, "dataset", None)
    dataframe = getattr(dataset, "df", None)

    with torch.no_grad():
        for index, item in enumerate(dataloader):
            visual = item[0].squeeze(0)
            length = int(item[2])
            original_length = length

            if original_length < max_length and visual.dim() == 2:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)
            lengths = split_lengths(original_length, max_length)
            padding_mask = get_batch_mask(lengths, max_length).to(device)

            _, anomaly_confidence, alignment_map = model(
                visual,
                padding_mask,
                prompt_text,
                lengths.to(device),
            )

            anomaly_confidence = anomaly_confidence.reshape(-1, anomaly_confidence.shape[-1])
            alignment_map = alignment_map.reshape(-1, alignment_map.shape[-1])

            branch1 = anomaly_confidence[:original_length].squeeze(-1).detach().cpu().numpy()
            branch2 = (
                1 - alignment_map[:original_length].softmax(dim=-1)[:, 0].squeeze(-1)
            ).detach().cpu().numpy()
            alignment_prob = alignment_map[:original_length].softmax(dim=-1).detach().cpu().numpy()

            branch1_segments.append(branch1)
            branch2_segments.append(branch2)
            alignment_stack.append(np.repeat(alignment_prob, 16, axis=0))

            if dataframe is not None:
                row = dataframe.iloc[index]
                video_meta.append(
                    {
                        "index": index,
                        "path": str(row.get("path", "")),
                        "label": str(row.get("label", "")),
                        "length": original_length,
                    }
                )
            else:
                video_meta.append({"index": index, "path": "", "label": "", "length": original_length})

    return {
        "branch1_segments": branch1_segments,
        "branch2_segments": branch2_segments,
        "alignment_stack": alignment_stack,
        "video_meta": video_meta,
    }


def repeated_frame_scores(segment_scores: Iterable[np.ndarray], repeat: int = 16) -> np.ndarray:
    if not segment_scores:
        return np.array([], dtype=np.float32)
    return np.repeat(np.concatenate(list(segment_scores)), repeat)


def align_to_ground_truth(gt: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if gt.shape[0] == scores.shape[0]:
        return gt, scores
    aligned_len = min(gt.shape[0], scores.shape[0])
    print(f"Warning: trimming scores/gt from {scores.shape[0]}/{gt.shape[0]} to {aligned_len}.")
    return gt[:aligned_len], scores[:aligned_len]


def binary_detection_metrics(gt: np.ndarray, scores: np.ndarray, prefix: str) -> dict[str, float]:
    gt_aligned, scores_aligned = align_to_ground_truth(gt, scores)
    return {
        f"{prefix}_auc": float(roc_auc_score(gt_aligned, scores_aligned)),
        f"{prefix}_ap": float(average_precision_score(gt_aligned, scores_aligned)),
    }


def evaluate_xd_predictions(
    predictions: dict[str, Any],
    gt: np.ndarray,
    gtsegments: np.ndarray,
    gtlabels: np.ndarray,
) -> dict[str, float]:
    branch1_scores = repeated_frame_scores(predictions["branch1_segments"])
    branch2_scores = repeated_frame_scores(predictions["branch2_segments"])

    metrics = {}
    metrics.update(binary_detection_metrics(gt, branch1_scores, "branch1"))
    metrics.update(binary_detection_metrics(gt, branch2_scores, "branch2"))

    dmap, iou = xd_detection_map(predictions["alignment_stack"], gtsegments, gtlabels, excludeNormal=False)
    for threshold, value in zip(iou, dmap):
        metrics[f"mAP@{threshold:.1f}"] = float(value)
    metrics["average_mAP"] = float(np.mean(dmap))
    return metrics


def evaluate_xd_model(model: CLIPVAD, settings: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    dataloader = build_xd_dataloader(settings)
    prompt_text = xd_prompt_text(settings.get("prompt_template"))
    gt = np.load(resolve_path(settings["gt_path"]))
    gtsegments = np.load(resolve_path(settings["gt_segment_path"]), allow_pickle=True)
    gtlabels = np.load(resolve_path(settings["gt_label_path"]), allow_pickle=True)
    predictions = collect_xd_predictions(
        model,
        dataloader,
        int(settings["visual_length"]),
        prompt_text,
        settings["device"],
    )
    return evaluate_xd_predictions(predictions, gt, gtsegments, gtlabels), predictions


def fusion_metrics(
    predictions: dict[str, Any],
    gt: np.ndarray,
    alphas: Iterable[float],
) -> list[dict[str, float]]:
    branch1 = repeated_frame_scores(predictions["branch1_segments"])
    branch2 = repeated_frame_scores(predictions["branch2_segments"])
    rows = []
    for alpha in alphas:
        fused = alpha * branch1 + (1.0 - alpha) * branch2
        metrics = binary_detection_metrics(gt, fused, "fusion")
        metrics["alpha"] = float(alpha)
        rows.append(metrics)
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    existing = output_path.exists() and output_path.stat().st_size > 0
    if existing:
        old_fieldnames: list[str] = []
        old_rows: list[dict[str, Any]] = []
        for encoding in ("utf-8-sig", "utf-16", "utf-8"):
            try:
                with output_path.open("r", newline="", encoding=encoding) as handle:
                    reader = csv.DictReader(handle)
                    old_fieldnames = reader.fieldnames or []
                    old_rows = list(reader)
                break
            except UnicodeDecodeError:
                continue
        if not old_fieldnames and not old_rows:
            raise UnicodeDecodeError(
                "utf-8",
                b"",
                0,
                1,
                f"Could not decode metrics CSV at {output_path}",
            )
        for key in old_fieldnames:
            if key not in fieldnames:
                fieldnames.insert(0, key)
        for row in old_rows:
            for key in fieldnames:
                row.setdefault(key, "")
        rows_to_write = old_rows + rows
    else:
        rows_to_write = rows

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)


def print_metrics(metrics: dict[str, Any]) -> None:
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
