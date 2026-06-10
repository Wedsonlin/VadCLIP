from __future__ import annotations

import argparse
import copy
import math
import random
import statistics
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from ablation_models import AdapterAblationCLIPVAD
from common import (
    PROJECT_ROOT,
    XD_LABEL_MAP,
    add_xd_runtime_args,
    append_metrics_csv,
    build_xd_model,
    config_get,
    evaluate_xd_model,
    load_config,
    print_metrics,
    resolve_path,
    write_json,
    xd_settings,
)
from utils.dataset import XDDataset
from utils.tools import get_batch_label_vector


LOSS_VARIANTS = {
    "full": ("bce", "nce", "cts"),
    "bce_nce": ("bce", "nce"),
    "bce_cts": ("bce", "cts"),
    "nce_cts": ("nce", "cts"),
    "bce_only": ("bce",),
    "nce_only": ("nce",),
}

METRIC_KEYS = (
    "branch1_auc",
    "branch1_ap",
    "branch2_auc",
    "branch2_ap",
    "mAP@0.1",
    "mAP@0.2",
    "mAP@0.3",
    "mAP@0.4",
    "mAP@0.5",
    "average_mAP",
)

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.16,
    14: 2.145,
    15: 2.131,
    16: 2.12,
    17: 2.11,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.08,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.06,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def bce_topk_loss(
    anomaly_confidence: torch.Tensor,
    instance_label: torch.Tensor,
    lengths: torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    video_level_scores = torch.zeros(0).to(device)
    binary_label = (1 - instance_label[:, 0]).reshape(-1).to(device)
    anomaly_confidence = anomaly_confidence.squeeze(-1)

    for i in range(anomaly_confidence.shape[0]):
        topk_scores, _ = torch.topk(
            anomaly_confidence[i, 0 : lengths[i]],
            k=int(lengths[i] / 16 + 1),
            largest=True,
        )
        video_level_scores = torch.cat([video_level_scores, torch.mean(topk_scores).view(1)], dim=0)

    return F.binary_cross_entropy(video_level_scores, binary_label)


def nce_topk_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    instance_logits = []
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        topk_score, _ = torch.topk(
            logits[i, 0 : lengths[i]],
            k=int(lengths[i] / 16 + 1),
            largest=True,
            dim=0,
        )
        instance_logits.append(torch.mean(topk_score, dim=0, keepdim=True))

    instance_logits = torch.cat(instance_logits, dim=0)
    return F.cross_entropy(instance_logits, labels, reduction="mean")


def cts_loss(text_features: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    text_features = F.normalize(text_features, p=2, dim=-1).to(device)
    normal_text_feature = text_features[0]
    abnormal_text_features = text_features[1:]
    return torch.mean(torch.abs(abnormal_text_features @ normal_text_feature))


def parse_list(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def parse_int_list(value) -> list[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XD-Violence adapter and loss ablations.")
    add_xd_runtime_args(parser)
    parser.add_argument("--adapter-variants", type=str, default=None)
    parser.add_argument("--skip-adapter-ablations", action="store_true")
    parser.add_argument("--benchmark-adapter-latency", action="store_true")
    parser.add_argument("--latency-warmup-iters", type=int, default=None)
    parser.add_argument("--latency-iters", type=int, default=None)
    parser.add_argument("--latency-batch-size", type=int, default=None)
    parser.add_argument("--train-loss-ablations", action="store_true")
    parser.add_argument("--loss-variants", type=str, default=None)
    parser.add_argument("--train-list", type=str, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--scheduler-rate", type=float, default=None)
    parser.add_argument("--scheduler-milestones", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=None, help="Number of sequential seeds to run per variant.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma list of explicit seeds to run per variant.")
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_seed_list(base_seed: int, num_seeds: int, seeds_value) -> list[int]:
    if seeds_value is not None:
        seeds = parse_int_list(seeds_value)
        if not seeds:
            raise ValueError("At least one seed must be provided when --seeds is set.")
        return seeds
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive.")
    return [base_seed + offset for offset in range(num_seeds)]


def loss_for_variant(
    variant: str,
    text_features: torch.Tensor,
    anomaly_confidence: torch.Tensor,
    alignment_map: torch.Tensor,
    instance_label: torch.Tensor,
    feature_length: torch.Tensor,
    device: str | torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    if variant not in LOSS_VARIANTS:
        raise ValueError(f"Unknown loss variant {variant!r}; choose from {sorted(LOSS_VARIANTS)}")

    terms = LOSS_VARIANTS[variant]
    total = torch.zeros((), device=device)
    values = {"bce_loss": 0.0, "nce_loss": 0.0, "cts_loss": 0.0}

    if "bce" in terms:
        bce = bce_topk_loss(anomaly_confidence, instance_label, feature_length, device)
        total = total + bce
        values["bce_loss"] = float(bce.detach().cpu())
    if "nce" in terms:
        nce = nce_topk_loss(alignment_map, instance_label, feature_length, device)
        total = total + nce
        values["nce_loss"] = float(nce.detach().cpu())
    if "cts" in terms:
        cts = cts_loss(text_features, device) * 1e-4
        total = total + cts
        values["cts_loss"] = float(cts.detach().cpu())

    return total, values


def build_train_loader(settings: dict, train_settings: dict) -> DataLoader:
    train_dataset = XDDataset(
        int(settings["visual_length"]),
        str(resolve_path(train_settings["train_list"])),
        False,
        XD_LABEL_MAP,
    )
    return DataLoader(
        train_dataset,
        batch_size=int(train_settings["train_batch_size"]),
        shuffle=True,
        num_workers=int(settings["num_workers"]),
    )


def ci95_half_width(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    std = statistics.stdev(values)
    t_value = T_CRITICAL_95.get(len(values) - 1, 1.96)
    return float(t_value * std / math.sqrt(len(values)))


def metric_values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def summarize_ablation_variant(
    rows: list[dict],
    variant: str,
    seeds: list[int],
    experiment_name: str,
    ablation_type: str,
) -> dict:
    summary = {
        "experiment": experiment_name,
        "ablation_type": ablation_type,
        "variant": variant,
        "seed_count": len(rows),
        "seeds": ",".join(str(seed) for seed in seeds),
    }
    for key in METRIC_KEYS:
        values = metric_values(rows, key)
        if not values:
            continue
        mean = float(statistics.mean(values))
        std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        variance = float(statistics.variance(values)) if len(values) > 1 else 0.0
        half_width = ci95_half_width(values)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
        summary[f"{key}_variance"] = variance
        summary[f"{key}_ci95_low"] = mean - half_width
        summary[f"{key}_ci95_high"] = mean + half_width
        summary[f"{key}_ci95_half_width"] = half_width
    return summary


def train_one_adapter_variant(settings: dict, train_settings: dict, variant: str) -> dict:
    from utils.logger import Logger

    seed = int(train_settings["seed"])
    print(f"Training adapter ablation: {variant} seed={seed}")
    setup_seed(seed)

    train_loader = build_train_loader(settings, train_settings)
    logger = Logger(
        project="WSAVD_adapter_variants",
        name=f"{settings['experiment_name']}_adapter_{variant}_seed{seed}",
    )

    eval_settings = dict(settings)
    eval_settings["batch_size"] = 1

    model = build_xd_model(eval_settings, AdapterAblationCLIPVAD, variant=variant)
    model.to(settings["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_settings["lr"]))
    scheduler = MultiStepLR(
        optimizer,
        milestones=[int(value) for value in train_settings["scheduler_milestones"]],
        gamma=float(train_settings["scheduler_rate"]),
    )
    label_classes = list(XD_LABEL_MAP.values())

    best_ap = -1.0
    best_state = None
    best_metrics: dict[str, float] = {}
    checkpoint = ""

    try:
        for epoch in range(int(train_settings["max_epoch"])):
            model.train()
            for step, item in enumerate(train_loader, start=1):
                video_clip_feature, text_labels, feature_length = item
                video_clip_feature = video_clip_feature.to(settings["device"])
                feature_length = feature_length.to(settings["device"])
                instance_label = get_batch_label_vector(text_labels, XD_LABEL_MAP).to(settings["device"])

                text_features, anomaly_confidence, alignment_map = model(
                    video_clip_feature,
                    None,
                    label_classes,
                    feature_length,
                )
                loss, values = loss_for_variant(
                    "full",
                    text_features,
                    anomaly_confidence,
                    alignment_map,
                    instance_label,
                    feature_length,
                    settings["device"],
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logger.log_train(
                    {
                        "bce_loss": values["bce_loss"],
                        "nce_loss": values["nce_loss"],
                        "cts_loss": values["cts_loss"],
                    }
                )

                if step % 50 == 0:
                    print(
                        f"variant={variant} seed={seed} epoch={epoch + 1} step={step} "
                        f"loss={float(loss.detach().cpu()):.4f} "
                        f"bce={values['bce_loss']:.4f} "
                        f"nce={values['nce_loss']:.4f} "
                        f"cts={values['cts_loss']:.6f}"
                    )

            scheduler.step()
            metrics, _ = evaluate_xd_model(model, eval_settings)
            logger.log_eval(
                {
                    "branch1_ap": metrics["branch1_ap"],
                    "branch2_ap": metrics["branch2_ap"],
                    "adapter_average_mAP": metrics["average_mAP"],
                }
            )
            print(f"variant={variant} seed={seed} epoch={epoch + 1} branch1_ap={metrics['branch1_ap']:.6f}, branch2_ap={metrics['branch2_ap']:.6f}, average_mAP={metrics['average_mAP']:.6f}")
            current_ap = metrics["branch2_ap"]
            if current_ap > best_ap:
                best_ap = current_ap
                best_metrics = metrics
                if bool(train_settings["save_checkpoints"]):
                    best_state = copy.deepcopy(model.state_dict())

        if bool(train_settings["save_checkpoints"]) and best_state is not None:
            checkpoint_dir = resolve_path(train_settings["checkpoint_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"xd_adapter_{variant}_seed{seed}.pth"
            checkpoint = str(checkpoint_path.relative_to(PROJECT_ROOT))
            torch.save(best_state, checkpoint_path)
    finally:
        logger.finish()

    if str(settings["device"]).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "experiment": settings["experiment_name"],
        "ablation_type": "adapter_train",
        "variant": variant,
        "seed": seed,
        "checkpoint": checkpoint,
        **best_metrics,
    }


def train_one_loss_variant(settings: dict, train_settings: dict, variant: str) -> dict:
    from utils.logger import Logger

    seed = int(train_settings["seed"])
    print(f"Training loss ablation: {variant} seed={seed}")
    setup_seed(seed)

    train_loader = build_train_loader(settings, train_settings)
    logger = Logger(
        project="WSAVD_loss_variants",
        name=f"{settings['experiment_name']}_loss_{variant}_seed{seed}",
    )

    eval_settings = dict(settings)
    eval_settings["batch_size"] = 1

    model = build_xd_model(eval_settings)
    model.to(settings["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_settings["lr"]))
    scheduler = MultiStepLR(
        optimizer,
        milestones=[int(value) for value in train_settings["scheduler_milestones"]],
        gamma=float(train_settings["scheduler_rate"]),
    )
    label_classes = list(XD_LABEL_MAP.values())

    best_ap = -1.0
    best_state = None
    best_metrics: dict[str, float] = {}
    checkpoint = ""

    try:
        for epoch in range(int(train_settings["max_epoch"])):
            model.train()
            for step, item in enumerate(train_loader, start=1):
                video_clip_feature, text_labels, feature_length = item
                video_clip_feature = video_clip_feature.to(settings["device"])
                feature_length = feature_length.to(settings["device"])
                instance_label = get_batch_label_vector(text_labels, XD_LABEL_MAP).to(settings["device"])

                text_features, anomaly_confidence, alignment_map = model(
                    video_clip_feature,
                    None,
                    label_classes,
                    feature_length,
                )
                loss, values = loss_for_variant(
                    variant,
                    text_features,
                    anomaly_confidence,
                    alignment_map,
                    instance_label,
                    feature_length,
                    settings["device"],
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logger.log_train(
                    {
                        "bce_loss": values["bce_loss"],
                        "nce_loss": values["nce_loss"],
                        "cts_loss": values["cts_loss"],
                    }
                )

                if step % 50 == 0:
                    print(
                        f"variant={variant} seed={seed} epoch={epoch + 1} step={step} "
                        f"loss={float(loss.detach().cpu()):.4f} "
                        f"bce={values['bce_loss']:.4f} "
                        f"nce={values['nce_loss']:.4f} "
                        f"cts={values['cts_loss']:.6f}"
                    )

            scheduler.step()
            metrics, _ = evaluate_xd_model(model, eval_settings)
            logger.log_eval(
                {
                    "branch1_ap": metrics["branch1_ap"],
                    "branch2_ap": metrics["branch2_ap"],
                    "loss_average_mAP": metrics["average_mAP"],
                }
            )
            print(
                f"variant={variant} seed={seed} epoch={epoch + 1} "
                f"branch1_ap={metrics['branch1_ap']:.6f}, "
                f"branch2_ap={metrics['branch2_ap']:.6f}, "
                f"average_mAP={metrics['average_mAP']:.6f}"
            )
            current_ap = metrics["branch1_ap"]
            if current_ap > best_ap:
                best_ap = current_ap
                best_metrics = metrics
                if bool(train_settings["save_checkpoints"]):
                    best_state = copy.deepcopy(model.state_dict())

        if bool(train_settings["save_checkpoints"]) and best_state is not None:
            checkpoint_dir = resolve_path(train_settings["checkpoint_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"xd_loss_{variant}_seed{seed}.pth"
            checkpoint = str(checkpoint_path.relative_to(PROJECT_ROOT))
            torch.save(best_state, checkpoint_path)
    finally:
        logger.finish()

    if str(settings["device"]).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "experiment": settings["experiment_name"],
        "ablation_type": "loss",
        "variant": variant,
        "seed": seed,
        "checkpoint": checkpoint,
        **best_metrics,
    }


def synchronize_if_cuda(device: str | torch.device) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_adapter_latency(
    settings: dict,
    variants: list[str],
    warmup_iters: int,
    timed_iters: int,
    batch_size: int,
) -> list[dict]:
    if warmup_iters < 0:
        raise ValueError("latency warmup iterations must be non-negative")
    if timed_iters <= 0:
        raise ValueError("latency timed iterations must be positive")
    if batch_size <= 0:
        raise ValueError("latency batch size must be positive")

    device = settings["device"]
    visual_length = int(settings["visual_length"])
    visual_width = int(settings["visual_width"])
    label_classes = list(XD_LABEL_MAP.values())
    lengths = torch.full((batch_size,), visual_length, dtype=torch.int, device=device)
    video_clip_feature = torch.randn(batch_size, visual_length, visual_width, device=device)

    rows = []
    for variant in variants:
        print(f"Benchmarking adapter latency: {variant}")
        model = build_xd_model(settings, AdapterAblationCLIPVAD, variant=variant)
        model.to(device)
        model.eval()

        with torch.inference_mode():
            for _ in range(warmup_iters):
                model(video_clip_feature, None, label_classes, lengths)
            synchronize_if_cuda(device)

            elapsed_ms: list[float] = []
            if str(device).startswith("cuda") and torch.cuda.is_available():
                for _ in range(timed_iters):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    model(video_clip_feature, None, label_classes, lengths)
                    end_event.record()
                    end_event.synchronize()
                    elapsed_ms.append(float(start_event.elapsed_time(end_event)))
            else:
                for _ in range(timed_iters):
                    start = time.perf_counter()
                    model(video_clip_feature, None, label_classes, lengths)
                    elapsed_ms.append((time.perf_counter() - start) * 1000.0)

        mean_ms = float(np.mean(elapsed_ms))
        row = {
            "experiment": settings["experiment_name"],
            "ablation_type": "adapter_latency",
            "variant": variant,
            "latency_mean_ms": mean_ms,
            "latency_std_ms": float(np.std(elapsed_ms)),
            "latency_median_ms": float(np.median(elapsed_ms)),
            "throughput_videos_per_s": float(batch_size * 1000.0 / mean_ms),
            "warmup_iters": warmup_iters,
            "timed_iters": timed_iters,
            "latency_batch_size": batch_size,
            "latency_visual_length": visual_length,
            "latency_visual_width": visual_width,
            "device": str(device),
        }
        print_metrics(row)
        rows.append(row)

    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = xd_settings(args, config)

    seed = args.seed if args.seed is not None else int(config_get(config, "seed", 234))
    num_seeds = args.num_seeds if args.num_seeds is not None else int(config_get(config, "num_seeds", 5))
    seeds_value = args.seeds if args.seeds is not None else config_get(config, "seeds", None)

    adapter_variants = parse_list(
        args.adapter_variants
        if args.adapter_variants is not None
        else config_get(
            config,
            "adapter_variants",
            [
                "baseline",
                "global_tf",
                "local_tf",
                "only_gcn",
                "local_global_tf",
                "global_tf_gcn",
                "lgt_adapter",
            ],
        )
    )
    loss_variants = parse_list(
        args.loss_variants
        if args.loss_variants is not None
        else config_get(config, "loss_variants", ["full", "bce_nce", "bce_cts", "nce_cts"])
    )

    train_settings = {
        "train_list": args.train_list or config_get(config, "train_list", "list/xd_CLIP_rgb.csv"),
        "train_batch_size": args.train_batch_size if args.train_batch_size is not None else int(config_get(config, "train_batch_size", 64)),
        "max_epoch": args.max_epoch if args.max_epoch is not None else int(config_get(config, "max_epoch", 20)),
        "lr": args.lr if args.lr is not None else float(config_get(config, "lr", 2e-5)),
        "scheduler_rate": args.scheduler_rate if args.scheduler_rate is not None else float(config_get(config, "scheduler_rate", 0.1)),
        "scheduler_milestones": parse_int_list(
            args.scheduler_milestones
            if args.scheduler_milestones is not None
            else config_get(config, "scheduler_milestones", [3, 6, 10])
        ),
        "seed": seed,
        "num_seeds": num_seeds,
        "seeds": build_seed_list(seed, num_seeds, seeds_value),
        "save_checkpoints": args.save_checkpoints
        if args.save_checkpoints is not None
        else bool(config_get(config, "save_checkpoints", False)),
        "checkpoint_dir": args.checkpoint_dir or config_get(config, "checkpoint_dir", "experiment/results/checkpoints"),
    }
    latency_settings = {
        "benchmark_adapter_latency": bool(
            args.benchmark_adapter_latency or config_get(config, "benchmark_adapter_latency", False)
        ),
        "latency_warmup_iters": (
            args.latency_warmup_iters
            if args.latency_warmup_iters is not None
            else int(config_get(config, "latency_warmup_iters", 10))
        ),
        "latency_iters": (
            args.latency_iters
            if args.latency_iters is not None
            else int(config_get(config, "latency_iters", 100))
        ),
        "latency_batch_size": (
            args.latency_batch_size
            if args.latency_batch_size is not None
            else int(config_get(config, "latency_batch_size", 1))
        ),
    }

    rows: list[dict] = []
    if not args.skip_adapter_ablations:
        for variant in adapter_variants:
            variant_rows: list[dict] = []
            for current_seed in train_settings["seeds"]:
                seed_train_settings = dict(train_settings)
                seed_train_settings["seed"] = int(current_seed)
                row = train_one_adapter_variant(settings, seed_train_settings, variant)
                print_metrics(row)
                variant_rows.append(row)
                rows.append(row)
            summary_row = summarize_ablation_variant(
                variant_rows,
                variant,
                [int(current_seed) for current_seed in train_settings["seeds"]],
                settings["experiment_name"],
                "adapter_train_summary",
            )
            print_metrics(summary_row)
            rows.append(summary_row)

    if latency_settings["benchmark_adapter_latency"]:
        rows.extend(
            benchmark_adapter_latency(
                settings,
                adapter_variants,
                int(latency_settings["latency_warmup_iters"]),
                int(latency_settings["latency_iters"]),
                int(latency_settings["latency_batch_size"]),
            )
        )

    if args.train_loss_ablations:
        for variant in loss_variants:
            variant_rows: list[dict] = []
            for current_seed in train_settings["seeds"]:
                seed_train_settings = dict(train_settings)
                seed_train_settings["seed"] = int(current_seed)
                row = train_one_loss_variant(settings, seed_train_settings, variant)
                print_metrics(row)
                variant_rows.append(row)
                rows.append(row)
            summary_row = summarize_ablation_variant(
                variant_rows,
                variant,
                [int(current_seed) for current_seed in train_settings["seeds"]],
                settings["experiment_name"],
                "loss_train_summary",
            )
            print_metrics(summary_row)
            rows.append(summary_row)

    append_metrics_csv(settings["metrics_csv"], rows)
    if settings["metrics_json"]:
        write_json(settings["metrics_json"], rows)


if __name__ == "__main__":
    main()
