from __future__ import annotations

import argparse
import copy
import math
import statistics

import torch
from torch.optim.lr_scheduler import MultiStepLR

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
from run_ablation import build_train_loader, loss_for_variant, setup_seed
from utils.tools import get_batch_label_vector


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


def parse_int_list(value) -> list[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def parse_str_list(value) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XD-Violence prompt-length ablations.")
    add_xd_runtime_args(parser)
    parser.add_argument("--prompt-lengths", type=str, default=None, help="Comma list of total learnable prompt lengths.")
    parser.add_argument("--prompt-placements", type=str, default=None, help="Comma list of placements: middle,end.")
    parser.add_argument("--train-list", type=str, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--scheduler-rate", type=float, default=None)
    parser.add_argument("--scheduler-milestones", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=None, help="Number of sequential seeds to run per variant.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma list of explicit seeds to run per variant.")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def build_prompt_variants(lengths: list[int], placements: list[str]) -> list[dict]:
    variants = [
        {
            "variant": "no_prompt",
            "prompt_length": 0,
            "prompt_placement": "none",
            "prompt_prefix": 0,
            "prompt_postfix": 0,
        }
    ]

    for length in lengths:
        if length <= 0:
            raise ValueError("Prompt lengths must be positive; no_prompt is added automatically.")
        for placement in placements:
            if placement == "middle":
                if length % 2 != 0:
                    raise ValueError(f"Middle placement requires an even total prompt length, got {length}.")
                prefix = length // 2
                postfix = length // 2
            elif placement == "end":
                prefix = length
                postfix = 0
            else:
                raise ValueError("Prompt placements must be chosen from: middle,end.")

            variants.append(
                {
                    "variant": f"{placement}_{length}",
                    "prompt_length": length,
                    "prompt_placement": placement,
                    "prompt_prefix": prefix,
                    "prompt_postfix": postfix,
                }
            )

    return variants


def build_seed_list(base_seed: int, num_seeds: int, seeds_value) -> list[int]:
    if seeds_value is not None:
        seeds = parse_int_list(seeds_value)
        if not seeds:
            raise ValueError("At least one seed must be provided when --seeds is set.")
        return seeds
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive.")
    return [base_seed + offset for offset in range(num_seeds)]


def build_train_settings(args: argparse.Namespace, config: dict) -> dict:
    base_seed = args.seed if args.seed is not None else int(config_get(config, "seed", 234))
    num_seeds = args.num_seeds if args.num_seeds is not None else int(config_get(config, "num_seeds", 5))
    seeds_value = args.seeds if args.seeds is not None else config_get(config, "seeds", None)
    return {
        "train_list": args.train_list or config_get(config, "train_list", "list/xd_CLIP_rgb.csv"),
        "train_batch_size": args.train_batch_size
        if args.train_batch_size is not None
        else int(config_get(config, "train_batch_size", 64)),
        "max_epoch": args.max_epoch if args.max_epoch is not None else int(config_get(config, "max_epoch", 10)),
        "lr": args.lr if args.lr is not None else float(config_get(config, "lr", 1e-5)),
        "scheduler_rate": args.scheduler_rate
        if args.scheduler_rate is not None
        else float(config_get(config, "scheduler_rate", 0.1)),
        "scheduler_milestones": parse_int_list(
            args.scheduler_milestones
            if args.scheduler_milestones is not None
            else config_get(config, "scheduler_milestones", [3, 6, 10])
        ),
        "seed": base_seed,
        "num_seeds": num_seeds,
        "seeds": build_seed_list(base_seed, num_seeds, seeds_value),
        "checkpoint_dir": args.checkpoint_dir or config_get(config, "checkpoint_dir", "experiment/results/checkpoints"),
    }


def relative_checkpoint_path(path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def train_one_prompt_variant(base_settings: dict, train_settings: dict, prompt_variant: dict) -> dict:
    from utils.logger import Logger

    variant = prompt_variant["variant"]
    seed = int(train_settings["seed"])
    print(f"Training prompt ablation: {variant} seed={seed}")

    setup_seed(seed)

    settings = dict(base_settings)
    settings["prompt_prefix"] = int(prompt_variant["prompt_prefix"])
    settings["prompt_postfix"] = int(prompt_variant["prompt_postfix"])

    eval_settings = dict(settings)
    eval_settings["batch_size"] = 1

    train_loader = build_train_loader(settings, train_settings)
    logger = Logger(
        project="WSAVD_prompt_variants",
        name=f"{settings['experiment_name']}_prompt_{variant}_seed{seed}",
    )
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
                    "prompt_average_mAP": metrics["average_mAP"],
                }
            )
            print(
                f"variant={variant} seed={seed} epoch={epoch + 1} "
                f"branch1_ap={metrics['branch1_ap']:.6f}, "
                f"branch2_ap={metrics['branch2_ap']:.6f}, "
                f"average_mAP={metrics['average_mAP']:.6f}"
            )
            current_ap = metrics["branch2_ap"]
            if current_ap > best_ap:
                best_ap = current_ap
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())

        checkpoint_dir = resolve_path(train_settings["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"xd_prompt_{variant}_seed{seed}.pth"
        if best_state is not None:
            torch.save(best_state, checkpoint_path)
    finally:
        logger.finish()

    if str(settings["device"]).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "experiment": settings["experiment_name"],
        "ablation_type": "prompt_train",
        "variant": variant,
        "seed": seed,
        "prompt_length": prompt_variant["prompt_length"],
        "prompt_placement": prompt_variant["prompt_placement"],
        "prompt_prefix": prompt_variant["prompt_prefix"],
        "prompt_postfix": prompt_variant["prompt_postfix"],
        "checkpoint": relative_checkpoint_path(checkpoint_path),
        **best_metrics,
    }


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


def summarize_prompt_variant(rows: list[dict], prompt_variant: dict, seeds: list[int], experiment_name: str) -> dict:
    summary = {
        "experiment": experiment_name,
        "ablation_type": "prompt_train_summary",
        "variant": prompt_variant["variant"],
        "prompt_length": prompt_variant["prompt_length"],
        "prompt_placement": prompt_variant["prompt_placement"],
        "prompt_prefix": prompt_variant["prompt_prefix"],
        "prompt_postfix": prompt_variant["prompt_postfix"],
        "seed_count": len(rows),
        "seeds": ",".join(str(seed) for seed in seeds),
    }
    for key in METRIC_KEYS:
        values = metric_values(rows, key)
        if not values:
            continue
        mean = float(statistics.mean(values))
        std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        half_width = ci95_half_width(values)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
        summary[f"{key}_ci95_low"] = mean - half_width
        summary[f"{key}_ci95_high"] = mean + half_width
        summary[f"{key}_ci95_half_width"] = half_width
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_settings = xd_settings(args, config)

    prompt_lengths = parse_int_list(
        args.prompt_lengths
        if args.prompt_lengths is not None
        else config_get(config, "prompt_lengths", [4, 6, 8, 10, 12, 14, 16, 18, 20])
    )
    prompt_placements = parse_str_list(
        args.prompt_placements
        if args.prompt_placements is not None
        else config_get(config, "prompt_placements", ["middle", "end"])
    )
    train_settings = build_train_settings(args, config)
    prompt_variants = build_prompt_variants(prompt_lengths, prompt_placements)

    rows: list[dict] = []
    for prompt_variant in prompt_variants:
        variant_rows: list[dict] = []
        for seed in train_settings["seeds"]:
            seed_train_settings = dict(train_settings)
            seed_train_settings["seed"] = int(seed)
            row = train_one_prompt_variant(base_settings, seed_train_settings, prompt_variant)
            print_metrics(row)
            variant_rows.append(row)
            rows.append(row)
        summary_row = summarize_prompt_variant(
            variant_rows,
            prompt_variant,
            [int(seed) for seed in train_settings["seeds"]],
            base_settings["experiment_name"],
        )
        print_metrics(summary_row)
        rows.append(summary_row)

    append_metrics_csv(base_settings["metrics_csv"], rows)
    if base_settings["metrics_json"]:
        write_json(base_settings["metrics_json"], rows)


if __name__ == "__main__":
    main()
