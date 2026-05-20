from __future__ import annotations

import argparse
import copy
import random
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
    build_xd_dataloader,
    build_xd_model,
    config_get,
    evaluate_xd_model,
    load_config,
    load_model_weights,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XD-Violence adapter and loss ablations.")
    add_xd_runtime_args(parser)
    parser.add_argument("--adapter-variants", type=str, default=None)
    parser.add_argument("--skip-adapter-ablations", action="store_true")
    parser.add_argument("--train-loss-ablations", action="store_true")
    parser.add_argument("--loss-variants", type=str, default=None)
    parser.add_argument("--train-list", type=str, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--scheduler-rate", type=float, default=None)
    parser.add_argument("--scheduler-milestones", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def run_adapter_ablations(settings: dict, variants: list[str]) -> list[dict]:
    rows = []
    for variant in variants:
        print(f"Running adapter ablation: {variant}")
        model = build_xd_model(settings, AdapterAblationCLIPVAD, variant=variant)
        load_model_weights(
            model,
            settings["model_path"],
            settings["device"],
            strict=bool(settings["strict_load"]),
        )
        metrics, _ = evaluate_xd_model(model, settings)
        row = {
            "experiment": settings["experiment_name"],
            "ablation_type": "adapter",
            "variant": variant,
            **metrics,
        }
        print_metrics(row)
        rows.append(row)
    return rows


def train_one_loss_variant(settings: dict, train_settings: dict, variant: str) -> dict:
    print(f"Training loss ablation: {variant}")
    train_dataset = XDDataset(
        int(settings["visual_length"]),
        str(resolve_path(train_settings["train_list"])),
        False,
        XD_LABEL_MAP,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_settings["train_batch_size"]),
        shuffle=True,
        num_workers=int(settings["num_workers"]),
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

    for epoch in range(int(train_settings["max_epoch"])):
        model.train()
        running = {"loss": 0.0, "bce_loss": 0.0, "nce_loss": 0.0, "cts_loss": 0.0}
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

            running["loss"] += float(loss.detach().cpu())
            for key, value in values.items():
                running[key] += value

            if step % 50 == 0:
                scale = 1.0 / step
                print(
                    f"epoch={epoch + 1} step={step} "
                    f"loss={running['loss'] * scale:.4f} "
                    f"bce={running['bce_loss'] * scale:.4f} "
                    f"nce={running['nce_loss'] * scale:.4f} "
                    f"cts={running['cts_loss'] * scale:.6f}"
                )

        scheduler.step()
        metrics, _ = evaluate_xd_model(model, eval_settings)
        current_ap = metrics["branch1_ap"]
        print(f"epoch={epoch + 1} branch1_ap={current_ap:.6f}")
        if current_ap > best_ap:
            best_ap = current_ap
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())

    checkpoint_dir = resolve_path(train_settings["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"xd_loss_{variant}.pth"
    if best_state is not None:
        torch.save(best_state, checkpoint_path)

    return {
        "experiment": settings["experiment_name"],
        "ablation_type": "loss",
        "variant": variant,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        **best_metrics,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = xd_settings(args, config)

    seed = args.seed if args.seed is not None else int(config_get(config, "seed", 234))
    setup_seed(seed)

    adapter_variants = parse_list(
        args.adapter_variants
        if args.adapter_variants is not None
        else config_get(
            config,
            "adapter_variants",
            ["full", "no_transformer", "no_gcn", "cosine_gcn_only", "distance_gcn_only"],
        )
    )
    loss_variants = parse_list(
        args.loss_variants
        if args.loss_variants is not None
        else config_get(config, "loss_variants", ["full", "bce_nce", "bce_cts", "nce_cts"])
    )

    train_settings = {
        "train_list": args.train_list or config_get(config, "train_list", "list/xd_CLIP_rgb.csv"),
        "train_batch_size": args.train_batch_size
        if args.train_batch_size is not None
        else int(config_get(config, "train_batch_size", 64)),
        "max_epoch": args.max_epoch if args.max_epoch is not None else int(config_get(config, "max_epoch", 20)),
        "lr": args.lr if args.lr is not None else float(config_get(config, "lr", 2e-5)),
        "scheduler_rate": args.scheduler_rate
        if args.scheduler_rate is not None
        else float(config_get(config, "scheduler_rate", 0.1)),
        "scheduler_milestones": parse_list(
            args.scheduler_milestones
            if args.scheduler_milestones is not None
            else config_get(config, "scheduler_milestones", [3, 6, 10])
        ),
        "checkpoint_dir": args.checkpoint_dir
        or config_get(config, "checkpoint_dir", "experiment/results/checkpoints"),
    }

    rows: list[dict] = []
    if not args.skip_adapter_ablations:
        rows.extend(run_adapter_ablations(settings, adapter_variants))

    if args.train_loss_ablations:
        for variant in loss_variants:
            rows.append(train_one_loss_variant(settings, train_settings, variant))

    append_metrics_csv(settings["metrics_csv"], rows)
    if settings["metrics_json"]:
        write_json(settings["metrics_json"], rows)


if __name__ == "__main__":
    main()
