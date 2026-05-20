from __future__ import annotations

import argparse

import numpy as np

from common import (
    add_xd_runtime_args,
    append_metrics_csv,
    build_xd_model,
    collect_xd_predictions,
    evaluate_xd_predictions,
    fusion_metrics,
    load_config,
    load_model_weights,
    print_metrics,
    resolve_path,
    write_json,
    xd_prompt_text,
    xd_settings,
    build_xd_dataloader,
)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare VadCLIP branches and branch-score fusion.")
    add_xd_runtime_args(parser)
    parser.add_argument("--alphas", type=str, default=None, help="Comma-separated branch1 weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = xd_settings(args, config)
    alphas = (
        parse_float_list(args.alphas)
        if args.alphas is not None
        else [float(value) for value in config.get("fusion_alphas", [0, 0.25, 0.5, 0.75, 1])]
    )

    model = build_xd_model(settings)
    load_model_weights(
        model,
        settings["model_path"],
        settings["device"],
        strict=bool(settings["strict_load"]),
    )

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
    branch_metrics = evaluate_xd_predictions(predictions, gt, gtsegments, gtlabels)
    fusion_rows = fusion_metrics(predictions, gt, alphas)

    rows = [{"experiment": settings["experiment_name"], "variant": "branches", **branch_metrics}]
    rows.extend(
        {
            "experiment": settings["experiment_name"],
            "variant": f"fusion_alpha_{row['alpha']:.2f}",
            **row,
        }
        for row in fusion_rows
    )

    for row in rows:
        print_metrics(row)
        print("")

    append_metrics_csv(settings["metrics_csv"], rows)
    if settings["metrics_json"]:
        write_json(settings["metrics_json"], rows)


if __name__ == "__main__":
    main()
