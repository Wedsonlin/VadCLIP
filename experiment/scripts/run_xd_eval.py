from __future__ import annotations

import argparse

from ablation_models import AdapterAblationCLIPVAD
from common import (
    add_xd_runtime_args,
    append_metrics_csv,
    build_xd_model,
    evaluate_xd_model,
    load_config,
    load_model_weights,
    print_metrics,
    write_json,
    xd_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VadCLIP on XD-Violence.")
    add_xd_runtime_args(parser)
    parser.add_argument(
        "--adapter-variant",
        type=str,
        default=None,
        help="Build AdapterAblationCLIPVAD with this variant. Omit for base CLIPVAD.",
    )
    return parser.parse_args()


def main() -> None:
    # load args from command line and merge with config file
    args = parse_args()
    config = load_config(args.config)
    settings = xd_settings(args, config)

    if args.adapter_variant:
        model = build_xd_model(settings, AdapterAblationCLIPVAD, variant=args.adapter_variant)
    else:
        model = build_xd_model(settings)

    load_model_weights(
        model,
        settings["model_path"],
        settings["device"],
        strict=bool(settings["strict_load"]),
    )

    metrics, _ = evaluate_xd_model(model, settings)
    if args.adapter_variant:
        row = {
            "experiment": settings["experiment_name"],
            "adapter_variant": args.adapter_variant,
            **metrics,
        }
    else:
        row = {"experiment": settings["experiment_name"], **metrics}
    print_metrics(row)

    append_metrics_csv(settings["metrics_csv"], [row])
    if settings["metrics_json"]:
        write_json(settings["metrics_json"], row)


if __name__ == "__main__":
    main()
