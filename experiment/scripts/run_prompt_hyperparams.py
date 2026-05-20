from __future__ import annotations

import argparse

from common import (
    add_xd_runtime_args,
    append_metrics_csv,
    build_xd_model,
    config_get,
    evaluate_xd_model,
    load_config,
    load_model_weights,
    print_metrics,
    write_json,
    xd_settings,
)


def parse_int_list(value) -> list[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def parse_str_list(value) -> list[str | None]:
    if value is None:
        return [None]
    if isinstance(value, str):
        return [item.strip() or None for item in value.split("|")]
    return [None if item in {"", None} else str(item) for item in value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep prompt and temporal hyperparameters on XD-Violence.")
    add_xd_runtime_args(parser)
    parser.add_argument("--prompt-token-pairs", type=str, default=None, help="Comma list like 0/0,5/5,10/10.")
    parser.add_argument("--attn-windows", type=str, default=None, help="Comma list like 16,32,64,128.")
    parser.add_argument(
        "--prompt-templates",
        type=str,
        default=None,
        help='Templates separated by "|"; use {label}, e.g. "{label}|a video of {label}".',
    )
    return parser.parse_args()


def parse_prompt_pairs(value) -> list[tuple[int, int]]:
    if isinstance(value, str):
        pairs = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            prefix, postfix = item.split("/")
            pairs.append((int(prefix), int(postfix)))
        return pairs
    return [(int(prefix), int(postfix)) for prefix, postfix in value]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_settings = xd_settings(args, config)

    prompt_pairs = parse_prompt_pairs(
        args.prompt_token_pairs
        if args.prompt_token_pairs is not None
        else config_get(config, "prompt_token_pairs", [(0, 0), (5, 5), (10, 10), (20, 20)])
    )
    attn_windows = parse_int_list(
        args.attn_windows
        if args.attn_windows is not None
        else config_get(config, "attn_windows", [16, 32, 64, 128])
    )
    prompt_templates = parse_str_list(
        args.prompt_templates
        if args.prompt_templates is not None
        else config_get(config, "prompt_templates", [None, "a video of {label}"])
    )

    rows: list[dict] = []

    for prefix, postfix in prompt_pairs:
        settings = dict(base_settings)
        settings["prompt_prefix"] = prefix
        settings["prompt_postfix"] = postfix
        settings["experiment_name"] = f"{base_settings['experiment_name']}_prompt_{prefix}_{postfix}"
        model = build_xd_model(settings)
        load_model_weights(model, settings["model_path"], settings["device"], strict=bool(settings["strict_load"]))
        metrics, _ = evaluate_xd_model(model, settings)
        row = {
            "experiment": base_settings["experiment_name"],
            "sweep_type": "prompt_tokens",
            "prompt_prefix": prefix,
            "prompt_postfix": postfix,
            **metrics,
        }
        print_metrics(row)
        rows.append(row)

    for attn_window in attn_windows:
        settings = dict(base_settings)
        settings["attn_window"] = attn_window
        settings["experiment_name"] = f"{base_settings['experiment_name']}_attn_{attn_window}"
        model = build_xd_model(settings)
        load_model_weights(model, settings["model_path"], settings["device"], strict=bool(settings["strict_load"]))
        metrics, _ = evaluate_xd_model(model, settings)
        row = {
            "experiment": base_settings["experiment_name"],
            "sweep_type": "attn_window",
            "attn_window": attn_window,
            **metrics,
        }
        print_metrics(row)
        rows.append(row)

    for template in prompt_templates:
        settings = dict(base_settings)
        settings["prompt_template"] = template
        name = "raw_labels" if template is None else template.replace(" ", "_").replace("{label}", "label")
        settings["experiment_name"] = f"{base_settings['experiment_name']}_template_{name}"
        model = build_xd_model(settings)
        load_model_weights(model, settings["model_path"], settings["device"], strict=bool(settings["strict_load"]))
        metrics, _ = evaluate_xd_model(model, settings)
        row = {
            "experiment": base_settings["experiment_name"],
            "sweep_type": "prompt_template",
            "prompt_template": template or "raw_label",
            **metrics,
        }
        print_metrics(row)
        rows.append(row)

    append_metrics_csv(base_settings["metrics_csv"], rows)
    if base_settings["metrics_json"]:
        write_json(base_settings["metrics_json"], rows)


if __name__ == "__main__":
    main()
