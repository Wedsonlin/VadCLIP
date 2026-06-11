# VadCLIP Course Experiments

This directory keeps course-project experiments separate from the original paper implementation in `src/`.

## Layout

```text
experiment/
  configs/          # Reproducible experiment settings
  scripts/          # Experiment entry points
  results/          # Metrics, checkpoints, and figures
```

## 1. Baseline Reproduction

Evaluate a trained XD-Violence checkpoint and write AUC/AP/mAP to `experiment/results/metrics.csv`.

```bash
python experiment/scripts/run_xd_eval.py --config experiment/configs/xd_baseline.yaml
```

The default config expects `model/model_xd.pth` and the XD feature paths listed in `list/xd_CLIP_rgbtest.csv`.

## 2. Branch Comparison and Fusion

Compare the classifier branch, visual-language alignment branch, and fused scores.

```bash
python experiment/scripts/run_branch_fusion.py --config experiment/configs/xd_ablation.yaml
```

Use `--alphas 0,0.25,0.5,0.75,1` to override fusion weights. `alpha` is the Branch 1 weight:

```text
score = alpha * branch1 + (1 - alpha) * branch2
```

## 3. LGT-Adapter and Loss Ablations

Train the seven LGT-Adapter variants in Table 5. Each variant runs five seeds by default and writes
per-seed rows plus a summary row with mean, variance, standard deviation, and 95% confidence intervals:

```bash
python experiment/scripts/run_ablation.py --config experiment/configs/xd_ablation.yaml
```

This is the paper-style adapter ablation. It now costs about 35 full XD training runs by default. Model
parameters are not saved by default; pass `--save-checkpoints` to write seed-specific checkpoints such as
`xd_adapter_lgt_adapter_seed234.pth`.

Evaluate trained adapter-variant checkpoints with their matching model structure:

```bash
python experiment/scripts/run_xd_eval.py \
  --config experiment/configs/xd_ablation.yaml \
  --adapter-variant lgt_adapter \
  --model-path experiment/results/checkpoints/xd_adapter_lgt_adapter_seed234.pth \
  --experiment-name xd_adapter_lgt_adapter_seed234
```

Pass `--adapter-variant` to build the matching `AdapterAblationCLIPVAD` architecture before loading the checkpoint.

Benchmark adapter inference latency without dataset I/O:

```bash
python experiment/scripts/run_ablation.py \
  --config experiment/configs/xd_ablation.yaml \
  --skip-adapter-ablations \
  --benchmark-adapter-latency
```

Latency rows report mean/std/median forward time and throughput for fixed dummy inputs. They measure compute cost, not detection performance.

Run loss-function ablations with the same five-seed protocol:

```bash
python experiment/scripts/run_ablation.py --config experiment/configs/xd_ablation.yaml --skip-adapter-ablations --train-loss-ablations
```

Loss variants include `full`, `bce_nce`, `bce_cts`, and `nce_cts`. Model parameters are not saved by
default; pass `--save-checkpoints` to write files such as `xd_loss_bce_nce_seed234.pth`.

## 4. Prompt Ablations

Train each prompt-length and placement variant independently:

```bash
python experiment/scripts/run_prompt_hyperparams.py --config experiment/configs/xd_ablation.yaml
```

The default run trains `no_prompt` plus `middle` and `end` placements for total learnable prompt lengths
`4,6,8,10,12,14,16,18,20`. Model parameters are not saved by default; pass `--save-checkpoints`
to write files such as `xd_prompt_middle_10_seed234.pth`.

Useful overrides:

```bash
python experiment/scripts/run_prompt_hyperparams.py \
  --config experiment/configs/xd_ablation.yaml \
  --prompt-lengths 4,8,12 \
  --prompt-placements middle,end \
  --max-epoch 10
```

## 5. Qualitative Visualization

Export per-frame score traces and paper-style PNG figures grouped by XD fine-grained category.
By default, the script runs inference on the full XD test set, ranks videos within each category,
and exports the top `3` highest-scoring cases per category (up to 21 figures total).

`--video-root` must point to the original XD-Violence test videos so the script can sample thumbnails.
It may contain either full movies, using the clip timestamps in the feature filename, or pre-cut clip videos.

By default the script builds the base `CLIPVAD` model and loads `model_path` from the config
(typically `model/model_xd.pth` via `xd_baseline.yaml`). To visualize an adapter ablation checkpoint,
pass `--adapter-variant` so the script builds the matching `AdapterAblationCLIPVAD` architecture
before loading weights (same pattern as `run_xd_eval.py`).

Fine-grained category map:


| Code | Name         |
| ---- | ------------ |
| A    | normal       |
| B1   | fighting     |
| B2   | shooting     |
| B4   | riot         |
| B5   | abuse        |
| B6   | car accident |
| G    | explosion    |


Pre-filter and ranking:

- `--max-anomaly-ratio` (default `0.8`): skip videos whose **anomaly frame ratio** exceeds this threshold before ranking. Pure normal clips (`label=A`) use an all-zero anomaly mask and are not filtered by this rule.
- `--rank-score` (default `abnormal`) controls which score is used to **select** top-k cases within each category.
  - `abnormal`: Branch 2 coarse anomaly scores.
  - `category`: fine-grained alignment probability of the target class (e.g. fighting for `B1`).
- `--plot-score` (default `category`) controls the **green curve** drawn in each PNG:
  - `category`: fine-grained class alignment (e.g. `fighting (B1)` in `B1_fighting/`).
  - `abnormal`: coarse Branch 2 anomaly score (`Branch 2 (abnormal)`).
- `--rank-metric` (default `auc`): ranking metric — `auc`, `f1` (F1@0.5), or `ap`.
- Normal category (`A`) when ranking:
  - `abnormal`: `1 - mean(abnormal_score)`.
  - `category`: `mean(normal_class_probability)`.

Figure layout and ground truth:

- PNGs have no long title bar at the top; the score label (e.g. `fighting (B1)`) is drawn **above** the chart, outside the plot area.
- `A_normal/` figures use a white chart background with **no pink GT overlay** (for pure normal clips, dataset intervals mark normal segments, not anomalies).
- Other categories highlight annotated anomaly intervals in pink.

Baseline checkpoint:

```bash
python experiment/scripts/run_visualize_cases.py \
  --config experiment/configs/xd_baseline.yaml \
  --video-root H:/Datasets/XD-Violence/test/videos \
  --test-list list/xd_CLIP_rgbtest_local.csv
```

Adapter checkpoint (`lgt_adapter` example):

```bash
python experiment/scripts/run_visualize_cases.py \
  --config experiment/configs/xd_ablation.yaml \
  --adapter-variant lgt_adapter \
  --model-path model/xd_adapter_lgt_adapter_seed234.pth \
  --video-root H:/Datasets/XD-Violence/test/videos \
  --test-list list/xd_CLIP_rgbtest_local.csv
```

Useful overrides:

```bash
# Export the top 5 cases per category instead of the default 3
python experiment/scripts/run_visualize_cases.py \
  --config experiment/configs/xd_baseline.yaml \
  --video-root H:/Datasets/XD-Violence/test/videos \
  --test-list list/xd_CLIP_rgbtest_local.csv \
  --top-k 5

# Rank by F1@0.5 instead of the default AUC
python experiment/scripts/run_visualize_cases.py \
  --config experiment/configs/xd_baseline.yaml \
  --video-root H:/Datasets/XD-Violence/test/videos \
  --test-list list/xd_CLIP_rgbtest_local.csv \
  --rank-metric f1

# Rank by fine-grained category alignment scores instead of Branch 2 abnormal scores
python experiment/scripts/run_visualize_cases.py \
  --config experiment/configs/xd_baseline.yaml \
  --video-root H:/Datasets/XD-Violence/test/videos \
  --test-list list/xd_CLIP_rgbtest_local.csv \
  --rank-score category

# Plot coarse Branch 2 scores in the green curve (ranking still uses --rank-score default)
python experiment/scripts/run_visualize_cases.py \
  --config experiment/configs/xd_ablation.yaml \
  --adapter-variant lgt_adapter \
  --model-path model/xd_adapter_lgt_adapter_seed234.pth \
  --video-root H:/Datasets/XD-Violence/test/videos \
  --test-list list/xd_CLIP_rgbtest_local.csv \
  --plot-score abnormal

# Rank only within a subset of test-video indices
python experiment/scripts/run_visualize_cases.py \
  --config experiment/configs/xd_baseline.yaml \
  --video-root D:/Datasets/XD-Violence/test/videos \
  --test-list list/xd_CLIP_rgbtest_local.csv \
  --indices 12,48,103
```

Outputs are written under `experiment/results/figures/`:

```text
experiment/results/figures/
  A_normal/
  B1_fighting/
  B2_shooting/
  B4_riot/
  B5_abuse/
  B6_car_accident/
  G_explosion/
  visualization_summary.json
```

- `.png`: sampled video thumbnails plus one green score curve (`--plot-score`); anomaly GT overlay on non-`A` categories only.
- `visualization_summary.json`: top-level `model_path`, `adapter_variant` (null for base `CLIPVAD`), `rank_score`, `rank_metric`, `plot_score`, and `max_anomaly_ratio`; per-category lists with `score_type`, `plot_score`, `rank_score`, `rank_metric`, `metric`, `metric_type`, source feature path, and PNG path.

## Notes

- The scripts reuse `src/model.py`, `src/utils/dataset.py`, and the XD mAP implementation without modifying them.
- Config files use a small key-value format parsed by `experiment/scripts/common.py`; no extra YAML dependency is required.
- Keep `strict_load: true` for fair checkpoint evaluation. Use `--no-strict-load` only when intentionally testing architecture sizes that do not match a checkpoint.

