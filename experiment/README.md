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

Export per-frame fine-grained category alignment score traces and paper-style coarse score PNGs
grouped by XD category. By default, the script runs inference on the full XD test set, ranks videos
within each category, and exports the top `3` highest-scoring cases per category (up to 21 figures total).
`--video-root` must point to the original XD-Violence test videos so the script can sample thumbnails.
It may contain either full movies, using the clip timestamps in the feature filename, or pre-cut clip videos.

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


Score curve and ranking metric:

- Figures and `.csv` always plot the alignment probability of the target fine-grained class in each
  subdirectory (e.g. `G_explosion/` uses explosion/G scores, `A_normal/` uses normal/A scores).
- Ranking score source (`--rank-score`, default `abnormal`):
  - `abnormal`: rank by Branch 2 scores.
  - `category`: rank by the alignment probability of the target fine-grained class (e.g. fighting for `B1`).
- Ranking metric (`--rank-metric`, default `auc`):
  - `auc`: frame-level AUC between the selected ranking score and the ground-truth mask.
  - `f1`: F1 at threshold 0.5.
  - `ap`: frame-level AP.
- Normal category (`A`):
  - `abnormal`: `1 - mean(abnormal_score)`.
  - `category`: `mean(normal_class_probability)`.

```bash
python experiment/scripts/run_visualize_cases.py \
  --config experiment/configs/xd_baseline.yaml \
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

- `.csv`: `frame`, `category_score`, `ground_truth`.
- `.png`: sampled video thumbnails plus the target category alignment score curve with ground-truth anomaly regions highlighted.
- `visualization_summary.json`: cases grouped by category code, with `score_type` (`category_alignment`), `rank_score`, `rank_metric`, `metric`, `metric_type` (e.g. `abnormal_auc`, `category_auc`, `mean_normal_prob`), source feature path, and output file paths.

## Notes

- The scripts reuse `src/model.py`, `src/utils/dataset.py`, and the XD mAP implementation without modifying them.
- Config files use a small key-value format parsed by `experiment/scripts/common.py`; no extra YAML dependency is required.
- Keep `strict_load: true` for fair checkpoint evaluation. Use `--no-strict-load` only when intentionally testing architecture sizes that do not match a checkpoint.

