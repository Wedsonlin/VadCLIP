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
python experiment/scripts/run_ablation.py \
  --config experiment/configs/xd_ablation.yaml \
  --skip-adapter-ablations \
  --eval-adapter-checkpoints
```

To evaluate one variant, pass `--adapter-variants lgt_adapter`. Adapter checkpoints cannot be loaded by `run_xd_eval.py` because that script builds the base `CLIPVAD` model.

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

Export per-frame score traces and paper-style coarse anomaly score PNGs for selected test videos.
`--video-root` must point to the original XD-Violence test videos so the script can sample thumbnails.
It may contain either full movies, using the clip timestamps in the feature filename, or pre-cut clip videos:

```bash
python experiment/scripts/run_visualize_cases.py \
  --config experiment/configs/xd_baseline.yaml \
  --video-root D:/Datasets/XD-Violence/test/videos \
  --indices 0,1,2,3,4
```

Outputs are written to `experiment/results/figures/`:

- `.csv`: frame, Branch 1 score, Branch 2 score, ground-truth mask.
- `.png`: sampled video thumbnails plus Branch 1 / Branch 2 score curves with ground-truth anomaly regions highlighted.

## Notes

- The scripts reuse `src/model.py`, `src/utils/dataset.py`, and the XD mAP implementation without modifying them.
- Config files use a small key-value format parsed by `experiment/scripts/common.py`; no extra YAML dependency is required.
- Keep `strict_load: true` for fair checkpoint evaluation. Use `--no-strict-load` only when intentionally testing architecture sizes that do not match a checkpoint.

