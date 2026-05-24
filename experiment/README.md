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

Train separate checkpoints for the seven LGT-Adapter variants in Table 5:

```bash
python experiment/scripts/run_ablation.py --config experiment/configs/xd_ablation.yaml
```

This is the paper-style adapter ablation. It trains one model per variant, so it costs about seven full XD training runs.

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

Run loss-function ablations by training separate models:

```bash
python experiment/scripts/run_ablation.py --config experiment/configs/xd_ablation.yaml --skip-adapter-ablations --train-loss-ablations
```

Loss variants include `full`, `bce_nce`, `bce_cts`, and `nce_cts`. Checkpoints are saved under `experiment/results/checkpoints/`.

## 4. Prompt Ablations

Train each prompt-length and placement variant independently:

```bash
python experiment/scripts/run_prompt_hyperparams.py --config experiment/configs/xd_ablation.yaml
```

The default run trains `no_prompt` plus `middle` and `end` placements for total learnable prompt lengths
`4,6,8,10,12,14,16,18,20`. Checkpoints are saved under `experiment/results/checkpoints/` as
`xd_prompt_<variant>.pth`.

Useful overrides:

```bash
python experiment/scripts/run_prompt_hyperparams.py \
  --config experiment/configs/xd_ablation.yaml \
  --prompt-lengths 4,8,12 \
  --prompt-placements middle,end \
  --max-epoch 10
```

## 5. Qualitative Visualization

Export per-frame score traces and SVG plots for selected test videos:

```bash
python experiment/scripts/run_visualize_cases.py --config experiment/configs/xd_baseline.yaml --indices 0,1,2,3,4
```

Outputs are written to `experiment/results/figures/`:

- `.csv`: frame, Branch 1 score, Branch 2 score, ground-truth mask.
- `.svg`: score curves with ground-truth anomaly regions highlighted.

## Notes

- The scripts reuse `src/model.py`, `src/utils/dataset.py`, and the XD mAP implementation without modifying them.
- Config files use a small key-value format parsed by `experiment/scripts/common.py`; no extra YAML dependency is required.
- Keep `strict_load: true` for fair checkpoint evaluation. Use `--no-strict-load` only when intentionally testing architecture sizes that do not match a checkpoint.

