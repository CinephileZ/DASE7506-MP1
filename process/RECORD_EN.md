# Experiment Record

## Notes

- All run paths point to `code/runs/`; abbreviated paths such as `exploration/...` and `trial1/...` in the tables are relative to `code/runs/`, while `final/...` points to `code/final/`.
- In `metrics.json`, `validation.bpb` is the validation score of the raw checkpoint at the end of training. If `averaged_validation.bpb` is present, it corresponds to the SWA/EMA checkpoint. Model selection should use the validation score for the checkpoint actually intended for submission.
- `steps` is the total number of training steps; numbers such as `6000` in directory names sometimes indicate the averaging start step, not necessarily the total number of training steps. Refer to `metrics.json` for the experiment settings.
- Records are organized by experimental question and stage and are not guaranteed to be in exact chronological order; directory names do not establish ordering.

## 1. Baseline and Early One-Factor Explorations

### Learning-Rate Schedule Horizon

Same seed (17), baseline, and 1200 training steps; only the cosine schedule horizon was changed:

| schedule steps | Validation BPB | Run |
|---:|---:|---|
| 1200 | 2.071087 | `code/runs/exploration/baseline-s17-step1200` |
| 2400 | 2.004458 | `code/runs/exploration/baseline-s17-horizon2400` |
| 4800 | 1.999483 | `code/runs/exploration/baseline-s17-horizon4800` |

These results only support the conclusion that "extending the schedule horizon helps when the number of training steps is limited."

### Width, Training Steps, and RoPE

| Comparison | Conditions | Validation BPB | Run |
|---|---|---:|---|
| baseline 128×4 vs 256×8 | 1200 steps, seed 17; schedule 4800 | 1.999483 vs 1.832846 | `exploration/baseline-s17-horizon4800`; `exploration/architecture-256x8-s17` |
| 256×8, 1200 vs 2400 steps | seed 17; schedule 4800 | 1.832846 vs 1.669807 | `exploration/architecture-256x8-s17`; `exploration/architecture-256x8-step2400` |
| RoPE, 1200 steps | 256×8, seed 17; schedule 4800 | without RoPE 1.832846; with RoPE 1.728150 | `exploration/architecture-256x8-s17`; `exploration/rope-256x8-s17` |
| RoPE, 2400 steps | 256×8, seed 17; schedule 4800 | without RoPE 1.669807; with RoPE 1.643764 | `exploration/architecture-256x8-step2400`; `exploration/rope-256x8-s17-step2400` |

### Normalization, MLP, Learning Rate, and Number of Heads

| Variable | Setting and Validation BPB | Run |
|---|---|---|
| SwiGLU | 256×8 + RoPE, 2400 steps: standard MLP 1.643764; SwiGLU 1.651917 | `exploration/rope-256x8-s17-step2400`; `exploration/rope-256x8-swiglu-s17-step2400` |
| RMSNorm | 256×8 + RoPE, 1200 steps: standard LayerNorm 1.728150; RMSNorm 1.728452 | `exploration/rope-256x8-s17`; `exploration/rope-256x8-rmsnorm-s17-step1200` |
| QK-Norm | 256×8 + RoPE, 1200 steps: without QK-Norm 1.728150; with QK-Norm 1.742784 | `exploration/rope-256x8-s17`; `exploration/rope-256x8-qknorm-s17-step1200` |
| Learning rate | 256×8 + RoPE, 2400 steps: 0.0005 yields 1.661655; 0.001 yields 1.643764; 0.0015 yields 1.629976 | `exploration/rope-256x8-lr05-s17-step2400`; `exploration/rope-256x8-s17-step2400`; `exploration/rope-256x8-lr015-s17-step2400` |
| Attention heads | 256×8 + RoPE, lr=0.0015, 2400 steps: 8 heads yield 1.629976; 16 heads yield 1.650949; parameter count is 6,842,880 for both | `exploration/rope-256x8-lr015-s17-step2400`; `exploration/rope-256x16-lr015-s17-step2400` |

## 2. Parallel Transformer and 256×8 Ablations

This stage mainly uses seed 17, 256×8, and 3000 steps. Unless otherwise specified in the table, lr=0.001, weight decay=0.1, and RoPE are enabled.

| Experiment | Change | Validation BPB | Run |
|---|---|---:|---|
| B1 | baseline 128×4 | 1.832139 | `exploration/baseline-original-s17-step3000` |
| A1 | Changed to 256×8 | 1.655924 | `exploration/architecture-only-256x8-s17-step3000` |
| R1 | 256×8 + RoPE | 1.625796 | `exploration/architecture256x8-rope-s17-step3000` |
| S1 | Same as R1, schedule horizon=5000 | 1.634966 | `exploration/architecture256x8-rope-s17-schedule5000-step3000` |
| P1 | Used parallel Transformer blocks on R1 | 1.612628 | `exploration/rope-parallel-256x8-s17-step3000` |
| QK-Norm | Added QK-Norm to P1 | 1.641189 | `exploration/rope-parallel-256x8-s17-step3000-qknorm` |
| SwiGLU | Switched P1 to SwiGLU | 1.618329 | `exploration/rope-parallel-256x8-s17-step3000-swiglu` |
| Learning rate | P1 configuration, lr=0.0015 | 1.616554 | `exploration/rope-parallel-256x8-s17-step3000-lr0015` |
| Seed | P1 configuration, seed=28 | 1.616698 | `exploration/rope-parallel-256x8-s28-step3000` |
| Longer training | P1 configuration, 3500 steps | 1.617289 | `exploration/rope-parallel-256x8-s17-step3500` |
| Longer training | P1 configuration, 6000 steps | 1.712921 | `exploration/rope-parallel-256x8-s17-step6000` |
| Longer training, higher learning rate | P1 configuration, lr=0.0015, 6000 steps | 1.735991 | `exploration/rope-parallel-256x8-s17-step6000-lr0015` |

**Stage conclusion:** P1 is the reference point for subsequent 3000-step averaging/weight-decay experiments, not the overall best configuration. Scores from longer training cannot be interpreted independently of the other training settings; the 6000-step comparison here actually performed worse.

## 3. Checkpoint Averaging and Weight Decay After P1

The validation column prioritizes `averaged_validation.bpb` for the actual averaged checkpoint; when there is no averaging, it shows the regular validation score.

| Run | Setting / checkpoint | Validation BPB |
|---|---|---:|
| `exploration/rope-parallel-s17-dense-step3000` | P1, 3000; raw checkpoint | 1.613392 |
| Same run | Checkpoint average 2800–3000 | 1.606744 |
| `exploration/rope-parallel-s17-dense-step3600` | P1, 3600; raw checkpoint | 1.618898 |
| Same run | Checkpoint average 2800–3200 | 1.603611 |
| Same run | Checkpoint average 3000–3400 | 1.606548 |
| `exploration/rope-parallel-wd02-s17-dense-step3000` | wd=0.2; raw checkpoint | 1.603691 |
| Same run | Checkpoint average 2800–3000 | 1.597010 |
| Same run | Checkpoint average 2700–3000 | 1.596128 |
| Same run | Checkpoint average 2600–3000 | 1.595227 |
| Same run | Checkpoint average 2500–3000 | 1.594629 |
| Same run | Checkpoint average 2400–3000 | 1.594629 |
| `exploration/rope-parallel-wd02-ema-step3000` | wd=0.2, EMA decay=0.999 | 1.606889 |
| `exploration/rope-parallel-wd02-late-ema-step3000` | wd=0.2, EMA decay=0.99, average 2000–3000 | 1.595911 |
| `exploration/rope-parallel-wd02-late2400-ema-step3000` | wd=0.2, EMA decay=0.99, average 2400–3000 | 1.595915 |
| `exploration/rope-parallel-wd02-swa-step3000` | wd=0.2, SWA 2000–3000 | 1.592483 |
| `exploration/rope-parallel-wd02-swa1500-step3000` | wd=0.2, SWA 1500–3000 | 1.594047 |
| `exploration/rope-parallel-wd02-swa2500-step3000` | wd=0.2, SWA 2500–3000 | 1.594177 |
| `exploration/rope-parallel-wd02-lr0015-s17-dense-step3000` | wd=0.2, lr=0.0015; raw checkpoint | 1.603730 |
| Same run | Checkpoint average 2800–3000 | 1.595305 |

## 4. Width-320 and SWA Search

All settings use RoPE, parallel blocks, and seed 17; refer to the specific configuration in the metrics. The validation column shows the SWA checkpoint validation score.

| Run | width×depth, steps, wd, SWA start step | Validation BPB |
|---|---|---:|
| `exploration/rope-320x8-swa-step2400-astart1500` | 320×8, 2400, 0.1, 1500 | 1.577299 |
| `exploration/rope-320x8-swa-step2400-astart1800` | 320×8, 2400, 0.1, 1800 | 1.580044 |
| `exploration/rope-320x8-swa-step2400-astart2000` | 320×8, 2400, 0.1, 2000 | 1.583686 |
| `exploration/rope-320x8-swa-wd02-step2400-astart1500` | 320×8, 2400, 0.2, 1500 | 1.571323 |
| `exploration/rope-320x8-swa-wd03-step2400-astart1500` | 320×8, 2400, 0.3, 1500 | 1.568290 |
| `exploration/rope-320x8-swa-step3000` | 320×8, 3000, 0.1, 2000 | 1.586817 |
| `exploration/rope-320x8-swa-wd025-step3000` | 320×8, 3000, 0.25, 2000 | 1.570090 |
| `exploration/rope-320x8-swa-wd02-step3000` | 320×8, 3000, 0.2, 2000 | 1.574776 |
| `exploration/rope-320x8-swa-wd02-step4000-start2000` | 320×8, 4000, 0.2, 2000 | 1.576424 |
| `exploration/rope-320x10-swa-step3000` | 320×10, 3000, 0.1, 2000 | 1.593707 |
| `exploration/rope-320x10-parallel-wd02-swa-step3000` | 320×10, 3000, 0.2, 2000 | 1.578497 |
| `exploration/rope-320x10-parallel-wd02-swa-step4000` | 320×10, 4000, 0.2, 2500 | 1.590796 |
| `exploration/rope-320x10-wd02-swa-step4000-start2000` | 320×10, 4000, 0.2, 2000 | 1.579779 |
| `exploration/rope-320x10-swa-wd03-step2400-astart1500` | 320×10, 2400, 0.3, 1500 | 1.571168 |

Among the tested 320-width models, the lowest validation score is 1.568290 for 320×8, wd=0.3, 2400 steps, and SWA start step 1500; the 320×8, wd=0.2, 3000-step configuration scores 1.574776. Rows with different width, depth, steps, and wd cannot be treated as strict one-factor comparisons.

## 5. 10,000-Step Long Runs, Component Ablations, and Averaging Start-Step Sweeps

The early-exploration runs named `rope-256x8-swa-dropout-step10000*` are 256×8, 10k-step SWA/dropout explorations, with averaged validation scores of 1.525053 (start 5000) and 1.526397 (start 3000), respectively.

The main later configuration is 256×6, 10k steps, seed 17, lr=0.001, wd=0.1, and dropout=0.1, with SWA updated every step. In `full-6000` below, `6000` refers to the SWA start step, not the total number of training steps; the metrics confirm 10,000 total steps.

| Run | Configuration difference | Averaged validation BPB |
|---|---|---:|
| `code/runs/full-6000` | parallel=false, cache=true, dropout=0.1, SWA start=6000 | 1.517091 |
| `code/runs/ema-6000` | Same architecture/components as full, EMA start=6000 | 1.517947 |
| `code/runs/no-cache-6000` | parallel=false, cache=false, dropout=0.1, SWA start=6000 | 1.523995 |
| `code/runs/no-dropout-6000` | parallel=false, cache=true, dropout=0, SWA start=6000 | 1.694840 |
| `code/runs/trial1/no-parallel-6000` | parallel=false, cache=true, dropout=0.1, SWA start=6000; duplicates the configuration/score of `full-6000` | 1.517091 |
| `code/runs/trial1/no-cache-6000` | parallel=true, cache=false, dropout=0.1, SWA start=6000 | 1.532419 |
| `code/runs/trial1/no-dropout-6000` | parallel=true, cache=true, dropout=0, SWA start=6000 | 1.711120 |

The `no-cache-6000` and `no-dropout-6000` groups do not have exactly the same parallel-structure settings; account for the difference in `parallel` when comparing them. Based on the available records, removing dropout causes a substantial performance regression, while the effect of cache is smaller; however, not all conditions in the existing groups are strictly paired.

### Start-Step Sweeps for Parallel-Structure and Cache/Dropout Comparisons

`code/runs/trial1/{no-parallel,no-cache,no-dropout}-{4000,5000,6000,7000}` are all 10k-step runs sweeping the SWA start step. The following lists validation BPB for the SWA checkpoints (in order: start=4000, 5000, 6000, 7000):

| Series | Averaged validation BPB |
|---|---|
| `no-parallel` | 1.523209, 1.522741, 1.522602, 1.523079 |
| `no-cache` | 1.538279, 1.538279, 1.538279, 1.538279 |
| `no-dropout` | 1.780439, 1.778403, 1.780675, 1.777959 |

`full-6000` and `trial1/no-parallel-6000` have the same configuration and validation score and should not be treated as two independent repetitions.

### EMA and SWA Start-Point Comparison

The EMA/SWA start-point sweeps for parallel structure + cache + dropout at 10k steps are in `code/runs/trial1/rope-256x8x6-cache-{ema,swa}-dropout-step10000-astart{4000,5000,6000,7000}`. Averaged validation BPB is as follows:

| Method | start=4000 | start=5000 | start=6000 | start=7000 |
|---|---:|---:|---:|---:|
| EMA | 1.524717 | 1.524628 | 1.524743 | 1.526027 |
| SWA | 1.525864 | 1.525935 | 1.525511 | 1.524564 |

## 6. Additional Notes

1. The searches over wd, SWA start step, and training steps for 320×8/320×10, as well as the 10k-step cache/dropout/parallel comparisons, component start-point sweeps, and EMA/SWA start-point sweeps, are listed using verifiable metrics from `metrics.json`.
2. `no-average-6000` contains only a test evaluation file for `full-6000/checkpoint.pt`; it is not an independent training run.
3. The checkpoint hash in the evaluation files for `final/baseline-10000` matches the checkpoint hash in the archived metrics; the SWA checkpoint hash for `final/full-10000` matches the SWA checkpoint/evaluation record for `runs/full-6000`. The latter is the currently archived candidate model; `10000` in the archive directory name is the number of training steps, while `6000` in `runs/full-6000` is the SWA start step.
4. Final model selection is based on validation bpb.