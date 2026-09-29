# MP1 — Small Language Model Challenge

This repository contains the implementation, experiments and final checkpoint for the MP1 language-model assignment.

The runnable project is in [`code/`](code/). The assignment rules and deadlines are in [`process/GUIDE.md`](process/GUIDE.md). The exploratory experimental records are in [`process/RECORD_EN.md`](process/RECORD_EN.md).

## Results

- **Selected checkpoint:** [`code/final/full-10000/swa-checkpoint.pt`](code/final/full-10000/swa-checkpoint.pt), averaged every step from step 6,000 through 10,000.
- **Changes from baseline:** width 256 / 8 heads / 6 blocks (baseline: width 128 / 4 heads / 4 blocks); RoPE replaces learned absolute position embeddings; adds a causal neural cache; dropout=0.1.

| Metric | Final model | Baseline |
|---|---:|---:|
| Parameters | 5,263,362 | 1,088,256 |
| Training steps | 10,000 | 10,000 |
| Processed training targets | 81,920,000 | 81,920,000 |
| Validation BPB | 1.517091 (SWA) | 1.689167 |
| Test BPB | 1.540968 | 1.712524 |

Both runs processed the same number of training targets (`10,000 × 32 × 256`). Validation and test scores use the fixed evaluation protocol and the submitted SWA checkpoint for the final model.

- **CPU time ≤5× baseline:** 44.34 s vs. 10.76 s = **4.12×** (scorer-reported CPU evaluation time).
- **Peak RAM ≤4 GiB:** **2.085 GiB** (2,186,176 KiB peak RSS).
- **Inference assets ≤64 MiB uncompressed:** **21M** as reported by the asset-size measurement.

Measurements: [final metrics](code/final/full-10000/metrics.json), [baseline metrics](code/final/baseline-10000/metrics.json), [final test evaluation](code/final/full-10000/test_cpu_fp32.json), [baseline test evaluation](code/final/baseline-10000/test_cpu_fp32.json), [peak RAM](code/final/full-10000/memory.txt), and [inference asset size](code/final/full-10000/inference-assets-size.txt).

## 1. Installation
All commands run from `code/`.

Use **Python 3.12**. From the extracted package directory:

```bash
cd code
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install PyTorch for **one** device:

```bash
# Linux/Windows CPU: recommended; no GPU needed
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a compatible driver, use this command **instead**:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

For macOS, install `torch==2.7.1` from the default PyPI index and run on CPU. After installing PyTorch, install the remaining dependencies and check the model:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Linux CPU commands were verified with Python 3.12 and PyTorch 2.7.1+cpu. Windows/macOS timings have not been measured.

## 2. Evaluate the submitted checkpoint
```bash
cd code && source .venv/bin/activate
python evaluate.py --checkpoint final/full-10000/swa-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

## 2. (Optional) Reproduce the training and evaluation

```bash
cd code && source .venv/bin/activate
python train.py \
  --implementation student \
  --config configs/architecture-256x8x6-rope-cache.json \
  --run-dir runs/my_model \
  --threads 4 \
  --seed 17 \
  --steps 10000 \
  --dropout 0.1 \
  --batch-size 32 \
  --averaging-method swa \
  --average-start-step 6000 \
  --average-every 1
```

Add `--device cuda` for GPU training runs.

Evaluate by running the following command: 

```bash
cd code && source .venv/bin/activate
python evaluate.py \
  --checkpoint runs/my_model/swa-checkpoint.pt \
  --device cpu \
  --precision fp32 \
  --split test \
  --output peer-test-reproduced.json
```

## 3. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in `data/manifest.json`. These dataset notices do not assign a new license to the surrounding classroom code.

## AI Use Disclosure

GitHub Copilot was used to assist with parts of the model code implementation and development.