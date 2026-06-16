# EvoMU

Official implementation of [EvoMU: Evolutionary Machine Unlearning](https://arxiv.org/abs/2602.02139).

## What EvoMU Does

The repository is organized around a simple loop:

1. Generate candidate unlearning losses with Qwen3-4B-Thinking through vLLM.
2. Train one LoRA adapter per generated loss.
3. Merge each adapter into an evaluation snapshot.
4. Evaluate with Open-Unlearning.
5. Select top losses and generate refined children.
6. Summarize the best loss functions across runs.

The current code includes entry points for TOFU, MUSE News, and MUSE Books. WMDP-Bio corpora are also prepared locally in JSONL format for experiments using the same forget/retain data interface.

## Repository Guide

| Path | Purpose |
| --- | --- |
| `train.py` | EvoMU search loop for TOFU-style unlearning experiments. |
| `train_muse_news.py` | EvoMU search loop for MUSE News. |
| `train_muse_books.py` | EvoMU search loop for MUSE Books. |
| `generate.py` | Generates and refines TOFU loss functions. |
| `generate_muse.py` | Generates and refines MUSE News loss functions. |
| `generate_books.py` | Generates and refines MUSE Books loss functions. |
| `utils.py`, `utils_books.py` | Shared data loading, loss loading, LoRA merge, and evaluation helpers. |
| `summarize.py` | Collects top runs and writes `results/final.txt`. |
| `scripts/` | Slurm launchers for cluster runs. |
| `ou/` | Nested Open-Unlearning evaluator used by the training scripts. |
| `tofu_data/` | Local TOFU JSON/JSONL splits. |
| `muse_news_data/` | Local MUSE News forget/retain JSONL splits. |
| `muse_books_data/` | Local MUSE Books forget/retain JSONL splits. |
| `wmdp_bio_data/` | Local WMDP-Bio forget/retain JSONL corpora. |

## Setup

The code expects Python 3.11 and CUDA-capable GPUs. The provided Slurm scripts request two A100 GPUs and load CUDA 12.6.

The WMDP-Bio JSONL files are large and are stored with Git LFS. Install Git LFS before cloning or pulling the repository:

```bash
git lfs install
```

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r ou/requirements.txt
pip install -e ou
pip install peft vllm
```

You also need Hugging Face access for the model and dataset weights used by your experiment:

```bash
huggingface-cli login
```

## Data

The main JSONL splits are already present in this workspace:

```text
tofu_data/
muse_news_data/
muse_books_data/
wmdp_bio_data/
```

Each training file follows the same simple format:

```json
{"prompt": "...", "response": "..."}
```

## Running EvoMU

All training entry points create a run-specific directory under `results/` and export it as `UNLEARN_RESULTS_DIR` for the corresponding generator. Generated losses, selected parents, refined children, evaluation summaries, and final rankings are stored there.

### TOFU

Run the default TOFU experiment:

```bash
python train.py
```

### MUSE News

```bash
python train_muse_news.py
```

This uses:

```text
model: muse-bench/MUSE-news_target
data:  muse_news_data/forget_train.jsonl
data:  muse_news_data/retain_train.jsonl
eval:  Open-Unlearning experiment eval/muse/default
```

### MUSE Books

```bash
python train_muse_books.py
```

This uses:

```text
model: muse-bench/MUSE-books_target
data:  muse_books_data/forget_train.jsonl
data:  muse_books_data/retain_train.jsonl
eval:  Open-Unlearning experiment eval/muse/books
```

## Generating Losses Directly

The training scripts call the generators automatically. You can also run them manually.

Initial TOFU losses:

```bash
export UNLEARN_RESULTS_DIR=results/manual_tofu
python generate.py --mode initial
```

Refine from selected seeds:

```bash
python generate.py \
  --mode refine \
  --seeds_file results/manual_tofu/refine_seeds.json
```

MUSE News:

```bash
export UNLEARN_RESULTS_DIR=results/manual_muse_news
python generate_muse.py --mode initial
```

MUSE Books:

```bash
export UNLEARN_RESULTS_DIR=results/manual_muse_books
python generate_books.py --mode initial
```

## Summarizing Results

After one or more runs finish:

```bash
python summarize.py
```

This writes:

```text
results/final.txt
```

For TOFU-style runs, `summarize.py` ranks by the average of model utility and forget quality terms. For MUSE runs, it ranks by `muse_assessment_score`.
