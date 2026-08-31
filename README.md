# Veto: A Training-free Framework for Fine-grained Visual Perception via Iterative Negative Filtering

[![ACM MM 2026](https://img.shields.io/badge/ACM%20MM-2026-0076A8.svg)](https://doi.org/10.1145/3767308.3836245)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg)](https://www.python.org/)

> **Veto** is a training-free, agentic framework that improves fine-grained visual perception by progressively removing image regions that are unlikely to matter for the task.

[[Paper](https://doi.org/10.1145/3767308.3836245)] [[Code](https://github.com/azureecho111/Veto)] [[Citation](#citation)]

## Overview

Vision-language models often struggle when a question depends on a small detail in a high-resolution or cluttered image. Instead of exhaustively searching every region, Veto follows a human-inspired **negative filtering** process: it verifies visible targets, identifies regions unlikely to contain the remaining targets, and iteratively simplifies the visual input before producing an answer.

Veto is training-free, uses no external vision expert, and works with OpenAI-compatible vision-language model endpoints.

### How Veto Works

1. **Target extraction.** The model identifies the visual objects needed to answer the question.
2. **Verifiable Visual Grounding (VVG).** Candidate targets are localized and then re-checked with local visual evidence before being accepted.
3. **Reasoning-Driven Negative Filtering (RDNF).** The model proposes plausible and implausible target locations, then removes the region least likely to contain the remaining targets.
4. **Answer generation.** Once all targets are grounded or the iteration budget is reached, Veto answers using the resulting visual evidence.

The two-stage VVG + RDNF design reduces the risk of accepting hallucinated targets while avoiding expensive exhaustive visual search.

## Main Results

The paper uses **Qwen2.5-VL-7B** as the primary backbone. Veto substantially improves fine-grained visual perception while maintaining or improving general VQA and hallucination-related performance.

| Method | V*Bench | HRBench-4K | HRBench-8K |
| :-- | --: | --: | --: |
| Qwen2.5-VL-7B | 76.4 | 71.4 | 67.3 |
| **Qwen2.5-VL-7B + Veto** | **90.6** | **75.4** | **71.6** |
| Improvement | **+14.2** | **+4.0** | **+4.3** |

For the main experiments, Veto uses temperature `0.4`, at most four iterations, and up to four candidate regions during negative filtering. See the paper for full comparisons, efficiency measurements, and ablation studies.

## Quick Start

### 1. Install

```bash
git clone https://github.com/azureecho111/Veto.git
cd Veto

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Veto requires Python 3.8 or newer. The evaluation scripts interact with a running OpenAI-compatible VLM endpoint; the repository does not download model checkpoints automatically.

### 2. Serve a Model

Start a local vLLM server or another compatible endpoint. For example, with Qwen2.5-VL-7B:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8000 \
  --trust-remote-code
```

If your endpoint requires authentication, set the credential in your shell. Do not commit credentials to the repository.

```bash
export OPENAI_API_KEY="your-api-key"
```

`OPENAI_API_KEY` is optional for local servers that accept `EMPTY` as a placeholder. For separate LLM-as-a-judge modes, use `JUDGE_API_KEY` or the corresponding command-line option.

### 3. Evaluate Veto

Run V*Bench after obtaining the benchmark from its official source and placing the question JSONL and images locally:

```bash
python Veto/eval_vstar.py \
  --data-path /path/to/test_questions.jsonl \
  --image-root /path/to/images \
  --api-url http://localhost:8000/v1 \
  --api-key "${OPENAI_API_KEY:-EMPTY}" \
  --model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --temperature 0.4 \
  --max-steps 4 \
  --output-path results/veto_vstar.jsonl
```

Evaluate HRBench with its Hugging Face dataset loader:

```bash
python Veto/eval_hr_bench.py \
  --split 4K \
  --api-url http://localhost:8000/v1 \
  --api-key "${OPENAI_API_KEY:-EMPTY}" \
  --model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --temperature 0.4 \
  --max-steps 4 \
  --output-path results/veto_hrbench_4k.jsonl
```

Use `--split 8K` for HRBench-8K. Add `--debug` to save per-sample trajectories and intermediate images.

## Repository Guide

| Path | Purpose |
| :-- | :-- |
| `Veto/base.py` | Main Veto implementation for Qwen2.5-VL-style endpoints. |
| `Veto/base_qwen3vl.py` | Qwen3-VL adaptation. |
| `Veto/base_internvl3.py` | InternVL3 adaptation. |
| `Veto/eval_vstar.py` | V*Bench evaluation. |
| `Veto/eval_hr_bench.py` | HRBench evaluation. |
| `Veto/eval_mmstar.py`, `Veto/eval_mme_realworld.py`, `Veto/eval_pope.py`, `Veto/eval_freak.py` | General VQA and hallucination benchmark evaluations. |
| `Veto/ablation/` | Ablation implementations and evaluation scripts. |
| `Veto/ic_examples/` | In-context examples used for target extraction. |
| `Veto/scripts/` | Experiment command templates. |

The `baseline/` directory contains comparison scripts. Generated outputs, local checkpoints, datasets, debug trajectories, IDE files, and credentials are excluded from Git by default.

## Data, Models, and Reproducibility

This repository does not redistribute datasets, images, model checkpoints, or third-party benchmark annotations. Please obtain V*Bench, HRBench, MMStar, MME-RealWorld, POPE, and FREAK from their official sources, and comply with their individual licenses and terms of use.

The paper evaluates Veto with Qwen2.5-VL, Qwen3-VL, MiMo-VL, and InternVL3 families. To reproduce the paper settings, use Qwen2.5-VL-7B, temperature `0.4`, a maximum of four Veto iterations, and the benchmark-specific evaluation scripts above.

## Paper

**Veto: A Training-free Framework for Fine-grained Visual Perception via Iterative Negative Filtering**

Ruyi Li*, Zhihan Yin*, Tan Yue, Jianxin Liang, Xian-Feng Han, Huishuai Zhang, and Dongyan Zhao.

*The first two authors contributed equally to this research.*

In *Proceedings of the 34th ACM International Conference on Multimedia (MM '26)*, November 10-14, 2026, Rio de Janeiro, Brazil.

DOI: [10.1145/3767308.3836245](https://doi.org/10.1145/3767308.3836245)

An arXiv version will be added when available.

## Citation

```bibtex
@inproceedings{li2026veto,
  title     = {Veto: A Training-free Framework for Fine-grained Visual Perception via Iterative Negative Filtering},
  author    = {Li, Ruyi and Yin, Zhihan and Yue, Tan and Liang, Jianxin and Han, Xian-Feng and Zhang, Huishuai and Zhao, Dongyan},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  doi       = {10.1145/3767308.3836245}
}
```

## License

The code is released under the [Apache License 2.0](LICENSE).

## Acknowledgements

We thank the authors and maintainers of the benchmark datasets and the open-source vision-language models used in this work.
