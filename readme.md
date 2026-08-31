# Veto: A Training-free Framework for Fine-grained Visual Perception via Iterative Negative Filtering

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository contains the official implementation of **Veto** (Visual Elimination via Task-Oriented reasoning), a training-free agentic framework for fine-grained visual perception. Veto alternates between **Verifiable Visual Grounding (VVG)** and **Reasoning-Driven Negative Filtering (RDNF)** to progressively discard task-irrelevant regions while preserving visual evidence needed to answer the question.

## Paper

**Veto: A Training-free Framework for Fine-grained Visual Perception via Iterative Negative Filtering**

Ruyi Li*, Zhihan Yin*, Tan Yue, Jianxin Liang, Xian-Feng Han, Huishuai Zhang, and Dongyan Zhao.

*The first two authors contributed equally to this research.*

In *Proceedings of the 34th ACM International Conference on Multimedia (MM '26)*, November 10-14, 2026, Rio de Janeiro, Brazil.
DOI: [10.1145/3767308.3836245](https://doi.org/10.1145/3767308.3836245)

An arXiv version will be added when available.

## Highlights

- Training-free and compatible with off-the-shelf vision-language models.
- Uses VVG to verify grounded objects before they are accepted as evidence.
- Uses RDNF to remove regions unlikely to contain the remaining target objects.
- Evaluated on V*Bench, HRBench, MMStar, MME-RealWorld, POPE, and FREAK.

## Setup

Create a Python environment (Python 3.8 or newer) and install the runtime dependencies:

```bash
git clone https://github.com/azureecho111/Veto.git
cd Veto
pip install -r requirements.txt
```

Veto uses an OpenAI-compatible inference endpoint. Start a local vLLM server or provide any compatible endpoint, then set the optional credential only when the server requires one:

```bash
export OPENAI_API_KEY="your-api-key"
```

Never commit API keys. Copy `.env.example` only for local reference; scripts read credentials from environment variables or their command-line flags.

## Evaluation

The implementation supports Qwen2.5-VL, Qwen3-VL, InternVL3, and related OpenAI-compatible VLM endpoints. The paper uses Qwen2.5-VL-7B as the main backbone, temperature `0.4`, and at most four Veto iterations.

Example: run V*Bench after obtaining the benchmark from its official source and placing its question JSONL and images locally:

```bash
python Veto/eval_vstar.py \
  --data-path /path/to/test_questions.jsonl \
  --image-root /path/to/images \
  --api-url http://localhost:8000/v1 \
  --api-key "${OPENAI_API_KEY:-EMPTY}" \
  --model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --max-steps 4 \
  --output-path results/veto_vstar.jsonl
```

Example: evaluate HRBench using the dataset loader:

```bash
python Veto/eval_hr_bench.py \
  --split 4K \
  --api-url http://localhost:8000/v1 \
  --api-key "${OPENAI_API_KEY:-EMPTY}" \
  --model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --max-steps 4 \
  --output-path results/veto_hrbench_4k.jsonl
```

For the optional LLM-based judge modes, pass `--judge-api-key "$JUDGE_API_KEY"`. Result files and debug trajectories are intentionally ignored by Git.

## Data and Models

This repository does not redistribute datasets, images, model checkpoints, or third-party benchmark annotations. Please obtain V*Bench, HRBench, MMStar, MME-RealWorld, POPE, and FREAK from their official sources and comply with their respective licenses. Download model checkpoints from their original model repositories.

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
