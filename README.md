# THOR
**THOR**: A Theta--Gamma Hierarchical Oscillatory Reasoning Framework for Multi-hop QA

## Abstract
Multi-hop question answering requires retrieving and integrating evidence from multiple contexts. Despite the rapid progress of current research, multi-hop reasoning remains constrained by two persistent limitations: attention decay, where the model's focus on main question degrades as the reasoning chain grows, and error accumulation, where mistakes propagate across hops and compounds into final failure. Inspired by Theta--Gamma hierarchical oscillation which decouples global planning from local retrieval, enabling efficient attention transfer between hops and a verification and repair mechanism that interrupts the accumulation of errors in the wrong paths, we present \textbf{THOR}, a brain-inspired Theta--Gamma hierarchical oscillatory reasoning framework. Extensive comparative experiments and specific validation experiments on multi-hop QA benchmarks demonstrate that THOR improves answer accuracy and robustness while mitigating limitations, showcasing its generalization across different backbones.

## Overview
Theta-Gamma is a two-rhythm QA system:
- Theta (PFC): planning, decomposition, repair, replan, final answer integration.
- Gamma (HPC + ACC): retrieval and answer extraction (HPC) plus self-checks (ACC).
- Rhythm oscillation: explicit state machine that switches between theta and gamma.

This repo includes dataset runners, evaluation utilities, and memory outputs.

## Project layout
- `Theta-Gamma/Theta/theta.py`: PFC (decompose, repair, replan, integrate answer).
- `Theta-Gamma/Gamma/gamma.py`: HPC (retrieval + answer) and ACC (checks).
- `Theta-Gamma/rhythm_oscillation.py`: rhythm state controller.
- `Theta-Gamma/pipeline.py`: dataset runner with rhythm trace logging.
- `Theta-Gamma/Memory/global_theta_memory/`: per-example global theta memory files.
- `Theta-Gamma/Memory/local_gamma_memory/`: per-call gamma memory files.
- `Scripts/test_pfc_memory.py`: lightweight PFC memory generation test.
- `Evaluation/metrics_em.py`, `Evaluation/metrics_f1.py`: evaluation utilities.

## Environment setup
Set API config (example, do not paste real keys):

```bash
export OPENROUTER_API_KEY="your_key"
export OPENROUTER_API_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_MODEL="openai/gpt-4o"
```

Optional theta via Ollama:

```bash
export THETA_OLLAMA_URL="http://127.0.0.1:11434/api/generate"
```

BGE retrieval models (HPC) require one of:
- `FlagEmbedding` (preferred)
- `sentence-transformers`

If you use BGE, you may also want:

```bash
export BGE_M3_MODEL="BAAI/bge-m3"
export BGE_RERANKER_MODEL="BAAI/bge-reranker-base"
```

## Run the pipeline
Run the full dataset (all examples in each dataset):

```bash
python Theta-Gamma/pipeline.py --datasets 2wiki,hotpotqa,musique
```

Run one example by index:

```bash
python Theta-Gamma/pipeline.py --datasets 2wiki --example-index 0
```

Tune behavior:
- `--max-steps`: max subquestions per plan (default 4)
- `--imax`: max retrieval retries before replan (default 3)
- `--limit`: cap examples per dataset

Logs:
- JSONL results in `logs/` (per dataset)
- `*_verbose.log` with rhythm state trace and module info
- per-example text logs in `logs/<dataset>_examples_txt/`

## PFC memory test
Generate global theta memory for a single question:

```bash
python Scripts/test_pfc_memory.py --question "Your question" --dataset 2wiki --example-index 0
```

## Rhythm logic
State transitions are driven by ACC check results (three 0/1 flags):
- `continue`: all checks pass, move to next subquestion
- `retrieval`: stay in gamma and replay retrieval paths
- `repair`: switch to theta to split the current subquestion
- `replan`: switch to theta to rebuild the full plan after too many retrieval retries

## Notes
- Datasets are expected under `Data/` (see `Theta-Gamma/pipeline.py`).
- The pipeline writes global theta memory at the start of each example.
- ACC checks are purely rule-based (no LLM calls).
