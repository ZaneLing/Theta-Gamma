# THOR

This package runs the Theta–Gamma pipeline with an ACC self-check on 2Wiki / HotpotQA / MuSiQue and derived hop splits.

## Components
- **gamma_gpt35.py** — Gamma agent; answers subquestions over provided facts using an embedded `gamma_answer` prompt.
- **theta_gpt35.py** — Theta agent; builds a question schema, plans/refines subquestions, runs Gamma, integrates answers (with a symbolic comparator), and calls ACC. BridgeManager enforces anchor alignment using schema entities and previous answers.
- **acc_gpt35.py** — ACC agent; lightweight judge/editor with actions KEEP / FLIP_BOOL / CHOOSE_FROM_CANDIDATES / ANSWER_UNKNOWN.
- **pipeline_gpt35.py** — CLI to run datasets end-to-end, compute EM/F1 + support metrics, and log outputs.
- **metrics_gpt35.py** — Helpers for EM/F1/support calculations.

## Running
Example (MuSiQue 2/3/4 hop with Ollama Theta and GPT-3.5 Gamma/ACC):
```bash
python GPT3.5turbo/pipeline_gpt35.py \
  --theta-provider ollama \
  --theta-ollama-url http://<ollama-host>:11434/api/generate \
  --theta-model deepseek-r1:14b \
  --log-dir logs \
  --datasets musique2hop,musique3hop,musique4hop
```
Common flags:
- `--datasets` comma list: 2wiki, hotpotqa, musique, musique2hop, musique3hop, musique4hop
- `--limit` N to cap samples per dataset
- `--theta-provider` openrouter (default) or ollama
- `--theta-model` model id for Theta (OpenRouter style or Ollama tag)
- `--theta-ollama-url` Ollama /api/generate endpoint when provider=ollama
- `--log-dir` output directory for results

## Outputs
- `log_dir/theta_gamma_<dataset>.jsonl` — per-example JSON results and metrics.
- `log_dir/theta_gamma_<dataset>_verbose.log` — human-readable trace per example.
- `log_dir/<dataset>_examples_txt/` — per-example plain-text IO summaries (question, schema, Gamma steps, Theta/ACC answers, metrics, LLM calls).

## Data
Dataset files are expected under `./Data/` (see `pipeline_gpt35.py` DATASET_CONFIG). The script will also look from repo root if launched inside `GPT3.5turbo/`.

## Environment
Gamma/ACC default to `openai/gpt-3.5-turbo` via OpenRouter/OpenAI:
- `MODEL_NAME` (default `openai/gpt-3.5-turbo`)
- `API_URL` (default `https://openrouter.ai/api/v1/chat/completions`)
- `OPENROUTER_API_KEY` or `OPENAI_API_KEY`
- Optional `FALLBACK_MODEL_NAME`
Theta can override via `--theta-model` or `THETA_MODEL_NAME`; Ollama endpoint via `--theta-ollama-url` or `THETA_OLLAMA_URL`.

## Flow per example
1) Theta builds a question schema (type, answer_form, entities/constraints).
2) Theta plans 2–4 subquestions (canonicalized to schema entities).
3) For each subquestion: refine with prior answers, call Gamma; BridgeManager checks anchor alignment.
4) Theta builds symbolic comparator hints and integrates a preliminary answer.
5) ACC self-check returns final answer (KEEP/flip/choose/unknown).
6) Metrics (answer EM/F1 and support metrics) are computed and logged with full traces.
