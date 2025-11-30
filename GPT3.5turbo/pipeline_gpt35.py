# pipeline.py
# Controls the end-to-end run over 2Wiki / HotpotQA / MuSiQue and reports six metrics

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import requests
from tqdm import tqdm

from gamma_gpt35 import LLMClient
from theta_gpt35 import ThetaAgent
from metrics_gpt35 import (
    get_gold_answers,
    get_gold_support_indices,
    answer_em,
    answer_f1,
    compute_support_metrics,
    extract_predicted_support_indices,
)

from dotenv import load_dotenv
load_dotenv()

DATASET_CONFIG = {
    "2wiki": {
        "path": "./Data/2wiki_500.json",
        "theta_dataset": "2wiki",
    },
    "hotpotqa": {
        "path": "./Data/hotpotqa_500.json",
        "theta_dataset": "hotpotqa",
    },
    "musique": {
        "path": "./Data/musique_500.json",
        "theta_dataset": "musique",
    },
    # Musique train splits with different hop counts (all share musique schema)
    "musique2hop": {
        "path": "./Data/musique_ans_train_2hop_samples.json",
        "theta_dataset": "musique",
    },
    "musique3hop": {
        "path": "./Data/musique_ans_train_3hop_samples.json",
        "theta_dataset": "musique",
    },
    "musique4hop": {
        "path": "./Data/musique_ans_train_4hop_samples.json",
        "theta_dataset": "musique",
    },
}


def ensure_gpt35_default() -> None:
    """
    If MODEL_NAME is not set, default to OpenRouter GPT-3.5 Turbo.
    Also reminds users to set OPENROUTER_API_KEY or OPENAI_API_KEY.
    """
    os.environ["MODEL_NAME"] = "openai/gpt-3.5-turbo"


class OllamaLLMClient:
    """
    Minimal Ollama /api/generate client (used when theta runs on an Ollama host).
    """

    def __init__(
        self,
        call_log: Optional[List[Dict[str, Any]]] = None,
        model_name: str = "deepseek-r1:14b",
        api_url: Optional[str] = None,
    ):
        self.api_url = api_url or os.getenv("THETA_OLLAMA_URL", "http://172.16.120.14:11434/api/generate")
        self.model_name = model_name or "deepseek-r1:14b"
        self.call_log: List[Dict[str, Any]] = call_log if call_log is not None else []

    def generate(
        self,
        prompt: str,
        meta: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        MAX_RETRY = 5
        for attempt in range(1, MAX_RETRY + 1):
            try:
                resp = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("response", "")
                self.call_log.append({
                    "type": "llm_call",
                    "meta": meta or {},
                    "request": payload,
                    "raw_response": data,
                    "response_text": text,
                })
                return text
            except Exception as e:
                if attempt == MAX_RETRY:
                    raise e
                print(f"[OllamaLLMClient] Timeout or error, retrying ({attempt}/{MAX_RETRY})...")


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of examples")
    return data


def load_processed_indices(log_path: str) -> Tuple[List[Dict[str, Any]], Set[int], Dict[str, float], int]:
    """
    Load processed sample indices, existing results, and accumulated metrics from a log file.
    
    Returns:
        existing_results: list of existing results (JSON array or merged from legacy JSONL)
        processed_indices: set of processed sample indices
        accumulated_metrics: accumulated metric sums
    """
    existing_results: List[Dict[str, Any]] = []
    processed_indices: Set[int] = set()
    accumulated_metrics = {
        "sum_answer_em": 0.0,
        "sum_answer_f1": 0.0,
        "sum_support_em": 0.0,
        "sum_support_f1": 0.0,
        "sum_support_prec": 0.0,
        "sum_support_rec": 0.0,
    }
    skipped_count = 0

    if not os.path.exists(log_path):
        return existing_results, processed_indices, accumulated_metrics, skipped_count

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return existing_results, processed_indices, accumulated_metrics, skipped_count

        # Support JSON array (preferred) or JSONL (legacy)
        if content.lstrip().startswith("["):
            try:
                loaded = json.loads(content)
                if not isinstance(loaded, list):
                    raise ValueError("Log JSON must be a list")
                existing_results = loaded
            except Exception as e:
                raise ValueError(f"Invalid JSON log format: {e}")
        else:
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        for result in existing_results:
            if not isinstance(result, dict):
                continue
            idx = result.get("example_index")
            if idx is None:
                continue
            processed_indices.add(int(idx))
            if result.get("skipped"):
                skipped_count += 1
                continue
            # accumulate metrics
            accumulated_metrics["sum_answer_em"] += float(result.get("answer_em", 0.0))
            accumulated_metrics["sum_answer_f1"] += float(result.get("answer_f1", 0.0))
            accumulated_metrics["sum_support_em"] += float(result.get("support_em", 0.0))
            accumulated_metrics["sum_support_f1"] += float(result.get("support_f1", 0.0))
            accumulated_metrics["sum_support_prec"] += float(result.get("support_precision", 0.0))
            accumulated_metrics["sum_support_rec"] += float(result.get("support_recall", 0.0))
    except Exception as e:
        print(f"Warning: Failed to load processed indices from {log_path}: {e}")

    return existing_results, processed_indices, accumulated_metrics, skipped_count


def run_dataset(
    dataset_name: str,
    data: List[Dict[str, Any]],
    log_dir: str,
    theta_dataset: str,
    limit: int = -1,
    theta_model_name: Optional[str] = "gpt-o3",
    theta_provider: str = "openrouter",
    theta_ollama_url: Optional[str] = None,
) -> None:
    ensure_gpt35_default()
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, f"theta_gamma_{dataset_name}.jsonl")
    verbose_log_path = os.path.join(log_dir, f"theta_gamma_{dataset_name}_verbose.log")

    if limit is not None and limit > 0:
        eval_data = data[:limit]
    else:
        eval_data = data

    total = len(eval_data)
    if total == 0:
        print(f"[{dataset_name}] No examples to run.")
        return

    # Load processed samples, existing results, and accumulated metrics
    _, processed_indices, accumulated_metrics, skipped_count = load_processed_indices(out_path)
    processed_count = len(processed_indices)
    
    if processed_count > 0:
        print(f"[{dataset_name}] Found {processed_count}/{total} processed samples, resuming from checkpoint...")
    
    # Accumulated sums for six metrics (restored from processed samples)
    sum_answer_em = accumulated_metrics["sum_answer_em"]
    sum_answer_f1 = accumulated_metrics["sum_answer_f1"]
    sum_support_em = accumulated_metrics["sum_support_em"]
    sum_support_f1 = accumulated_metrics["sum_support_f1"]
    sum_support_prec = accumulated_metrics["sum_support_prec"]
    sum_support_rec = accumulated_metrics["sum_support_rec"]

    # Filter remaining samples to process
    remaining_data = [(idx, ex) for idx, ex in enumerate(eval_data) if idx not in processed_indices]
    remaining_count = len(remaining_data)
    
    if remaining_count == 0:
        print(f"[{dataset_name}] All samples already processed. Nothing to do.")
        return

    # Progress bar over remaining samples
    pbar = tqdm(
        remaining_data,
        total=total,
        initial=processed_count,
        desc=f"[{dataset_name}]",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    current_processed = processed_count  # running count of processed samples

    # Append to JSONL to avoid rewriting the whole file
    with open(out_path, "a", encoding="utf-8") as fout:
        vlog = open(verbose_log_path, "a", encoding="utf-8")
        for idx, ex in pbar:
            call_log: List[Dict[str, Any]] = []
            # Gamma + ACC stay on baseline GPT-3.5; theta can optionally use a different model.
            gamma_llm = LLMClient(call_log=call_log)
            desired_theta_model = theta_model_name or os.getenv("THETA_MODEL_NAME")
            if theta_provider == "ollama":
                theta_llm = OllamaLLMClient(
                    call_log=call_log,
                    model_name=desired_theta_model or "deepseek-r1:14b",
                    api_url=theta_ollama_url or os.getenv("THETA_OLLAMA_URL"),
                )
            else:
                if desired_theta_model and desired_theta_model != gamma_llm.model_name:
                    # If the theta model is unavailable, auto-fallback to gamma's model.
                    theta_llm = LLMClient(
                        call_log=call_log,
                        model_name=desired_theta_model,
                        fallback_model_name=gamma_llm.model_name,
                    )
                else:
                    theta_llm = gamma_llm
            theta_model_id = getattr(theta_llm, "model_name", "unknown")
            gamma_model_id = getattr(gamma_llm, "model_name", "unknown")
            theta_model_short = theta_model_id.split("/")[-1]
            gamma_model_short = gamma_model_id.split("/")[-1]
            theta = ThetaAgent(
                dataset_name=theta_dataset,
                llm_client=gamma_llm,
                theta_llm_client=theta_llm,
            )

            result = theta.solve_one(ex, example_index=idx)

            if result.get("skipped"):
                skipped_count += 1
                # 标记样本已跳过，写入日志方便后续追溯
                result.setdefault("example_index", idx)
                result.setdefault("question", ex.get("question", ""))
                result["dataset"] = dataset_name
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                skip_stage = result.get("skip_stage", "schema")
                resp_body = (result.get("skip_error_body") or "").strip()
                question_txt = result.get("question", "")
                vlog_lines = [
                    f"==== Example {idx} ====",
                    f"[SKIPPED {skip_stage}] {result.get('skip_reason')}",
                    f"Question: {question_txt}",
                    f"LLMs: theta={theta_model_id} ({theta_model_short}), gamma/acc={gamma_model_id} ({gamma_model_short})",
                ]
                if resp_body:
                    vlog_lines.append(f"Response body: {resp_body}")
                vlog_lines.append("")  # separator
                vlog.write("\n".join(vlog_lines) + "\n")
                vlog.flush()
                current_processed += 1
                if current_processed > 0:
                    answered_count = current_processed - skipped_count
                    if answered_count > 0:
                        pbar.set_postfix({
                            "ans_EM": f"{sum_answer_em/answered_count:.3f}",
                            "ans_F1": f"{sum_answer_f1/answered_count:.3f}",
                            "sup_EM": f"{sum_support_em/answered_count:.3f}",
                            "sup_F1": f"{sum_support_f1/answered_count:.3f}"
                        })
                continue

            # Compute metrics outside Theta
            gold_answers = get_gold_answers(ex)
            pred_answer = result.get("predicted_answer", "")
            theta_answer = result.get("theta_answer", "")
            # EM is 1 if either Theta or ACC answer matches exactly per dataset rules
            ans_em_theta = answer_em(theta_dataset, theta_answer, gold_answers)
            ans_em_acc = answer_em(theta_dataset, pred_answer, gold_answers)
            ans_em_val = 1.0 if max(ans_em_theta, ans_em_acc) >= 1.0 else 0.0
            # F1 still computed on final predicted answer
            ans_f1_val = answer_f1(pred_answer, gold_answers)

            gold_support = get_gold_support_indices(theta_dataset, ex)
            gamma_results = result.get("theta_gamma_trace", {}).get("gamma_results", [])
            pred_support = extract_predicted_support_indices(gamma_results)
            support_vals = compute_support_metrics(pred_support, gold_support)

            # Attach metrics and gold info
            result.update({
                "gold_answers": gold_answers,
                "answer_em": ans_em_val,
                "answer_f1": ans_f1_val,
                "support_em": support_vals["support_em"],
                "support_f1": support_vals["support_f1"],
                "support_precision": support_vals["support_precision"],
                "support_recall": support_vals["support_recall"],
                "predicted_support_indices": pred_support,
                "gold_support_indices": gold_support,
            })

            sum_answer_em += float(result.get("answer_em", 0.0))
            sum_answer_f1 += float(result.get("answer_f1", 0.0))
            sum_support_em += float(result.get("support_em", 0.0))
            sum_support_f1 += float(result.get("support_f1", 0.0))
            sum_support_prec += float(result.get("support_precision", 0.0))
            sum_support_rec += float(result.get("support_recall", 0.0))

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()  # flush immediately for checkpoint safety

            # ---- Human-readable log: per-question view question / theta->gamma / gamma->theta / final decision ----
            trace = result.get("theta_gamma_trace", {})
            gamma_steps = trace.get("gamma_results", [])
            final = trace.get("theta_final", {})
            acc = trace.get("acc_result", {}) or {}
            comparator = trace.get("symbolic_comparator", {}) or {}
            gold = result.get("gold_answers", [])
            status = "✅" if float(result.get("answer_em", 0.0)) == 1.0 else "❌"
            theta_ans = result.get("theta_answer")
            acc_action = acc.get("action")
            acc_final = acc.get("final_answer")
            acc_changed = (str(acc_final).strip() != str(theta_ans).strip()) if theta_ans is not None else False

            log_lines: List[str] = []
            log_lines.append(f"==== Example {idx} ====")
            log_lines.append(f"Question: {result.get('question')}")
            log_lines.append(
                f"[LLMs] theta={theta_model_id} ({theta_model_short}), gamma/acc={gamma_model_id} ({gamma_model_short})"
            )
            for step in gamma_steps:
                planned = step.get("subquestion", "")
                refined = step.get("refined_subquestion") or planned
                gres = step.get("gamma_result", {}) or {}
                log_lines.append(
                    f"[Theta -> Gamma | llm={gamma_model_short}] step {step.get('step_index')}: {refined} (planned: {planned})"
                )
                log_lines.append(
                    f"[Gamma -> Theta | llm={gamma_model_short}] found={gres.get('found')} answer={gres.get('answer')} "
                    f"reason={gres.get('reasoning')} used_facts={gres.get('selected_fact_indices')}"
                )
            # Question schema / comparator hints (if present)
            if comparator:
                log_lines.append(
                    f"[Schema] comparative={comparator.get('is_comparative')} "
                    f"keywords={comparator.get('keywords')} "
                    f"summary={comparator.get('summary', '').replace(chr(10), ' / ')}"
                )
            # Theta's own integrated answer (before ACC)
            log_lines.append(
                f"[Theta answer | llm={theta_model_short}] {result.get('theta_answer')}"
            )
            # ACC self-check
            log_lines.append(
                f"[ACC | llm={gamma_model_short}] action={acc_action} final_answer={acc_final} "
                f"flags={acc.get('flags')} "
                f"revised={'yes' if acc_changed else 'no-op'}"
            )
            log_lines.append(f"[ACC] explanation: {acc.get('explanation')}")
            log_lines.append(
                f"[Theta final] answer={result.get('predicted_answer')} (gold={gold}) status={status}"
            )
            log_lines.append(f"Reasoning: {final.get('reasoning', '')}")
            log_lines.append("")  # separator
            vlog.write("\n".join(log_lines))
            vlog.flush()

            # Update processed count
            current_processed += 1
            
            # Update progress bar with running averages
            if current_processed > 0:
                answered_count = current_processed - skipped_count
                if answered_count > 0:
                    pbar.set_postfix({
                        "ans_EM": f"{sum_answer_em/answered_count:.3f}",
                        "ans_F1": f"{sum_answer_f1/answered_count:.3f}",
                        "sup_EM": f"{sum_support_em/answered_count:.3f}",
                        "sup_F1": f"{sum_support_f1/answered_count:.3f}"
                    })

    answered_total = total - skipped_count
    print(f"[{dataset_name}] DONE on {total} examples (answered {answered_total}, skipped {skipped_count}).")
    if answered_total > 0:
        n = float(answered_total)
        print(
            f"  answer_em = {sum_answer_em/n:.4f}, "
            f"answer_f1 = {sum_answer_f1/n:.4f}"
        )
        print(
            f"  support_em = {sum_support_em/n:.4f}, "
            f"support_f1 = {sum_support_f1/n:.4f}, "
            f"support_precision = {sum_support_prec/n:.4f}, "
            f"support_recall = {sum_support_rec/n:.4f}"
        )
    else:
        print("  No answered samples; all skipped during schema stage.")
    print(f"  Results saved to: {out_path}")


def main():
    default_data_dir = Path(__file__).resolve().parent.parent  # repo root
    parser = argparse.ArgumentParser(
        description="Theta-Gamma dual-agent pipeline for 2Wiki / HotpotQA / MuSiQue"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(default_data_dir),
        help="Directory containing the dataset json files",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="2wiki,hotpotqa,musique",
        help="Comma-separated list of datasets: "
             "2wiki, hotpotqa, musique, musique2hop, musique3hop, musique4hop",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of examples per dataset (-1 = all)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to store JSONL logs",
    )
    parser.add_argument(
        "--theta-model",
        type=str,
        default="gpt-o3",
        help="Model identifier for theta (OpenRouter style). "
             "Gamma/ACC stay on GPT-3.5-turbo. Example: gpt-o3 (alias for openai/o3-mini)",
    )
    parser.add_argument(
        "--theta-provider",
        type=str,
        choices=["openrouter", "ollama"],
        default="openrouter",
        help="Theta LLM backend: openrouter (default) or ollama (for local/remote hosts).",
    )
    parser.add_argument(
        "--theta-ollama-url",
        type=str,
        default=os.getenv("THETA_OLLAMA_URL"),
        help="Ollama /api/generate endpoint for theta (e.g., http://172.16.120.14:11434/api/generate).",
    )

    args = parser.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    data_dir = Path(args.data_dir).expanduser()
    repo_root = Path(__file__).resolve().parent.parent
    for dataset_name in datasets:
        if dataset_name not in DATASET_CONFIG:
            print(f"Unknown dataset: {dataset_name}, skip.")
            continue
        cfg = DATASET_CONFIG[dataset_name]
        data_path = data_dir / cfg["path"]
        if not data_path.exists():
            # Fallback: try repo root (useful when running inside GPT3.5turbo/ but data lives at repo root Data/)
            alt_path = repo_root / cfg["path"]
            if alt_path.exists():
                data_path = alt_path
        theta_dataset = cfg.get("theta_dataset", dataset_name)
        if not os.path.exists(data_path):
            print(f"File not found for {dataset_name}: {data_path}, skip.")
            continue

        print(f"Loading {dataset_name} from {data_path} ...")
        data = load_dataset(data_path)

        limit = args.limit if args.limit and args.limit > 0 else -1
        if args.theta_provider == "ollama":
            theta_model_name = args.theta_model or os.getenv("THETA_MODEL_NAME") or "deepseek-r1:14b"
        else:
            theta_model_name = args.theta_model or os.getenv("THETA_MODEL_NAME") or "gpt-o3"
        run_dataset(
            dataset_name,
            data,
            log_dir=args.log_dir,
            theta_dataset=theta_dataset,
            limit=limit,
            theta_model_name=theta_model_name,
            theta_provider=args.theta_provider,
            theta_ollama_url=args.theta_ollama_url,
        )


if __name__ == "__main__":
    main()
