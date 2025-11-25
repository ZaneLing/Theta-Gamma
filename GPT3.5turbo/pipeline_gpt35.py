# pipeline.py
# Controls the end-to-end run over 2Wiki / HotpotQA / MuSiQue and reports six metrics

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
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

DATASET_FILES = {
    "2wiki": "./Data/2wiki_500.json",
    "hotpotqa": "./Data/hotpotqa_500.json",
    "musique": "./Data/musique_500.json",
}


def ensure_gpt35_default() -> None:
    """
    If MODEL_NAME is not set, default to OpenRouter GPT-3.5 Turbo.
    Also reminds users to set OPENROUTER_API_KEY or OPENAI_API_KEY.
    """
    os.environ["MODEL_NAME"] = "openai/gpt-3.5-turbo"


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of examples")
    return data


def load_processed_indices(log_path: str) -> Tuple[List[Dict[str, Any]], Set[int], Dict[str, float]]:
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
    
    if not os.path.exists(log_path):
        return existing_results, processed_indices, accumulated_metrics
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return existing_results, processed_indices, accumulated_metrics

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
            # accumulate metrics
            accumulated_metrics["sum_answer_em"] += float(result.get("answer_em", 0.0))
            accumulated_metrics["sum_answer_f1"] += float(result.get("answer_f1", 0.0))
            accumulated_metrics["sum_support_em"] += float(result.get("support_em", 0.0))
            accumulated_metrics["sum_support_f1"] += float(result.get("support_f1", 0.0))
            accumulated_metrics["sum_support_prec"] += float(result.get("support_precision", 0.0))
            accumulated_metrics["sum_support_rec"] += float(result.get("support_recall", 0.0))
    except Exception as e:
        print(f"Warning: Failed to load processed indices from {log_path}: {e}")
    
    return existing_results, processed_indices, accumulated_metrics


def run_dataset(
    dataset_name: str,
    data: List[Dict[str, Any]],
    log_dir: str,
    limit: int = -1,
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
    _, processed_indices, accumulated_metrics = load_processed_indices(out_path)
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
            llm_client = LLMClient(call_log=call_log)
            theta = ThetaAgent(dataset_name=dataset_name, llm_client=llm_client)

            result = theta.solve_one(ex, example_index=idx)

            # Compute metrics outside Theta
            gold_answers = get_gold_answers(ex)
            pred_answer = result.get("predicted_answer", "")
            ans_em_val = answer_em(dataset_name, pred_answer, gold_answers)
            ans_f1_val = answer_f1(pred_answer, gold_answers)

            gold_support = get_gold_support_indices(dataset_name, ex)
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
            acc_verdict = acc.get("verdict")
            acc_final = acc.get("final_answer")
            acc_changed = (str(acc_final).strip() != str(theta_ans).strip()) if theta_ans is not None else False

            log_lines: List[str] = []
            log_lines.append(f"==== Example {idx} ====")
            log_lines.append(f"Question: {result.get('question')}")
            for step in gamma_steps:
                planned = step.get("subquestion", "")
                refined = step.get("refined_subquestion") or planned
                gres = step.get("gamma_result", {}) or {}
                log_lines.append(
                    f"[Theta -> Gamma] step {step.get('step_index')}: {refined} (planned: {planned})"
                )
                log_lines.append(
                    f"[Gamma -> Theta] found={gres.get('found')} answer={gres.get('answer')} "
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
                f"[Theta answer] {result.get('theta_answer')}"
            )
            # ACC self-check
            log_lines.append(
                f"[ACC] verdict={acc_verdict} final_answer={acc_final} "
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
                pbar.set_postfix({
                    "ans_EM": f"{sum_answer_em/current_processed:.3f}",
                    "ans_F1": f"{sum_answer_f1/current_processed:.3f}",
                    "sup_EM": f"{sum_support_em/current_processed:.3f}",
                    "sup_F1": f"{sum_support_f1/current_processed:.3f}"
                })

    n = float(total)
    print(f"[{dataset_name}] DONE on {total} examples.")
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
        help="Comma-separated list of datasets: 2wiki,hotpotqa,musique",
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

    args = parser.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for dataset_name in datasets:
        if dataset_name not in DATASET_FILES:
            print(f"Unknown dataset: {dataset_name}, skip.")
            continue
        data_dir = Path(args.data_dir).expanduser()
        data_path = data_dir / DATASET_FILES[dataset_name]
        if not os.path.exists(data_path):
            print(f"File not found for {dataset_name}: {data_path}, skip.")
            continue

        print(f"Loading {dataset_name} from {data_path} ...")
        data = load_dataset(data_path)

        limit = args.limit if args.limit and args.limit > 0 else -1
        run_dataset(dataset_name, data, log_dir=args.log_dir, limit=limit)


if __name__ == "__main__":
    main()
