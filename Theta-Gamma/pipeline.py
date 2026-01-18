# pipeline.py
# End-to-end dataset runner for Theta-Gamma with rhythm oscillation and metrics.

import os
import json
import argparse
import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import requests
from requests.exceptions import HTTPError, SSLError, ConnectionError, Timeout
from dotenv import load_dotenv
from tqdm import tqdm

from rhythm_oscillation import (
    RhythmOscillation,
    STATE_CONTINUE,
    STATE_RETRIEVAL,
    STATE_REPAIR,
    STATE_REPLAN,
)


load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Evaluation.metrics_em import (  # noqa: E402
    get_gold_answers,
    get_gold_support_indices,
    answer_em,
    compute_support_metrics,
    extract_predicted_support_indices,
)
from Evaluation.metrics_f1 import answer_f1  # noqa: E402


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
    """
    os.environ.setdefault("MODEL_NAME", "openai/gpt-3.5-turbo")


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
        self.api_url = api_url or os.getenv("THETA_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
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
        max_retry = 5
        for attempt in range(1, max_retry + 1):
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
                if attempt == max_retry:
                    raise e
                print(f"[OllamaLLMClient] Timeout or error, retrying ({attempt}/{max_retry})...")


def _load_module(alias: str, path: Path):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {alias} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def load_theta_gamma_modules(base_dir: Path):
    gamma_path = base_dir / "Gamma" / "gamma.py"
    theta_path = base_dir / "Theta" / "theta.py"
    if not gamma_path.exists():
        raise FileNotFoundError(f"Missing gamma module at {gamma_path}")
    if not theta_path.exists():
        raise FileNotFoundError(f"Missing theta module at {theta_path}")

    gamma_mod = _load_module("gamma_gpt35", gamma_path)
    theta_mod = _load_module("theta_gpt35", theta_path)
    return gamma_mod, theta_mod


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
            # Accumulate metrics
            accumulated_metrics["sum_answer_em"] += float(result.get("answer_em", 0.0))
            accumulated_metrics["sum_answer_f1"] += float(result.get("answer_f1", 0.0))
            accumulated_metrics["sum_support_em"] += float(result.get("support_em", 0.0))
            accumulated_metrics["sum_support_f1"] += float(result.get("support_f1", 0.0))
            accumulated_metrics["sum_support_prec"] += float(result.get("support_precision", 0.0))
            accumulated_metrics["sum_support_rec"] += float(result.get("support_recall", 0.0))
    except Exception as e:
        print(f"Warning: Failed to load processed indices from {log_path}: {e}")

    return existing_results, processed_indices, accumulated_metrics, skipped_count


def _skip_on_llm_error(
    err: Exception,
    stage: str,
    example_index: int,
    question: str,
) -> Dict[str, Any]:
    status = None
    resp_text = ""
    if isinstance(err, HTTPError) and err.response is not None:
        status = err.response.status_code
        try:
            resp_text = err.response.text or ""
        except Exception:
            resp_text = ""

    if isinstance(err, (HTTPError, SSLError, ConnectionError, Timeout)) or status == 403 or "403" in str(err):
        return {
            "example_index": example_index,
            "question": question,
            "skipped": True,
            "skip_stage": stage,
            "skip_reason": f"{stage} failed: {err}",
            "skip_error_status": status,
            "skip_error_body": resp_text.strip(),
        }

    raise err


def _build_memory_entries(
    pfc: Any,
    question: str,
    subquestions: List[str],
    schema: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    memory = pfc._build_global_theta_memory(question, subquestions, schema)
    pfc._global_theta_memory = memory
    entries = memory.get("sub_questions", []) if isinstance(memory, dict) else []

    plan: List[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            subq = str(entry.get("sub_question", "")).strip()
            if subq:
                plan.append(subq)

    if not plan:
        plan = list(subquestions)

    return plan, entries, memory


def _extract_actual_type(gamma_result: Dict[str, Any]) -> str:
    local_mem = gamma_result.get("local_gamma_memory")
    if isinstance(local_mem, dict):
        retrieval = local_mem.get("retrieval")
        if isinstance(retrieval, dict):
            atype = str(retrieval.get("sub_answer_type", "")).strip()
            if atype:
                return atype
    return "unknown"


def _format_rhythm_event(event: Dict[str, Any]) -> str:
    state = event.get("state", "")
    rhythm = event.get("rhythm", "")
    module = event.get("module", "")
    detail = event.get("detail", "")
    step_idx = event.get("step_index")
    attempt = event.get("attempt")

    header = f"[state={state} rhythm={rhythm} module={module}]"
    tail_parts = []
    if step_idx is not None:
        tail_parts.append(f"step={step_idx}")
    if attempt is not None:
        tail_parts.append(f"attempt={attempt}")
    tail = " ".join(tail_parts)
    if tail and detail:
        return f"{header} {tail} {detail}".strip()
    if tail:
        return f"{header} {tail}".strip()
    if detail:
        return f"{header} {detail}".strip()
    return header


def run_one_example(
    example: Dict[str, Any],
    example_index: int,
    dataset_label: str,
    theta_dataset: str,
    pfc: Any,
    acc: Any,
    oscillator: RhythmOscillation,
    max_steps: int = 4,
) -> Dict[str, Any]:
    question = str(example.get("question", ""))
    if theta_dataset == "2wiki":
        ex_id = example.get("_id", f"2wiki_{example_index}")
    else:
        ex_id = example.get("id", f"{theta_dataset}_{example_index}")

    rhythm_trace: List[Dict[str, Any]] = []

    # 0) Schema + working memory
    try:
        schema = pfc._ensure_schema(question)
        wm = pfc._ensure_working_memory(question)
    except Exception as e:
        return _skip_on_llm_error(e, "schema", example_index, question)

    schema_summary = pfc._schema_summary(schema)

    # 1) Decompose
    try:
        subquestions = pfc.decompose_question(question, max_steps=max_steps)
    except Exception as e:
        return _skip_on_llm_error(e, "decompose_question", example_index, question)

    subquestions, memory_entries, global_memory = _build_memory_entries(pfc, question, subquestions, schema)
    memory_path = pfc._write_global_theta_memory(global_memory, dataset_name=dataset_label, example_index=example_index)

    rhythm_trace.append({
        "state": oscillator.state,
        "rhythm": "theta",
        "module": "PFC.decompose_question",
        "detail": f"planned_steps={len(subquestions)}",
    })

    plan_history = [
        {
            "event": "decompose",
            "subquestions": list(subquestions),
        }
    ]

    accepted_steps: List[Dict[str, Any]] = []
    gamma_attempts: List[Dict[str, Any]] = []
    executed_subquestions: List[str] = []
    gamma_call_count = 0
    gamma_success_count = 0

    step_idx = 0
    while step_idx < len(subquestions):
        planned_subq = subquestions[step_idx]

        if step_idx >= len(memory_entries):
            repaired_plan, new_entries, _ = _build_memory_entries(pfc, question, [planned_subq], schema)
            entry = new_entries[0] if new_entries else {
                "sub_question": planned_subq,
                "core_entity": "",
                "expected_answer_type": "unknown",
                "sub_answer": "",
                "completion_flag": 0,
            }
            memory_entries.append(entry)
            if repaired_plan:
                planned_subq = repaired_plan[0]
                subquestions[step_idx] = planned_subq
        else:
            entry = memory_entries[step_idx]

        core_entity = str(entry.get("core_entity", "")).strip()
        expected_type = str(entry.get("expected_answer_type", "")).strip() or "unknown"

        if step_idx > 0:
            refined_subq = pfc.refine_subquestion(
                question=question,
                planned_subquestion=planned_subq,
                previous_steps=accepted_steps,
            )
        else:
            refined_subq = planned_subq

        attempt_num = 0
        core_override: Optional[str] = None
        last_state = None

        while True:
            gamma_call_count += 1
            call_id = f"{ex_id}_step{step_idx + 1}_try{attempt_num + 1}"
            used_core = core_override or core_entity

            gamma_result = pfc.hpc.answer_subquestion(
                example=example,
                subquestion=refined_subq,
                call_id=call_id,
                core_entity=used_core,
                schema=schema,
            )

            if gamma_result.get("found"):
                gamma_success_count += 1

            actual_type = _extract_actual_type(gamma_result)
            subanswer = gamma_result.get("answer") or ""
            evidence = gamma_result.get("selected_fact_texts") or []
            reason = gamma_result.get("reasoning", "")
            checks = acc.check_subanswer(
                subquestion=refined_subq,
                core_entity=core_entity,
                expected_type=expected_type,
                subanswer=subanswer,
                actual_type=actual_type,
                evidence=evidence,
                reason=reason,
            )
            decision = oscillator.update(checks)
            last_state = decision["state"]

            gamma_attempts.append({
                "step_index": step_idx + 1,
                "attempt": attempt_num + 1,
                "planned_subquestion": planned_subq,
                "refined_subquestion": refined_subq,
                "core_entity": core_entity,
                "core_entity_used": used_core,
                "expected_answer_type": expected_type,
                "actual_answer_type": actual_type,
                "gamma_result": gamma_result,
                "acc_checks": checks,
                "acc_decision": decision,
            })

            rhythm_trace.append({
                "state": decision["state"],
                "rhythm": "gamma",
                "module": "HPC",
                "step_index": step_idx + 1,
                "attempt": attempt_num + 1,
                "detail": (
                    f"subq={refined_subq} core_entity={used_core} found={gamma_result.get('found')} "
                    f"answer={gamma_result.get('answer')} facts={gamma_result.get('selected_fact_indices')}"
                ),
            })
            rhythm_trace.append({
                "state": decision["state"],
                "rhythm": "gamma",
                "module": "ACC",
                "step_index": step_idx + 1,
                "attempt": attempt_num + 1,
                "detail": (
                    f"checks={checks} expected_type={expected_type} actual_type={actual_type} "
                    f"reason={decision.get('reason')}"
                ),
            })

            if decision["state"] == STATE_CONTINUE:
                entry["sub_answer"] = subanswer
                entry["completion_flag"] = 1
                memory_entries[step_idx] = entry
                global_memory["sub_questions"] = memory_entries
                memory_path = pfc._write_global_theta_memory(
                    global_memory,
                    dataset_name=dataset_label,
                    example_index=example_index,
                )

                accepted_steps.append({
                    "step_index": step_idx + 1,
                    "subquestion": planned_subq,
                    "refined_subquestion": refined_subq,
                    "core_entity": core_entity,
                    "expected_answer_type": expected_type,
                    "gamma_result": gamma_result,
                    "acc_checks": checks,
                    "acc_decision": decision,
                })
                executed_subquestions.append(refined_subq)
                step_idx += 1
                break

            if decision["state"] == STATE_RETRIEVAL:
                replay = gamma_result.get("retrieval_replay")
                if not replay:
                    attempts = gamma_result.get("retrieval_attempts") or []
                    if attempts:
                        facts = pfc.hpc.build_facts(example)
                        replay = pfc.hpc.path_replay(
                            subquestion=refined_subq,
                            core_entity=core_entity,
                            attempts=attempts,
                            facts=facts,
                        )
                if replay:
                    rhythm_trace.append({
                        "state": decision["state"],
                        "rhythm": "gamma",
                        "module": "HPC.path_replay",
                        "step_index": step_idx + 1,
                        "attempt": attempt_num + 1,
                        "detail": (
                            f"next_query={replay.get('next_query')} avoid_indices={replay.get('avoid_fact_indices')}"
                        ),
                    })
                    core_override = replay.get("next_query") or core_entity
                else:
                    core_override = core_entity
                attempt_num += 1
                continue

            if decision["state"] == STATE_REPAIR:
                new_subqs = pfc.repair_subquestion(
                    question=question,
                    current_subquestion=refined_subq,
                    previous_steps=accepted_steps,
                    max_steps=max_steps,
                )
                new_subqs, new_entries, _ = _build_memory_entries(pfc, question, new_subqs, schema)
                subquestions = subquestions[:step_idx] + new_subqs + subquestions[step_idx + 1:]
                memory_entries = memory_entries[:step_idx] + new_entries + memory_entries[step_idx + 1:]
                global_memory["sub_questions"] = memory_entries
                pfc._global_theta_memory = global_memory
                memory_path = pfc._write_global_theta_memory(
                    global_memory,
                    dataset_name=dataset_label,
                    example_index=example_index,
                )
                plan_history.append({
                    "event": "repair",
                    "step_index": step_idx + 1,
                    "subquestions": list(subquestions),
                })
                rhythm_trace.append({
                    "state": STATE_REPAIR,
                    "rhythm": "theta",
                    "module": "PFC.repair_subquestion",
                    "step_index": step_idx + 1,
                    "detail": f"old_subq={planned_subq} new_subqs={new_subqs}",
                })
                oscillator.reset()
                break

            if decision["state"] == STATE_REPLAN:
                new_subqs = pfc.replan_subquestions(
                    question=question,
                    current_plan=subquestions,
                    max_steps=max_steps,
                )
                new_subqs, new_entries, global_memory = _build_memory_entries(pfc, question, new_subqs, schema)
                subquestions = new_subqs
                memory_entries = new_entries
                memory_path = pfc._write_global_theta_memory(
                    global_memory,
                    dataset_name=dataset_label,
                    example_index=example_index,
                )
                plan_history.append({
                    "event": "replan",
                    "subquestions": list(subquestions),
                })
                rhythm_trace.append({
                    "state": STATE_REPLAN,
                    "rhythm": "theta",
                    "module": "PFC.replan_subquestions",
                    "detail": f"new_plan_steps={len(subquestions)}",
                })
                accepted_steps = []
                executed_subquestions = []
                step_idx = 0
                oscillator.reset()
                break

        if last_state in (STATE_REPAIR, STATE_REPLAN):
            continue

    comparator = pfc.build_symbolic_schema(question, accepted_steps)
    comparator_summary = comparator.get("summary", "") if comparator.get("is_comparative") else ""
    final = pfc.integrate_answer(question, accepted_steps, comparator_summary=comparator_summary)

    theta_answer = final.get("answer", "")
    if not isinstance(theta_answer, str):
        theta_answer = str(theta_answer)
    theta_answer = theta_answer.strip()

    predicted_answer = theta_answer

    rhythm_trace.append({
        "state": STATE_CONTINUE,
        "rhythm": "theta",
        "module": "PFC.integrate_answer",
        "detail": f"answer={theta_answer}",
    })

    trace = {
        "question_schema": schema,
        "question_schema_summary": schema_summary,
        "working_memory": wm,
        "planned_subquestions": plan_history[0]["subquestions"],
        "final_subquestions": list(subquestions),
        "plan_history": plan_history,
        "executed_subquestions": executed_subquestions,
        "global_theta_memory": global_memory,
        "global_theta_memory_path": memory_path,
        "gamma_results": accepted_steps,
        "gamma_attempts": gamma_attempts,
        "rhythm_trace": rhythm_trace,
        "theta_final": final,
        "theta_initial_answer": theta_answer,
        "gamma_call_count": gamma_call_count,
        "gamma_success_count": gamma_success_count,
        "symbolic_comparator": comparator,
    }

    llm_calls = getattr(pfc.llm, "call_log", [])

    return {
        "dataset": dataset_label,
        "example_index": example_index,
        "id": ex_id,
        "question": question,
        "theta_answer": theta_answer,
        "predicted_answer": predicted_answer,
        "theta_gamma_trace": trace,
        "llm_calls": llm_calls,
    }


def run_dataset(
    dataset_name: str,
    data: List[Dict[str, Any]],
    log_dir: str,
    theta_dataset: str,
    gamma_mod: Any,
    theta_mod: Any,
    limit: int = -1,
    theta_model_name: Optional[str] = "gpt-o3",
    theta_provider: str = "openrouter",
    theta_ollama_url: Optional[str] = None,
    example_index: Optional[int] = None,
    max_steps: int = 4,
    imax: int = 3,
) -> None:
    ensure_gpt35_default()
    os.makedirs(log_dir, exist_ok=True)

    single_run = example_index is not None and example_index >= 0
    if single_run:
        out_path = os.path.join(log_dir, f"theta_gamma_{dataset_name}_example_{example_index}.jsonl")
        verbose_log_path = os.path.join(log_dir, f"theta_gamma_{dataset_name}_example_{example_index}_verbose.log")
        per_example_dir = os.path.join(log_dir, f"{dataset_name}_example_{example_index}_txt")
    else:
        out_path = os.path.join(log_dir, f"theta_gamma_{dataset_name}.jsonl")
        verbose_log_path = os.path.join(log_dir, f"theta_gamma_{dataset_name}_verbose.log")
        per_example_dir = os.path.join(log_dir, f"{dataset_name}_examples_txt")
    os.makedirs(per_example_dir, exist_ok=True)

    if single_run:
        if example_index >= len(data):
            print(f"[{dataset_name}] example_index {example_index} out of range (size={len(data)}).")
            return
        eval_data = data
        target_indices = {int(example_index)}
    else:
        if limit is not None and limit > 0:
            eval_data = data[:limit]
        else:
            eval_data = data
        target_indices = set(range(len(eval_data)))

    total = len(target_indices)
    if total == 0:
        print(f"[{dataset_name}] No examples to run.")
        return

    # Load processed samples, existing results, and accumulated metrics
    if single_run:
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
        processed_count = 0
    else:
        _, processed_indices, accumulated_metrics, skipped_count = load_processed_indices(out_path)
        processed_indices = processed_indices & target_indices
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
    remaining_data = [(idx, eval_data[idx]) for idx in sorted(target_indices) if idx not in processed_indices]
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
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    current_processed = processed_count

    # Create shared LLMs (HPC/ACC on baseline, theta optionally separate)
    gamma_llm = gamma_mod.LLMClient()
    desired_theta_model = theta_model_name or os.getenv("THETA_MODEL_NAME")
    if theta_provider == "ollama":
        theta_llm = OllamaLLMClient(
            call_log=[],
            model_name=desired_theta_model or "deepseek-r1:14b",
            api_url=theta_ollama_url or os.getenv("THETA_OLLAMA_URL"),
        )
    else:
        if desired_theta_model and desired_theta_model != gamma_llm.model_name:
            theta_llm = gamma_mod.LLMClient(
                call_log=[],
                model_name=desired_theta_model,
                fallback_model_name=gamma_llm.model_name,
            )
        else:
            theta_llm = gamma_llm

    pfc = theta_mod.PFC(
        dataset_name=theta_dataset,
        llm_client=gamma_llm,
        theta_llm_client=theta_llm,
    )
    acc = gamma_mod.ACC(dataset_name=theta_dataset)

    with open(out_path, "a", encoding="utf-8") as fout:
        vlog = open(verbose_log_path, "a", encoding="utf-8")
        for idx, ex in pbar:
            call_log: List[Dict[str, Any]] = []
            gamma_llm.call_log = call_log
            if theta_llm is not gamma_llm:
                theta_llm.call_log = call_log

            oscillator = RhythmOscillation(imax=imax)
            result = run_one_example(
                example=ex,
                example_index=idx,
                dataset_label=dataset_name,
                theta_dataset=theta_dataset,
                pfc=pfc,
                acc=acc,
                oscillator=oscillator,
                max_steps=max_steps,
            )

            if result.get("skipped"):
                skipped_count += 1
                # Mark skipped samples in the log for traceability
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
                ]
                if resp_body:
                    vlog_lines.append(f"Response body: {resp_body}")
                vlog_lines.append("")
                vlog.write("\n".join(vlog_lines) + "\n")
                vlog.flush()

                per_path = os.path.join(per_example_dir, f"example_{idx:05d}.txt")
                per_lines = [
                    f"==== Example {idx} (SKIPPED) ====",
                    f"Dataset: {dataset_name}",
                    f"Question: {question_txt}",
                    f"Skip stage: {skip_stage}",
                    f"Skip reason: {result.get('skip_reason')}",
                    f"Skip error status: {result.get('skip_error_status')}",
                    f"Skip error body: {resp_body}",
                ]
                with open(per_path, "w", encoding="utf-8") as pf:
                    pf.write("\n".join(per_lines) + "\n")

                current_processed += 1
                if current_processed > 0:
                    answered_count = current_processed - skipped_count
                    if answered_count > 0:
                        pbar.set_postfix({
                            "ans_EM": f"{sum_answer_em/answered_count:.3f}",
                            "ans_F1": f"{sum_answer_f1/answered_count:.3f}",
                            "sup_EM": f"{sum_support_em/answered_count:.3f}",
                            "sup_F1": f"{sum_support_f1/answered_count:.3f}",
                        })
                continue

            # Compute metrics
            gold_answers = get_gold_answers(ex)
            pred_answer = result.get("predicted_answer", "")
            ans_em_val = answer_em(theta_dataset, pred_answer, gold_answers)
            ans_f1_val = answer_f1(pred_answer, gold_answers)

            gold_support = get_gold_support_indices(theta_dataset, ex)
            gamma_results = result.get("theta_gamma_trace", {}).get("gamma_results", [])
            pred_support = extract_predicted_support_indices(gamma_results)
            support_vals = compute_support_metrics(pred_support, gold_support)

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
            fout.flush()

            trace = result.get("theta_gamma_trace", {})
            rhythm_trace = trace.get("rhythm_trace", [])
            final = trace.get("theta_final", {})
            gold = result.get("gold_answers", [])
            status = "OK" if float(result.get("answer_em", 0.0)) == 1.0 else "FAIL"

            theta_model_id = getattr(theta_llm, "model_name", "unknown")
            gamma_model_id = getattr(gamma_llm, "model_name", "unknown")
            theta_model_short = theta_model_id.split("/")[-1]
            gamma_model_short = gamma_model_id.split("/")[-1]

            log_lines: List[str] = []
            log_lines.append(f"==== Example {idx} ====")
            log_lines.append(f"Question: {result.get('question')}")
            log_lines.append(
                f"[LLMs] pfc={theta_model_id} ({theta_model_short}), hpc/acc={gamma_model_id} ({gamma_model_short})"
            )
            log_lines.append("-- Rhythm trace --")
            for event in rhythm_trace:
                log_lines.append(_format_rhythm_event(event))
            log_lines.append(f"[PFC answer | llm={theta_model_short}] {result.get('theta_answer')}")
            log_lines.append(
                f"[Final answer] answer={result.get('predicted_answer')} (gold={gold}) status={status}"
            )
            log_lines.append(f"Reasoning: {final.get('reasoning', '')}")
            log_lines.append("")
            vlog.write("\n".join(log_lines))
            vlog.flush()

            per_lines: List[str] = []
            per_lines.append(f"==== Example {idx} ====")
            per_lines.append(f"Dataset: {dataset_name}")
            per_lines.append(f"Question: {result.get('question')}")
            per_lines.append(
                f"LLMs: pfc={theta_model_id} ({theta_model_short}), hpc/acc={gamma_model_id} ({gamma_model_short})"
            )
            schema_summary = trace.get("question_schema_summary", "")
            if schema_summary:
                per_lines.append("-- Question schema summary --")
                per_lines.extend(schema_summary.splitlines())

            plan_history = trace.get("plan_history") or []
            if plan_history:
                per_lines.append("-- Plan history --")
                for entry in plan_history:
                    label = entry.get("event")
                    step_ref = entry.get("step_index")
                    steps = entry.get("subquestions", [])
                    header = f"{label}"
                    if step_ref:
                        header += f" (step {step_ref})"
                    per_lines.append(header + ":")
                    for i, subq in enumerate(steps, 1):
                        per_lines.append(f"  {i}. {subq}")

            executed = trace.get("executed_subquestions") or []
            if executed:
                per_lines.append("-- Executed subquestions --")
                for i, sub in enumerate(executed, 1):
                    per_lines.append(f"  {i}. {sub}")

            per_lines.append("-- Rhythm trace --")
            for event in rhythm_trace:
                per_lines.append(_format_rhythm_event(event))

            per_lines.append("-- Metrics --")
            per_lines.append(
                f"answer_em={result.get('answer_em')} answer_f1={result.get('answer_f1')} "
                f"support_em={result.get('support_em')} support_f1={result.get('support_f1')} "
                f"support_precision={result.get('support_precision')} support_recall={result.get('support_recall')}"
            )
            per_lines.append(f"Gold answers: {result.get('gold_answers')}")
            per_lines.append(f"Predicted answer: {result.get('predicted_answer')}")
            per_lines.append(f"Gold support indices: {result.get('gold_support_indices')}")
            per_lines.append(f"Predicted support indices: {result.get('predicted_support_indices')}")

            llm_calls = result.get("llm_calls") or []
            if llm_calls:
                per_lines.append("-- LLM calls --")
                for i, call in enumerate(llm_calls, 1):
                    meta = call.get("meta") or {}
                    req = call.get("request") or {}
                    per_lines.append(
                        f"  Call {i}: rhythm={meta.get('rhythm')} kind={meta.get('kind')} "
                        f"dataset={meta.get('dataset')} type={call.get('type')}"
                    )
                    if req:
                        per_lines.append(f"    request: {req}")
                    resp_text = call.get("response_text")
                    if resp_text:
                        per_lines.append(f"    response_text: {resp_text}")
                    raw_resp = call.get("raw_response")
                    if raw_resp and not resp_text:
                        per_lines.append(f"    raw_response: {raw_resp}")

            per_path = os.path.join(per_example_dir, f"example_{idx:05d}.txt")
            with open(per_path, "w", encoding="utf-8") as pf:
                pf.write("\n".join(per_lines) + "\n")

            current_processed += 1

            if current_processed > 0:
                answered_count = current_processed - skipped_count
                if answered_count > 0:
                    pbar.set_postfix({
                        "ans_EM": f"{sum_answer_em/answered_count:.3f}",
                        "ans_F1": f"{sum_answer_f1/answered_count:.3f}",
                        "sup_EM": f"{sum_support_em/answered_count:.3f}",
                        "sup_F1": f"{sum_support_f1/answered_count:.3f}",
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


def main() -> None:
    default_data_dir = REPO_ROOT
    parser = argparse.ArgumentParser(
        description="Theta-Gamma pipeline with rhythm oscillation for 2Wiki / HotpotQA / MuSiQue"
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
        "--example-index",
        type=int,
        default=None,
        help="Run a single example index from the dataset (overrides limit).",
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
        help="Model identifier for PFC (OpenRouter style).",
    )
    parser.add_argument(
        "--theta-provider",
        type=str,
        choices=["openrouter", "ollama"],
        default="openrouter",
        help="PFC LLM backend: openrouter (default) or ollama.",
    )
    parser.add_argument(
        "--theta-ollama-url",
        type=str,
        default=os.getenv("THETA_OLLAMA_URL"),
        help="Ollama /api/generate endpoint for PFC.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Maximum number of subquestions per plan.",
    )
    parser.add_argument(
        "--imax",
        type=int,
        default=3,
        help="Maximum retrieval retries before replan.",
    )

    args = parser.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    data_dir = Path(args.data_dir).expanduser()
    base_dir = Path(__file__).resolve().parent
    gamma_mod, theta_mod = load_theta_gamma_modules(base_dir)

    for dataset_name in datasets:
        if dataset_name not in DATASET_CONFIG:
            print(f"Unknown dataset: {dataset_name}, skip.")
            continue
        cfg = DATASET_CONFIG[dataset_name]
        data_path = data_dir / cfg["path"]
        if not data_path.exists():
            alt_path = REPO_ROOT / cfg["path"]
            if alt_path.exists():
                data_path = alt_path
        theta_dataset = cfg.get("theta_dataset", dataset_name)
        if not os.path.exists(data_path):
            print(f"File not found for {dataset_name}: {data_path}, skip.")
            continue

        print(f"Loading {dataset_name} from {data_path} ...")
        data = load_dataset(str(data_path))

        limit = args.limit if args.limit and args.limit > 0 else -1
        if args.theta_provider == "ollama":
            theta_model_name = args.theta_model or os.getenv("THETA_MODEL_NAME") or "deepseek-r1:14b"
        else:
            theta_model_name = args.theta_model or os.getenv("THETA_MODEL_NAME") or "gpt-o3"

        run_dataset(
            dataset_name=dataset_name,
            data=data,
            log_dir=args.log_dir,
            theta_dataset=theta_dataset,
            gamma_mod=gamma_mod,
            theta_mod=theta_mod,
            limit=limit,
            theta_model_name=theta_model_name,
            theta_provider=args.theta_provider,
            theta_ollama_url=args.theta_ollama_url,
            example_index=args.example_index,
            max_steps=args.max_steps,
            imax=args.imax,
        )


if __name__ == "__main__":
    main()
