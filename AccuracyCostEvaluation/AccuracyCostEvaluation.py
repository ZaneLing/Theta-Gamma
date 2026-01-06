#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple


def _to_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_usage(call: Dict) -> Tuple[int, int, int]:
    raw = call.get("raw_response") or call.get("response") or call
    prompt = completion = total = 0

    if isinstance(raw, dict):
        usage = raw.get("usage") or raw.get("token_usage") or raw.get("usage_metadata")
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion = int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            total = int(usage.get("total_tokens") or 0)
        else:
            prompt = int(raw.get("prompt_eval_count") or raw.get("prompt_tokens") or 0)
            completion = int(raw.get("eval_count") or raw.get("completion_tokens") or 0)
            total = int(raw.get("total_tokens") or 0)

    if total <= 0 and (prompt or completion):
        total = prompt + completion
    return prompt, completion, total


def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _discover_inputs(paths: Optional[List[str]]) -> List[str]:
    if paths:
        files = []
        for p in paths:
            if os.path.isdir(p):
                for root, _, names in os.walk(p):
                    for name in names:
                        if name.endswith(".jsonl"):
                            files.append(os.path.join(root, name))
            elif os.path.isfile(p):
                files.append(p)
        return files

    files = []
    for root, _, names in os.walk(os.getcwd()):
        for name in names:
            if name.startswith("theta_gamma") and name.endswith(".jsonl"):
                files.append(os.path.join(root, name))
    return files


def _record_metrics(rec: Dict) -> Tuple[Optional[float], Optional[float]]:
    em = rec.get("answer_em")
    if em is None:
        em = rec.get("ans_EM")
    f1 = rec.get("answer_f1")
    if f1 is None:
        f1 = rec.get("ans_F1")
    return _to_float(em), _to_float(f1)


def build_rows(paths: List[str]) -> List[Dict]:
    rows = []
    for path in paths:
        for rec in _iter_jsonl(path):
            prompt_sum = completion_sum = total_sum = 0
            missing = 0
            calls = rec.get("llm_calls") or []
            for call in calls:
                p, c, t = _extract_usage(call)
                if p == 0 and c == 0 and t == 0:
                    missing += 1
                prompt_sum += p
                completion_sum += c
                total_sum += t if t else (p + c)

            em, f1 = _record_metrics(rec)
            rows.append(
                {
                    "dataset": rec.get("dataset"),
                    "example_index": rec.get("example_index"),
                    "id": rec.get("id"),
                    "answer_em": em,
                    "answer_f1": f1,
                    "prompt_tokens": prompt_sum,
                    "completion_tokens": completion_sum,
                    "total_tokens": total_sum,
                    "llm_calls": len(calls),
                    "missing_token_calls": missing,
                    "source_file": os.path.basename(path),
                }
            )
    return rows


def write_csv(rows: List[Dict], out_path: str) -> None:
    if not rows:
        raise SystemExit("No rows to write. Check input paths.")
    fieldnames = [
        "dataset",
        "example_index",
        "id",
        "answer_em",
        "answer_f1",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "llm_calls",
        "missing_token_calls",
        "source_file",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-case token usage with final EM/F1."
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="*",
        help="JSONL file(s) or directories; defaults to theta_gamma*.jsonl under cwd.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="accuracy_cost_cases.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    inputs = _discover_inputs(args.input)
    rows = build_rows(inputs)
    write_csv(rows, args.output)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
