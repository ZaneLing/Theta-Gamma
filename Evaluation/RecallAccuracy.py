#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

from metrics_gpt35 import extract_predicted_support_indices


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


def _coerce_indices(values: Optional[List]) -> List[int]:
    if not values:
        return []
    out: List[int] = []
    for v in values:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _extract_indices(rec: Dict) -> Tuple[List[int], List[int]]:
    trace = rec.get("theta_gamma_trace") or {}

    pred = rec.get("predicted_support_indices")
    if pred is None:
        pred = trace.get("predicted_support_indices")
    gold = rec.get("gold_support_indices")
    if gold is None:
        gold = trace.get("gold_support_indices")

    pred_idx = _coerce_indices(pred)
    gold_idx = _coerce_indices(gold)

    if not pred_idx:
        gamma_results = trace.get("gamma_results") or rec.get("gamma_results") or []
        pred_idx = _coerce_indices(extract_predicted_support_indices(gamma_results))

    return pred_idx, gold_idx


def recall_at_k(pred: List[int], gold: List[int], k: int) -> Optional[float]:
    if not gold:
        return None
    topk = pred[:k]
    return len(set(topk) & set(gold)) / len(set(gold))


def build_rows(paths: List[str], k: int, skip_empty_gold: bool) -> List[Dict]:
    rows: List[Dict] = []
    for path in paths:
        for rec in _iter_jsonl(path):
            pred_idx, gold_idx = _extract_indices(rec)
            rec_val = recall_at_k(pred_idx, gold_idx, k)
            if rec_val is None and skip_empty_gold:
                continue
            rows.append(
                {
                    "dataset": rec.get("dataset"),
                    "example_index": rec.get("example_index"),
                    "id": rec.get("id"),
                    "recall_at_k": rec_val if rec_val is not None else 1.0,
                    "gold_count": len(set(gold_idx)),
                    "pred_count": len(pred_idx),
                    "pred_topk": pred_idx[:k],
                    "gold_indices": gold_idx,
                    "source_file": os.path.basename(path),
                }
            )
    return rows


def write_csv(rows: List[Dict], out_path: str) -> None:
    fieldnames = [
        "dataset",
        "example_index",
        "id",
        "recall_at_k",
        "gold_count",
        "pred_count",
        "pred_topk",
        "gold_indices",
        "source_file",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict]) -> None:
    if not rows:
        print("No rows to summarize.")
        return
    total = len(rows)
    avg = sum(r["recall_at_k"] for r in rows) / total
    print(f"Recall@k avg: {avg:.4f} (n={total})")

    by_dataset: Dict[str, List[float]] = {}
    for r in rows:
        ds = r.get("dataset") or "unknown"
        by_dataset.setdefault(ds, []).append(r["recall_at_k"])
    for ds, vals in sorted(by_dataset.items()):
        ds_avg = sum(vals) / len(vals)
        print(f"  {ds}: {ds_avg:.4f} (n={len(vals)})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute recall@k between retrieved support indices and gold."
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="*",
        help="JSONL file(s) or directories; defaults to theta_gamma*.jsonl under cwd.",
    )
    parser.add_argument("-k", type=int, default=15, help="k for recall@k.")
    parser.add_argument(
        "--skip-empty-gold",
        action="store_true",
        help="Skip examples with empty gold support.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output CSV path for per-example recall@k.",
    )
    args = parser.parse_args()

    inputs = _discover_inputs(args.input)
    rows = build_rows(inputs, args.k, args.skip_empty_gold)
    summarize(rows)
    if args.output:
        write_csv(rows, args.output)
        print(f"Saved per-example results to {args.output}")


if __name__ == "__main__":
    main()
