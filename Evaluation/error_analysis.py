#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
error_analysis.py

对 theta-gamma 结果 json/jsonl 做整体评估 + 错误类型分析，并输出若干 matplotlib 图。

用法示例：
    python error_analysis.py \
        --inputs theta_gamma_2wiki.jsonl theta_gamma_hotpotqa.jsonl theta_gamma_musique.jsonl
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


# -----------------------------
# 数据加载
# -----------------------------

def load_results(path: str) -> List[Dict[str, Any]]:
    """
    同时支持：
    - JSONL: 每行一个 JSON 对象
    - JSON:   顶层是一个 list
    """
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # JSON list
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} 顶层不是 list")
            return data
        else:
            # JSONL
            data = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
            return data


# -----------------------------
# 错误分类
# -----------------------------

ERROR_TYPES = [
    "perfect",
    "answer_correct_support_wrong",
    "partial_answer",
    "retrieval_failure",
    "integration_failure",
    "other",
]


def classify_record(rec: Dict[str, Any]) -> str:
    """
    按照用户定义的规则对一条样本进行错误类型分类。
    """
    # 使用 float + 容忍度，避免 0.999999 之类问题
    em = float(rec.get("answer_em", 0.0) or 0.0)
    f1 = float(rec.get("answer_f1", 0.0) or 0.0)
    sup_em = float(rec.get("support_em", 0.0) or 0.0)

    trace = rec.get("theta_gamma_trace") or {}
    gamma_calls = int(trace.get("gamma_call_count", 0) or 0)
    gamma_success = int(trace.get("gamma_success_count", 0) or 0)

    def is_one(x: float) -> bool:
        return x >= 0.999

    def is_zero(x: float) -> bool:
        return x <= 1e-6

    # 1) perfect：答案对 + 证据也完全匹配 gold
    if is_one(em) and is_one(sup_em):
        return "perfect"

    # 2) answer_correct_support_wrong：答案对，但证据集合不完全对
    if is_one(em) and sup_em < 1.0:
        return "answer_correct_support_wrong"

    # 3) partial_answer：EM=0 但 F1>0
    if is_zero(em) and f1 > 0.0:
        return "partial_answer"

    # 4) retrieval_failure：F1=0 且 Gamma 从来没成功过（gamma_success=0 且 gamma_call>0）
    if is_zero(f1) and gamma_calls > 0 and gamma_success == 0:
        return "retrieval_failure"

    # 5) integration_failure：F1=0 但是 Gamma 至少成功过一次（有线索，但 Theta 最终答错）
    if is_zero(f1) and gamma_success > 0:
        return "integration_failure"

    # 其它情况归为 other（例如模型完全没调用 gamma 就错了）
    return "other"


# -----------------------------
# 统计与分析
# -----------------------------

def summarize_metrics(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    计算平均指标：answer_em, answer_f1, support_em, support_f1, support_precision, support_recall
    """
    def avg(name: str) -> float:
        vals = [float(d.get(name, 0.0) or 0.0) for d in data]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    return {
        "answer_em": avg("answer_em"),
        "answer_f1": avg("answer_f1"),
        "support_em": avg("support_em"),
        "support_f1": avg("support_f1"),
        "support_precision": avg("support_precision"),
        "support_recall": avg("support_recall"),
    }


def summarize_error_types(data: List[Dict[str, Any]]) -> Tuple[Counter, Dict[str, float]]:
    """
    返回：
      - Counter: 每种类型的样本数
      - pct_dict: 每种类型占比（0~1）
    """
    ctr = Counter()
    n = len(data)
    for rec in data:
        t = classify_record(rec)
        ctr[t] += 1

    pct = {}
    for t in ERROR_TYPES:
        pct[t] = ctr.get(t, 0) / n if n > 0 else 0.0
    return ctr, pct


# -----------------------------
# 可视化
# -----------------------------

def plot_metrics(datasets_stats: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """
    为每个数据集画一张指标柱状图，保存到 output_dir/metrics_<dataset>.png
    """
    os.makedirs(output_dir, exist_ok=True)
    metric_names = [
        "answer_em",
        "answer_f1",
        "support_em",
        "support_f1",
        "support_precision",
        "support_recall",
    ]

    for dataset_name, stats in datasets_stats.items():
        values = [stats[m] for m in metric_names]

        plt.figure(figsize=(8, 4))
        xs = range(len(metric_names))
        plt.bar(xs, values)
        plt.xticks(xs, metric_names, rotation=30, ha="right")
        plt.ylim(0.0, 1.0)
        plt.ylabel("score")
        plt.title(f"Metrics for {dataset_name}")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"metrics_{dataset_name}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_error_types_stacked(
    datasets_error_pct: Dict[str, Dict[str, float]],
    output_dir: str,
) -> None:
    """
    所有数据集的错误类型分布堆叠柱状图，保存到 output_dir/error_types_stacked.png
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset_names = list(datasets_error_pct.keys())
    num_ds = len(dataset_names)

    # 按固定顺序叠加（便于阅读）
    types_for_plot = [
        "perfect",
        "answer_correct_support_wrong",
        "partial_answer",
        "retrieval_failure",
        "integration_failure",
        "other",
    ]

    # 每一种错误类型在每个数据集上的百分比
    type_to_values: Dict[str, List[float]] = {}
    for et in types_for_plot:
        vals = []
        for ds in dataset_names:
            vals.append(datasets_error_pct[ds].get(et, 0.0) * 100.0)
        type_to_values[et] = vals

    xs = list(range(num_ds))
    bottom = [0.0] * num_ds

    plt.figure(figsize=(8, 5))

    for et in types_for_plot:
        vals = type_to_values[et]
        plt.bar(xs, vals, bottom=bottom, label=et)
        bottom = [bottom[i] + vals[i] for i in range(num_ds)]

    plt.xticks(xs, dataset_names, rotation=0)
    plt.ylabel("Percentage of examples (%)")
    plt.ylim(0.0, 100.0)
    plt.title("Error Type Distribution (Stacked)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "error_types_stacked.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# CSV 导出
# -----------------------------

def write_error_summary_csv(
    datasets_error_counts: Dict[str, Counter],
    datasets_error_pct: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    """
    将每个数据集的错误类型统计导出为 CSV：
    dataset,error_type,count,percent
    """
    lines = ["dataset,error_type,count,percent"]
    for ds, ctr in datasets_error_counts.items():
        n_ds = sum(ctr.values())
        for et in ERROR_TYPES:
            c = ctr.get(et, 0)
            p = datasets_error_pct[ds].get(et, 0.0)
            lines.append(f"{ds},{et},{c},{p:.6f}")
    text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


# -----------------------------
# 主流程
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="对 theta-gamma 结果 json/jsonl 做评估和错误分析"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="一个或多个结果文件路径（json / jsonl）",
    )
    parser.add_argument(
        "--outdir",
        default="error_analysis_outputs",
        help="输出图表和 CSV 的目录（默认：error_analysis_outputs）",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    datasets_metrics: Dict[str, Dict[str, float]] = {}
    datasets_error_counts: Dict[str, Counter] = {}
    datasets_error_pct: Dict[str, Dict[str, float]] = {}

    print("========== Error Analysis ==========")
    for path in args.inputs:
        data = load_results(path)
        dataset_name = os.path.splitext(os.path.basename(path))[0]

        print(f"\n--- Dataset: {dataset_name} ---")
        print(f"  #examples = {len(data)}")

        # 1) 总体指标
        metrics = summarize_metrics(data)
        datasets_metrics[dataset_name] = metrics

        print("  Metrics:")
        for k, v in metrics.items():
            print(f"    {k:18s}: {v:.4f}")

        # 2) 错误类型分布
        ctr, pct = summarize_error_types(data)
        datasets_error_counts[dataset_name] = ctr
        datasets_error_pct[dataset_name] = pct

        print("  Error types:")
        n_ds = len(data)
        for et in ERROR_TYPES:
            c = ctr.get(et, 0)
            p = pct.get(et, 0.0)
            print(f"    {et:28s}: {c:4d} ({p*100:5.1f}%)")
        # 额外提示一下 partial_answer 的平均 F1
        partial_f1_vals = [
            float(rec.get("answer_f1", 0.0) or 0.0)
            for rec in data
            if classify_record(rec) == "partial_answer"
        ]
        if partial_f1_vals:
            avg_partial_f1 = sum(partial_f1_vals) / len(partial_f1_vals)
            print(f"    partial_answer avg F1 : {avg_partial_f1:.4f}")

    # 3) 画图
    plot_metrics(datasets_metrics, args.outdir)
    plot_error_types_stacked(datasets_error_pct, args.outdir)

    # 4) 写 CSV
    csv_path = os.path.join(args.outdir, "error_summary.csv")
    write_error_summary_csv(datasets_error_counts, datasets_error_pct, csv_path)

    print(f"\n图表和 CSV 已输出到目录：{args.outdir}")
    print(f"错误统计 CSV：{csv_path}")


if __name__ == "__main__":
    main()
