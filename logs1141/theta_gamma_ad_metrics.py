#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute Attention Decay metrics (QHD, QSA-Decay, CRD)
for your theta-gamma MuSiQue JSONL outputs.

假设每行是一个 JSON，对应字段结构大致为：
{
  "id": "...",
  "question": "...",
  "theta_gamma_trace": {
      "executed_subquestions": [...],
      "planned_subquestions": [...]
      ...
  },
  ...
}

Usage 示例：
    python theta_gamma_ad_metrics.py \
        --input theta_gamma_musique2hop.jsonl \
        --output-dir ./tg_2hop_metrics

依赖：
    pip install sentence-transformers pandas matplotlib
"""

import os
import json
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer


# --------------------------
# 文本工具 & token 抽取
# --------------------------

STOPWORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "a", "an", "the", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "its", "our", "their",
    "of", "in", "on", "at", "by", "for", "with", "about", "into",
    "from", "to", "up", "down", "over", "under", "after", "before",
    "and", "or", "but", "if", "because", "as", "while", "though",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing",
    "have", "has", "had", "having",
    "then", "than", "so", "such",
}

def normalize_text(text: str) -> str:
    """小写 + 保留字母数字 + 合并空格"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_content_words(text: str):
    """
    把文本转成一组“约束 token”：
    - 仅保留字母数字
    - 去停用词
    """
    norm = normalize_text(text)
    if not norm:
        return []
    tokens = norm.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# --------------------------
# 基础相似度 & hop 推断
# --------------------------

def cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    """cosine similarity"""
    denom = (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8)
    return float(np.dot(u, v) / denom)

def infer_hop_from_id(sample_id: str, default_hop: int) -> int:
    """
    从 id 前缀解析 hop 数（如 "2hop__..."），失败则用 default_hop。
    """
    m = re.match(r"(\d)hop", sample_id)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return int(default_hop)


# --------------------------
# 指标 1: QHD (Question vs History Drift)
# --------------------------

def compute_qhd(question_vec: np.ndarray, step_vecs: np.ndarray) -> float:
    """
    QHD: Question vs History Drift

    对每个 step_t (t>=2) 计算:
        d_Q = sim(s_t, q)
        d_H = sim(s_t, s_{t-1})
        delta_t = d_H - d_Q
    最后:
        AD_QHD = 平均(max(0, delta_t))
    """
    T = len(step_vecs)
    if T <= 1:
        return 0.0

    deltas = []
    for t in range(1, T):
        d_q = cosine_sim(step_vecs[t], question_vec)
        d_h = cosine_sim(step_vecs[t], step_vecs[t - 1])
        delta = d_h - d_q
        if delta > 0:
            deltas.append(delta)

    if not deltas:
        return 0.0
    return float(np.mean(deltas))


# --------------------------
# 指标 2: QSA-Decay (Question–Step Alignment Decay)
# --------------------------

def compute_qsa_decay(question_vec: np.ndarray, step_vecs: np.ndarray):
    """
    QSA-Decay:
        a_t = sim(s_t, q)
        对 t = 1..T 做线性拟合 a_t ≈ alpha * t + beta
        AD_QSA = max(0, -alpha)

    返回:
        (AD_QSA, slope_alpha)
    """
    T = len(step_vecs)
    if T == 0:
        return 0.0, 0.0

    sims = [cosine_sim(step_vecs[t], question_vec) for t in range(T)]
    t_pos = np.arange(1, T + 1, dtype=np.float32)

    if T == 1:
        # 只有一个 step，没法看衰减
        return 0.0, 0.0

    alpha, beta = np.polyfit(t_pos, np.array(sims, dtype=np.float32), 1)
    ad_qsa = max(0.0, -float(alpha))
    return ad_qsa, float(alpha)


# --------------------------
# 指标 3: CRD (Constraint Retention Decay)
# --------------------------

def compute_crd(question: str, step_texts):
    """
    CRD: Constraint Retention Decay

    1) 从 question 中抽取约束 token 集合 C(q)
    2) 对每个 step_t 计算:
           cov_t = |C ∩ tokens(step_t)| / |C|
    3) AD_CRD = max(0, cov_1 - cov_T)

    返回:
        (AD_CRD, cov_1, cov_T)
    """
    constraint_tokens = tokenize_content_words(question)
    C = set(constraint_tokens)

    if len(C) == 0 or len(step_texts) == 0:
        return 0.0, np.nan, np.nan

    covs = []
    for s in step_texts:
        tokens = set(tokenize_content_words(s))
        covs.append(len(C & tokens) / float(len(C)))

    cov_1 = covs[0]
    cov_T = covs[-1]
    ad_crd = max(0.0, cov_1 - cov_T)
    return float(ad_crd), float(cov_1), float(cov_T)


# --------------------------
# JSONL 读取
# --------------------------

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# --------------------------
# 画图工具
# --------------------------

def plot_metric_hist_by_hop(df: pd.DataFrame, metric: str, output_dir: str):
    plt.figure(figsize=(8, 5))
    hops = sorted(df["hop"].dropna().unique())
    for h in hops:
        sub = df[df["hop"] == h][metric].dropna()
        if len(sub) == 0:
            continue
        plt.hist(sub, bins=30, alpha=0.5, label=f"{int(h)}-hop")
    plt.xlabel(metric)
    plt.ylabel("Count")
    plt.title(f"{metric} distribution by hop")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{metric}_hist_by_hop.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved {metric} histogram to {out_path}")

def plot_metric_box_by_hop(df: pd.DataFrame, metric: str, output_dir: str):
    plt.figure(figsize=(6, 5))
    df.boxplot(column=metric, by="hop")
    plt.title(f"{metric} by hop")
    plt.suptitle("")
    plt.xlabel("hop")
    plt.ylabel(metric)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{metric}_box_by_hop.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved {metric} boxplot to {out_path}")

def plot_metrics(df: pd.DataFrame, output_dir: str):
    for metric in ["AD_QHD", "AD_QSA", "AD_CRD"]:
        if metric not in df:
            continue
        plot_metric_hist_by_hop(df, metric, output_dir)
        plot_metric_box_by_hop(df, metric, output_dir)


# --------------------------
# 主流程：对你的 jsonl 计算三个指标
# --------------------------

def compute_metrics_for_jsonl(input_path: str,
                              output_dir: str,
                              model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                              sample_limit: int = None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    records = []

    print(f"[INFO] Reading from {input_path}")
    for idx, ex in enumerate(load_jsonl(input_path)):
        if sample_limit is not None and idx >= sample_limit:
            break

        sid = ex.get("id", f"sample_{idx}")
        question = ex.get("question", "")

        trace = ex.get("theta_gamma_trace", {}) or {}
        # 优先用 executed_subquestions，如果为空则退回 planned_subquestions
        step_texts = trace.get("executed_subquestions") \
                      or trace.get("planned_subquestions") \
                      or []

        if not step_texts:
            continue

        T = len(step_texts)
        hop = infer_hop_from_id(sid, T)

        texts = [question] + list(step_texts)
        vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        q_vec = vecs[0]
        step_vecs = vecs[1:]

        # 三个指标
        ad_qhd = compute_qhd(q_vec, step_vecs)
        ad_qsa, slope_alpha = compute_qsa_decay(q_vec, step_vecs)
        ad_crd, cov_1, cov_T = compute_crd(question, step_texts)

        rec = {
            "id": sid,
            "hop": hop,
            "num_steps": T,
            "AD_QHD": ad_qhd,
            "AD_QSA": ad_qsa,
            "QSA_slope": slope_alpha,
            "AD_CRD": ad_crd,
            "CRD_cov_first": cov_1,
            "CRD_cov_last": cov_T,
        }
        records.append(rec)

        if (idx + 1) % 50 == 0:
            print(f"[INFO] Processed {idx + 1} examples")

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "theta_gamma_ad_metrics.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[INFO] Saved per-sample metrics to {csv_path}")

    # 汇总统计
    stats = {}
    for metric in ["AD_QHD", "AD_QSA", "AD_CRD"]:
        if metric not in df:
            continue
        stats[metric] = {
            "mean": float(df[metric].mean()),
            "std": float(df[metric].std()),
        }
        for hop, sub in df.groupby("hop"):
            stats.setdefault(metric + "_by_hop", {})[int(hop)] = {
                "mean": float(sub[metric].mean()),
                "std": float(sub[metric].std()),
                "count": int(len(sub)),
            }

    stats_path = os.path.join(output_dir, "theta_gamma_ad_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Saved aggregate stats to {stats_path}")

    # 画图
    if len(df) > 0:
        plot_metrics(df, output_dir)


# --------------------------
# CLI
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to theta-gamma MuSiQue JSONL file.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save metrics and plots.")
    parser.add_argument("--model-name", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence embedding model name.")
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Optional: only use first N samples for quick debug.")
    args = parser.parse_args()

    compute_metrics_for_jsonl(
        input_path=args.input,
        output_dir=args.output_dir,
        model_name=args.model_name,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
