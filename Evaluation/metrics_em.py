# Normalization across different datasets and then evaluation utilities for answer/support metrics.

import re
import string
from typing import Any, Dict, List


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    s = s or ""
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_gold_answers(example: Dict[str, Any]) -> List[str]:
    golds: List[str] = []
    aliases = example.get("answer_aliases")
    if isinstance(aliases, list) and aliases:
        golds.extend([str(a) for a in aliases])
    if "answer" in example:
        golds.append(str(example["answer"]))

    uniq: List[str] = []
    seen = set()
    for g in golds:
        key = normalize_answer(g)
        if key not in seen:
            uniq.append(g)
            seen.add(key)
    return uniq


def get_gold_support_indices(dataset_name: str, example: Dict[str, Any]) -> List[int]:
    """
    Normalize supporting facts into a set of fact indices:
    - 2wiki: supporting_facts: [[title, sent_id], ...], map title to context index
    - hotpotqa: supporting_facts: {"title": [...], "sent_id": [...]}
    - musique: question_decomposition[*]["paragraph_support_idx"]
    """
    if dataset_name == "2wiki":
        title_to_idx = {}
        for idx, pair in enumerate(example.get("context", [])):
            if not isinstance(pair, list) or len(pair) < 1:
                continue
            title = pair[0]
            title_to_idx[title] = idx
        indices = set()
        for title, _ in example.get("supporting_facts", []):
            if title in title_to_idx:
                indices.add(title_to_idx[title])
        return sorted(indices)

    if dataset_name == "hotpotqa":
        ctx = example.get("context", {})
        titles = ctx.get("title", [])
        title_to_idx = {t: i for i, t in enumerate(titles)}
        indices = set()
        sf = example.get("supporting_facts", {})
        if isinstance(sf, dict):
            st = sf.get("title", [])
            for t in st:
                if t in title_to_idx:
                    indices.add(title_to_idx[t])
        else:
            # Also support legacy list[[title, sent_id], ...] format
            for t, _ in sf:
                if t in title_to_idx:
                    indices.add(title_to_idx[t])
        return sorted(indices)

    if dataset_name == "musique":
        indices = set()
        for qd in example.get("question_decomposition", []):
            if "paragraph_support_idx" in qd:
                indices.add(int(qd["paragraph_support_idx"]))
        return sorted(indices)

    return []


def answer_em(dataset_name: str, pred: str, golds: List[str]) -> float:
    if not pred or not golds:
        return 0.0
    npred = normalize_answer(pred)
    ptoks = set(npred.split())
    for g in golds:
        ng = normalize_answer(g)
        gtoks = set(ng.split())
        if npred == ng:
            return 1.0
        if npred and ng and (npred in ng or ng in npred):
            return 1.0
        if ptoks and gtoks and ptoks & gtoks:
            return 1.0
    return 0.0


def compute_support_metrics(
    pred_indices: List[int],
    gold_indices: List[int],
) -> Dict[str, float]:
    pset = set(pred_indices)
    gset = set(gold_indices)

    # If both are empty -> treat as perfect
    if not pset and not gset:
        return {
            "support_em": 1.0,
            "support_precision": 1.0,
            "support_recall": 1.0,
            "support_f1": 1.0,
        }

    tp = len(pset & gset)
    precision = tp / len(pset) if pset else 0.0
    recall = tp / len(gset) if gset else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    em = 1.0 if pset == gset and bool(gset) else 0.0

    return {
        "support_em": em,
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
    }


def extract_predicted_support_indices(gamma_results: List[Dict[str, Any]]) -> List[int]:
    pred_support: List[int] = []
    seen = set()
    for gr in gamma_results:
        for idx in gr.get("gamma_result", {}).get("selected_fact_indices", []):
            if idx not in seen:
                seen.add(idx)
                pred_support.append(idx)
    return pred_support
