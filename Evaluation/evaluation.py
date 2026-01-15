#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py
Compute EM / F1 / Frame Shift Rate (FSR) / Anchor Shift Rate (ASR) for multi-hop QA.

FSR/ASR follow THOR paper definitions:
- FSR: off-frame step proportion using a fixed LLM judge (e.g., GPT-4o, temp=0). :contentReference[oaicite:4]{index=4}
  Practical details: evaluate first h gold hops; missing predicted hops are off-frame; extra predicted hops ignored. :contentReference[oaicite:5]{index=5}
- ASR: anchor-missing hop proportion; anchor extracted from predicted sub-question, checked in retrieved evidence. :contentReference[oaicite:6]{index=6}

Usage examples:
1) One-file (combined) JSONL:
   python evaluation.py --data_path run_outputs.jsonl --out_path metrics.json --enable_fsr --enable_asr --openai_model gpt-4o

2) Separate gold/pred:
   python evaluation.py --gold_path dev_gold.jsonl --pred_path preds.jsonl --out_path metrics.json --enable_fsr --enable_asr

Notes:
- For FSR with OpenAI judge, set OPENAI_API_KEY in env.
- You can cache judge results to avoid repeated calls: --judge_cache judge_cache.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# -------------------------
# IO helpers
# -------------------------

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _is_jsonl(path: str) -> bool:
    return path.endswith(".jsonl")

def _load_any(path: str) -> Any:
    if _is_jsonl(path):
        return _read_jsonl(path)
    return _read_json(path)

# -------------------------
# EM / F1 (SQuAD-style)
# -------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    if s is None:
        return ""
    s = str(s)

    def remove_articles(text: str) -> str:
        return re.sub(_ARTICLES, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s: str) -> List[str]:
    s = normalize_answer(s)
    return s.split() if s else []

def compute_exact(a_pred: str, a_gold: str) -> int:
    return int(normalize_answer(a_pred) == normalize_answer(a_gold))

def compute_f1(a_pred: str, a_gold: str) -> float:
    pred_toks = get_tokens(a_pred)
    gold_toks = get_tokens(a_gold)
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    return max(metric_fn(prediction, gt) for gt in ground_truths) if ground_truths else 0.0

# -------------------------
# Field extraction (robust)
# -------------------------

def _first_present(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def get_id(ex: Dict[str, Any]) -> str:
    v = _first_present(ex, ["id", "_id", "qid", "question_id", "uid"])
    return str(v) if v is not None else ""

def get_question(ex: Dict[str, Any]) -> str:
    v = _first_present(ex, ["question", "query", "Q"])
    return str(v) if v is not None else ""

def get_gold_answers(ex: Dict[str, Any]) -> List[str]:
    v = _first_present(ex, ["answers", "gold_answers", "ground_truth", "gold", "answer", "gold_answer"])
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]

def get_pred_answer(ex: Dict[str, Any]) -> str:
    v = _first_present(ex, ["prediction", "pred", "pred_answer", "predicted_answer", "answer_pred"])
    return "" if v is None else str(v)

def get_gold_hops(ex: Dict[str, Any]) -> List[str]:
    v = _first_present(ex, ["gold_decomposition", "gold_steps", "gold_subquestions", "reference_decomposition", "decomposition_gold"])
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    # occasionally stored as a single string with separators
    if isinstance(v, str):
        # try split by newline/semicolon
        parts = [p.strip() for p in re.split(r"[\n;]+", v) if p.strip()]
        return parts
    return []

def get_pred_hops(ex: Dict[str, Any]) -> List[str]:
    v = _first_present(ex, ["pred_decomposition", "pred_steps", "pred_subquestions", "decomposition_pred", "subquestions"])
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        parts = [p.strip() for p in re.split(r"[\n;]+", v) if p.strip()]
        return parts
    return []

def get_hop_evidence(ex: Dict[str, Any]) -> List[List[str]]:
    """
    Returns evidence per hop: List[hop] -> List[str passages/snippets].
    Accepts multiple possible shapes.
    """
    v = _first_present(ex, ["retrieved_evidence", "hop_evidence", "evidence_by_hop", "evidence", "E_hat", "E"])
    if v is None:
        return []

    # If already List[List[str]]
    if isinstance(v, list):
        if len(v) == 0:
            return []
        if all(isinstance(x, list) for x in v):
            out: List[List[str]] = []
            for hop_list in v:
                out.append([str(p) for p in hop_list if p is not None])
            return out
        # List[str] => treat as single hop
        if all(isinstance(x, (str, int, float, dict)) for x in v):
            # if dict entries, stringify
            return [[json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x) for x in v]]
    # Dict keyed by hop idx
    if isinstance(v, dict):
        # sort by numeric key if possible
        items = []
        for k, val in v.items():
            try:
                idx = int(k)
            except Exception:
                idx = k
            items.append((idx, val))
        items.sort(key=lambda x: x[0])
        out = []
        for _, val in items:
            if isinstance(val, list):
                out.append([str(p) for p in val if p is not None])
            else:
                out.append([str(val)])
        return out

    # Fallback
    return []

def get_pred_anchors(ex: Dict[str, Any]) -> List[str]:
    v = _first_present(ex, ["pred_anchors", "anchors", "core_entities", "pred_core_entities", "ce", "ce_list"])
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        parts = [p.strip() for p in re.split(r"[;,\n]+", v) if p.strip()]
        return parts
    return []

def get_anchor_aliases(ex: Dict[str, Any]) -> Union[List[List[str]], List[str]]:
    """
    Optional aliases. Supports either:
    - List[str] global aliases
    - List[List[str]] per-hop aliases
    """
    v = _first_present(ex, ["anchor_aliases", "aliases", "alias_set"])
    if v is None:
        return []
    return v

# -------------------------
# ASR: lightweight anchor judge + mention matcher
# -------------------------

def _norm_for_match(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    # keep alnum and spaces
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = " ".join(s.split())
    return s

def _acronym_from_phrase(phrase: str) -> str:
    toks = re.split(r"\s+", _norm_for_match(phrase))
    toks = [t for t in toks if t]
    if len(toks) <= 1:
        return ""
    ac = "".join(t[0] for t in toks if t[0].isalnum())
    return ac.upper() if 2 <= len(ac) <= 10 else ""

def build_alias_set(anchor: str, extra_aliases: Optional[List[str]] = None) -> List[str]:
    aliases = set()
    anchor = anchor or ""
    if anchor.strip():
        aliases.add(anchor.strip())

    # parentheses variants: "A (B)" -> A, B
    m = re.match(r"^(.*?)\((.*?)\)\s*$", anchor.strip())
    if m:
        a = m.group(1).strip()
        b = m.group(2).strip()
        if a:
            aliases.add(a)
        if b:
            aliases.add(b)

    # acronym
    ac = _acronym_from_phrase(anchor)
    if ac:
        aliases.add(ac)

    if extra_aliases:
        for a in extra_aliases:
            if a and str(a).strip():
                aliases.add(str(a).strip())

    # normalized variants (case-insensitive handled later)
    return list(aliases)

def mention_in_evidence(anchor: str, evidence_text: str, aliases: Optional[List[str]] = None) -> bool:
    """
    Robust mention check (exact/case-insensitive/alias) as described in appendix. :contentReference[oaicite:7]{index=7}
    """
    if not anchor or not evidence_text:
        return False

    ev_norm = _norm_for_match(evidence_text)
    if not ev_norm:
        return False

    alias_set = build_alias_set(anchor, extra_aliases=aliases)

    for a in alias_set:
        a_norm = _norm_for_match(a)
        if not a_norm:
            continue
        # word-boundary-ish substring match on normalized text
        if re.search(rf"(?<![a-z0-9]){re.escape(a_norm)}(?![a-z0-9])", ev_norm):
            return True
        # fallback: plain substring (covers cases with missing spaces)
        if a_norm in ev_norm:
            return True

    return False

def extract_anchor_from_subquestion(subq: str) -> str:
    """
    Lightweight rule-based anchor extraction fallback.
    If you already log/store per-hop core entities, prefer providing pred_anchors/core_entities in input. :contentReference[oaicite:8]{index=8}
    """
    if not subq:
        return ""

    # 1) quoted spans
    for pat in [r"\"([^\"]+)\"", r"'([^']+)'", r"“([^”]+)”", r"‘([^’]+)’", r"\[([^\]]+)\]"]:
        m = re.search(pat, subq)
        if m:
            cand = m.group(1).strip()
            if len(cand) >= 2:
                return cand

    # 2) longest capitalized span (English-centric)
    caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5})\b", subq)
    if caps:
        caps.sort(key=lambda x: len(x), reverse=True)
        return caps[0].strip()

    # 3) fallback: first noun-ish token sequence
    toks = subq.strip().split()
    return toks[0].strip() if toks else ""

# -------------------------
# FSR: LLM judge with caching
# -------------------------

@dataclass
class JudgeConfig:
    provider: str = "openai"   # currently only openai implemented
    model: str = "gpt-4o"
    temperature: float = 0.0

class JudgeCache:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.data: Dict[str, int] = {}
        if path and os.path.exists(path):
            try:
                self.data = _read_json(path)
                # ensure int values
                self.data = {k: int(v) for k, v in self.data.items()}
            except Exception:
                self.data = {}

    def get(self, key: str) -> Optional[int]:
        return self.data.get(key)

    def set(self, key: str, value: int) -> None:
        self.data[key] = int(value)

    def save(self) -> None:
        if self.path:
            _write_json(self.path, self.data)

def _hash_key(parts: List[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\n---\n")
    return h.hexdigest()

class FrameAlignmentJudge:
    def __init__(self, cfg: JudgeConfig, cache: JudgeCache):
        self.cfg = cfg
        self.cache = cache
        self._client = None

    def _ensure_openai(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAI python package not found. Install with: pip install openai"
            ) from e
        self._client = OpenAI()

    def judge(self, question: str, gold_hop: str, pred_hop: str) -> int:
        """
        Return 1 if pred hop aligns with gold hop objective, else 0. :contentReference[oaicite:9]{index=9}
        Deterministic setting (temp=0) recommended. :contentReference[oaicite:10]{index=10}
        """
        key = _hash_key([question, gold_hop, pred_hop, self.cfg.provider, self.cfg.model, str(self.cfg.temperature)])
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if self.cfg.provider != "openai":
            raise ValueError(f"Unsupported judge provider: {self.cfg.provider}")

        self._ensure_openai()
        assert self._client is not None

        system = (
            "You are a strict evaluator for multi-hop QA hop decomposition alignment.\n"
            "Given an original question, a gold hop sub-question, and a predicted hop sub-question,\n"
            "output ONLY a single character: '1' if the predicted hop is semantically equivalent to the gold hop objective\n"
            "(same target entity/relation and same information need), otherwise output '0'.\n"
            "Do not output any other text."
        )
        user = (
            f"Original question:\n{question}\n\n"
            f"Gold hop sub-question:\n{gold_hop}\n\n"
            f"Predicted hop sub-question:\n{pred_hop}\n\n"
            "Return 1 or 0 only."
        )

        # Chat Completions API (works for most OpenAI models)
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=1,
        )
        content = (resp.choices[0].message.content or "").strip()
        val = 1 if content.startswith("1") else 0

        self.cache.set(key, val)
        return val

# -------------------------
# Core computation
# -------------------------

def merge_gold_pred(
    gold: Union[List[Dict[str, Any]], Dict[str, Any], None],
    pred: Union[List[Dict[str, Any]], Dict[str, Any], None],
    data: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Returns list of merged examples.
    Priority: data (already merged) OR merge gold & pred by id.
    Supports:
      - gold jsonl list of dicts
      - pred jsonl list of dicts
      - pred json dict {id: answer} or {id: {prediction:..., ...}}
    """
    if data is not None:
        return data

    gold_list: List[Dict[str, Any]] = []
    if isinstance(gold, list):
        gold_list = gold
    elif isinstance(gold, dict):
        # if gold provided as dict keyed by id
        gold_list = []
        for k, v in gold.items():
            if isinstance(v, dict):
                ex = dict(v)
                ex.setdefault("id", k)
                gold_list.append(ex)
            else:
                gold_list.append({"id": k, "answer": v})
    else:
        gold_list = []

    pred_map: Dict[str, Any] = {}
    if isinstance(pred, list):
        for ex in pred:
            pid = get_id(ex)
            if pid:
                pred_map[pid] = ex
    elif isinstance(pred, dict):
        pred_map = pred
    else:
        pred_map = {}

    merged: List[Dict[str, Any]] = []
    for g in gold_list:
        gid = get_id(g)
        out = dict(g)
        if gid in pred_map:
            p = pred_map[gid]
            if isinstance(p, dict):
                out.update(p)
            else:
                # p is answer string
                out["prediction"] = p
        merged.append(out)

    # If gold is empty but pred is list, evaluate pred-only (useful for debugging)
    if not merged and isinstance(pred, list):
        merged = pred

    return merged

def compute_em_f1(examples: List[Dict[str, Any]]) -> Dict[str, float]:
    total = 0
    em_sum = 0.0
    f1_sum = 0.0
    for ex in examples:
        golds = get_gold_answers(ex)
        pred = get_pred_answer(ex)
        if not golds:
            # skip if no gold answers
            continue
        em = metric_max_over_ground_truths(compute_exact, pred, golds)
        f1 = metric_max_over_ground_truths(compute_f1, pred, golds)
        em_sum += em
        f1_sum += f1
        total += 1
    return {
        "count_answered": total,
        "em": (em_sum / total) if total else 0.0,
        "f1": (f1_sum / total) if total else 0.0,
        "em_percent": (100.0 * em_sum / total) if total else 0.0,
        "f1_percent": (100.0 * f1_sum / total) if total else 0.0,
    }

def compute_fsr_asr(
    examples: List[Dict[str, Any]],
    enable_fsr: bool,
    enable_asr: bool,
    judge: Optional[FrameAlignmentJudge] = None,
    bucket_by_hops: bool = False,
) -> Dict[str, Any]:
    """
    Returns overall + (optional) buckets by gold hop count.

    FSR definition and aggregation: :contentReference[oaicite:11]{index=11}
    Details on horizon/missing hops: :contentReference[oaicite:12]{index=12}
    ASR definition and aggregation: :contentReference[oaicite:13]{index=13}
    """
    # Overall counters (evaluated on gold hop horizon h)
    total_steps = 0
    off_frame_steps = 0

    total_hops = 0
    anchor_missing_hops = 0

    # Buckets
    buckets: Dict[int, Dict[str, int]] = collections.defaultdict(lambda: {
        "total_steps": 0, "off_frame_steps": 0,
        "total_hops": 0, "anchor_missing_hops": 0,
        "count_examples": 0
    })

    for ex in examples:
        q = get_question(ex)
        gold_hops = get_gold_hops(ex)
        if not gold_hops:
            # Without gold decomposition, FSR (and horizon-based ASR) cannot be computed reliably.
            # We still allow ASR computed on predicted hops horizon if user really wants, but default is skip.
            if enable_asr and not enable_fsr:
                # optional fallback: evaluate ASR on predicted hop count
                pred_hops = get_pred_hops(ex)
                h = len(pred_hops)
            else:
                continue
        else:
            h = len(gold_hops)

        pred_hops = get_pred_hops(ex)
        evid_by_hop = get_hop_evidence(ex)
        pred_anchors = get_pred_anchors(ex)
        alias_obj = get_anchor_aliases(ex)

        if bucket_by_hops:
            buckets[h]["count_examples"] += 1

        # -------- FSR --------
        if enable_fsr:
            if judge is None:
                raise RuntimeError("FSR enabled but judge is None. Provide an LLM judge (e.g., OpenAI).")

            # Evaluate first h predicted hops; missing => off-frame; extra ignored. :contentReference[oaicite:14]{index=14}
            for t in range(h):
                gold_t = gold_hops[t] if t < len(gold_hops) else ""
                pred_t = pred_hops[t] if t < len(pred_hops) else ""  # missing -> empty
                if not pred_t:
                    fat = 0
                else:
                    fat = judge.judge(q, gold_t, pred_t)
                total_steps += 1
                off_frame_steps += (1 - fat)

                if bucket_by_hops:
                    buckets[h]["total_steps"] += 1
                    buckets[h]["off_frame_steps"] += (1 - fat)

        # -------- ASR --------
        if enable_asr:
            for t in range(h):
                # sub-question for anchor extraction
                subq_t = pred_hops[t] if t < len(pred_hops) else ""
                # 1) anchor from logged field if present, else rule-based extraction :contentReference[oaicite:15]{index=15}
                if t < len(pred_anchors) and pred_anchors[t]:
                    anchor = pred_anchors[t]
                else:
                    anchor = extract_anchor_from_subquestion(subq_t)

                # aliases: per-hop list or global list
                hop_aliases: List[str] = []
                if isinstance(alias_obj, list) and alias_obj:
                    if all(isinstance(x, list) for x in alias_obj):
                        # per-hop
                        if t < len(alias_obj) and isinstance(alias_obj[t], list):
                            hop_aliases = [str(a) for a in alias_obj[t] if a is not None]
                    elif all(isinstance(x, (str, int, float)) for x in alias_obj):
                        hop_aliases = [str(a) for a in alias_obj if a is not None]

                # evidence string (concat top-k)
                hop_evs: List[str] = evid_by_hop[t] if t < len(evid_by_hop) else []
                ev_text = "\n".join(str(p) for p in hop_evs) if hop_evs else ""

                ok = mention_in_evidence(anchor, ev_text, aliases=hop_aliases)
                aat = 1 if ok else 0

                total_hops += 1
                anchor_missing_hops += (1 - aat)

                if bucket_by_hops:
                    buckets[h]["total_hops"] += 1
                    buckets[h]["anchor_missing_hops"] += (1 - aat)

    out: Dict[str, Any] = {}
    if enable_fsr:
        fsr = (off_frame_steps / total_steps) if total_steps else 0.0
        out["fsr"] = fsr
        out["fsr_percent"] = 100.0 * fsr
        out["fsr_total_steps"] = total_steps
    if enable_asr:
        asr = (anchor_missing_hops / total_hops) if total_hops else 0.0
        out["asr"] = asr
        out["asr_percent"] = 100.0 * asr
        out["asr_total_hops"] = total_hops

    if bucket_by_hops:
        bucket_out: Dict[str, Any] = {}
        for h, b in sorted(buckets.items(), key=lambda x: x[0]):
            entry: Dict[str, Any] = {"count_examples": b["count_examples"]}
            if enable_fsr:
                fsr_h = (b["off_frame_steps"] / b["total_steps"]) if b["total_steps"] else 0.0
                entry.update({
                    "fsr": fsr_h,
                    "fsr_percent": 100.0 * fsr_h,
                    "fsr_total_steps": b["total_steps"],
                })
            if enable_asr:
                asr_h = (b["anchor_missing_hops"] / b["total_hops"]) if b["total_hops"] else 0.0
                entry.update({
                    "asr": asr_h,
                    "asr_percent": 100.0 * asr_h,
                    "asr_total_hops": b["total_hops"],
                })
            bucket_out[str(h)] = entry
        out["by_gold_hops"] = bucket_out

    return out

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default=None, help="JSONL with both gold and pred fields merged")
    ap.add_argument("--gold_path", type=str, default=None, help="Gold JSON/JSONL")
    ap.add_argument("--pred_path", type=str, default=None, help="Pred JSON/JSONL (json dict or jsonl records)")
    ap.add_argument("--out_path", type=str, required=True, help="Output metrics JSON")

    ap.add_argument("--enable_fsr", action="store_true", help="Compute FSR using LLM judge")
    ap.add_argument("--enable_asr", action="store_true", help="Compute ASR using anchor mention check")
    ap.add_argument("--bucket_by_hops", action="store_true", help="Report metrics bucketed by gold hop count")

    # Judge options
    ap.add_argument("--judge_provider", type=str, default="openai", choices=["openai"])
    ap.add_argument("--openai_model", type=str, default="gpt-4o")
    ap.add_argument("--judge_temperature", type=float, default=0.0)
    ap.add_argument("--judge_cache", type=str, default=None, help="Path to cache judge results (json)")

    args = ap.parse_args()

    # Load data
    data = None
    gold = None
    pred = None

    if args.data_path:
        raw = _load_any(args.data_path)
        if not isinstance(raw, list):
            raise ValueError("--data_path must be a JSONL (list of dicts)")
        data = raw
    else:
        if args.gold_path:
            gold = _load_any(args.gold_path)
        if args.pred_path:
            pred = _load_any(args.pred_path)

    examples = merge_gold_pred(gold=gold, pred=pred, data=data)

    # EM/F1
    em_f1 = compute_em_f1(examples)

    # FSR/ASR
    judge = None
    cache = JudgeCache(args.judge_cache)
    if args.enable_fsr:
        # FSR uses fixed LLM judge (e.g., GPT-4o, temp=0). :contentReference[oaicite:16]{index=16}
        cfg = JudgeConfig(provider=args.judge_provider, model=args.openai_model, temperature=args.judge_temperature)
        judge = FrameAlignmentJudge(cfg, cache)

    fsr_asr = compute_fsr_asr(
        examples=examples,
        enable_fsr=args.enable_fsr,
        enable_asr=args.enable_asr,
        judge=judge,
        bucket_by_hops=args.bucket_by_hops,
    )

    # Save cache
    cache.save()

    out = {
        "count_examples_total": len(examples),
        "em_f1": em_f1,
        "fsr_asr": fsr_asr,
        "meta": {
            "enable_fsr": args.enable_fsr,
            "enable_asr": args.enable_asr,
            "judge_provider": args.judge_provider if args.enable_fsr else None,
            "judge_model": args.openai_model if args.enable_fsr else None,
            "judge_temperature": args.judge_temperature if args.enable_fsr else None,
            "judge_cache": args.judge_cache if args.enable_fsr else None,
        }
    }

    _write_json(args.out_path, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
