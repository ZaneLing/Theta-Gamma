# theta.py
# Theta agent：控制主节律 + 调用 Gamma，并计算六个指标

from typing import Any, Dict, List
import re
import string
from collections import Counter

from gamma_gpt35 import LLMClient, GammaAgent, extract_json_from_text


def normalize_answer(s: str) -> str:
    """
    SQuAD 风格答案归一化
    """
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


class ThetaAgent:
    """
    Theta agent:
    - 规划子问题（theta）
    - 按需调度 gamma
    - 整合最终答案
    - 计算 answer / support 六个指标
    """

    def __init__(self, dataset_name: str, llm_client: LLMClient):
        assert dataset_name in {"2wiki", "hotpotqa", "musique"}
        self.dataset_name = dataset_name
        self.llm = llm_client
        self.gamma = GammaAgent(dataset_name=dataset_name, llm_client=llm_client)

    # ---------- 答案指标 ----------

    def answer_em(self, pred: str, golds: List[str]) -> float:
        if not pred or not golds:
            return 0.0
        npred = normalize_answer(pred)
        for g in golds:
            if npred == normalize_answer(g):
                return 1.0
        return 0.0

    def answer_f1(self, pred: str, golds: List[str]) -> float:
        if not golds:
            return 0.0
        pred_tokens = normalize_answer(pred).split()
        best = 0.0
        for g in golds:
            gold_tokens = normalize_answer(g).split()
            if not pred_tokens and not gold_tokens:
                f1 = 1.0
            elif not pred_tokens or not gold_tokens:
                f1 = 0.0
            else:
                common = Counter(pred_tokens) & Counter(gold_tokens)
                num_same = sum(common.values())
                if num_same == 0:
                    f1 = 0.0
                else:
                    prec = num_same / len(pred_tokens)
                    rec = num_same / len(gold_tokens)
                    f1 = 2 * prec * rec / (prec + rec)
            if f1 > best:
                best = f1
        return best

    # ---------- 支持集（support）指标 ----------

    def get_gold_support_indices(self, example: Dict[str, Any]) -> List[int]:
        """
        把不同数据集的 supporting facts 统一成 “fact index” 集合：
        - 2wiki: supporting_facts: [[title, sent_id], ...]，映射到 context 里 title 的索引
        - hotpotqa: supporting_facts: {"title": [...], "sent_id": [...]}
        - musique: question_decomposition[*]["paragraph_support_idx"]
        """
        if self.dataset_name == "2wiki":
            title_to_idx = {}
            for idx, pair in enumerate(example.get("context", [])):
                if not isinstance(pair, list) or len(pair) < 1:
                    continue
                title = pair[0]
                title_to_idx[title] = idx
            indices = set()
            for title, sent_id in example.get("supporting_facts", []):
                if title in title_to_idx:
                    indices.add(title_to_idx[title])
            return sorted(indices)

        if self.dataset_name == "hotpotqa":
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
                # 兼容 list[[title, sent_id], ...] 格式
                for t, sid in sf:
                    if t in title_to_idx:
                        indices.add(title_to_idx[t])
            return sorted(indices)

        if self.dataset_name == "musique":
            indices = set()
            for qd in example.get("question_decomposition", []):
                if "paragraph_support_idx" in qd:
                    indices.add(int(qd["paragraph_support_idx"]))
            return sorted(indices)

        return []

    def compute_support_metrics(
        self,
        pred_indices: List[int],
        gold_indices: List[int],
    ) -> Dict[str, float]:
        pset = set(pred_indices)
        gset = set(gold_indices)

        # 两边都空 -> 认为 perfect
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

    # ---------- gold 答案 ----------

    def get_gold_answers(self, example: Dict[str, Any]) -> List[str]:
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

    # ---------- 阶段 1：theta 拆子问题 ----------

    def plan_subquestions(self, question: str, max_steps: int = 4) -> List[str]:
        prompt = f"""
You are the THETA agent in a dual-agent multi-hop reasoning system.

Your job:
1. Read the original QUESTION.
2. Decompose it into 2 to {max_steps} ordered SUBQUESTIONS.
   - Each subquestion should be a simple fact-based question.
   - Later, the GAMMA agent will try to fetch evidence for each subquestion.

CRITICAL OUTPUT:
- Respond with a SINGLE JSON object.
- JSON keys:
  - "subquestions": array of 2 to {max_steps} strings.

Example:
{{
  "subquestions": [
    "Who directed the film 'Move (1970 film)'?",
    "What country is that director from?",
    "Who directed 'Méditerranée (1963 film)'?",
    "What country is that director from, and are they the same?"
  ]
}}

QUESTION:
{question}

Respond with JSON ONLY:
""".strip()

        raw_text = self.llm.generate(
            prompt,
            meta={
                "agent": "theta",
                "kind": "plan_subquestions",
                "dataset": self.dataset_name,
            },
            temperature=0.3,
        )

        try:
            obj = extract_json_from_text(raw_text)
            subs = obj.get("subquestions", [])
            subs = [str(x).strip() for x in subs if str(x).strip()]
            if not subs:
                subs = [question]
            return subs[:max_steps]
        except Exception:
            return [question]

    # ---------- 阶段 2：theta 整合最终答案 ----------

    def integrate_answer(
        self,
        question: str,
        gamma_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        lines = []
        for i, gr in enumerate(gamma_results, start=1):
            gres = gr.get("gamma_result", {})
            lines.append(
                f"Step {i}:\n"
                f"  subquestion: {gr.get('subquestion')}\n"
                f"  gamma_found: {gres.get('found')}\n"
                f"  gamma_answer: {gres.get('answer')}\n"
                f"  gamma_reasoning: {gres.get('reasoning')}\n"
                f"  used_facts: {gres.get('selected_fact_indices')}"
            )
        gamma_summary = "\n\n".join(lines)

        prompt = f"""
You are the THETA agent. You have received evidence from the GAMMA agent for a multi-hop question.

Your tasks:
1. Carefully read the ORIGINAL QUESTION.
2. Review the GAMMA_RESULTS (each step's subquestion, whether gamma found an answer, and gamma's answer).
3. Produce a FINAL_ANSWER: a short phrase/text that directly answers the original question.
4. Provide a short REASONING summary (1-3 sentences in English) explaining how you used the gamma results.

CRITICAL OUTPUT:
- Respond with a SINGLE JSON object.
- JSON keys:
  - "answer": string
  - "reasoning": string

Example:
{{
  "answer": "no",
  "reasoning": "One director is American and the other is French, so they are not from the same country."
}}

ORIGINAL QUESTION:
{question}

GAMMA_RESULTS:
{gamma_summary}

Respond with JSON ONLY:
""".strip()

        raw_text = self.llm.generate(
            prompt,
            meta={
                "agent": "theta",
                "kind": "integrate_answer",
                "dataset": self.dataset_name,
            },
            temperature=0.2,
        )

        try:
            obj = extract_json_from_text(raw_text)
        except Exception as e:
            obj = {
                "answer": "",
                "reasoning": f"JSON parse error: {e}",
            }

        answer = obj.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)
        answer = answer.strip()

        reasoning = obj.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)
        reasoning = reasoning.strip()

        return {
            "answer": answer,
            "reasoning": reasoning,
            "raw_json": obj,
        }

    # ---------- 总流程：解答一个样本并返回所有信息 ----------

    def solve_one(
        self,
        example: Dict[str, Any],
        example_index: int,
    ) -> Dict[str, Any]:
        question = str(example.get("question", ""))
        if self.dataset_name == "2wiki":
            ex_id = example.get("_id", f"2wiki_{example_index}")
        else:
            ex_id = example.get("id", f"{self.dataset_name}_{example_index}")

        gold_answers = self.get_gold_answers(example)

        # 1) theta: 拆子问题
        subquestions = self.plan_subquestions(question)
        gamma_results: List[Dict[str, Any]] = []

        gamma_call_count = 0
        gamma_success_count = 0
        for step_idx, subq in enumerate(subquestions, start=1):
            call_id = f"{ex_id}_step{step_idx}"
            gamma_call_count += 1
            gr = self.gamma.answer_subquestion(
                example=example,
                subquestion=subq,
                call_id=call_id,
            )
            if gr.get("found"):
                gamma_success_count += 1
            gamma_results.append(
                {
                    "step_index": step_idx,
                    "subquestion": subq,
                    "gamma_result": gr,
                }
            )

        # 2) theta: 整合最终答案
        final = self.integrate_answer(question, gamma_results)
        predicted_answer = final.get("answer", "").strip()

        # 3) 答案指标
        ans_em = self.answer_em(predicted_answer, gold_answers)
        ans_f1 = self.answer_f1(predicted_answer, gold_answers)

        # 4) 支持集指标
        gold_support = self.get_gold_support_indices(example)
        pred_support: List[int] = []
        seen = set()
        for gr in gamma_results:
            for idx in gr.get("gamma_result", {}).get("selected_fact_indices", []):
                if idx not in seen:
                    seen.add(idx)
                    pred_support.append(idx)

        support_metric_vals = self.compute_support_metrics(pred_support, gold_support)

        # 5) trace + log
        trace = {
            "subquestions": subquestions,
            "gamma_results": gamma_results,
            "theta_final": final,
            "gamma_call_count": gamma_call_count,
            "gamma_success_count": gamma_success_count,
            "predicted_support_indices": pred_support,
            "gold_support_indices": gold_support,
        }

        llm_calls = getattr(self.llm, "call_log", [])

        result = {
            "dataset": self.dataset_name,
            "example_index": example_index,
            "id": ex_id,
            "question": question,
            "gold_answers": gold_answers,
            "predicted_answer": predicted_answer,
            # 六个指标
            "answer_em": ans_em,
            "answer_f1": ans_f1,
            "support_em": support_metric_vals["support_em"],
            "support_f1": support_metric_vals["support_f1"],
            "support_precision": support_metric_vals["support_precision"],
            "support_recall": support_metric_vals["support_recall"],
            # 详细轨迹
            "theta_gamma_trace": trace,
            "llm_calls": llm_calls,
        }
        return result
