# acc_gpt35.py
# ACC = 前扣带皮层风格的“裁判 + 轻量编辑器”
#
# 设计原则：
# 1. ACC 不再完整重答问题，而是只在以下有限 action 里选择：
#    - KEEP: 保留 Theta 原答案
#    - FLIP_BOOL: 翻转 yes/no
#    - CHOOSE_FROM_CANDIDATES: 在候选集合中重新选一个
#    - ANSWER_UNKNOWN: 明确标记证据不足
# 2. ACC 不允许发明新实体，只能在候选集中选。
# 3. ACC 主要职责是：
#    - 发现明显冲突（同/不同国家、早/晚、生卒年比较等）
#    - 在冲突明显时，做最小修改
# 4. 所有决策和中间信息都写进 acc_result 方便后续分析。

from typing import Any, Dict, List, Optional

from gamma_gpt35 import LLMClient, extract_json_from_text


ACTION_KEEP = "KEEP"
ACTION_FLIP_BOOL = "FLIP_BOOL"
ACTION_CHOOSE = "CHOOSE_FROM_CANDIDATES"
ACTION_UNKNOWN = "ANSWER_UNKNOWN"

BOOL_YES = "yes"
BOOL_NO = "no"
BOOL_UNKNOWN = "unknown"


class ACCAgent:
    """
    ACCAgent: 前扣带皮层风格的自检模块（裁判 + 轻量编辑器）

    主要接口:
        check_answer(question, gamma_results, initial_answer) -> dict

    返回字典示例:
    {
        "action": "KEEP",
        "final_answer": "yes",
        "explanation": "...",
        "flags": {
            "entity_mismatch": False,
            "logic_inconsistency": False,
            "insufficient_evidence": False
        },
        "candidates": ["yes", "no", "unknown"],
        "raw_llm_output": "<原始LLM输出文本>",
        "raw_json": {...}  # 从LLM解析出的JSON
    }
    """

    def __init__(self, dataset_name: str, llm_client: LLMClient):
        assert dataset_name in {"2wiki", "hotpotqa", "musique"}
        self.dataset_name = dataset_name
        self.llm = llm_client

    # ------------------------------------------------------------------
    # 工具函数：格式化 Gamma 证据
    # ------------------------------------------------------------------
    def _format_gamma_evidence(self, gamma_results: List[Dict[str, Any]]) -> str:
        """
        把 theta-gamma 的 step 结果整理成一段文本，提供给 ACC 使用。
        """
        lines: List[str] = []
        for i, step in enumerate(gamma_results, start=1):
            subq = step.get("refined_subquestion") or step.get("subquestion")
            gres = step.get("gamma_result", {}) or {}
            lines.append(f"Step {i}:")
            lines.append(f"  subquestion: {subq}")
            lines.append(f"  gamma_found: {gres.get('found')}")
            lines.append(f"  gamma_answer: {gres.get('answer')}")
            lines.append(f"  gamma_reasoning: {gres.get('reasoning')}")

            fact_texts = gres.get("selected_fact_texts") or []
            if fact_texts:
                lines.append("  selected_facts:")
                for ft in fact_texts:
                    ft_str = str(ft)
                    if len(ft_str) > 400:
                        ft_str = ft_str[:400] + " ..."
                    lines.append(f"    - {ft_str}")
            lines.append("")

        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # 工具函数：判断是否是 yes/no 问题
    # ------------------------------------------------------------------
    def _looks_like_yes_no_question(self, question: str) -> bool:
        q = question.strip().lower()
        if not q.endswith("?"):
            q += "?"

        starters = (
            "is ", "are ", "was ", "were ",
            "do ", "does ", "did ",
            "has ", "have ", "had ",
            "can ", "could ", "should ", "would "
        )
        if any(q.startswith(s) for s in starters):
            return True

        # 一些典型模式
        patterns = [
            "same country", "same nationality", "same state",
            "is it true", "whether", "if it is"
        ]
        if any(p in q for p in patterns):
            return True

        return False

    # ------------------------------------------------------------------
    # 工具函数：从问题中抽取“X or Y or Z”候选
    # ------------------------------------------------------------------
    def _extract_options_from_question(self, question: str) -> List[str]:
        """
        从包含 "... A or B" 结构的问题中抽取候选实体。
        这不是针对某一题的 hack，而是一个通用的 option parser。
        """
        q = question.strip().strip("?")
        # 粗暴一点：只看最后一句里的 " or "
        segment = q
        if "," in q:
            # 截取最后一个逗号之后的部分，通常是 "... A or B"
            segment = q.split(",")[-1]

        parts = [p.strip() for p in segment.split(" or ")]
        # 过滤掉一些明显不是实体的东西
        options: List[str] = []
        for p in parts:
            if not p:
                continue
            # 去掉前导的连接词
            for bad_prefix in ("which film", "which movie", "which", "who", "what"):
                bp = bad_prefix + " "
                if p.lower().startswith(bp):
                    p = p[len(bp):].strip()
            if p:
                options.append(p)

        # 去重
        seen = set()
        uniq: List[str] = []
        for o in options:
            key = o.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(o)
        return uniq

    # ------------------------------------------------------------------
    # 工具函数：答案规范化
    # ------------------------------------------------------------------
    def _norm(self, s: str) -> str:
        return s.strip().lower()

    def _normalize_bool_answer(self, ans: str) -> str:
        a = self._norm(ans)
        if a in {"yes", "y", "true"}:
            return BOOL_YES
        if a in {"no", "n", "false"}:
            return BOOL_NO
        if a in {"unknown", "cannot be determined", "not sure"}:
            return BOOL_UNKNOWN
        return ans.strip()

    # ------------------------------------------------------------------
    # 主函数：自检 + 轻量编辑
    # ------------------------------------------------------------------
    def check_answer(
        self,
        question: str,
        gamma_results: List[Dict[str, Any]],
        initial_answer: str,
    ) -> Dict[str, Any]:
        """
        ACC 自检主函数。

        说明：
        - ACC 不重新回答问题，只在有限动作中选择：
            KEEP / FLIP_BOOL / CHOOSE_FROM_CANDIDATES / ANSWER_UNKNOWN
        - 对于 yes/no 问题，候选集就是 {"yes","no","unknown"}。
        - 对于带显式 options（A or B）的比较问题，候选集是 {A, B, ...} ∪ {initial_answer}。
        - ACC 不允许输出候选集之外的实体。
        """

        if initial_answer is None:
            initial_answer = ""
        if not isinstance(initial_answer, str):
            initial_answer = str(initial_answer)
        initial_answer = initial_answer.strip()

        # 1) 基本判断：是否 yes/no 题，是否有 options
        is_bool_q = self._looks_like_yes_no_question(question)
        options = self._extract_options_from_question(question)

        # 构造候选集
        candidates: List[str] = []
        if is_bool_q:
            # 明确 yes/no/unknown 三种
            candidates = [BOOL_YES, BOOL_NO, BOOL_UNKNOWN]
            init_norm = self._normalize_bool_answer(initial_answer)
            if init_norm in {BOOL_YES, BOOL_NO, BOOL_UNKNOWN}:
                initial_answer_normed = init_norm
            else:
                initial_answer_normed = initial_answer
        else:
            # 非 boolean：options + initial_answer
            candidates = options[:] if options else []
            if initial_answer and all(self._norm(initial_answer) != self._norm(o) for o in candidates):
                candidates.append(initial_answer)

            initial_answer_normed = initial_answer

        # 2) 格式化 Gamma 证据
        evidence_block = self._format_gamma_evidence(gamma_results)

        # 3) 构造 ACC 的 prompt
        #    强调：只能在给定 candidates 中选择 / 做有限 action
        candidate_list_str = ", ".join(f'"{c}"' for c in candidates) if candidates else ""
        bool_or_not = "YES_NO" if is_bool_q else "GENERAL"

        prompt = f"""
You are an ACC-like self-checking module in a multi-hop QA system.

Your role:
- You are NOT a full problem solver.
- You act as a JUDGE + LIGHT EDITOR over the Theta agent's INITIAL_ANSWER.
- You have a very LIMITED ACTION SPACE.

You receive:
- ORIGINAL QUESTION
- INITIAL_ANSWER from Theta
- STEPWISE EVIDENCE from Gamma
- A CANDIDATE_ANSWER_SET that you MUST respect.

Your allowed actions (ACTION):
1. "KEEP":
    - Keep INITIAL_ANSWER as the final answer.
2. "FLIP_BOOL":
    - Only for YES/NO type questions.
    - Flip "yes" to "no" or "no" to "yes".
    - If INITIAL_ANSWER is not clearly yes/no, do NOT use this.
3. "CHOOSE_FROM_CANDIDATES":
    - Choose a different answer from the candidate set.
    - You MUST NOT invent any answer outside the candidate set.
4. "ANSWER_UNKNOWN":
    - Use this only if the evidence is clearly insufficient to decide.
    - In that case, final_answer should be "unknown" (or equivalent).

Guidelines:
- Prefer ACTION="KEEP" unless you see a CLEAR and STRONG conflict between INITIAL_ANSWER and the evidence.
- For YES/NO questions, normalised answers are:
    - "{BOOL_YES}" for yes/true,
    - "{BOOL_NO}" for no/false,
    - "{BOOL_UNKNOWN}" for unknown / cannot be determined.
- For non-YES/NO questions, you must choose within the provided candidate set:
    [{candidate_list_str}]

You MUST answer ONLY the ORIGINAL QUESTION. Do NOT change its meaning.

You must output a SINGLE JSON object with fields:
{{
  "action": "KEEP" | "FLIP_BOOL" | "CHOOSE_FROM_CANDIDATES" | "ANSWER_UNKNOWN",
  "candidate_answer": "string or empty",
  "explanation": "short explanation in English",
  "flags": {{
    "entity_mismatch": true/false,
    "logic_inconsistency": true/false,
    "insufficient_evidence": true/false
  }}
}}

IMPORTANT CONSTRAINTS:
- If action="KEEP":
    - You MUST set candidate_answer to an empty string or to INITIAL_ANSWER.
- If action="FLIP_BOOL":
    - This only makes sense for YES/NO questions.
    - You should flip yes<->no based on the evidence.
- If action="CHOOSE_FROM_CANDIDATES":
    - candidate_answer MUST be exactly one of the provided candidates.
- If action="ANSWER_UNKNOWN":
    - candidate_answer SHOULD be "unknown" (or empty).
- Never invent new entities or facts not suggested by the evidence.

METADATA:
- QUESTION_TYPE = {bool_or_not}
- CANDIDATES = [{candidate_list_str}]

ORIGINAL QUESTION:
{question}

INITIAL_ANSWER:
{initial_answer}

GAMMA STEPWISE EVIDENCE:
{evidence_block}

Now respond with JSON ONLY:
""".strip()

        raw_llm_output = ""
        try:
            raw_llm_output = self.llm.generate(
                prompt,
                meta={
                    "agent": "acc",
                    "dataset": self.dataset_name,
                    "kind": "self_check_limited",
                },
                temperature=0.0,
            )
        except Exception as e:
            # 调用失败：保守选择 KEEP
            return {
                "action": ACTION_KEEP,
                "final_answer": initial_answer_normed,
                "explanation": f"ACC call error: {e}",
                "flags": {
                    "entity_mismatch": False,
                    "logic_inconsistency": False,
                    "insufficient_evidence": False,
                },
                "candidates": candidates,
                "raw_llm_output": raw_llm_output,
                "raw_json": {},
            }

        # 4) 解析 JSON
        try:
            obj = extract_json_from_text(raw_llm_output)
            if not isinstance(obj, dict):
                raise ValueError("ACC output is not a dict")
        except Exception as e:
            # 解析失败：保守 KEEP
            return {
                "action": ACTION_KEEP,
                "final_answer": initial_answer_normed,
                "explanation": f"ACC JSON parse error: {e}",
                "flags": {
                    "entity_mismatch": False,
                    "logic_inconsistency": False,
                    "insufficient_evidence": False,
                },
                "candidates": candidates,
                "raw_llm_output": raw_llm_output,
                "raw_json": {},
            }

        action = str(obj.get("action", ACTION_KEEP)).strip().upper()
        if action not in {ACTION_KEEP, ACTION_FLIP_BOOL, ACTION_CHOOSE, ACTION_UNKNOWN}:
            action = ACTION_KEEP

        candidate_answer = obj.get("candidate_answer", "")
        if candidate_answer is None:
            candidate_answer = ""
        if not isinstance(candidate_answer, str):
            candidate_answer = str(candidate_answer)
        candidate_answer = candidate_answer.strip()

        explanation = obj.get("explanation", "")
        if explanation is None:
            explanation = ""
        if not isinstance(explanation, str):
            explanation = str(explanation)
        explanation = explanation.strip()

        flags_raw = obj.get("flags", {}) or {}
        flags = {
            "entity_mismatch": bool(flags_raw.get("entity_mismatch", False)),
            "logic_inconsistency": bool(flags_raw.get("logic_inconsistency", False)),
            "insufficient_evidence": bool(flags_raw.get("insufficient_evidence", False)),
        }

        # 5) 根据 action 和 candidates 计算最终答案（加入强约束防呆）
        final_answer = initial_answer_normed

        # boolean 题再规范化一次
        if is_bool_q:
            final_answer = self._normalize_bool_answer(initial_answer_normed)
            # 同时也规范 candidates
            candidates = [self._normalize_bool_answer(c) for c in candidates]

        # safety: 如果 evidence 本身很混乱(比如 flags.logic_inconsistency=True)，
        # 可以对 action 进行降级（这里先只打 flag，不强制降级，方便后续分析）。
        # 若你希望更加保守，可以在这里强制 action=KEEP。

        if action == ACTION_KEEP:
            final_answer = initial_answer_normed

        elif action == ACTION_FLIP_BOOL and is_bool_q:
            norm_init = self._normalize_bool_answer(initial_answer_normed)
            if norm_init == BOOL_YES:
                final_answer = BOOL_NO
            elif norm_init == BOOL_NO:
                final_answer = BOOL_YES
            else:
                # 初始答案本身不是明确 yes/no，就不要乱翻
                action = ACTION_KEEP
                final_answer = initial_answer_normed

        elif action == ACTION_CHOOSE and candidates:
            # 只能在 candidates 中选择
            # 如果 LLM 没给 candidate_answer，或者不在候选集中，就退回 KEEP
            if candidate_answer:
                # 对 boolean 题做规范化匹配
                if is_bool_q:
                    cand_norm = self._normalize_bool_answer(candidate_answer)
                    candidate_answer = cand_norm

                if any(self._norm(candidate_answer) == self._norm(c) for c in candidates):
                    final_answer = candidate_answer
                else:
                    # 非法 candidate，回退 KEEP
                    action = ACTION_KEEP
                    final_answer = initial_answer_normed
            else:
                action = ACTION_KEEP
                final_answer = initial_answer_normed

        elif action == ACTION_UNKNOWN:
            # 使用统一的 unknown 标记
            final_answer = BOOL_UNKNOWN if is_bool_q else "unknown"

        # 6) 返回结果，供 pipeline / theta 写入 log
        return {
            "action": action,
            "final_answer": final_answer,
            "explanation": explanation,
            "flags": flags,
            "candidates": candidates,
            "raw_llm_output": raw_llm_output,
            "raw_json": obj,
        }
