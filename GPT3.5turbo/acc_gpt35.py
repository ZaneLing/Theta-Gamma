# acc_gpt35.py
# ACC = Anterior Cingulate Cortex style “judge + lightweight editor”
#
# Design principles:
# 1. ACC never fully re-answers the question; it only chooses among limited actions:
#    - KEEP: keep Theta’s original answer
#    - FLIP_BOOL: flip yes/no
#    - CHOOSE_FROM_CANDIDATES: pick one from the candidate set
#    - ANSWER_UNKNOWN: explicitly mark insufficient evidence
# 2. ACC must not invent new entities; it only chooses from the candidate set.
# 3. ACC’s main duties:
#    - detect obvious conflicts (same/different country, earlier/later, birth/death comparisons, etc.)
#    - apply the smallest edit when conflict is clear
# 4. All decisions and intermediate info are written into acc_result for later analysis.

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
    ACCAgent: ACC-style self-check module (judge + lightweight editor)

    Main interface:
        check_answer(question, gamma_results, initial_answer) -> dict

    Example return dict:
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
        "raw_llm_output": "<raw llm output text>",
        "raw_json": {...}  # JSON parsed from the LLM
    }
    """

    def __init__(self, dataset_name: str, llm_client: LLMClient):
        assert dataset_name in {"2wiki", "hotpotqa", "musique"}
        self.dataset_name = dataset_name
        self.llm = llm_client

    # ------------------------------------------------------------------
    # Helper: format Gamma evidence
    # ------------------------------------------------------------------
    def _format_gamma_evidence(self, gamma_results: List[Dict[str, Any]]) -> str:
        """
        Format theta-gamma step results into text for ACC.
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
    # Helper: detect if it looks like a yes/no question
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

        # Some typical patterns
        patterns = [
            "same country", "same nationality", "same state",
            "is it true", "whether", "if it is"
        ]
        if any(p in q for p in patterns):
            return True

        return False

    # ------------------------------------------------------------------
    # Helper: extract “A or B” options from the question
    # NOTE: key fix to avoid treating the whole question as a candidate
    # ------------------------------------------------------------------
    def _extract_options_from_question(self, question: str) -> List[str]:
        """
        Extract option entities from questions containing an "... A or B" structure.
        This is a general option parser, not a per-question hack.

        Important guardrails:
        - Extract only when the last segment truly contains " A or B ".
        - Questions without " or " (e.g., "What city is the person...") produce no candidates,
          avoiding treating question fragments as answers.
        """
        q = question.strip().strip("?")
        lower_q = q.lower()

        # If no " or ", it is not a choice question
        if " or " not in lower_q:
            return []

        # Only look at the last clause for " or "
        segment = q
        if "," in q:
            # Take the part after the last comma, usually "... A or B"
            segment = q.split(",")[-1]

        # If the last clause lacks " or " (comma elsewhere), skip extraction
        if " or " not in segment.lower():
            return []

        parts = [p.strip() for p in segment.split(" or ")]
        options: List[str] = []
        for p in parts:
            if not p:
                continue
            # Drop leading connectives (only when options exist)
            for bad_prefix in ("which film", "which movie", "which", "who", "what"):
                bp = bad_prefix + " "
                if p.lower().startswith(bp):
                    p = p[len(bp):].strip()
            if p:
                options.append(p)

        # Deduplicate
        seen = set()
        uniq: List[str] = []
        for o in options:
            key = o.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(o)
        return uniq

    # ------------------------------------------------------------------
    # Helper: normalize answers
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
    # Helper: detect if a proposed answer looks like a question (final guardrail)
    # ------------------------------------------------------------------
    def _looks_like_questionish(self, text: str) -> bool:
        """
        Return True if an answer still looks like a question fragment.
        Used as the last sanity check: ACC must not return such answers.
        """
        t = (text or "").strip().lower()
        if not t:
            return False

        # Contains a question mark
        if "?" in t:
            return True

        # Starts with a wh- word
        wh_prefixes = ("who ", "what ", "when ", "where ", "why ", "how ", "which ")
        if any(t.startswith(p) for p in wh_prefixes):
            return True

        # Contains wh- words inside and is fairly long
        tokens = t.split()
        if len(tokens) > 10 and any(
            f" {w} " in f" {t} " for w in ("who", "what", "when", "where", "why", "how", "which")
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Main function: self-check + light edit
    # ------------------------------------------------------------------
    def check_answer(
        self,
        question: str,
        gamma_results: List[Dict[str, Any]],
        initial_answer: str,
    ) -> Dict[str, Any]:
        """
        ACC main self-check function.

        Notes:
        - ACC does not re-answer the question; it only chooses among:
          KEEP / FLIP_BOOL / CHOOSE_FROM_CANDIDATES / ANSWER_UNKNOWN
        - For yes/no questions, the candidates are {"yes", "no", "unknown"}.
        - For explicit option questions (A or B), candidates = {A, B, ...} ∪ {initial_answer}.
        - ACC must not output anything outside the candidate set.
        """

        if initial_answer is None:
            initial_answer = ""
        if not isinstance(initial_answer, str):
            initial_answer = str(initial_answer)
        initial_answer = initial_answer.strip()

        # 1) Basic checks: is it yes/no, and does it have options
        is_bool_q = self._looks_like_yes_no_question(question)
        options = self._extract_options_from_question(question)

        # Build candidate set
        candidates: List[str] = []
        if is_bool_q:
            # Explicit yes/no/unknown set
            candidates = [BOOL_YES, BOOL_NO, BOOL_UNKNOWN]
            init_norm = self._normalize_bool_answer(initial_answer)
            if init_norm in {BOOL_YES, BOOL_NO, BOOL_UNKNOWN}:
                initial_answer_normed = init_norm
            else:
                initial_answer_normed = initial_answer
        else:
            # Non-boolean: options + initial_answer
            candidates = options[:] if options else []
            if initial_answer and all(self._norm(initial_answer) != self._norm(o) for o in candidates):
                candidates.append(initial_answer)

            initial_answer_normed = initial_answer

        # 2) Format Gamma evidence
        evidence_block = self._format_gamma_evidence(gamma_results)

        # 3) Build ACC prompt (must stay within given candidates/actions)
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
            # Call failed: conservatively KEEP
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

        # 4) Parse JSON
        try:
            obj = extract_json_from_text(raw_llm_output)
            if not isinstance(obj, dict):
                raise ValueError("ACC output is not a dict")
        except Exception as e:
            # Parse failed: conservatively KEEP
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

        # 5) Compute final answer based on action and candidates (strict guardrails)
        final_answer = initial_answer_normed

        # Normalize boolean answers again
        if is_bool_q:
            final_answer = self._normalize_bool_answer(initial_answer_normed)
            # Normalize candidates as well
            candidates = [self._normalize_bool_answer(c) for c in candidates]

        if action == ACTION_KEEP:
            final_answer = initial_answer_normed

        elif action == ACTION_FLIP_BOOL and is_bool_q:
            norm_init = self._normalize_bool_answer(initial_answer_normed)
            if norm_init == BOOL_YES:
                final_answer = BOOL_NO
            elif norm_init == BOOL_NO:
                final_answer = BOOL_YES
            else:
                # Initial answer not clearly yes/no; do not flip
                action = ACTION_KEEP
                final_answer = initial_answer_normed

        elif action == ACTION_CHOOSE and candidates:
            # Must choose within candidates
            # If missing or invalid candidate_answer, fall back to KEEP
            if candidate_answer:
                # Normalize boolean candidate before matching
                if is_bool_q:
                    cand_norm = self._normalize_bool_answer(candidate_answer)
                    candidate_answer = cand_norm

                if any(self._norm(candidate_answer) == self._norm(c) for c in candidates):
                    final_answer = candidate_answer
                else:
                    # Invalid candidate; fall back to KEEP
                    action = ACTION_KEEP
                    final_answer = initial_answer_normed
            else:
                action = ACTION_KEEP
                final_answer = initial_answer_normed

        elif action == ACTION_UNKNOWN:
            # Use a unified unknown label
            final_answer = BOOL_UNKNOWN if is_bool_q else "unknown"

        # 6) Final guardrail: ACC cannot return the question (or fragments) as an answer
        if (not is_bool_q) and self._looks_like_questionish(final_answer):
            # Directly fall back to Theta's original answer
            if explanation:
                explanation = (
                    explanation
                    + " NOTE: ACC candidate looked like a question, fallback to Theta initial answer."
                )
            else:
                explanation = (
                    "ACC candidate looked like a question, fallback to Theta initial answer."
                )
            action = ACTION_KEEP
            final_answer = initial_answer_normed

        # 7) Return result for pipeline/theta logging
        return {
            "action": action,
            "final_answer": final_answer,
            "explanation": explanation,
            "flags": flags,
            "candidates": candidates,
            "raw_llm_output": raw_llm_output,
            "raw_json": obj,
        }
