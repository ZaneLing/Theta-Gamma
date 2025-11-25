# acc.py
# ACC self-check module: sanity-check logic/entity consistency before Theta finalizes the answer.

from typing import Any, Dict, List, Optional

from gamma_gpt35 import LLMClient, extract_json_from_text


class ACCAgent:
    """
    ACCAgent emulates an “ACC self-check” module.

    Inputs:
      - original question
      - theta-gamma stepwise trace (subquestion + gamma_result)
      - theta's initial answer

    Outputs:
      - verdict: "keep" or "revise"
      - final_answer: final answer to score (revised if needed)
      - explanation: why ACC made the decision
      - flags: error type markers (entity_mismatch / logic_inconsistency / insufficient_evidence)
    """

    def __init__(self, dataset_name: str, llm_client: LLMClient):
        assert dataset_name in {"2wiki", "hotpotqa", "musique"}
        self.dataset_name = dataset_name
        self.llm = llm_client

    # ----------------- Helper: format gamma evidence -----------------

    def _format_gamma_evidence(self, gamma_results: List[Dict[str, Any]]) -> str:
        """
        Turn theta-gamma step results into a readable block for ACC:
          - subquestion / refined_subquestion
          - found / answer / reasoning
          - selected supporting fact texts (if any)
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
                    # Trim long fact text to keep prompt compact
                    ft_str = str(ft)
                    if len(ft_str) > 400:
                        ft_str = ft_str[:400] + " ..."
                    lines.append(f"    - {ft_str}")
            lines.append("")  # blank line between steps

        return "\n".join(lines).strip()

    # ----------------- Main entry: self-check + possible correction -----------------

    def check_answer(
        self,
        question: str,
        gamma_results: List[Dict[str, Any]],
        initial_answer: str,
    ) -> Dict[str, Any]:
        """
        ACC self-check core routine.

        Returns:
          {
            "verdict": "keep" | "revise",
            "final_answer": str,
            "explanation": str,
            "flags": {
               "entity_mismatch": bool,
               "logic_inconsistency": bool,
               "insufficient_evidence": bool
            },
            "raw_json": {...}
          }
        """
        evidence_block = self._format_gamma_evidence(gamma_results)

        prompt = f"""
        You are the ACC (anterior cingulate cortex) style self-checking module in a dual-agent multi-hop QA system.

        You receive:
        - ORIGINAL QUESTION
        - STEPWISE EVIDENCE from the Gamma agent
        - An INITIAL_ANSWER produced by the Theta agent

        Your tasks:
        1. Decide whether INITIAL_ANSWER correctly answers the ORIGINAL QUESTION, given ONLY the evidence.
        2. Explicitly check for:
        (a) ENTITY MISMATCH:
            - The answer refers to the wrong entity (wrong film/person/place),
                or uses evidence about a different but similar entity (e.g., another film in the same series).
        (b) LOGICAL / NUMERICAL INCONSISTENCY:
            - Especially for comparative questions ("older/younger", "earlier/later", "more/less", etc.),
                verify the comparison by reasoning over dates, years, or other numbers extracted from the evidence.
            - For "who is younger/older", compare BIRTH DATES or BIRTH YEARS.
                Example: if A was born in 1887 and B was born in 1914, then B is younger (because 1914 > 1887).
                Do NOT compare lifespan length unless the question explicitly asks "who lived longer".
        (c) INSUFFICIENT EVIDENCE:
            - If you cannot confidently determine the correct answer from the provided evidence, mark this flag.

        3. If INITIAL_ANSWER is wrong, not fully supported, or suffers from the above issues,
        you must compute a CORRECTED ANSWER using ONLY the evidence.

        Output a SINGLE JSON object with this exact structure:
        {{
        "verdict": "keep" or "revise",
        "final_answer": "your final answer as a short phrase, or empty string if unknown",
        "explanation": "short explanation in English",
        "flags": {{
            "entity_mismatch": true or false,
            "logic_inconsistency": true or false,
            "insufficient_evidence": true or false
        }}
        }}

        Rules:
        - If you choose "keep", final_answer MUST EQUAL the INITIAL_ANSWER (after trimming spaces).
        - If you choose "revise", final_answer SHOULD be your new corrected answer.
        - If evidence is clearly insufficient, set insufficient_evidence=true and you MAY set final_answer="" or "unknown".

        ORIGINAL QUESTION:
        {question}

        INITIAL_ANSWER (from Theta):
        {initial_answer}

        GAMMA STEPWISE EVIDENCE:
        {evidence_block}

        Respond with JSON ONLY:
        """.strip()

        try:
            raw_text = self.llm.generate(
                prompt,
                meta={
                    "agent": "acc",
                    "dataset": self.dataset_name,
                    "kind": "self_check",
                },
                temperature=0.0,
            )
        except Exception as e:
            # If LLM call fails, fall back to the initial answer
            return {
                "verdict": "keep",
                "final_answer": initial_answer,
                "explanation": f"ACC call error: {e}",
                "flags": {
                    "entity_mismatch": False,
                    "logic_inconsistency": False,
                    "insufficient_evidence": False,
                },
                "raw_json": {},
            }

        try:
            obj = extract_json_from_text(raw_text)
            if not isinstance(obj, dict):
                raise ValueError("ACC output is not a dict")
        except Exception as e:
            # If parsing fails, keep the initial answer
            return {
                "verdict": "keep",
                "final_answer": initial_answer,
                "explanation": f"ACC JSON parse error: {e}",
                "flags": {
                    "entity_mismatch": False,
                    "logic_inconsistency": False,
                    "insufficient_evidence": False,
                },
                "raw_json": {},
            }

        verdict = str(obj.get("verdict", "keep")).strip().lower()
        if verdict not in {"keep", "revise"}:
            verdict = "keep"

        final_answer = obj.get("final_answer", "")
        if not isinstance(final_answer, str):
            final_answer = str(final_answer)
        final_answer = final_answer.strip()

        explanation = obj.get("explanation", "")
        if not isinstance(explanation, str):
            explanation = str(explanation)
        explanation = explanation.strip()

        flags = obj.get("flags", {}) or {}
        entity_mismatch = bool(flags.get("entity_mismatch", False))
        logic_inconsistency = bool(flags.get("logic_inconsistency", False))
        insufficient_evidence = bool(flags.get("insufficient_evidence", False))

        # safety: if verdict=keep but ACC produced a different final_answer, force initial_answer
        if verdict == "keep":
            final_answer = initial_answer.strip()

        # if verdict=revise but final_answer is empty, fall back to initial_answer
        if verdict == "revise" and not final_answer:
            final_answer = initial_answer.strip()

        return {
            "verdict": verdict,
            "final_answer": final_answer,
            "explanation": explanation,
            "flags": {
                "entity_mismatch": entity_mismatch,
                "logic_inconsistency": logic_inconsistency,
                "insufficient_evidence": insufficient_evidence,
            },
            "raw_json": obj,
        }
