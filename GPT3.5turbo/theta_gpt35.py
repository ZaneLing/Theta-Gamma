# theta_gpt35.py
# Theta agent: orchestrates Gamma calls, applies ACC self-check,
# question schema + slot graph, BridgeManager, and returns trace.

from typing import Any, Dict, List, Optional
from pathlib import Path
import re
from requests.exceptions import HTTPError, SSLError, ConnectionError, Timeout
import yaml

from gamma_gpt35 import LLMClient, GammaAgent, extract_json_from_text
from acc_gpt35 import ACCAgent


_PROMPT_CACHE: Dict[str, str] = {}


def _load_prompts() -> Dict[str, str]:
    global _PROMPT_CACHE
    if _PROMPT_CACHE:
        return _PROMPT_CACHE
    prompt_path = Path(__file__).resolve().parent / "prompts" / "theta_prompts.yaml"
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _PROMPT_CACHE = {str(k): str(v) for k, v in data.items()}
    return _PROMPT_CACHE


def get_prompt(name: str) -> str:
    prompts = _load_prompts()
    if name not in prompts:
        raise KeyError(f"Prompt '{name}' not found")
    return prompts[name]


class BridgeManager:
    """
    BridgeManager: lightweight “bridge” manager.

    Goals:
    - Use schema entities plus previous Gamma entity answers as anchors for each subquestion.
    - Check whether Gamma-selected facts actually mention these anchors.
    - If a subquestion clearly depends on anchors but selected facts omit them, mark the bridge
      as unreliable: set bridge_ok=False and found=False (gating).

    This version adds:
    - Support for entity aliases (schema.entities[*].aliases) as anchor forms.
    - Simple pronoun alignment: if subquestions contain he/she/they/this film/that movie, use
      the most recent Gamma answer as an anchor.
    - Track hits by “wiki page group”; when the same anchor appears on a new page group, note it
      to reduce cross-article drift (logging hint, not enforced gating).

    Output per step:
      {
        "step_index": int,
        "anchors": [...],          # anchors inferred for the current subquestion
        "matched_anchors": [...],  # anchors actually matched in selected_fact_texts
        "bridge_ok": bool,
        "bridge_reason": str,
        "page_titles": [...],      # rough page titles hit in this step
      }
    The trace’s "bridge_manager" contains summary + anchor_pages for logging/analysis.
    """

    def __init__(self, question: str, schema: Dict[str, Any]):
        self.question = question
        self.schema = schema or {}

        # Pull explicit entity names + aliases from the schema
        # schema_entities: canonical names
        # entity_alias_map: canonical name -> [alias1, alias2, ...]
        # all_anchor_forms: every surface form usable as an anchor (name + alias)
        self.schema_entities: List[str] = []
        self.entity_alias_map: Dict[str, List[str]] = {}
        self.all_anchor_forms: List[str] = []

        for e in self.schema.get("entities", []) or []:
            name = (e.get("name") or "").strip()
            aliases = e.get("aliases") or e.get("alias") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]

            if name:
                self.schema_entities.append(name)
                self.all_anchor_forms.append(name)
                if aliases:
                    self.entity_alias_map[name] = aliases
                    self.all_anchor_forms.extend(aliases)

        # Entity-style answers from previous Gamma steps
        self.prev_answers: List[str] = []

        # Record which rough page titles each anchor hit
        # anchor_pages[anchor] = ["Title1", "Title2", ...]
        self.anchor_pages: Dict[str, List[str]] = {}

        # Per-step bridge records
        self.steps: List[Dict[str, Any]] = []

    def _extract_anchors_for_subquestion(self, subq: str) -> List[str]:
        """
        Extract anchor entities for the current refined_subquestion:
        - Schema entity names or aliases that appear in the subquestion.
        - Explicit answers from prior steps that appear in the subquestion.
        - Simple pronoun alignment: if he/she/they/this film/etc. appear and there are
          previous answers, use the most recent answer as an anchor.
        """
        anchors = set()
        sq = (subq or "")
        sq_lower = sq.lower()

        # Schema entities (name + alias)
        for form in self.all_anchor_forms:
            f = form.strip()
            if f and f.lower() in sq_lower:
                anchors.add(f)

        # Prior answers as potential intermediate entities
        for ans in self.prev_answers:
            a = ans.strip()
            if a and a.lower() in sq_lower:
                anchors.add(a)

        # Pronoun alignment: with pronouns + history, use the most recent answer as anchor
        pronoun_markers = [
            " he ", " she ", " they ", " him ", " her ", " them ",
            " this person", " that person", " the person",
            " this man", " that man", " this woman", " that woman",
            " this film", " that film", " this movie", " that movie",
            " the film", " the movie",
        ]
        padded = f" {sq_lower} "
        if any(p in padded for p in pronoun_markers) and self.prev_answers:
            last_ans = self.prev_answers[-1].strip()
            if last_ans:
                anchors.add(last_ans)

        return list(anchors)

    def _extract_page_titles(self, selected_texts: List[Any]) -> List[str]:
        """
        Roughly extract “page titles” from selected_fact_texts:
        prefer the part before a colon; otherwise take a short prefix.
        """
        titles = set()
        for t in selected_texts or []:
            s = str(t)
            if ":" in s:
                head = s.split(":", 1)[0].strip()
                if head:
                    titles.add(head)
            else:
                snippet = s[:80].strip()
                if snippet:
                    titles.add(snippet)
        return sorted(titles)

    def update_step(
        self,
        step_index: int,
        refined_subq: str,
        gamma_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call after each Gamma step:
        - Extract anchors from refined_subq.
        - Check whether selected_fact_texts mention these anchors.
        - If anchors exist but none are mentioned:
            - bridge_ok=False
            - annotate gamma_result["reasoning"]
            - force gamma_result["found"]=False (gating)
        - If still found and answer is a string, record it in prev_answers.
        - Record page_titles and accumulate in anchor_pages.
        """
        anchors = self._extract_anchors_for_subquestion(refined_subq)
        selected_texts = gamma_result.get("selected_fact_texts") or []
        big_text = " ".join(str(t) for t in selected_texts).lower()
        page_titles = self._extract_page_titles(selected_texts)

        matched_anchors: List[str] = []
        if big_text:
            matched_anchors = [a for a in anchors if a.lower() in big_text]

        bridge_ok = True
        bridge_reason_parts: List[str] = []

        # Main gating: anchors exist but facts never mention them
        if anchors:
            if gamma_result.get("found"):
                if not matched_anchors:
                    bridge_ok = False
                    bridge_reason_parts.append(
                        "Gamma selected facts do not mention any anchor entities "
                        "from schema/previous answers."
                    )
                    gamma_result["found"] = False
                    gamma_result.setdefault("reasoning", "")
                    if "BridgeManager" not in gamma_result["reasoning"]:
                        gamma_result["reasoning"] += (
                            " [BridgeManager: anchor entities not found in selected facts; "
                            "treat as not found]"
                        )
            else:
                # Already not found, and the subquestion needs anchors -> bridge breaks here
                bridge_ok = False
                bridge_reason_parts.append(
                    "Gamma could not find an answer for a subquestion that depends on "
                    "anchor entities."
                )
        else:
            # No anchors detected: pass, no gating
            bridge_ok = True

        # Track page groups, note cross-page jumps
        if matched_anchors and page_titles:
            for a in matched_anchors:
                prev_pages = set(self.anchor_pages.get(a, []))
                new_pages = set(page_titles)
                if prev_pages and not (prev_pages & new_pages):
                    # Same anchor appears in a new page group: add a cross-page note
                    bridge_reason_parts.append(
                        f"Anchor '{a}' moved from pages {sorted(prev_pages)} "
                        f"to new pages {sorted(new_pages)}."
                    )
                merged = sorted(prev_pages | new_pages) if prev_pages else sorted(new_pages)
                self.anchor_pages[a] = merged

        bridge_reason = " ".join(bridge_reason_parts).strip()

        # Keep entity answers that are still considered found, for later steps
        ans = gamma_result.get("answer")
        if gamma_result.get("found") and isinstance(ans, str) and ans.strip():
            self.prev_answers.append(ans.strip())
            # Trim list to avoid overgrowth
            if len(self.prev_answers) > 6:
                self.prev_answers = self.prev_answers[-6:]

        meta = {
            "step_index": step_index,
            "anchors": anchors,
            "matched_anchors": matched_anchors,
            "bridge_ok": bridge_ok,
            "bridge_reason": bridge_reason,
            "page_titles": page_titles,
        }
        self.steps.append(meta)
        return meta

    def summary(self) -> str:
        """
        Return a short summary for logs/prompts.
        """
        if not self.steps:
            return "BridgeManager: no steps recorded."
        lines: List[str] = []
        bad = 0
        for s in self.steps:
            ok = bool(s.get("bridge_ok", True))
            if not ok:
                bad += 1
            lines.append(
                f"step {s.get('step_index')}: bridge_ok={ok}, "
                f"anchors={s.get('anchors')}, matched={s.get('matched_anchors')}, "
                f"pages={s.get('page_titles')}, "
                f"reason={s.get('bridge_reason')}"
            )
        if bad == 0:
            head = "BridgeManager summary: all steps passed anchor-entity bridge checks."
        else:
            head = (
                f"BridgeManager summary: {bad}/{len(self.steps)} steps had weak or "
                f"missing anchor-entity alignment."
            )
        return head + "\n" + "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "schema_entities": list(self.schema_entities),
            "entity_alias_map": dict(self.entity_alias_map),
            "anchor_pages": dict(self.anchor_pages),
            "steps": list(self.steps),
            "summary": self.summary(),
        }

class ThetaAgent:
    """
    Theta agent:
    - Question schema analysis (task type, answer role, entities, slots, subgoals)
    - Plan subquestions (theta)
    - Dispatch gamma as needed
    - Dynamically rewrite later subquestions using previous answers
    - Integrate the final answer (with symbolic & multi-constraint hints)
    - Call ACC for lightweight self-check
    - Return full trace (for logging / analysis)
    """

    def __init__(self, dataset_name: str, llm_client: LLMClient, theta_llm_client: Optional[LLMClient] = None):
        assert dataset_name in {"2wiki", "hotpotqa", "musique"}
        self.dataset_name = dataset_name
        # Gamma/ACC use the baseline client (e.g., gpt-3.5), while theta can use a separate model.
        self.gamma_llm = llm_client
        self.theta_llm = theta_llm_client or llm_client
        self.acc = ACCAgent(dataset_name=dataset_name, llm_client=self.gamma_llm)
        self.llm = self.theta_llm
        self.gamma = GammaAgent(dataset_name=dataset_name, llm_client=self.gamma_llm)

        # BridgeManager class (swap implementation easily if needed)
        self.bridge_manager_cls = BridgeManager

        # Cache for the current sample's question schema (avoid repeat calls)
        self._schema_question: Optional[str] = None
        self._schema: Optional[Dict[str, Any]] = None

    # ---------- Question Schema Construction ----------

    def _ensure_schema(self, question: str) -> Dict[str, Any]:
        """
        Ensure that self._schema is built for this question.
        """
        if self._schema is not None and self._schema_question == question:
            return self._schema
        schema = self.build_question_schema(question)
        self._schema_question = question
        self._schema = schema
        return schema

    def build_question_schema(self, question: str) -> Dict[str, Any]:
        """
        Build a Question Schema + Slot Graph:
        - question_type: yes_no / comparison / fact / count / other
        - answer_form: yes_no / entity_name / number / date / span / other
        - focus: person / film / location / organization / other (hint only)
        - entities: [{"name": ..., "type": ..., "role": ..., "aliases": [...]}, ...]
        - constraints / notes: task rules & other hints
        - slots: intermediate slots (id / description / required / expected_type)
        - subgoals: subgoals (id / description / fills_slots)
        - final_composition: how to combine slot values into the final answer
        """
        prompt = get_prompt("question_schema").format(
            dataset_name=self.dataset_name,
            question=question,
        ).strip()

        raw_text = self.llm.generate(
            prompt,
            meta={
                "agent": "theta",
                "kind": "question_schema",
                "dataset": self.dataset_name,
            },
            temperature=0.2,
        )

        # Robust parsing
        try:
            obj = extract_json_from_text(raw_text)
            if not isinstance(obj, dict):
                raise ValueError("schema must be an object")
        except Exception:
            obj = {}

        # Extract slots / subgoals / final_composition with correct types
        slots = obj.get("slots") or []
        if not isinstance(slots, list):
            slots = []
        subgoals = obj.get("subgoals") or []
        if not isinstance(subgoals, list):
            subgoals = []
        final_comp = obj.get("final_composition") or obj.get("final_answer_composition") or ""
        if not isinstance(final_comp, str):
            final_comp = str(final_comp)

        # Enforce: every required slot has at least one subgoal bound
        slot_ids_required = {str(s.get("id")).strip()
                             for s in slots
                             if isinstance(s, dict) and s.get("required") and str(s.get("id")).strip()}
        filled_ids = set()
        for g in subgoals:
            if not isinstance(g, dict):
                continue
            for sid in g.get("fills_slots") or []:
                sid_str = str(sid).strip()
                if sid_str:
                    filled_ids.add(sid_str)
        missing_ids = sorted(slot_ids_required - filled_ids)
        if missing_ids:
            for mid in missing_ids:
                # Find the slot description
                desc = ""
                for s in slots:
                    if isinstance(s, dict) and str(s.get("id")).strip() == mid:
                        desc = (s.get("description") or "").strip()
                        break
                auto_id = f"auto_{mid}"
                subgoals.append(
                    {
                        "id": auto_id,
                        "description": desc or f"Resolve slot {mid}",
                        "fills_slots": [mid],
                    }
                )

        # Fill defaults to avoid downstream KeyErrors
        schema: Dict[str, Any] = {
            "question_type": obj.get("question_type", "other"),
            "answer_form": obj.get("answer_form", "other"),
            "focus": obj.get("focus", "generic"),
            "entities": obj.get("entities", []),
            "constraints": obj.get("constraints", []),
            "notes": obj.get("notes", ""),
            "slots": slots,
            "subgoals": subgoals,
            "final_composition": final_comp,
            "raw_json": obj,
        }
        return schema

    def _schema_summary(self, schema: Dict[str, Any]) -> str:
        """
        Render schema + slot graph into a short text hint for prompts.
        """
        qtype = schema.get("question_type", "other")
        ans_form = schema.get("answer_form", "other")
        focus = schema.get("focus", "generic")
        ents = schema.get("entities", [])
        cons = schema.get("constraints", [])
        notes = schema.get("notes", "")
        slots = schema.get("slots", [])
        subgoals = schema.get("subgoals", [])
        final_comp = schema.get("final_composition", "")

        ent_lines = []
        for e in ents:
            name = (e.get("name") or "").strip()
            etype = (e.get("type") or "other").strip()
            role = (e.get("role") or "other").strip()
            aliases = e.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            alias_txt = ""
            if aliases:
                alias_txt = f", aliases={aliases}"
            ent_lines.append(f"- {role}: '{name}' ({etype}{alias_txt})")

        ent_block = "\n".join(ent_lines) if ent_lines else "  (no explicit entities parsed)"
        cons_block = "\n".join(f"- {c}" for c in cons) if cons else "  (no explicit constraints)"
        notes_block = notes or "(no extra notes)"

        slot_lines = []
        for s in slots:
            if not isinstance(s, dict):
                continue
            sid = (s.get("id") or "").strip() or "S?"
            desc = (s.get("description") or "").strip()
            req = bool(s.get("required"))
            etype = (s.get("expected_type") or "unknown").strip()
            req_flag = "required" if req else "optional"
            slot_lines.append(f"- {sid} ({req_flag}, type={etype}): {desc}")
        slot_block = "\n".join(slot_lines) if slot_lines else "  (no explicit slots)"

        sg_lines = []
        for g in subgoals:
            if not isinstance(g, dict):
                continue
            gid = (g.get("id") or "").strip() or "G?"
            desc = (g.get("description") or "").strip()
            fills = [str(x).strip() for x in (g.get("fills_slots") or []) if str(x).strip()]
            sg_lines.append(f"- {gid}: {desc} [fills_slots={fills}]")
        subgoal_block = "\n".join(sg_lines) if sg_lines else "  (no explicit subgoals)"

        final_comp_txt = final_comp or "(not specified)"

        return (
            f"Question type: {qtype}\n"
            f"Answer form: {ans_form}\n"
            f"Answer focus: {focus}\n"
            f"Entities:\n{ent_block}\n"
            f"Constraints:\n{cons_block}\n"
            f"Notes: {notes_block}\n"
            f"Slots:\n{slot_block}\n"
            f"Subgoals:\n{subgoal_block}\n"
            f"Final composition: {final_comp_txt}"
        )

    # ---------- Stage 1: theta decomposes the question ----------

    def plan_subquestions(self, question: str, max_steps: int = 4) -> List[str]:
        # Ensure schema exists
        schema = self._ensure_schema(question)
        schema_summary = self._schema_summary(schema)

        prompt = get_prompt("plan_subquestions").format(
            max_steps=max_steps,
            schema_summary=schema_summary,
            question=question,
        ).strip()

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

    def refine_subquestion(
        self,
        question: str,
        planned_subquestion: str,
        previous_steps: List[Dict[str, Any]],
    ) -> str:
        """
        Rewrite the planned_subquestion using earlier Gamma answers to be more specific,
        while respecting the QUESTION_SCHEMA (answer role, entity alignment).
        If no reliable answer yet (found=false or answer empty), keep it unchanged.
        """
        if not previous_steps:
            return planned_subquestion

        schema = self._ensure_schema(question)
        schema_summary = self._schema_summary(schema)

        # Summarize previous steps
        lines = []
        for step in previous_steps:
            gres = step.get("gamma_result", {}) or {}
            lines.append(
                f"step {step.get('step_index')}: "
                f"subquestion='{step.get('refined_subquestion', step.get('subquestion'))}', "
                f"found={gres.get('found')}, answer={gres.get('answer')}"
            )
        history_block = "\n".join(lines)

        prompt = get_prompt("refine_subquestion").format(
            schema_summary=schema_summary,
            history_block=history_block,
            planned_subquestion=planned_subquestion,
        ).strip()

        raw_text = self.llm.generate(
            prompt,
            meta={
                "agent": "theta",
                "kind": "refine_subquestion",
                "dataset": self.dataset_name,
            },
            temperature=0.2,
        )

        try:
            obj = extract_json_from_text(raw_text)
            refined = obj.get("subquestion", "").strip()
            return refined if refined else planned_subquestion
        except Exception:
            return planned_subquestion

    # ---------- Stage 2: symbolic comparator + multi-constraint hints ----------

    def _extract_years(self, text: str) -> List[int]:
        years: List[int] = []
        for m in re.findall(r"(1[5-9]\d{2}|20\d{2})", text or ""):
            try:
                years.append(int(m))
            except Exception:
                continue
        return years

    def build_symbolic_schema(self, question: str, gamma_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Lightweight symbolic comparator + multi-constraint hints:
        - detect comparative keywords (older/younger/earlier/later/before/after).
        - detect intersection / multi-constraint keywords (both, in common, shared, etc.).
        - extract numeric cues (years) from gamma answers and selected facts.
        - detect tokens that appear in answers of multiple steps (rough intersection candidates).
        - provide a compact hint string for LLM to avoid small comparison slips
          and to respect intersection-style constraints.
        """
        comparative_keywords = [
            "older", "younger", "earlier", "later", "before", "after",
            "first", "second", "younger than", "older than", "earlier than", "later than",
        ]
        multi_keywords = [
            "both", "all of the following", "all of", "each of",
            "in common", "shared", "common to", "intersection",
            "that are also", "who are also", "as well as",
        ]

        q_lower = question.lower()
        comp_hits = [kw for kw in comparative_keywords if kw in q_lower]
        multi_hits = [kw for kw in multi_keywords if kw in q_lower]
        is_comp = bool(comp_hits)
        is_multi = bool(multi_hits)

        entities = []
        answer_tokens_per_step: List[List[str]] = []

        for step in gamma_results:
            gres = step.get("gamma_result", {}) or {}
            answer = str(gres.get("answer", "") or "")
            years = set(self._extract_years(answer))
            for ft in gres.get("selected_fact_texts", []) or []:
                years.update(self._extract_years(str(ft)))

            # Roughly extract possible candidate entity tokens from answers
            tokens: List[str] = []
            if answer:
                parts = re.split(r",| and | & ", answer)
                for p in parts:
                    t = p.strip()
                    if len(t) >= 3 and any(c.isalpha() for c in t):
                        tokens.append(t)
            answer_tokens_per_step.append(tokens)

            entities.append({
                "step": step.get("step_index"),
                "subq": step.get("refined_subquestion") or step.get("subquestion"),
                "answer": answer,
                "years": sorted(years),
            })

        # Collect intersection candidates: tokens appearing in answers of >=2 steps
        token_counts: Dict[str, int] = {}
        for ts in answer_tokens_per_step:
            for t in set(ts):
                token_counts[t] = token_counts.get(t, 0) + 1
        intersection_candidates = [t for t, c in token_counts.items() if c >= 2]

        # Build a human-readable summary to feed into prompts
        lines = []
        if is_comp:
            lines.append(f"Comparative keywords detected: {', '.join(comp_hits)}.")
        else:
            lines.append("No obvious comparative keywords detected.")

        if is_multi:
            lines.append(f"Multi-constraint / intersection keywords detected: {', '.join(multi_hits)}.")
            if intersection_candidates:
                lines.append(
                    "Tokens that appear in answers of multiple steps "
                    "(possible intersection candidates): "
                    + "; ".join(intersection_candidates)
                )
            else:
                lines.append(
                    "No obvious intersection candidates detected from gamma answers; "
                    "if the question requires satisfying multiple constraints, "
                    "prefer entities supported by more than one step."
                )
        else:
            lines.append("No obvious multi-constraint / intersection pattern detected.")

        for ent in entities:
            ytxt = ", ".join(str(y) for y in ent["years"]) if ent["years"] else "none"
            lines.append(f"Step {ent['step']}: answer='{ent['answer']}', years=[{ytxt}].")
        summary = "\n".join(lines)

        return {
            "is_comparative": is_comp,
            "keywords": comp_hits,
            "entities": entities,
            "summary": summary,
            "multi_constraint": {
                "is_multi_constraint": is_multi,
                "keywords": multi_hits,
                "intersection_candidates": intersection_candidates,
            },
        }

    def integrate_answer(
        self,
        question: str,
        gamma_results: List[Dict[str, Any]],
        comparator_summary: str = "",
    ) -> Dict[str, Any]:
        """
        Feed question schema + gamma results + comparator hints + multi-constraint hints
        to the LLM and request a final answer with strict formatting:
        - Answer must follow schema.answer_form.
        - If yes/no: only “yes” or “no”.
        - If “which film/person...”: output the entity, not intermediate reasoning nodes.
        """
        schema = self._ensure_schema(question)
        schema_summary = self._schema_summary(schema)

        lines = []
        for i, gr in enumerate(gamma_results, start=1):
            gres = gr.get("gamma_result", {}) or {}
            asked_subq = gr.get("refined_subquestion") or gr.get("subquestion")
            lines.append(
                f"Step {i}:\n"
                f"  subquestion: {asked_subq}\n"
                f"  planned_subquestion: {gr.get('subquestion')}\n"
                f"  gamma_found: {gres.get('found')}\n"
                f"  gamma_answer: {gres.get('answer')}\n"
                f"  gamma_reasoning: {gres.get('reasoning')}\n"
                f"  used_facts: {gres.get('selected_fact_indices')}"
            )
        gamma_summary = "\n\n".join(lines)

        prompt = get_prompt("integrate_answer").format(
            schema_summary=schema_summary,
            comparator_summary=comparator_summary,
            question=question,
            gamma_summary=gamma_summary,
        ).strip()

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

        # Include the original question inside the reasoning for clearer logs.
        combined_reasoning = (
            f"Question: {question}\n"
            f"Reasoning: {reasoning}\n"
            f"Final answer: {answer}"
        )

        return {
            "answer": answer,
            "reasoning": combined_reasoning,
            "raw_json": obj,
        }

    # ---------- Full pipeline: solve one example and return all details ----------

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

        # 0) Build Question Schema first (everything else runs within this frame)
        try:
            schema = self._ensure_schema(question)
        except Exception as e:
            # In some environments the schema call may be blocked/network-failed; skip sample to avoid halting the run
            status = None
            resp_text = ""
            if isinstance(e, HTTPError) and e.response is not None:
                status = e.response.status_code
                try:
                    resp_text = e.response.text or ""
                except Exception:
                    resp_text = ""
            if isinstance(e, (HTTPError, SSLError, ConnectionError, Timeout)) or status == 403 or "403" in str(e):
                return {
                    "example_index": example_index,
                    "question": question,
                    "skipped": True,
                    "skip_stage": "schema",
                    "skip_reason": f"Schema request failed: {e}",
                    "skip_error_status": status,
                    "skip_error_body": resp_text.strip(),
                }
            raise
        schema_summary = self._schema_summary(schema)  # log-only

        # Initialize BridgeManager
        bridge_manager = self.bridge_manager_cls(question=question, schema=schema)

        # 1) theta: plan subquestions
        try:
            subquestions = self.plan_subquestions(question)
        except Exception as e:
            status = None
            resp_text = ""
            if isinstance(e, HTTPError) and e.response is not None:
                status = e.response.status_code
                try:
                    resp_text = e.response.text or ""
                except Exception:
                    resp_text = ""
            # If planning hits 403 / network errors, skip this sample
            if isinstance(e, (HTTPError, SSLError, ConnectionError, Timeout)) or status == 403 or "403" in str(e):
                return {
                    "example_index": example_index,
                    "question": question,
                    "skipped": True,
                    "skip_stage": "plan_subquestions",
                    "skip_reason": f"Planning failed: {e}",
                    "skip_error_status": status,
                    "skip_error_body": resp_text.strip(),
                }
            raise
        gamma_results: List[Dict[str, Any]] = []
        executed_subquestions: List[str] = []

        gamma_call_count = 0
        gamma_success_count = 0
        for step_idx, subq in enumerate(subquestions, start=1):
            refined_subq = (
                self.refine_subquestion(
                    question=question,
                    planned_subquestion=subq,
                    previous_steps=gamma_results,
                )
                if step_idx > 1
                else subq
            )
            executed_subquestions.append(refined_subq)
            call_id = f"{ex_id}_step{step_idx}"
            gamma_call_count += 1
            gr = self.gamma.answer_subquestion(
                example=example,
                subquestion=refined_subq,
                call_id=call_id,
            )
            # BridgeManager: check anchor alignment for this step and apply gating
            bridge_meta = bridge_manager.update_step(
                step_index=step_idx,
                refined_subq=refined_subq,
                gamma_result=gr,
            )

            if gr.get("found"):
                gamma_success_count += 1
            gamma_results.append(
                {
                    "step_index": step_idx,
                    "subquestion": subq,
                    "refined_subquestion": refined_subq,
                    "gamma_result": gr,
                    "bridge": bridge_meta,
                }
            )

        # 2) Symbolic comparator / question schema (PFC+parietal helper)
        comparator = self.build_symbolic_schema(question, gamma_results)
        comparator_summary = comparator.get("summary", "")

        # 3) theta: integrate a preliminary answer (with comparator & schema hints)
        final = self.integrate_answer(question, gamma_results, comparator_summary=comparator_summary)
        initial_answer = final.get("answer", "")
        if not isinstance(initial_answer, str):
            initial_answer = str(initial_answer)
        initial_answer = initial_answer.strip()

        # 4) ACC self-check: detect logical/entity issues and optionally revise
        acc_result = self.acc.check_answer(
            question=question,
            gamma_results=gamma_results,
            initial_answer=initial_answer,
        )
        predicted_answer = acc_result.get("final_answer", "") or initial_answer

        # 5) trace + log
        trace = {
            "question_schema": schema,
            "question_schema_summary": schema_summary,
            "planned_subquestions": subquestions,
            "executed_subquestions": executed_subquestions,
            "gamma_results": gamma_results,
            "theta_final": final,              # Theta raw integration
            "theta_initial_answer": initial_answer,
            "acc_result": acc_result,          # ACC self-check details
            "gamma_call_count": gamma_call_count,
            "gamma_success_count": gamma_success_count,
            "symbolic_comparator": comparator,
            "bridge_manager": bridge_manager.to_dict(),
        }

        theta_calls = getattr(self.theta_llm, "call_log", []) if hasattr(self, "theta_llm") else getattr(self.llm, "call_log", [])
        gamma_calls = getattr(self.gamma_llm, "call_log", []) if hasattr(self, "gamma_llm") else theta_calls
        if theta_calls is gamma_calls:
            llm_calls = theta_calls
        else:
            llm_calls = (theta_calls or []) + (gamma_calls or [])

        result = {
            "dataset": self.dataset_name,
            "example_index": example_index,
            "id": ex_id,
            "question": question,
            "theta_answer": initial_answer,     # Theta answer before ACC
            "predicted_answer": predicted_answer,
            "theta_gamma_trace": trace,
            "llm_calls": llm_calls,
        }
        return result
