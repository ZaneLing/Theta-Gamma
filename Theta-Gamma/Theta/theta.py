# theta_gpt35.py
# Theta agent: orchestrates Gamma calls, applies ACC self-check,
# question schema + slot graph, and returns trace.

from typing import Any, Dict, List, Optional
import re
import json
from requests.exceptions import HTTPError, SSLError, ConnectionError, Timeout

from gamma_gpt35 import LLMClient, GammaAgent, extract_json_from_text
from acc_gpt35 import ACCAgent


class BridgeManager:
    """
    BridgeManager: lightweight anchor-alignment checker.

    Goals:
    - Use entities from the question schema plus explicit entity answers from earlier
      Gamma steps as anchors for each subquestion.
    - Check whether Gamma-selected facts actually mention these anchors.
    - If a subquestion clearly depends on anchors but the selected facts do not mention
      them, treat the bridge as unreliable: set bridge_ok=False and found=False (gating).

    Output:
    - Per-step metadata:
      {
        "step_index": int,
        "anchors": [...],
        "matched_anchors": [...],
        "bridge_ok": bool,
        "bridge_reason": str
      }
    - trace["bridge_manager"] includes a summary for logging/analysis.
    """

    def __init__(self, question: str, schema: Dict[str, Any]):
        self.question = question
        self.schema = schema or {}
        # Extract explicit entity names (and aliases if present) from the schema
        self.schema_anchor_map: Dict[str, str] = self._extract_schema_anchors(schema)
        self.schema_entities: List[str] = list(self.schema_anchor_map.values())
        # Entity-style answers from previous Gamma steps
        self.prev_answers: List[str] = []
        # Bridge metadata per step
        self.steps: List[Dict[str, Any]] = []

    def _extract_schema_anchors(self, schema: Dict[str, Any]) -> Dict[str, str]:
        """
        Build a canonical anchor map from schema.entities:
        key = lowercase surface form or alias; value = canonical surface form.
        """
        anchors: Dict[str, str] = {}
        for e in schema.get("entities", []) or []:
            name = (e.get("name") or "").strip()
            aliases = e.get("aliases") or e.get("alias") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            forms = [name] if name else []
            forms.extend([(a or "").strip() for a in aliases])
            for form in forms:
                if not form:
                    continue
                key = form.lower()
                # prefer the longest surface form as canonical if conflicts
                if key not in anchors or len(form) > len(anchors[key]):
                    anchors[key] = form
        return anchors

    def _extract_anchors_for_subquestion(self, subq: str) -> List[str]:
        """
        Extract anchor entities for the current refined_subquestion:
        - entity names from the schema that appear in this subquestion
        - explicit answers from previous steps that appear in this subquestion
        """
        anchors = set()
        sq_lower = (subq or "").lower()

        # Schema entities (canonicalized)
        for key, canonical in self.schema_anchor_map.items():
            if key and key in sq_lower:
                anchors.add(canonical)

        # Previous answers as potential intermediate anchors
        for ans in self.prev_answers:
            a = ans.strip()
            if a and a.lower() in sq_lower:
                anchors.add(a)

        return list(anchors)

    def update_step(
        self,
        step_index: int,
        refined_subq: str,
        gamma_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run after each Gamma step:
        - extract anchors from refined_subq
        - check whether selected_fact_texts mention these anchors
        - if anchors exist but none are matched:
            - set bridge_ok=False
            - annotate gamma_result["reasoning"]
            - force gamma_result["found"]=False (gating)
        - if this step is still found and answer is a string, save the answer for later steps
        """
        anchors = self._extract_anchors_for_subquestion(refined_subq)
        selected_texts = gamma_result.get("selected_fact_texts") or []
        big_text = " ".join(str(t) for t in selected_texts).lower()
        matched_anchors: List[str] = []
        if big_text:
            matched_anchors = [a for a in anchors if a.lower() in big_text]

        bridge_ok = True
        bridge_reason = ""

        if anchors:
            if gamma_result.get("found"):
                if not matched_anchors:
                    # Clear anchors exist but selected facts never mention them: gate this step
                    bridge_ok = False
                    bridge_reason = (
                        "Gamma selected facts do not mention any anchor entities from "
                        "schema/previous answers."
                    )
                    gamma_result["found"] = False
                    gamma_result.setdefault("reasoning", "")
                    if "BridgeManager" not in gamma_result["reasoning"]:
                        gamma_result["reasoning"] += (
                            " [BridgeManager: anchor entities not found in selected facts; "
                            "treat as not found]"
                        )
            else:
                # No answer found and the subquestion has anchors -> bridge broken here
                bridge_ok = False
                bridge_reason = (
                    "Gamma could not find an answer for a subquestion that depends on "
                    "anchor entities."
                )
        else:
            # No anchors detected: pass through without gating
            bridge_ok = True
            bridge_reason = ""

        # Track entity answers still treated as found for later steps
        ans = gamma_result.get("answer")
        if gamma_result.get("found") and isinstance(ans, str) and ans.strip():
            self.prev_answers.append(ans.strip())
            # Trim history to avoid unbounded growth
            if len(self.prev_answers) > 6:
                self.prev_answers = self.prev_answers[-6:]

        meta = {
            "step_index": step_index,
            "anchors": anchors,
            "matched_anchors": matched_anchors,
            "bridge_ok": bridge_ok,
            "bridge_reason": bridge_reason,
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
            "steps": list(self.steps),
            "summary": self.summary(),
        }


class ThetaAgent:
    """
    Theta agent:
    - Question schema analysis (task type, answer role, entities)
    - Plan subquestions (theta)
    - Dispatch gamma as needed
    - Dynamically rewrite later subquestions using previous answers
    - Integrate the final answer (with symbolic comparator hints)
    - Call ACC for lightweight self-check
    - Return full trace (for logging / analysis)
    """

    def __init__(
        self,
        dataset_name: str,
        llm_client: LLMClient,
        theta_llm_client: Optional[Any] = None,
    ):
        assert dataset_name in {"2wiki", "hotpotqa", "musique"}
        self.dataset_name = dataset_name
        # Gamma / ACC stay on the baseline llm_client; theta can optionally use
        # a separate client (e.g., Ollama) for planning/answering.
        self.gamma_llm = llm_client
        self.theta_llm = theta_llm_client or llm_client
        self.acc = ACCAgent(dataset_name=dataset_name, llm_client=self.gamma_llm)
        self.llm = self.theta_llm
        self.gamma = GammaAgent(dataset_name=dataset_name, llm_client=self.gamma_llm)

        # BridgeManager class (kept swappable for future implementations)
        self.bridge_manager_cls = BridgeManager

        # Cache the schema per question to avoid recomputation
        self._schema_question: Optional[str] = None
        self._schema: Optional[Dict[str, Any]] = None

        # Working memory cache (per question)
        self._wm_question: Optional[str] = None
        self._working_memory: Optional[Dict[str, Any]] = None

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

    def _normalize_entities_from_obj(self, obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Robustly extract a list of {name,type,role} entities from the raw JSON object:
        - If obj["entities"] is a list -> normalize each item.
        - If obj itself looks like a single entity (has name/type/role) -> wrap into a list.
        """
        entities: List[Dict[str, Any]] = []

        raw_entities = obj.get("entities", None)

        # Case 1: well-formed entities list
        if isinstance(raw_entities, list):
            for e in raw_entities:
                if not isinstance(e, dict):
                    continue
                name = str(e.get("name", "")).strip()
                etype = str(e.get("type", "other")).strip() or "other"
                role = str(e.get("role", "other")).strip() or "other"
                if name:
                    entities.append(
                        {
                            "name": name,
                            "type": etype,
                            "role": role,
                        }
                    )
        # Case 2: entities is a single dict
        elif isinstance(raw_entities, dict):
            name = str(raw_entities.get("name", "")).strip()
            etype = str(raw_entities.get("type", "other")).strip() or "other"
            role = str(raw_entities.get("role", "other")).strip() or "other"
            if name:
                entities.append(
                    {
                        "name": name,
                        "type": etype,
                        "role": role,
                    }
                )

        # Case 3: no "entities" field, but top-level object looks like an entity
        if not entities and isinstance(obj, dict):
            if {"name", "type", "role"}.issubset(set(obj.keys())):
                name = str(obj.get("name", "")).strip()
                etype = str(obj.get("type", "other")).strip() or "other"
                role = str(obj.get("role", "other")).strip() or "other"
                if name:
                    entities.append(
                        {
                            "name": name,
                            "type": etype,
                            "role": role,
                        }
                    )

        return entities

    def build_question_schema(self, question: str) -> Dict[str, Any]:
        """
        Build a lightweight Question Schema:
        - question_type: yes_no / comparison / fact / count / other
        - answer_form: yes_no / entity_name / number / date / span / other
        - focus: person / film / location / organization / other (hint only)
        - entities: [{"name": ..., "type": ..., "role": ...}, ...]
        - constraints / notes: task rules and other notes
        """
        prompt = f"""
        You are a QUESTION SCHEMA ANALYZER for a multi-hop QA system (THETA-GAMMA).

        Given the ORIGINAL QUESTION, you must produce a compact schema that describes:
        - The task type (yes/no, comparison, fact lookup, count, etc.).
        - The expected answer form (yes/no, entity name, number, date, etc.).
        - The main entities involved and their roles.
        - Any key constraints or multi-hop structure.

        CRITICAL:
        Return a SINGLE JSON object with keys:
        - "question_type": one of ["yes_no", "comparison", "fact", "count", "other"]
        - "answer_form": one of ["yes_no", "entity_name", "number", "date", "span", "other"]
        - "focus": short phrase for what the answer is about (e.g., "film", "person", "city", "organization", "generic").
        - "entities": array of objects, each:
                {{
                "name": string,              # surface form in the question if any
                "type": string,              # e.g., "person", "film", "city", "organization", "other"
                "role": string               # e.g., "subject", "candidate1", "candidate2", "pivot", "target"
                }}
        - "constraints": array of short strings describing important logical conditions.
        - "notes": short English description of how the question should be solved and what the final answer must look like.

        Example output:
        {{
        "question_type": "comparison",
        "answer_form": "entity_name",
        "focus": "film",
        "entities": [
            {{"name": "Charge It To Me", "type": "film", "role": "candidate1"}},
            {{"name": "Danger: Diabolik", "type": "film", "role": "candidate2"}},
            {{"name": "", "type": "person", "role": "director"}}
        ],
        "constraints": [
            "compare which director is younger",
            "answer must be the film whose director is younger"
        ],
        "notes": "Two-hop: find each film's director, then compare their ages and answer with the film title."
        }}

        ORIGINAL QUESTION:
        {question}

        Respond with JSON ONLY:
        """.strip()

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
            # Some models wrap the schema inside a 'schema' field; unwrap if present.
            if isinstance(obj.get("schema"), dict):
                obj = obj["schema"]
        except Exception:
            obj = {}

        # Fallback: try to parse a fenced ```json ... ``` block if keys are missing
        if not obj.get("question_type") or not obj.get("answer_form"):
            import re  # local import to avoid polluting module scope

            code_blocks = re.findall(r"```json(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
            for block in reversed(code_blocks):
                try:
                    cand = json.loads(block)
                    if isinstance(cand, dict):
                        if isinstance(cand.get("schema"), dict):
                            cand = cand["schema"]
                        obj = cand
                        break
                except Exception:
                    continue

        # Normalize entities from whatever shape the model returned
        entities = self._normalize_entities_from_obj(obj)

        # Normalize constraints to a list of strings
        raw_constraints = obj.get("constraints", [])
        if isinstance(raw_constraints, list):
            constraints = [str(c).strip() for c in raw_constraints if str(c).strip()]
        elif isinstance(raw_constraints, str) and raw_constraints.strip():
            constraints = [raw_constraints.strip()]
        else:
            constraints = []

        schema: Dict[str, Any] = {
            "question_type": obj.get("question_type", "other"),
            "answer_form": obj.get("answer_form", "other"),
            "focus": obj.get("focus", "generic"),
            "entities": entities,
            "constraints": constraints,
            "notes": obj.get("notes", ""),
            "raw_json": obj,
            "raw_text": raw_text,
        }
        return schema

    def _schema_summary(self, schema: Dict[str, Any]) -> str:
        """
        Convert the schema into short text for prompt hints.
        """
        qtype = schema.get("question_type", "other")
        ans_form = schema.get("answer_form", "other")
        focus = schema.get("focus", "generic")
        ents = schema.get("entities", [])
        cons = schema.get("constraints", [])
        notes = schema.get("notes", "")

        ent_lines = []
        for e in ents:
            name = (e.get("name") or "").strip()
            etype = (e.get("type") or "other").strip()
            role = (e.get("role") or "other").strip()
            ent_lines.append(f"- {role}: '{name}' ({etype})")

        ent_block = "\n".join(ent_lines) if ent_lines else "  (no explicit entities parsed)"
        cons_block = "\n".join(f"- {c}" for c in cons) if cons else "  (no explicit constraints)"
        notes_block = notes or "(no extra notes)"

        return (
            f"Question type: {qtype}\n"
            f"Answer form: {ans_form}\n"
            f"Answer focus: {focus}\n"
            f"Entities:\n{ent_block}\n"
            f"Constraints:\n{cons_block}\n"
            f"Notes: {notes_block}"
        )

    # ---------- Working Memory (explicit variable-style memory) ----------

    def _build_working_memory(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a structured WORKING_MEMORY object from the schema.
        This is the "variable-style" working memory:
        - entities have stable ids E1, E2, ... and canonical names/types/roles.
        """
        wm_entities: List[Dict[str, Any]] = []
        for i, e in enumerate(schema.get("entities", []) or []):
            wm_entities.append(
                {
                    "id": f"E{i + 1}",
                    "name": (e.get("name") or "").strip(),
                    "type": (e.get("type") or "other").strip() or "other",
                    "role": (e.get("role") or "other").strip() or "other",
                }
            )

        wm = {
            "question_type": schema.get("question_type", "other"),
            "answer_form": schema.get("answer_form", "other"),
            "focus": schema.get("focus", "generic"),
            "entities": wm_entities,
            "constraints": schema.get("constraints", []),
        }
        return wm

    def _ensure_working_memory(self, question: str) -> Dict[str, Any]:
        """
        Ensure that working memory is built and cached for this question.
        """
        if self._working_memory is not None and self._wm_question == question:
            return self._working_memory
        schema = self._ensure_schema(question)
        wm = self._build_working_memory(schema)
        self._working_memory = wm
        self._wm_question = question
        return wm

    # ---------- Entity canonicalization helpers ----------

    def _build_entity_canonical_map(self, schema: Dict[str, Any]) -> Dict[str, str]:
        """
        Build a canonical map from schema.entities:
        - key: lowercase surface forms (name / alias)
        - value: canonical name from the schema (prefer longer/more complete names)
        """
        mapping: Dict[str, str] = {}
        for e in schema.get("entities", []) or []:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            forms = {name}
            # Include aliases when provided (raw_json fields)
            aliases = e.get("aliases") or e.get("alias") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            for a in aliases:
                a = (a or "").strip()
                if a:
                    forms.add(a)

            for form in forms:
                key = form.lower()
                # Keep the longer name if the key collides
                if key in mapping:
                    if len(name) > len(mapping[key]):
                        mapping[key] = name
                else:
                    mapping[key] = name
        return mapping

    def _canonicalize_with_schema(self, text: str, schema: Dict[str, Any]) -> str:
        """
        Normalize entity mentions in text to canonical forms from the schema.
        For example:
        - "the game" / "The Game" / "THE GAME" -> "The Game"
        - Prevent the LLM from drifting "The Game" into a vague mention
        """
        if not text:
            return text
        mapping = self._build_entity_canonical_map(schema)
        if not mapping:
            return text

        # Replace from longest to shortest keys to avoid truncating longer entities
        items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
        result = text
        for form_lower, canonical in items:
            if not form_lower:
                continue
            # \b + IGNORECASE keeps the schema surface form instead of LLM paraphrases
            pattern = re.compile(rf"\b{re.escape(form_lower)}\b", flags=re.IGNORECASE)
            result = pattern.sub(canonical, result)
        return result

    # ---------- Stage 1: theta decomposes the question ----------

    def plan_subquestions(self, question: str, max_steps: int = 4) -> List[str]:
        # Ensure schema & working memory are prepared
        schema = self._ensure_schema(question)
        wm = self._ensure_working_memory(question)
        schema_summary = self._schema_summary(schema)
        wm_json = json.dumps(wm, ensure_ascii=False, indent=2)

        prompt = f"""
You are the THETA agent in a dual-agent multi-hop reasoning system.

You have two inputs:
1) QUESTION_SCHEMA_JSON: a compact analysis of the question.
2) WORKING_MEMORY: a variable-style memory that contains canonical entities from the question.

WORKING_MEMORY:
- You MUST treat WORKING_MEMORY.entities as the authoritative list of entities.
- Each entity has an id (E1, E2, ...) and a canonical "name", "type", "role".
- When you write SUBQUESTIONS, you MUST NOT invent or paraphrase entity names.
  Instead, you MUST copy the exact "name" field from WORKING_MEMORY.entities[*].
  For example, if WORKING_MEMORY.entities has:
    {{"id": "E1", "name": "The Game", "type": "film", "role": "subject"}}
  then in your subquestions you MUST use "The Game" exactly (same casing and spacing),
  NOT "the game", NOT "that game", NOT any paraphrase.

TASK:
Decompose the ORIGINAL QUESTION into 2 to {max_steps} ordered SUBQUESTIONS.

Guidelines:
- Each subquestion should be a simple fact-based question.
- The sequence of subquestions should follow the schema:
  - Respect the question_type and constraints.
  - Ensure that the LAST subquestion directly prepares the information needed
    to produce the final answer in the required ANSWER_FORM.

Do NOT answer the question here; only plan subquestions.

QUESTION_SCHEMA_JSON (for reference, do NOT rewrite it):
{json.dumps(schema, ensure_ascii=False, indent=2)}

QUESTION_SCHEMA_SUMMARY (for human readability):
{schema_summary}

WORKING_MEMORY (authoritative, do NOT modify):
{wm_json}

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
    "What country is that director from?"
  ]
}}

ORIGINAL QUESTION:
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
            # Canonicalize entity names with the schema to avoid drift
            subs = [self._canonicalize_with_schema(s, schema) for s in subs]
            return subs[:max_steps]
        except Exception:
            # Fallback: still canonicalize to keep entity names stable
            return [self._canonicalize_with_schema(question, schema)]

    def refine_subquestion(
        self,
        question: str,
        planned_subquestion: str,
        previous_steps: List[Dict[str, Any]],
    ) -> str:
        """
        Rewrite the planned_subquestion using earlier Gamma answers to be more specific,
        while respecting the QUESTION_SCHEMA and WORKING_MEMORY (entity alignment).
        If no reliable answer yet (found=false or answer empty), keep it unchanged.
        """
        if not previous_steps:
            return planned_subquestion

        schema = self._ensure_schema(question)
        wm = self._ensure_working_memory(question)
        schema_summary = self._schema_summary(schema)
        wm_json = json.dumps(wm, ensure_ascii=False, indent=2)

        # Summarize previous steps
        lines = []
        for step in previous_steps:
            gres = step.get("gamma_result", {}) or {}
            lines.append(
                f"step {step.get('step_index')}: subquestion='{step.get('refined_subquestion', step.get('subquestion'))}', "
                f"found={gres.get('found')}, answer={gres.get('answer')}"
            )
        history_block = "\n".join(lines)

        prompt = f"""
You are the THETA agent refining the next subquestion in a multi-hop QA task.

You have WORKING_MEMORY that encodes entities as variables (E1, E2, ...),
and QUESTION_SCHEMA that describes the overall task.

RULES:
- Use QUESTION_SCHEMA and WORKING_MEMORY to preserve entity alignment and answer role.
- When mentioning entities in the refined subquestion, ALWAYS use the exact "name"
  from WORKING_MEMORY.entities[*].name. Do NOT change "The Game" into "the game",
  and do NOT drop qualifiers like "(film)".
- Use PREVIOUS STEPS (gamma answers) to rewrite the NEXT_SUBQUESTION to be as specific as possible
  (e.g., replace pronouns like "that director" with the actual name if confidently known),
  BUT you MUST still respect WORKING_MEMORY.entities for the canonical surface forms.
- If previous steps did NOT find a reliable answer (found=false or empty answer), KEEP the next subquestion unchanged.

QUESTION_SCHEMA_SUMMARY:
{schema_summary}

WORKING_MEMORY (authoritative, do NOT modify):
{wm_json}

PREVIOUS STEPS:
{history_block}

NEXT_SUBQUESTION (planned):
{planned_subquestion}

CRITICAL OUTPUT:
- Respond with a SINGLE JSON object:
  - "subquestion": string (the refined subquestion to ask gamma)

Respond with JSON ONLY:
""".strip()

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
            refined_raw = obj.get("subquestion", "").strip()
            refined = refined_raw if refined_raw else planned_subquestion
            # Canonicalize again with the schema so entity names stay unchanged
            refined = self._canonicalize_with_schema(refined, schema)
            return refined
        except Exception:
            # Fallback: canonicalize the planned subquestion
            return self._canonicalize_with_schema(planned_subquestion, schema)

    # ---------- Stage 2: symbolic comparator + answer integration ----------

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
        Lightweight symbolic comparator / numeric hints:
        - detect comparative keywords (older/younger/earlier/later/before/after).
        - extract numeric cues (years) from gamma answers and selected facts.
        - provide a compact hint string for LLM to avoid small comparison slips.
        """
        comparative_keywords = [
            "older", "younger", "earlier", "later", "before", "after",
            "first", "second", "younger than", "older than", "earlier than", "later than",
        ]
        q_lower = question.lower()
        hits = [kw for kw in comparative_keywords if kw in q_lower]
        is_comp = bool(hits)

        entities = []
        for step in gamma_results:
            gres = step.get("gamma_result", {}) or {}
            answer = str(gres.get("answer", "") or "")
            years = set(self._extract_years(answer))
            for ft in gres.get("selected_fact_texts", []) or []:
                years.update(self._extract_years(str(ft)))
            entities.append({
                "step": step.get("step_index"),
                "subq": step.get("refined_subquestion") or step.get("subquestion"),
                "answer": answer,
                "years": sorted(years),
            })

        # Build a human-readable summary to feed into prompts
        lines = []
        if is_comp:
            lines.append(f"Comparative keywords detected: {', '.join(hits)}.")
        else:
            lines.append("No obvious comparative keywords detected.")
        for ent in entities:
            ytxt = ", ".join(str(y) for y in ent["years"]) if ent["years"] else "none"
            lines.append(f"Step {ent['step']}: answer='{ent['answer']}', years=[{ytxt}].")
        summary = "\n".join(lines)

        return {
            "is_comparative": is_comp,
            "keywords": hits,
            "entities": entities,
            "summary": summary,
        }

    def integrate_answer(
        self,
        question: str,
        gamma_results: List[Dict[str, Any]],
        comparator_summary: str = "",
    ) -> Dict[str, Any]:
        """
        Feed the question schema, gamma results, and symbolic comparator hints to the LLM,
        with explicit requirements:
        - Answer form must follow schema.answer_form.
        - If yes/no, answer strictly yes or no.
        - If asking for a specific entity (film/person/etc.), output that entity, not an intermediate node.
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

        prompt = f"""
You are the THETA agent. You must produce a FINAL ANSWER to the ORIGINAL QUESTION
based on GAMMA's evidence, the QUESTION_SCHEMA, and the SYMBOLIC_COMPARATOR hints.

STRICT RULES:
1. Respect QUESTION_SCHEMA.answer_form:
   - If "yes_no": the answer MUST be exactly "yes" or "no".
   - If "entity_name": the answer MUST be the name of the required entity (film/person/city/etc.),
     NOT a description, NOT an explanation, and NOT an intermediate node.
   - If "number" or "date": output ONLY the number or date required.
2. Your reasoning text should match the final answer logically.
3. Do NOT change the task type. If the question asks for a FILM, do NOT answer with a PERSON.

CRITICAL OUTPUT:
- Respond with a SINGLE JSON object:
  - "answer": string
  - "reasoning": string (1-3 sentences in English)

QUESTION_SCHEMA_SUMMARY:
{schema_summary}

SYMBOLIC_COMPARATOR_HINTS:
{comparator_summary}

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

        # 0) Build the Question Schema first so later steps share the same frame
        try:
            schema = self._ensure_schema(question)
            wm = self._ensure_working_memory(question)
        except Exception as e:
            # In some environments schema calls may fail (blocked or network); skip the sample to keep the run going
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
        schema_summary = self._schema_summary(schema)  # logging only

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
            # If planning hits 403/network issues, skip this sample
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
                schema=schema,  # Pass schema so Gamma can prioritize titles/anchors
            )
            # BridgeManager: check anchor alignment for this step and gate if needed
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
        comparator_summary = comparator.get("summary", "") if comparator.get("is_comparative") else ""

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
            "working_memory": wm,
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

        llm_calls = getattr(self.llm, "call_log", [])

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
