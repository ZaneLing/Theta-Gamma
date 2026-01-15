# theta_gpt35.py
# PFC: orchestrates HPC calls, applies ACC self-check,
# question schema + slot graph, and returns trace.

from typing import Any, Dict, List, Optional
import re
import json
from pathlib import Path
import os
from requests.exceptions import HTTPError, SSLError, ConnectionError, Timeout

from gamma_gpt35 import LLMClient, HPC, ACC, extract_json_from_text
from dotenv import load_dotenv
load_dotenv()

class BridgeManager:
    """
    BridgeManager: lightweight anchor-alignment checker.

    Goals:
    - Use entities from the question schema plus explicit entity answers from earlier
      HPC steps as anchors for each subquestion.
    - Check whether HPC-selected facts actually mention these anchors.
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
        # Entity-style answers from previous HPC steps
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
        Run after each HPC step:
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
                        "HPC selected facts do not mention any anchor entities from "
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
                    "HPC could not find an answer for a subquestion that depends on "
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


class PFC:
    """
    PFC:
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
        # HPC / ACC stay on the baseline llm_client; theta can optionally use
        # a separate client (e.g., Ollama) for planning/answering.
        self.gamma_llm = llm_client
        self.theta_llm = theta_llm_client or llm_client
        self._apply_openrouter_config(self.gamma_llm)
        if self.theta_llm is not self.gamma_llm:
            self._apply_openrouter_config(self.theta_llm)
        self.acc = ACC(dataset_name=dataset_name, llm_client=self.gamma_llm)
        self.llm = self.theta_llm
        self.hpc = HPC(dataset_name=dataset_name, llm_client=self.gamma_llm)

        # BridgeManager class (kept swappable for future implementations)
        self.bridge_manager_cls = BridgeManager

        # Cache the schema per question to avoid recomputation
        self._schema_question: Optional[str] = None
        self._schema: Optional[Dict[str, Any]] = None

        # Working memory cache (per question)
        self._wm_question: Optional[str] = None
        self._working_memory: Optional[Dict[str, Any]] = None

        # Global theta memory (per question)
        self._global_theta_memory: Optional[Dict[str, Any]] = None

    def _apply_openrouter_config(self, llm_client: Any) -> None:
        """
        Apply OPENROUTER_* environment settings to LLMClient instances.
        """
        if not isinstance(llm_client, LLMClient):
            return
        base_url = os.getenv("OPENROUTER_API_BASE_URL", "").strip()
        if base_url:
            base_url = base_url.rstrip("/")
            if not base_url.endswith("/chat/completions"):
                base_url = base_url + "/chat/completions"
            llm_client.api_url = base_url

        model = os.getenv("OPENROUTER_MODEL", "").strip()
        if model:
            aliases = getattr(llm_client, "MODEL_ALIASES", {})
            llm_client.model_name = aliases.get(model, model)

        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            llm_client.api_key = api_key

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
        You are a QUESTION SCHEMA ANALYZER for a multi-hop QA system (PFC-HPC).

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
                "rhythm": "PFC",
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

    def _infer_core_entity(self, subq: str, schema: Dict[str, Any]) -> str:
        """
        Heuristic fallback: pick the first schema entity mentioned in the subquestion.
        """
        sq_lower = (subq or "").lower()
        for e in schema.get("entities", []) or []:
            name = (e.get("name") or "").strip()
            if name and name.lower() in sq_lower:
                return name
        return ""

    def _infer_expected_answer_type(self, subq: str) -> str:
        """
        Heuristic fallback: infer answer type from wh-words.
        """
        sq = (subq or "").strip().lower()
        if sq.startswith("who "):
            return "person"
        if sq.startswith("where "):
            return "location"
        if sq.startswith("when "):
            return "date"
        if sq.startswith("how many ") or sq.startswith("how much "):
            return "number"
        if sq.startswith("what year "):
            return "date"
        if sq.startswith("which film ") or sq.startswith("which movie "):
            return "film"
        return "unknown"

    def _build_global_theta_memory(
        self,
        question: str,
        sub_items: List[Any],
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build global theta memory with placeholders for sub-answers and completion flags.
        """
        entries: List[Dict[str, Any]] = []
        for item in sub_items:
            if isinstance(item, dict):
                subq = str(item.get("subquestion", "")).strip()
                core_entity = str(item.get("core_entity", "")).strip()
                expected_type = str(item.get("expected_answer_type", "")).strip()
            else:
                subq = str(item).strip()
                core_entity = ""
                expected_type = ""

            if not subq:
                continue

            subq = self._canonicalize_with_schema(subq, schema)
            if not core_entity:
                core_entity = self._infer_core_entity(subq, schema)
            core_entity = self._canonicalize_with_schema(core_entity, schema).strip()
            if not expected_type:
                expected_type = self._infer_expected_answer_type(subq)

            entries.append(
                {
                    "sub_question": subq,
                    "core_entity": core_entity,
                    "expected_answer_type": expected_type or "unknown",
                    "sub_answer": "",
                    "completion_flag": 0,
                }
            )

        return {
            "main_question": question,
            "sub_questions": entries,
        }

    def _write_global_theta_memory(
        self,
        memory: Dict[str, Any],
        dataset_name: str,
        example_index: int,
    ) -> str:
        base_dir = Path(__file__).resolve().parent.parent / "Memory" / "global_theta_memory"
        base_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{dataset_name}_{example_index:05d}.json"
        path = base_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=True, indent=2)
        return str(path)

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

    def _collect_found_answers(self, previous_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collect reliable answers from previous steps in order.
        """
        found_answers: List[Dict[str, Any]] = []
        for step in previous_steps:
            gres = step.get("gamma_result", {}) or {}
            answer = gres.get("answer")
            if gres.get("found") and isinstance(answer, str) and answer.strip():
                found_answers.append(
                    {
                        "step_index": step.get("step_index"),
                        "answer": answer.strip(),
                        "subquestion": step.get("refined_subquestion")
                        or step.get("subquestion")
                        or "",
                    }
                )
        return found_answers

    def _latest_found_answer(self, previous_steps: List[Dict[str, Any]]) -> str:
        """
        Return the most recent reliable answer, or empty string.
        """
        for step in reversed(previous_steps):
            gres = step.get("gamma_result", {}) or {}
            answer = gres.get("answer")
            if gres.get("found") and isinstance(answer, str) and answer.strip():
                return answer.strip()
        return ""

    def _replace_placeholders_with_answer(self, subq: str, answer: str) -> str:
        """
        Replace ambiguous placeholders (e.g., "that director") with a concrete answer.
        """
        if not subq or not answer:
            return subq
        roles = [
            "director", "actor", "writer", "author", "composer", "artist",
            "person", "politician", "president", "prime minister", "king", "queen",
            "city", "country", "state", "province", "company", "organization",
            "team", "club", "school", "university",
            "river", "mountain", "island",
            "film", "movie", "book", "album", "song", "band", "work",
            "show", "series", "episode", "season",
        ]
        role_pattern = "|".join(re.escape(r) for r in roles)
        pattern = re.compile(
            rf"\b(?:that|this|the)\s+(?:{role_pattern})\b(?!\s+of\b)",
            flags=re.IGNORECASE,
        )
        return pattern.sub(answer, subq)

    def _extract_comparison_candidates(
        self,
        schema: Dict[str, Any],
        wm: Dict[str, Any],
    ) -> List[str]:
        """
        Extract candidate entities for comparison tasks from schema or working memory.
        """
        candidates: List[str] = []
        for e in schema.get("entities", []) or []:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            role = (e.get("role") or "").strip().lower()
            if "candidate" in role or "option" in role or "choice" in role:
                candidates.append(name)

        if len(candidates) < 2:
            for e in wm.get("entities", []) or []:
                name = (e.get("name") or "").strip()
                if name and name not in candidates:
                    candidates.append(name)
                if len(candidates) >= 2:
                    break

        seen = set()
        ordered: List[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        return ordered

    def _type_guidance(self, schema: Dict[str, Any], wm: Dict[str, Any]) -> str:
        """
        Return type-specific prompt guidance for decomposition.
        """
        qtype = (schema.get("question_type") or "other").strip().lower()
        if qtype == "comparison":
            candidates = self._extract_comparison_candidates(schema, wm)
            if candidates:
                cand_line = "Candidates: " + ", ".join(f"'{c}'" for c in candidates[:2])
            else:
                cand_line = "Candidates: use the two comparison entities in WORKING_MEMORY."
            return "\n".join(
                [
                    "COMPARISON RULES:",
                    "- Ask for the target attribute of each candidate separately.",
                    "- Add a final subquestion that compares them and yields the final choice.",
                    "- Ensure each candidate appears at least once.",
                    f"- {cand_line}",
                ]
            )
        if qtype == "count":
            return "\n".join(
                [
                    "COUNT RULES:",
                    "- Identify the items to be counted.",
                    "- Add a final subquestion that computes the count.",
                ]
            )
        if qtype == "yes_no":
            return "\n".join(
                [
                    "YES/NO RULES:",
                    "- Ask for the supporting facts.",
                    "- Add a final subquestion that decides yes or no.",
                ]
            )
        return ""

    def _looks_like_comparison(self, text: str, candidates: List[str]) -> bool:
        """
        Heuristic check for a comparison-style subquestion.
        """
        t = (text or "").lower()
        keywords = [
            "compare", "comparison", "which", "older", "younger", "earlier", "later",
            "before", "after", "same", "different", "more", "less", "greater", "fewer",
            "higher", "lower", "longer", "shorter", "bigger", "smaller",
        ]
        if any(k in t for k in keywords):
            return True
        if len(candidates) >= 2:
            c0 = candidates[0].lower()
            c1 = candidates[1].lower()
            if c0 in t and c1 in t:
                return True
        return False

    def _validate_comparison_plan(
        self,
        items: List[Dict[str, str]],
        candidates: List[str],
        min_steps: int,
    ) -> Optional[str]:
        """
        Return an issue string if the comparison plan is incomplete, else None.
        """
        subqs = [
            (item.get("subquestion") or "").strip()
            for item in items
            if isinstance(item, dict)
        ]
        if len(subqs) < max(min_steps, 3):
            return "comparison tasks require at least 3 subquestions"

        if len(candidates) >= 2:
            missing = []
            for c in candidates[:2]:
                if not any(c.lower() in (s or "").lower() for s in subqs):
                    missing.append(c)
            if missing:
                return "missing candidate subquestion(s) for: " + ", ".join(missing)

        if subqs:
            last = subqs[-1]
            if not self._looks_like_comparison(last, candidates):
                return "final subquestion does not look like a comparison step"

        return None

    def _sanitize_subquestion_items(
        self,
        items: List[Any],
        schema: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Normalize subquestion items into dicts, canonicalize entity mentions,
        and drop empty or duplicate subquestions.
        """
        sanitized: List[Dict[str, str]] = []
        seen: set = set()
        for item in items:
            if isinstance(item, dict):
                subq = str(item.get("subquestion", "")).strip()
                core_entity = str(item.get("core_entity", "")).strip()
                expected_type = str(item.get("expected_answer_type", "")).strip()
            else:
                subq = str(item).strip()
                core_entity = ""
                expected_type = ""

            if not subq:
                continue

            subq = self._canonicalize_with_schema(subq, schema)
            key = subq.lower()
            if key in seen:
                continue
            seen.add(key)

            sanitized.append(
                {
                    "subquestion": subq,
                    "core_entity": core_entity,
                    "expected_answer_type": expected_type,
                }
            )
        return sanitized

    def _subquestion_repair_prompt(
        self,
        question: str,
        schema: Dict[str, Any],
        wm: Dict[str, Any],
        items: List[Dict[str, str]],
        max_steps: int,
        min_steps: int,
        issue_hint: str,
    ) -> str:
        wm_json = json.dumps(wm, ensure_ascii=False, indent=2)
        items_json = json.dumps({"subquestions": items}, ensure_ascii=False, indent=2)
        type_guidance = self._type_guidance(schema, wm)
        type_block = f"\n{type_guidance}\n" if type_guidance else ""

        return f"""
You are PFC.

The current decomposition is invalid or incomplete: {issue_hint}

Rewrite into {min_steps} to {max_steps} ordered, step-by-step subquestions.
Each subquestion should ask for one missing fact.

Rules:
- Use exact entity names from WORKING_MEMORY.entities when referring to known entities.
- If a step depends on a previous answer, you may use a short placeholder (e.g., "that person").
- The last subquestion should enable the final answer.
{type_block}

CANDIDATE_SUBQUESTIONS (invalid):
{items_json}

QUESTION_SCHEMA_JSON:
{json.dumps(schema, ensure_ascii=False, indent=2)}

WORKING_MEMORY (authoritative, do NOT modify):
{wm_json}

ORIGINAL QUESTION:
{question}

Return JSON ONLY with:
- "subquestions": array of {min_steps} to {max_steps} objects:
  - "subquestion": string
  - "core_entity": string (use WORKING_MEMORY.entities[*].name when applicable)
  - "expected_answer_type": string (e.g., person, location, date, number, film, organization)

Respond with JSON ONLY:
""".strip()

    def _split_single_subquestion_prompt(
        self,
        question: str,
        schema: Dict[str, Any],
        wm: Dict[str, Any],
        subquestion: str,
        max_steps: int,
        min_steps: int,
    ) -> str:
        wm_json = json.dumps(wm, ensure_ascii=False, indent=2)
        type_guidance = self._type_guidance(schema, wm)
        type_block = f"\n{type_guidance}\n" if type_guidance else ""

        return f"""
You are PFC.

Split the SINGLE_SUBQUESTION into {min_steps} to {max_steps} ordered, step-by-step subquestions.
Each subquestion should ask for one missing fact.

Rules:
- Use exact entity names from WORKING_MEMORY.entities when referring to known entities.
- If a step depends on a previous answer, you may use a short placeholder (e.g., "that person").
- The last subquestion should enable the final answer.
{type_block}

SINGLE_SUBQUESTION:
{subquestion}

QUESTION_SCHEMA_JSON:
{json.dumps(schema, ensure_ascii=False, indent=2)}

WORKING_MEMORY (authoritative, do NOT modify):
{wm_json}

ORIGINAL QUESTION:
{question}

Return JSON ONLY with:
- "subquestions": array of {min_steps} to {max_steps} objects:
  - "subquestion": string
  - "core_entity": string (use WORKING_MEMORY.entities[*].name when applicable)
  - "expected_answer_type": string (e.g., person, location, date, number, film, organization)

Respond with JSON ONLY:
""".strip()

    # ---------- Stage 1: theta decomposes the question ----------

    def plan_subquestions(self, question: str, max_steps: int = 4) -> List[str]:
        # Ensure schema & working memory are prepared
        schema = self._ensure_schema(question)
        wm = self._ensure_working_memory(question)
        wm_json = json.dumps(wm, ensure_ascii=False, indent=2)
        question_type = (schema.get("question_type") or "other").strip().lower()
        min_steps = 2 if max_steps >= 2 else max_steps
        if question_type == "comparison":
            min_steps = max(min_steps, 3)
        type_guidance = self._type_guidance(schema, wm)
        type_block = f"\n{type_guidance}\n" if type_guidance else ""

        prompt = f"""
You are PFC.

Decompose the ORIGINAL QUESTION into {min_steps} to {max_steps} ordered, step-by-step subquestions.
Each subquestion should ask for one missing fact. Do NOT answer the question.

Rules:
- Use exact entity names from WORKING_MEMORY.entities when referring to known entities.
- If a step depends on a previous answer, you may use a short placeholder (e.g., "that person").
- The last subquestion should enable the final answer.
{type_block}

QUESTION_SCHEMA_JSON:
{json.dumps(schema, ensure_ascii=False, indent=2)}

WORKING_MEMORY (authoritative, do NOT modify):
{wm_json}

ORIGINAL QUESTION:
{question}

Return JSON ONLY with:
- "subquestions": array of {min_steps} to {max_steps} objects:
  - "subquestion": string
  - "core_entity": string (use WORKING_MEMORY.entities[*].name when applicable)
  - "expected_answer_type": string (e.g., person, location, date, number, film, organization)

Respond with JSON ONLY:
""".strip()

        raw_text = self.llm.generate(
            prompt,
            meta={
                "rhythm": "PFC",
                "kind": "plan_subquestions",
                "dataset": self.dataset_name,
            },
            temperature=0.3,
        )

        try:
            obj = extract_json_from_text(raw_text)
            items = obj.get("subquestions", [])
            if not isinstance(items, list):
                items = []

            items = self._sanitize_subquestion_items(items, schema)
            candidates = []
            if question_type == "comparison":
                candidates = self._extract_comparison_candidates(schema, wm)

            # Enforce multi-subquestion planning with a repair pass if needed
            issue_parts: List[str] = []
            if len(items) < min_steps:
                issue_parts.append(
                    f"only {len(items)} subquestion(s); must be at least {min_steps}"
                )
            if question_type == "comparison":
                cmp_issue = self._validate_comparison_plan(items, candidates, min_steps)
                if cmp_issue:
                    issue_parts.append(cmp_issue)
            if issue_parts:
                issue = "; ".join(issue_parts)
                repair_prompt = self._subquestion_repair_prompt(
                    question=question,
                    schema=schema,
                    wm=wm,
                    items=items,
                    max_steps=max_steps,
                    min_steps=min_steps,
                    issue_hint=issue,
                )
                repaired_raw = self.llm.generate(
                    repair_prompt,
                    meta={
                        "rhythm": "PFC",
                        "kind": "plan_subquestions_repair",
                        "dataset": self.dataset_name,
                    },
                    temperature=0.2,
                )
                try:
                    repaired_obj = extract_json_from_text(repaired_raw)
                    repaired_items = repaired_obj.get("subquestions", [])
                    if isinstance(repaired_items, list):
                        items = self._sanitize_subquestion_items(repaired_items, schema)
                except Exception:
                    pass

            # Final safeguard: ensure at least min_steps by forcing a second repair pass
            issue_parts = []
            if len(items) < min_steps:
                issue_parts.append(
                    f"still fewer than {min_steps} subquestions after repair"
                )
            if question_type == "comparison":
                cmp_issue = self._validate_comparison_plan(items, candidates, min_steps)
                if cmp_issue:
                    issue_parts.append(cmp_issue)
            if issue_parts:
                issue = "; ".join(issue_parts)
                force_prompt = self._subquestion_repair_prompt(
                    question=question,
                    schema=schema,
                    wm=wm,
                    items=items,
                    max_steps=max_steps,
                    min_steps=min_steps,
                    issue_hint=issue,
                )
                force_raw = self.llm.generate(
                    force_prompt,
                    meta={
                        "rhythm": "PFC",
                        "kind": "plan_subquestions_force",
                        "dataset": self.dataset_name,
                    },
                    temperature=0.1,
                )
                try:
                    force_obj = extract_json_from_text(force_raw)
                    force_items = force_obj.get("subquestions", [])
                    if isinstance(force_items, list):
                        items = self._sanitize_subquestion_items(force_items, schema)
                except Exception:
                    pass

            # If we still only have one subquestion, explicitly split it into multiple hops
            if len(items) < min_steps and min_steps >= 2:
                seed_subq = items[0]["subquestion"] if items else question
                split_prompt = self._split_single_subquestion_prompt(
                    question=question,
                    schema=schema,
                    wm=wm,
                    subquestion=seed_subq,
                    max_steps=max_steps,
                    min_steps=min_steps,
                )
                split_raw = self.llm.generate(
                    split_prompt,
                    meta={
                        "rhythm": "PFC",
                        "kind": "plan_subquestions_split",
                        "dataset": self.dataset_name,
                    },
                    temperature=0.1,
                )
                try:
                    split_obj = extract_json_from_text(split_raw)
                    split_items = split_obj.get("subquestions", [])
                    if isinstance(split_items, list):
                        items = self._sanitize_subquestion_items(split_items, schema)
                except Exception:
                    pass

            # Hard fallback: enforce multiple subquestions even if repairs fail
            if len(items) < min_steps and min_steps >= 2:
                seed_subq = items[0]["subquestion"] if items else question
                items = self._sanitize_subquestion_items(
                    [
                        {"subquestion": seed_subq},
                        {
                            "subquestion": (
                                f"Using the answer from the previous step, {question}"
                            )
                        },
                    ],
                    schema,
                )

            if not items:
                items = [{"subquestion": question}]

            # Build global theta memory from planned subquestions
            global_memory = self._build_global_theta_memory(question, items, schema)
            self._global_theta_memory = global_memory

            # Extract subquestion strings
            subquestions = [
                str(entry.get("sub_question", "")).strip()
                for entry in global_memory.get("sub_questions", [])
                if isinstance(entry, dict) and str(entry.get("sub_question", "")).strip()
            ]
            if not subquestions:
                subquestions = [question]

            return subquestions[:max_steps]
        except Exception:
            # Fallback: still canonicalize to keep entity names stable
            fallback_subq = self._canonicalize_with_schema(question, schema)
            self._global_theta_memory = self._build_global_theta_memory(
                question,
                [fallback_subq],
                schema,
            )
            return [fallback_subq]

    def refine_subquestion(
        self,
        question: str,
        planned_subquestion: str,
        previous_steps: List[Dict[str, Any]],
    ) -> str:
        """
        Rewrite the planned_subquestion using earlier HPC answers to be more specific,
        while respecting the QUESTION_SCHEMA and WORKING_MEMORY (entity alignment).
        If no reliable answer yet (found=false or answer empty), keep it unchanged.
        """
        if not previous_steps:
            return planned_subquestion

        schema = self._ensure_schema(question)
        wm = self._ensure_working_memory(question)
        wm_json = json.dumps(wm, ensure_ascii=False, indent=2)

        found_answers = self._collect_found_answers(previous_steps)
        if not found_answers:
            return planned_subquestion
        last_answer = found_answers[-1]["answer"]

        history_lines = []
        for entry in found_answers:
            history_lines.append(
                f"step {entry.get('step_index')}: answer='{entry.get('answer')}', "
                f"subquestion='{entry.get('subquestion')}'"
            )
        history_block = "\n".join(history_lines)

        prompt = f"""
You are PFC.

Refine NEXT_SUBQUESTION using PREVIOUS ANSWERS so it is more specific.

Rules:
- Replace ambiguous placeholders (e.g., "that director", "that person", "that film")
  with a concrete answer from PREVIOUS ANSWERS when relevant.
- Use exact entity names from WORKING_MEMORY.entities for known entities.
- If a needed answer is missing, return NEXT_SUBQUESTION unchanged.

WORKING_MEMORY (authoritative, do NOT modify):
{wm_json}

PREVIOUS ANSWERS:
{history_block}

NEXT_SUBQUESTION (planned):
{planned_subquestion}

Return JSON ONLY with:
- "subquestion": string

Respond with JSON ONLY:
""".strip()

        raw_text = self.llm.generate(
            prompt,
            meta={
                "rhythm": "PFC",
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
            refined = self._replace_placeholders_with_answer(refined, last_answer)
            return refined
        except Exception:
            # Fallback: canonicalize the planned subquestion
            fallback = self._canonicalize_with_schema(planned_subquestion, schema)
            return self._replace_placeholders_with_answer(fallback, last_answer)

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
You are PFC. You must produce a FINAL ANSWER to the ORIGINAL QUESTION
based on HPC's evidence, the QUESTION_SCHEMA, and the SYMBOLIC_COMPARATOR hints.

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

HPC_RESULTS:
{gamma_summary}

Respond with JSON ONLY:
""".strip()

        raw_text = self.llm.generate(
            prompt,
            meta={
                "rhythm": "PFC",
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
        global_memory = self._global_theta_memory or self._build_global_theta_memory(
            question,
            subquestions,
            schema,
        )
        memory_path = self._write_global_theta_memory(
            global_memory,
            dataset_name=self.dataset_name,
            example_index=example_index,
        )

        gamma_results: List[Dict[str, Any]] = []
        executed_subquestions: List[str] = []
        memory_items = global_memory.get("sub_questions", []) if isinstance(global_memory, dict) else []

        gamma_call_count = 0
        gamma_success_count = 0
        for step_idx, subq in enumerate(subquestions, start=1):
            core_entity = ""
            if isinstance(memory_items, list) and (step_idx - 1) < len(memory_items):
                entry = memory_items[step_idx - 1]
                if isinstance(entry, dict):
                    core_entity = str(entry.get("core_entity", "")).strip()
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
            gr = self.hpc.answer_subquestion(
                example=example,
                subquestion=refined_subq,
                call_id=call_id,
                core_entity=core_entity,
                schema=schema,  # Pass schema so HPC can prioritize titles/anchors
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
                    "core_entity": core_entity,
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
            "global_theta_memory": global_memory,
            "global_theta_memory_path": memory_path,
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
            "theta_answer": initial_answer,     # PFC answer before ACC
            "predicted_answer": predicted_answer,
            "theta_gamma_trace": trace,
            "llm_calls": llm_calls,
        }
        return result
