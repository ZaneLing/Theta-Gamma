# theta_gpt35.py
# Theta agent: orchestrates Gamma calls, applies ACC self-check, question schema, and returns trace.

from typing import Any, Dict, List, Optional
import re
from requests.exceptions import HTTPError, SSLError, ConnectionError, Timeout

from gamma_gpt35 import LLMClient, GammaAgent, extract_json_from_text
from acc_gpt35 import ACCAgent


class BridgeManager:
    """
    BridgeManager: 轻量“桥接管理器”

    目标：
    - 利用 question schema 中的实体 + 之前 Gamma 找到的显式实体答案，
      作为每一步子问题的 anchor（锚点实体）。
    - 检查 Gamma 选中的 facts 是否真的提到了这些 anchor。
    - 如果一个子问题明显依赖 anchor，但选中的 facts 里完全没出现，
      则认为 bridge 不可靠：把该步标记为 bridge_ok=False，并把 found=False（gating）。

    输出：
    - 每步一个 meta:
      {
        "step_index": int,
        "anchors": [...],
        "matched_anchors": [...],
        "bridge_ok": bool,
        "bridge_reason": str
      }
    - trace 中的 "bridge_manager" 会包含 summary 方便 log/分析。
    """

    def __init__(self, question: str, schema: Dict[str, Any]):
        self.question = question
        self.schema = schema or {}
        # 从 schema 中抽取显式实体名字
        self.schema_entities: List[str] = []
        for e in self.schema.get("entities", []) or []:
            name = (e.get("name") or "").strip()
            if name:
                self.schema_entities.append(name)
        # 之前各步 Gamma 的 answer（实体类答案）
        self.prev_answers: List[str] = []
        # 每步的 bridge 记录
        self.steps: List[Dict[str, Any]] = []

    def _extract_anchors_for_subquestion(self, subq: str) -> List[str]:
        """
        对当前 refined_subquestion 抽取“应当对齐的锚点实体”：
        - schema 中在该子问题里出现的 entity name；
        - 之前各步显式 answer 在该子问题中出现的。
        """
        anchors = set()
        sq_lower = (subq or "").lower()

        # schema 实体
        for name in self.schema_entities:
            n = name.strip()
            if n and n.lower() in sq_lower:
                anchors.add(n)

        # 之前 answer 作为潜在中间实体
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
        在 Gamma 每一步之后调用：
        - 根据 refined_subq 抽 anchor；
        - 看 selected_fact_texts 里有没有提到这些 anchor；
        - 如有 anchor 但完全没命中，则：
            - bridge_ok=False
            - 在 gamma_result["reasoning"] 里标记
            - 并把 gamma_result["found"] 强制改为 False（gating）
        - 如果本步仍然 found 且 answer 是字符串，则把 answer 记入 prev_answers。
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
                    # 有明确锚点，但选的 facts 里完全没出现：认为桥接不可靠，直接 gate 掉
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
                # 本来就没找到答案，但这个子问题带锚点 -> 说明桥接链在此断裂
                bridge_ok = False
                bridge_reason = (
                    "Gamma could not find an answer for a subquestion that depends on "
                    "anchor entities."
                )
        else:
            # 没有检测到锚点：默认通过，不做 gating
            bridge_ok = True
            bridge_reason = ""

        # 记录“仍然当做 found 的实体答案”，供后续步使用
        ans = gamma_result.get("answer")
        if gamma_result.get("found") and isinstance(ans, str) and ans.strip():
            self.prev_answers.append(ans.strip())
            # 简单截断，避免列表过长
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
        返回一个简短 summary，给 log/后续 prompt 使用。
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

    def __init__(self, dataset_name: str, llm_client: LLMClient):
        assert dataset_name in {"2wiki", "hotpotqa", "musique"}
        self.dataset_name = dataset_name
        self.acc = ACCAgent(dataset_name=dataset_name, llm_client=llm_client)
        self.llm = llm_client
        self.gamma = GammaAgent(dataset_name=dataset_name, llm_client=llm_client)

        # BridgeManager 类（方便之后如果想换实现）
        self.bridge_manager_cls = BridgeManager

        # 当前样本的问题 schema 缓存（避免重复推一次）
        self._schema_question: Optional[str] = None
        self._schema: Optional[Dict[str, Any]] = None

    # ---------- Question Schema 构建 ----------

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
        构造一个轻量 Question Schema：
        - question_type: yes_no / comparison / fact / count / other
        - answer_form: yes_no / entity_name / number / date / span / other
        - focus: 人 / 电影 / 地点 / 机构 / other（仅提示用）
        - entities: [{"name": ..., "type": ..., "role": ...}, ...]
        - constraints / notes: 任务规则 & 其他说明
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

        # 容错解析
        try:
            obj = extract_json_from_text(raw_text)
            if not isinstance(obj, dict):
                raise ValueError("schema must be an object")
        except Exception:
            obj = {}

        # 填默认值，避免下游 KeyError
        schema: Dict[str, Any] = {
            "question_type": obj.get("question_type", "other"),
            "answer_form": obj.get("answer_form", "other"),
            "focus": obj.get("focus", "generic"),
            "entities": obj.get("entities", []),
            "constraints": obj.get("constraints", []),
            "notes": obj.get("notes", ""),
            "raw_json": obj,
        }
        return schema

    def _schema_summary(self, schema: Dict[str, Any]) -> str:
        """
        把 schema 转成短文本，方便塞进 prompt 作为 hint
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

    # ---------- Stage 1: theta decomposes the question ----------

    def plan_subquestions(self, question: str, max_steps: int = 4) -> List[str]:
        # 确保有 schema
        schema = self._ensure_schema(question)
        schema_summary = self._schema_summary(schema)

        prompt = f"""
You are the THETA agent in a dual-agent multi-hop reasoning system.

FIRST, read the QUESTION_SCHEMA (already analyzed for you).
Then, decompose the ORIGINAL QUESTION into 2 to {max_steps} ordered SUBQUESTIONS.

Guidelines:
- Each subquestion should be a simple fact-based question.
- The sequence of subquestions should follow the schema:
  - Respect the question_type and constraints.
  - Ensure that the LAST subquestion directly prepares the information needed
    to produce the final answer in the required ANSWER_FORM.
- Do NOT answer the question here; only plan subquestions.

QUESTION_SCHEMA (for reference, DO NOT rewrite it):
{schema_summary}

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
                f"step {step.get('step_index')}: subquestion='{step.get('refined_subquestion', step.get('subquestion'))}', "
                f"found={gres.get('found')}, answer={gres.get('answer')}"
            )
        history_block = "\n".join(lines)

        prompt = f"""
You are the THETA agent refining the next subquestion in a multi-hop QA task.

You MUST:
- Use the QUESTION_SCHEMA to preserve entity alignment and answer role.
- Use PREVIOUS STEPS (gamma answers) to rewrite the NEXT_SUBQUESTION to be as specific as possible
  (e.g., replace pronouns like "that director" with the actual name if confidently known).
- If previous steps did NOT find a reliable answer (found=false or empty answer), KEEP the next subquestion unchanged.

QUESTION_SCHEMA:
{schema_summary}

PREVIOUS STEPS:
{history_block}

NEXT_SUBQUESTION (planned):
{planned_subquestion}

CRITICAL OUTPUT:
- Respond with a SINGLE JSON object:
  - "subquestion": string (the rewritten subquestion to ask gamma)

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
            refined = obj.get("subquestion", "").strip()
            return refined if refined else planned_subquestion
        except Exception:
            return planned_subquestion

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
        把 question schema + gamma 结果 + 符号比较提示 三者一起喂给 LLM，
        明确要求它：
        - 回答形式要符合 schema.answer_form；
        - 如果是 yes/no，就只回答 yes / no；
        - 如果是 “which film/person...”，就输出对应实体，而不是中间推理节点。
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

QUESTION_SCHEMA:
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

        # 0) 先构建 Question Schema（让后续都在这个 frame 下运转）
        try:
            schema = self._ensure_schema(question)
        except Exception as e:
            # 某些环境下 schema 调用可能被封禁或网络异常，直接跳过该样本，避免中断全局
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
        schema_summary = self._schema_summary(schema)  # 纯 log 用

        # 初始化 BridgeManager
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
            # 如果分解调用出现 403 / 网络错误，直接跳过当前样本
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
            # BridgeManager: 检测这一步的锚点对齐情况，并做 gating
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
