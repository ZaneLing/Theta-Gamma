# gamma_gpt35.py
# HPC + LLM utilities

import os
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from an LLM response:
    - Try loading the full string; if it's a list take the last dict.
    - Strip <think>/<analysis> reasoning blocks and retry.
    - If it still fails, pull the last {...} substring in the text.
    """

    def _strip_reasoning_blocks(raw: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<analysis>.*?</analysis>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        return cleaned

    def _load_obj(candidate: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                obj = next((item for item in reversed(obj) if isinstance(item, dict)), {})
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        matches = re.findall(r"\{.*?\}", candidate, flags=re.DOTALL)
        for m in reversed(matches):
            try:
                obj = json.loads(m)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None

    text = text.strip()
    for cand in (text, _strip_reasoning_blocks(text)):
        obj = _load_obj(cand)
        if obj is not None:
            return obj
    raise ValueError(f"Cannot parse JSON from LLM output: {text[:200]}...")


class LLMClient:
    """
    Thin wrapper around the OpenRouter / OpenAI chat API.
    """

    MODEL_ALIASES = {
        # Friendly alias -> OpenRouter model id
        "gpt-o3": "openai/o3-mini",
        "openai/gpt-o3": "openai/o3-mini",
    }

    def __init__(
        self,
        call_log: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None,
        api_url: Optional[str] = None,
        fallback_model_name: Optional[str] = None,
    ):
        self.api_url = api_url or os.getenv("API_URL", "https://openrouter.ai/api/v1/chat/completions")
        # Default to OpenRouter GPT-3.5 Turbo identifier
        requested_model = model_name or os.getenv("MODEL_NAME", "openai/gpt-3.5-turbo")
        self.model_name = self.MODEL_ALIASES.get(requested_model, requested_model)
        fb_requested = fallback_model_name or os.getenv("FALLBACK_MODEL_NAME")
        self.fallback_model_name = self.MODEL_ALIASES.get(fb_requested, fb_requested) if fb_requested else None
        self.api_key = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        self.call_log: List[Dict[str, Any]] = call_log if call_log is not None else []

    def generate(
        self,
        prompt: str,
        meta: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }

        MAX_RETRY = 5
        attempted_fallback = False
        for attempt in range(1, MAX_RETRY + 1):
            try:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                resp = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                text = ""
                try:
                    text = data["choices"][0]["message"]["content"]
                except Exception:
                    text = data.get("response", "")

                self.call_log.append({
                    "type": "llm_call",
                    "meta": meta or {},
                    "request": payload,
                    "raw_response": data,
                    "response_text": text,
                })
                return text

            except Exception as e:
                # Auto-fallback on missing endpoint if configured
                if (
                    not attempted_fallback
                    and self.fallback_model_name
                    and self.model_name != self.fallback_model_name
                ):
                    message = ""
                    status = None
                    if hasattr(e, "response") and e.response is not None:
                        try:
                            status = e.response.status_code
                            message = e.response.text or ""
                        except Exception:
                            message = ""
                    msg_lower = (message or "").lower()
                    if status == 404 or "no endpoints found" in msg_lower:
                        attempted_fallback = True
                        self.model_name = self.fallback_model_name
                        payload["model"] = self.model_name
                        print(f"[LLMClient] Model endpoint missing, switching to fallback model '{self.model_name}'.")
                        continue
                if attempt == MAX_RETRY:
                    raise e
                print(f"[LLMClient] Timeout or error, retrying ({attempt}/{MAX_RETRY})...")
                time.sleep(3)  # wait 3 seconds then retry


class HPC:
    """
    HPC: search within facts and answer subquestions.
    """

    PROMPTS: Dict[str, str] = {
        "gamma_answer": """
You are HPC in a multi-hop QA system.

Your role:
- Given a SUBQUESTION and a list of NUMBERED FACTS, pick the relevant facts.
- If the facts are sufficient, extract a SHORT ANSWER phrase (a few words).
- If the facts are NOT sufficient, you must say found = false and answer = null.

CRITICAL OUTPUT REQUIREMENTS:
- You MUST respond with a SINGLE valid JSON object.
- Do NOT write any explanation outside JSON.
- JSON keys:
  - "found": a boolean
  - "selected_fact_indices": array of integers (indices of used facts, possibly empty)
  - "answer": a string or null
  - "reasoning": a short one-sentence explanation in English

Example of valid output:
{{
  "found": true,
  "selected_fact_indices": [1, 3],
  "answer": "Thomas Jefferson",
  "reasoning": "Facts [1] and [3] show his role and birth state."
}}

Now process the input.

SUBQUESTION:
{subquestion}

FACTS:
{facts_block}

Respond with JSON ONLY:
""".strip(),
    }

    def __init__(self, dataset_name: str, llm_client: LLMClient):
        assert dataset_name in {"2wiki", "hotpotqa", "musique"}
        self.dataset_name = dataset_name
        self.llm = llm_client
        self._prompt_cache: Dict[str, str] = {}

    def _get_prompt(self, name: str) -> str:
        if name in self._prompt_cache:
            return self._prompt_cache[name]
        if name not in self.PROMPTS:
            raise KeyError(f"Prompt '{name}' not found")
        self._prompt_cache[name] = self.PROMPTS[name]
        return self._prompt_cache[name]

    # ---------- Build facts for different datasets ----------

    def _build_facts_2wiki(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        for idx, pair in enumerate(example.get("context", [])):
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            title, sentences = pair
            if isinstance(sentences, list):
                text = " ".join(sentences)
            else:
                text = str(sentences)
            facts.append({"idx": idx, "title": str(title), "text": text})
        return facts

    def _build_facts_hotpotqa(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        ctx = example.get("context", {})
        titles = ctx.get("title", [])
        sents = ctx.get("sentences", [])
        for idx, title in enumerate(titles):
            if idx < len(sents) and isinstance(sents[idx], list):
                text = " ".join(sents[idx])
            else:
                text = ""
            facts.append({"idx": idx, "title": str(title), "text": text})
        return facts

    def _build_facts_musique(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        for para in example.get("paragraphs", []):
            idx = para.get("idx")
            title = para.get("title", "")
            text = para.get("paragraph_text", "")
            facts.append({"idx": int(idx), "title": str(title), "text": str(text)})
        return facts

    def build_facts(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.dataset_name == "2wiki":
            return self._build_facts_2wiki(example)
        if self.dataset_name == "hotpotqa":
            return self._build_facts_hotpotqa(example)
        if self.dataset_name == "musique":
            return self._build_facts_musique(example)
        raise ValueError(f"Unknown dataset: {self.dataset_name}")

    # ---------- 内部工具：构造 facts_block 并调用一次 LLM ----------

    def _build_facts_block(self, facts: List[Dict[str, Any]]) -> str:
        fact_lines: List[str] = []
        for f in facts:
            fact_lines.append(
                f"[{f['idx']}] Title: {f['title']}\nText: {f['text']}"
            )
        return "\n\n".join(fact_lines)

    def _infer_answer_type(self, subquestion: str, answer: Optional[str]) -> str:
        """
        Heuristic answer-type inference from the subquestion and answer string.
        """
        if not answer:
            return "unknown"
        ans = answer.strip()
        ans_lower = ans.lower()
        sq = (subquestion or "").strip().lower()

        if ans_lower in {"yes", "no"}:
            return "boolean"
        if re.fullmatch(r"\d+", ans):
            return "number"
        if re.fullmatch(r"(1[5-9]\d{2}|20\d{2})", ans):
            return "date"

        if sq.startswith("who "):
            return "person"
        if sq.startswith("where "):
            return "location"
        if sq.startswith("when "):
            return "date"
        if sq.startswith("how many ") or sq.startswith("how much "):
            return "number"
        if sq.startswith("which film ") or sq.startswith("which movie "):
            return "film"

        return "unknown"

    def _build_local_gamma_memory(
        self,
        subquestion: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        answer = result.get("answer")
        if not isinstance(answer, str):
            answer = ""
        facts = result.get("selected_fact_texts") or []
        reasoning = result.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)
        answer_type = self._infer_answer_type(subquestion, answer)

        return {
            "sub_question": subquestion,
            "retrieval": {
                "sub_answer": answer,
                "retrieved_facts": facts,
                "sub_answer_type": answer_type,
            },
            "reasoning": reasoning,
        }

    def _write_local_gamma_memory(self, memory: Dict[str, Any], call_id: str) -> str:
        base_dir = Path(__file__).resolve().parent.parent / "Memory" / "local_gamma_memory"
        base_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", call_id)
        filename = f"{safe_name}.json"
        path = base_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=True, indent=2)
        return str(path)

    def _attach_local_gamma_memory(
        self,
        result: Dict[str, Any],
        subquestion: str,
        call_id: str,
    ) -> Dict[str, Any]:
        memory = self._build_local_gamma_memory(subquestion, result)
        result["local_gamma_memory"] = memory
        try:
            result["local_gamma_memory_path"] = self._write_local_gamma_memory(memory, call_id)
        except Exception:
            pass
        return result

    def _run_gamma_once(
        self,
        subquestion: str,
        candidate_facts: List[Dict[str, Any]],
        call_id: str,
        all_facts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not candidate_facts:
            return {
                "found": False,
                "answer": None,
                "selected_fact_indices": [],
                "selected_fact_texts": [],
                "reasoning": "No facts available for this subquestion",
                "raw_json": {},
            }

        facts_block = self._build_facts_block(candidate_facts)
        prompt = self._get_prompt("gamma_answer").format(
            subquestion=subquestion,
            facts_block=facts_block,
        ).strip()

        raw_text = self.llm.generate(
            prompt,
            meta={
                "rhythm": "HPC",
                "dataset": self.dataset_name,
                "call_id": call_id,
                "subquestion": subquestion,
            },
            temperature=0.1,
        )

        try:
            obj = extract_json_from_text(raw_text)
            if isinstance(obj, list):
                # Some models may return a single-element list; take the last dict
                obj = next((item for item in reversed(obj) if isinstance(item, dict)), {})
            if not isinstance(obj, dict):
                obj = {}
        except Exception as e:
            obj = {
                "found": False,
                "selected_fact_indices": [],
                "answer": None,
                "reasoning": f"JSON parse error: {e}",
            }

        found = bool(obj.get("found", False))
        answer = obj.get("answer", None)
        if isinstance(answer, str):
            answer = answer.strip()
        else:
            answer = None

        idxs = obj.get("selected_fact_indices", [])
        if not isinstance(idxs, list):
            idxs = []

        selected_fact_texts: List[str] = []
        idx_set = set()
        fact_map = {f["idx"]: f for f in all_facts}
        for idx in idxs:
            if idx in fact_map and idx not in idx_set:
                f = fact_map[idx]
                selected_fact_texts.append(
                    f"[{f['idx']}] {f['title']}: {f['text']}"
                )
                idx_set.add(idx)

        reasoning = obj.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)

        return {
            "found": found,
            "answer": answer,
            "selected_fact_indices": sorted(idx_set),
            "selected_fact_texts": selected_fact_texts,
            "reasoning": reasoning,
            "raw_json": obj,
        }

    # ---------- MusiQue 专用：从 schema + 子问题中抽锚点实体 ----------

    def _extract_anchors_from_schema(
        self,
        schema: Optional[Dict[str, Any]],
        subquestion: str,
    ) -> List[str]:
        if not schema:
            return []
        sq_lower = (subquestion or "").lower()
        anchors: List[str] = []
        for e in schema.get("entities", []) or []:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            # 只把真的在子问题里出现的实体，当做当前子问题的锚点
            if name.lower() in sq_lower:
                anchors.append(name)
        return anchors

    def _measure_title_relevance(self, title: str, anchors: List[str]) -> int:
        """
        一个简单但偏向“精确 title 匹配”的打分：
        - 完全相等（忽略大小写）: 100
        - 互为包含: 60
        - token 交集 >0: 10 + 交集长度
        这样能让 "John J. Collins" >> "John Collins" >> "Lucy John"
        """
        t = (title or "").lower()
        if not t or not anchors:
            return 0
        tokens_t = set(re.findall(r"\w+", t))
        score = 0
        for a in anchors:
            al = (a or "").lower()
            if not al:
                continue
            if t == al or t.strip() == al.strip():
                score = max(score, 100)
                continue
            if al in t or t in al:
                score = max(score, 60)
            ats = set(re.findall(r"\w+", al))
            if tokens_t and ats:
                inter = len(tokens_t & ats)
                if inter > 0:
                    score = max(score, 10 + inter)
        return score

    # ---------- Retrieve + answer a single subquestion ----------

    def answer_subquestion(
        self,
        example: Dict[str, Any],
        subquestion: str,
        call_id: str,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        对 2wiki / hotpotqa：仍然是“全部 facts”一次性喂给 LLM。
        对 MusiQue：
          - 如果能从 schema + 子问题中抽到实体锚点：
              1) 按标题相关度对所有段落 rerank，选出 top-5 作为 primary_facts。
              2) 用 primary_facts 做第一轮 HPC 调用。
              3) 若未找到答案（found=False 或没选中任何 fact），
                 再用剩余段落做第二轮 HPC 调用。
          - 如果没有锚点，则退化为“全量 facts 一次性调用”。
        """
        facts = self.build_facts(example)
        if not facts:
            result = {
                "found": False,
                "answer": None,
                "selected_fact_indices": [],
                "selected_fact_texts": [],
                "reasoning": "No facts available",
                "raw_json": {},
            }
            return self._attach_local_gamma_memory(result, subquestion, call_id)

        # MusiQue: 标题优先 + 两阶段检索
        if self.dataset_name == "musique" and schema is not None:
            anchors = self._extract_anchors_from_schema(schema, subquestion)
            if anchors:
                # 1) 按标题相关度打分
                scored: List[Dict[str, Any]] = []
                for f in facts:
                    s = self._measure_title_relevance(f.get("title", ""), anchors)
                    scored.append({"score": s, "fact": f})

                scored.sort(key=lambda x: x["score"], reverse=True)

                # 只保留有分数的 top-5；若全部 0 分，就按原顺序取前 5
                primary_facts: List[Dict[str, Any]] = [
                    item["fact"] for item in scored if item["score"] > 0
                ][:5]
                if not primary_facts:
                    primary_facts = [item["fact"] for item in scored[:5]]

                primary_idx_set = {f["idx"] for f in primary_facts}

                # 2) 第一轮：只用 top-5 段落
                first = self._run_gamma_once(
                    subquestion=subquestion,
                    candidate_facts=primary_facts,
                    call_id=f"{call_id}_p1",
                    all_facts=facts,
                )
                first["retrieval_pass"] = "primary"
                first["retrieval_anchors"] = anchors
                first["retrieval_primary_indices"] = sorted(primary_idx_set)

                if (not first.get("found")) or (not first.get("selected_fact_indices")):
                    # 3) 如果没找到，再用剩余段落做第二轮
                    fallback_facts = [f for f in facts if f["idx"] not in primary_idx_set]
                    if fallback_facts:
                        second = self._run_gamma_once(
                            subquestion=subquestion,
                            candidate_facts=fallback_facts,
                            call_id=f"{call_id}_p2",
                            all_facts=facts,
                        )
                        second["retrieval_pass"] = "fallback"
                        second["retrieval_anchors"] = anchors
                        second["retrieval_primary_indices"] = sorted(primary_idx_set)
                        return self._attach_local_gamma_memory(second, subquestion, call_id)
                return self._attach_local_gamma_memory(first, subquestion, call_id)

        # 默认路径：直接把所有 facts 一次性喂给 LLM（2wiki/hotpotqa 或没有锚点的 MusiQue）
        result = self._run_gamma_once(
            subquestion=subquestion,
            candidate_facts=facts,
            call_id=call_id,
            all_facts=facts,
        )
        return self._attach_local_gamma_memory(result, subquestion, call_id)


ACTION_KEEP = "KEEP"
ACTION_FLIP_BOOL = "FLIP_BOOL"
ACTION_CHOOSE = "CHOOSE_FROM_CANDIDATES"
ACTION_UNKNOWN = "ANSWER_UNKNOWN"

BOOL_YES = "yes"
BOOL_NO = "no"
BOOL_UNKNOWN = "unknown"

ACC_PROMPTS: Dict[str, str] = {
    "acc_check": """
You are an ACC-like self-checking module in a multi-hop QA system.

Your role:
- You are NOT a full problem solver.
- You act as a JUDGE + LIGHT EDITOR over PFC's INITIAL_ANSWER.
- You have a very LIMITED ACTION SPACE.

You receive:
- ORIGINAL QUESTION
- INITIAL_ANSWER from PFC
- STEPWISE EVIDENCE from HPC
- A CANDIDATE_ANSWER_SET that you MUST respect.
- For each HPC step, you may also see a line like:
    "bridge_ok: <true/false> anchors=[...] matched_anchors=[...]"
  These come from a BridgeManager that checks whether the selected facts actually
  mention the anchor entities from the question/schema/previous steps.
- If bridge_ok=false, you should treat that step as UNRELIABLE EVIDENCE.
- Do NOT use such steps to aggressively flip or override the INITIAL_ANSWER.
- Prefer KEEP or ANSWER_UNKNOWN when evidence is weak or bridge_ok is mostly false.

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
- Prefer ACTION="KEEP" unless you see a CLEAR and STRONG conflict between INITIAL_ANSWER and the reliable evidence.
- For YES/NO questions, normalised answers are:
    - "yes" for yes/true,
    - "no" for no/false,
    - "unknown" for unknown / cannot be determined.
- For non-YES/NO questions, you must choose within the provided candidate set.

You MUST answer ONLY the ORIGINAL QUESTION. Do NOT change its meaning.
Do NOT output intermediate entities if the question asks for a film, person, country, etc.
For example, if the question asks "Which film ...", the final answer must be a FILM TITLE, not a director name.

You must output a SINGLE JSON object with fields:
{
  "action": "KEEP" | "FLIP_BOOL" | "CHOOSE_FROM_CANDIDATES" | "ANSWER_UNKNOWN",
  "candidate_answer": "string or empty",
  "explanation": "short explanation in English",
  "flags": {
    "entity_mismatch": true/false,
    "logic_inconsistency": true/false,
    "insufficient_evidence": true/false
  }
}

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
- QUESTION_TYPE = {question_type}
- CANDIDATES = {candidates}

ORIGINAL QUESTION:
{question}

INITIAL_ANSWER:
{initial_answer}

GAMMA STEPWISE EVIDENCE:
{gamma_evidence}

Now respond with JSON ONLY:
""".strip(),
}


class ACC:
    """
    ACC: ACC-style self-check module (judge + lightweight editor)

    Main interface:
        check_answer(question, gamma_results, initial_answer) -> dict
    """

    def __init__(self, dataset_name: str, llm_client: LLMClient):
        assert dataset_name in {"2wiki", "hotpotqa", "musique"}
        self.dataset_name = dataset_name
        self.llm = llm_client

    # ------------------------------------------------------------------
    # Helper: format HPC evidence
    # ------------------------------------------------------------------
    def _format_gamma_evidence(self, gamma_results: List[Dict[str, Any]]) -> str:
        """
        Format PFC-HPC step results into text for ACC.
        Include BridgeManager info so ACC can treat bridge_ok=False steps as suspect evidence.
        """
        lines: List[str] = []
        for i, step in enumerate(gamma_results, start=1):
            subq = step.get("refined_subquestion") or step.get("subquestion")
            gres = step.get("gamma_result", {}) or {}
            bridge = step.get("bridge", {}) or {}
            lines.append(f"Step {i}:")
            lines.append(f"  subquestion: {subq}")
            lines.append(f"  gamma_found: {gres.get('found')}")
            lines.append(f"  gamma_answer: {gres.get('answer')}")
            lines.append(f"  gamma_reasoning: {gres.get('reasoning')}")

            # Bridge info: bridge_ok / anchors / matched_anchors
            if bridge:
                lines.append(
                    f"  bridge_ok: {bridge.get('bridge_ok')} "
                    f"anchors={bridge.get('anchors')} "
                    f"matched_anchors={bridge.get('matched_anchors')}"
                )
                if bridge.get("bridge_reason"):
                    lines.append(f"  bridge_note: {bridge.get('bridge_reason')}")

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

        # 2) Format HPC evidence (includes bridge info)
        evidence_block = self._format_gamma_evidence(gamma_results)

        # 3) Build ACC prompt (must stay within given candidates/actions)
        candidate_list_str = ", ".join(f'"{c}"' for c in candidates) if candidates else ""
        bool_or_not = "YES_NO" if is_bool_q else "GENERAL"

        prompt = ACC_PROMPTS["acc_check"].format(
            BOOL_YES=BOOL_YES,
            BOOL_NO=BOOL_NO,
            BOOL_UNKNOWN=BOOL_UNKNOWN,
            candidate_list_str=candidate_list_str,
            bool_or_not=bool_or_not,
            question=question,
            initial_answer=initial_answer,
            evidence_block=evidence_block,
        ).strip()

        raw_llm_output = ""
        try:
            raw_llm_output = self.llm.generate(
                prompt,
                meta={
                    "rhythm": "ACC",
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
            # Directly fall back to PFC's original answer
            if explanation:
                explanation = (
                    explanation
                    + " NOTE: ACC candidate looked like a question, fallback to PFC initial answer."
                )
            else:
                explanation = (
                    "ACC candidate looked like a question, fallback to PFC initial answer."
                )
            action = ACTION_KEEP
            final_answer = initial_answer_normed

        # 7) Return result for pipeline/PFC logging
        return {
            "action": action,
            "final_answer": final_answer,
            "explanation": explanation,
            "flags": flags,
            "candidates": candidates,
            "raw_llm_output": raw_llm_output,
            "raw_json": obj,
        }
