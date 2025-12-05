# gamma_gpt35.py
# Gamma agent + LLM utilities

import os
import json
import re
import time
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


class GammaAgent:
    """
    Gamma agent: search within facts and answer subquestions.
    """

    PROMPTS: Dict[str, str] = {
        "gamma_answer": """
You are the gamma agent in a dual-agent multi-hop QA system.

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
                "agent": "gamma",
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
              2) 用 primary_facts 做第一轮 Gamma 调用。
              3) 若未找到答案（found=False 或没选中任何 fact），
                 再用剩余段落做第二轮 Gamma 调用。
          - 如果没有锚点，则退化为“全量 facts 一次性调用”。
        """
        facts = self.build_facts(example)
        if not facts:
            return {
                "found": False,
                "answer": None,
                "selected_fact_indices": [],
                "selected_fact_texts": [],
                "reasoning": "No facts available",
                "raw_json": {},
            }

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
                        return second
                return first

        # 默认路径：直接把所有 facts 一次性喂给 LLM（2wiki/hotpotqa 或没有锚点的 MusiQue）
        return self._run_gamma_once(
            subquestion=subquestion,
            candidate_facts=facts,
            call_id=call_id,
            all_facts=facts,
        )
