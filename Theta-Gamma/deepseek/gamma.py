# gamma.py
# Gamma agent + LLM 工具封装

import os
import json
import re
import time
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional


def load_dotenv(path: str = ".env") -> None:
    """
    简单 .env 加载：KEY=VALUE 行
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
    except Exception:
        pass


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    从 LLM 输出中鲁棒提取 JSON：
    - 先整体解析
    - 针对 phi4-reasoning 这类会输出 <think>...</think>{...} 的格式，先去掉思考块
    - 再用非贪婪正则取最后一个 {...}
    """
    def _strip_reasoning_blocks(raw: str) -> str:
        # phi4-reasoning 会把思考放在 <think>...</think> 里，去掉后再试解析
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        # 也顺便去掉可能的 <analysis>...</analysis> 之类块
        cleaned = re.sub(r"<analysis>.*?</analysis>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        return cleaned

    text = text.strip()

    candidates = [text]
    stripped = _strip_reasoning_blocks(text).strip()
    if stripped and stripped != text:
        candidates.append(stripped)

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            pass

        matches = re.findall(r"\{.*?\}", cand, flags=re.DOTALL)
        for m in reversed(matches):
            try:
                return json.loads(m)
            except Exception:
                continue

    raise ValueError(f"Cannot parse JSON from LLM output: {text[:200]}...")


class LLMClient:
    """
    封装 Ollama /api/generate
    """

    def __init__(self, call_log: Optional[List[Dict[str, Any]]] = None):
        load_dotenv()
        self.api_url = os.getenv("API_URL", "http://172.16.120.14:11434/api/generate")
        # 默认使用 deepseek-r1:14b，如果需要改模型可通过环境变量 MODEL_NAME 覆盖
        self.model_name = os.getenv("MODEL_NAME", "deepseek-r1:14b")
        self.call_log: List[Dict[str, Any]] = call_log if call_log is not None else []

    def generate(
        self,
        prompt: str,
        meta: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        MAX_RETRY = 5
        for attempt in range(1, MAX_RETRY + 1):
            try:
                resp = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=120,       
                )
                resp.raise_for_status()
                data = resp.json()
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
                if attempt == MAX_RETRY:
                    raise e
                print(f"[LLMClient] Timeout or error, retrying ({attempt}/{MAX_RETRY})...")
                time.sleep(3)  # 等3秒继续 retry



class GammaAgent:
    """
    Gamma agent：负责在 facts 中找线索 + 局部回答
    """

    def __init__(self, dataset_name: str, llm_client: LLMClient):
        assert dataset_name in {"2wiki", "hotpotqa", "musique"}
        self.dataset_name = dataset_name
        self.llm = llm_client

    # ---------- 不同数据集构造 facts ----------

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

    # ---------- 对单个子问题进行检索 + 回答 ----------

    def answer_subquestion(
        self,
        example: Dict[str, Any],
        subquestion: str,
        call_id: str,
    ) -> Dict[str, Any]:
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

        fact_lines = []
        for f in facts:
            fact_lines.append(
                f"[{f['idx']}] Title: {f['title']}\nText: {f['text']}"
            )
        facts_block = "\n\n".join(fact_lines)

        prompt = f"""
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
""".strip()

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
        fact_map = {f["idx"]: f for f in facts}
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
