# gamma.py
# Gamma agent + LLM utilities

import os
import json
import re
import time
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
import yaml


def load_dotenv(path: str = ".env") -> None:
    """
    Minimal .env loader that reads KEY=VALUE per line.
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
        load_dotenv()
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

    def __init__(self, dataset_name: str, llm_client: LLMClient):
        assert dataset_name in {"2wiki", "hotpotqa", "musique"}
        self.dataset_name = dataset_name
        self.llm = llm_client
        self._prompt_cache: Dict[str, str] = {}

    # ---------- Prompt loading ----------

    def _load_prompts(self) -> Dict[str, str]:
        if self._prompt_cache:
            return self._prompt_cache
        prompt_path = Path(__file__).resolve().parent / "prompts" / "gamma_prompts.yaml"
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self._prompt_cache = {str(k): str(v) for k, v in data.items()}
        return self._prompt_cache

    def _get_prompt(self, name: str) -> str:
        prompts = self._load_prompts()
        if name not in prompts:
            raise KeyError(f"Prompt '{name}' not found")
        return prompts[name]

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

    # ---------- Retrieve + answer a single subquestion ----------

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
