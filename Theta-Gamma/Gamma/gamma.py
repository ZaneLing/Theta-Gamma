# gamma_gpt35.py
# HPC + ACC utilities

import os
import json
import re
import time
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    HPC: retrieval + answer for subquestions, with path replay guidance.
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
        self._bge_backend: Optional[str] = None
        self._bge_m3 = None
        self._bge_reranker = None
        self._bge_m3_name = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")
        self._bge_reranker_name = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-base")
        self._bge_batch_size = int(os.getenv("BGE_BATCH_SIZE", "16"))
        self._bge_max_length = int(os.getenv("BGE_MAX_LENGTH", "512"))

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

    def _ensure_bge_models(self) -> None:
        if self._bge_backend is not None:
            if self._bge_backend == "none":
                raise RuntimeError(
                    "BGE models unavailable. Install FlagEmbedding or sentence-transformers."
                )
            return

        last_err: Optional[Exception] = None
        try:
            from FlagEmbedding import BGEM3FlagModel, FlagReranker

            self._bge_m3 = BGEM3FlagModel(self._bge_m3_name, use_fp16=True)
            self._bge_reranker = FlagReranker(self._bge_reranker_name, use_fp16=True)
            self._bge_backend = "flagembedding"
            return
        except Exception as e:
            last_err = e

        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder

            self._bge_m3 = SentenceTransformer(self._bge_m3_name)
            self._bge_reranker = CrossEncoder(self._bge_reranker_name)
            self._bge_backend = "sentence_transformers"
            return
        except Exception as e:
            last_err = e

        self._bge_backend = "none"
        raise RuntimeError(
            "BGE models unavailable. Install FlagEmbedding or sentence-transformers."
        ) from last_err

    def _encode_bge_texts(self, texts: List[str]) -> List[Any]:
        self._ensure_bge_models()
        if self._bge_backend == "flagembedding":
            try:
                emb = self._bge_m3.encode(
                    texts,
                    batch_size=self._bge_batch_size,
                    max_length=self._bge_max_length,
                )
            except TypeError:
                emb = self._bge_m3.encode(texts)
            if isinstance(emb, dict) and "dense_vecs" in emb:
                return list(emb["dense_vecs"])
            return list(emb)

        emb = self._bge_m3.encode(
            texts,
            batch_size=self._bge_batch_size,
            normalize_embeddings=True,
        )
        return list(emb)

    def _cosine_scores(self, query_vec: Any, doc_vecs: List[Any]) -> List[float]:
        try:
            import numpy as np

            q = np.asarray(query_vec)
            d = np.asarray(doc_vecs)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            denom = (np.linalg.norm(d, axis=1) * np.linalg.norm(q)) + 1e-8
            scores = (d @ q.T).reshape(-1) / denom
            return scores.tolist()
        except Exception:
            q = list(query_vec)
            q_norm = math.sqrt(sum(x * x for x in q)) + 1e-8
            scores: List[float] = []
            for v in doc_vecs:
                vv = list(v)
                dot = sum(a * b for a, b in zip(q, vv))
                v_norm = math.sqrt(sum(x * x for x in vv)) + 1e-8
                scores.append(dot / (q_norm * v_norm))
            return scores

    def _bge_similarity_scores(self, query: str, texts: List[str]) -> List[float]:
        if not query or not texts:
            return []
        vectors = self._encode_bge_texts([query] + texts)
        if not vectors or len(vectors) < 2:
            return [0.0 for _ in texts]
        query_vec = vectors[0]
        doc_vecs = vectors[1:]
        return self._cosine_scores(query_vec, doc_vecs)

    def _bge_rerank_scores(self, query: str, texts: List[str]) -> List[float]:
        self._ensure_bge_models()
        pairs = [[query, t] for t in texts]
        if self._bge_backend == "flagembedding":
            scores = self._bge_reranker.compute_score(pairs)
        else:
            scores = self._bge_reranker.predict(pairs)
        return [float(s) for s in scores]

    def _rerank_facts(
        self,
        query: str,
        facts: List[Dict[str, Any]],
        use_title_only: bool,
        top_k: int,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        if not facts:
            return [], []
        docs = []
        for f in facts:
            if use_title_only:
                docs.append(str(f.get("title", "")))
            else:
                docs.append(f"{f.get('title', '')}. {f.get('text', '')}")
        scores = self._bge_rerank_scores(query, docs)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        reranked = [facts[i] for i in order[:top_k]]
        rerank_scores = [scores[i] for i in order[:top_k]]
        return reranked, rerank_scores

    def _retrieve_topk_title_bge(
        self,
        facts: List[Dict[str, Any]],
        core_entity: str,
        top_k: int = 3,
    ) -> Tuple[List[Dict[str, Any]], List[float], List[int]]:
        titles = [str(f.get("title", "")) for f in facts]
        scores = self._bge_similarity_scores(core_entity, titles)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = order[:top_k]
        candidates = [facts[i] for i in top]
        reranked, rerank_scores = self._rerank_facts(
            core_entity,
            candidates,
            use_title_only=True,
            top_k=top_k,
        )
        reranked_indices = [f.get("idx") for f in reranked]
        return reranked, rerank_scores, reranked_indices

    def _retrieve_topk_context_bge(
        self,
        facts: List[Dict[str, Any]],
        core_entity: str,
        top_k: int = 3,
    ) -> Tuple[List[Dict[str, Any]], List[float], List[int]]:
        docs = [f"{f.get('title', '')}. {f.get('text', '')}" for f in facts]
        scores = self._bge_similarity_scores(core_entity, docs)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = order[:top_k]
        candidates = [facts[i] for i in top]
        reranked, rerank_scores = self._rerank_facts(
            core_entity,
            candidates,
            use_title_only=False,
            top_k=top_k,
        )
        reranked_indices = [f.get("idx") for f in reranked]
        return reranked, rerank_scores, reranked_indices

    def _is_retrieval_failure(self, result: Dict[str, Any]) -> bool:
        return (not result.get("found")) or (not result.get("selected_fact_indices"))

    def _extract_focus_terms(self, subquestion: str, core_entity: str) -> List[str]:
        sub_tokens = [t for t in re.findall(r"[A-Za-z0-9]+", (subquestion or "").lower())]
        core_tokens = set(re.findall(r"[A-Za-z0-9]+", (core_entity or "").lower()))
        focus = [t for t in sub_tokens if t not in core_tokens and len(t) > 2]
        seen = set()
        ordered: List[str] = []
        for t in focus:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        return ordered[:6]

    # ---------- Retrieval (bge-m3 + bge-reranker) ----------
    def retrieve(
        self,
        facts: List[Dict[str, Any]],
        subquestion: str,
        call_id: str,
        core_entity: str,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        query = (core_entity or "").strip()
        if not query and schema is not None:
            anchors = self._extract_anchors_from_schema(schema, subquestion)
            if anchors:
                query = anchors[0].strip()
        if not query:
            query = subquestion.strip()

        attempts: List[Dict[str, Any]] = []

        # Pass 1: title-top3
        title_candidates: List[Dict[str, Any]] = []
        title_scores: List[float] = []
        title_indices: List[int] = []
        if query:
            title_candidates, title_scores, title_indices = self._retrieve_topk_title_bge(
                facts=facts,
                core_entity=query,
                top_k=3,
            )

        first = self._run_gamma_once(
            subquestion=subquestion,
            candidate_facts=title_candidates if title_candidates else facts,
            call_id=f"{call_id}_t1",
            all_facts=facts,
        )
        first["retrieval_pass"] = "title_top3"
        first["retrieval_query"] = query
        first["retrieval_candidate_indices"] = title_indices
        first["retrieval_rerank_scores"] = title_scores
        attempts.append(
            {
                "pass": "title_top3",
                "query": query,
                "candidate_indices": title_indices,
                "rerank_scores": title_scores,
                "selected_fact_indices": first.get("selected_fact_indices", []),
                "found": first.get("found"),
                "reasoning": first.get("reasoning"),
            }
        )

        if not self._is_retrieval_failure(first):
            return first, attempts

        # Pass 2: context-top3
        context_candidates, context_scores, context_indices = self._retrieve_topk_context_bge(
            facts=facts,
            core_entity=query,
            top_k=3,
        )
        second = self._run_gamma_once(
            subquestion=subquestion,
            candidate_facts=context_candidates if context_candidates else facts,
            call_id=f"{call_id}_t2",
            all_facts=facts,
        )
        second["retrieval_pass"] = "context_top3"
        second["retrieval_query"] = query
        second["retrieval_candidate_indices"] = context_indices
        second["retrieval_rerank_scores"] = context_scores
        attempts.append(
            {
                "pass": "context_top3",
                "query": query,
                "candidate_indices": context_indices,
                "rerank_scores": context_scores,
                "selected_fact_indices": second.get("selected_fact_indices", []),
                "found": second.get("found"),
                "reasoning": second.get("reasoning"),
            }
        )
        return second, attempts

    # ---------- Path replay ----------
    def path_replay(
        self,
        subquestion: str,
        core_entity: str,
        attempts: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Replay failed retrieval attempts and produce guidance for the next round.
        """
        candidate_indices: List[int] = []
        selected_indices: List[int] = []
        for att in attempts:
            candidate_indices.extend(att.get("candidate_indices", []) or [])
            selected_indices.extend(att.get("selected_fact_indices", []) or [])

        candidate_set = sorted(set(int(i) for i in candidate_indices))
        selected_set = sorted(set(int(i) for i in selected_indices))
        avoid_indices = selected_set if selected_set else candidate_set

        no_selected = all(not (att.get("selected_fact_indices") or []) for att in attempts)
        all_failed = all(not att.get("found") for att in attempts)
        if no_selected:
            reason = "no_selected_facts"
        elif all_failed:
            reason = "answer_not_found"
        else:
            reason = "insufficient_evidence"

        focus_terms = self._extract_focus_terms(subquestion, core_entity)
        if core_entity:
            next_query = f"{core_entity} " + " ".join(focus_terms)
            next_query = next_query.strip()
        else:
            next_query = " ".join([subquestion] + focus_terms).strip()

        fact_map = {f.get("idx"): f for f in facts}
        avoid_titles = [
            fact_map[i].get("title", "")
            for i in avoid_indices
            if i in fact_map
        ]

        return {
            "failure_reason": reason,
            "avoid_fact_indices": avoid_indices,
            "avoid_titles": avoid_titles,
            "focus_terms": focus_terms,
            "next_query": next_query,
        }

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

    # ---------- Anchor extraction ----------

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

    # ---------- Retrieve + answer a single subquestion ----------

    def answer_subquestion(
        self,
        example: Dict[str, Any],
        subquestion: str,
        call_id: str,
        core_entity: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieval strategy:
        1) title-top3 by bge-m3 similarity between core_entity and fact titles + bge-reranker.
        2) if failed, context-top3 by bge-m3 similarity between core_entity and full fact text + bge-reranker.
        If core_entity is empty, fall back to full-context retrieval with the subquestion.
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

        result, attempts = self.retrieve(
            facts=facts,
            subquestion=subquestion,
            call_id=call_id,
            core_entity=(core_entity or ""),
            schema=schema,
        )
        result["retrieval_attempts"] = attempts
        if self._is_retrieval_failure(result):
            result["retrieval_replay"] = self.path_replay(
                subquestion=subquestion,
                core_entity=(core_entity or ""),
                attempts=attempts,
                facts=facts,
            )
        return self._attach_local_gamma_memory(result, subquestion, call_id)

class ACC:
    """
    ACC: sub-answer checker.

    Main interface:
        check_subanswer(...) -> [int, int, int]
    """

    def __init__(self, dataset_name: str, llm_client: Optional[LLMClient] = None):
        assert dataset_name in {"2wiki", "hotpotqa", "musique"}
        self.dataset_name = dataset_name
        self.llm = llm_client

    def _normalize_text(self, text: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9\s]+", " ", (text or "").lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]+", (text or "").lower())

    def _core_entity_in_evidence(self, core_entity: str, evidence_text: str) -> bool:
        if not core_entity or not evidence_text:
            return False
        core_norm = self._normalize_text(core_entity)
        evidence_norm = self._normalize_text(evidence_text)
        if not core_norm or not evidence_norm:
            return False
        if core_norm in evidence_norm:
            return True
        core_tokens = self._tokenize(core_norm)
        evidence_tokens = set(self._tokenize(evidence_norm))
        return bool(core_tokens) and all(t in evidence_tokens for t in core_tokens)

    def _reason_supports(
        self,
        answer: str,
        evidence_text: str,
        reason: str,
    ) -> bool:
        if not answer or not evidence_text or not reason:
            return False
        answer_norm = self._normalize_text(answer)
        evidence_norm = self._normalize_text(evidence_text)
        reason_norm = self._normalize_text(reason)
        if not answer_norm or not evidence_norm or not reason_norm:
            return False

        answer_tokens = self._tokenize(answer_norm)
        evidence_tokens = set(self._tokenize(evidence_norm))
        reason_tokens = set(self._tokenize(reason_norm))

        answer_in_evidence = (
            answer_norm in evidence_norm
            or (answer_tokens and all(t in evidence_tokens for t in answer_tokens))
        )
        if not answer_in_evidence:
            return False

        stopwords = {
            "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
            "is", "are", "was", "were", "be", "by", "as", "at", "from", "that",
            "this", "these", "those", "it", "its", "he", "she", "they", "them",
            "his", "her", "their", "we", "you", "i", "our", "your",
            "fact", "facts", "evidence", "shows", "show", "indicates",
        }
        evidence_keywords = [
            t for t in self._tokenize(evidence_norm)
            if t not in stopwords and len(t) > 2
        ]
        keyset = set(evidence_keywords[:50])

        reason_mentions_answer = (
            answer_norm in reason_norm
            or any(t in reason_tokens for t in answer_tokens)
        )
        reason_mentions_evidence = any(t in reason_tokens for t in keyset)

        return reason_mentions_answer or reason_mentions_evidence

    def check_subanswer(
        self,
        subquestion: str,
        core_entity: str,
        expected_type: str,
        subanswer: str,
        actual_type: str,
        evidence: List[str],
        reason: str,
    ) -> List[int]:
        """
        Return three checks as 0/1:
        1) evidence contains core entity
        2) expected type matches actual type
        3) reason supports the answer from evidence
        """
        evidence_text = " ".join(str(e) for e in (evidence or []))

        check1 = 1 if self._core_entity_in_evidence(core_entity, evidence_text) else 0

        exp = (expected_type or "").strip().lower()
        act = (actual_type or "").strip().lower()
        check2 = 1 if exp and act and exp == act else 0

        check3 = 1 if self._reason_supports(subanswer, evidence_text, reason) else 0

        return [check1, check2, check3]
