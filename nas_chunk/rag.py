"""
rag.py — موديول RAG كامل (Retrieval + Generation) فوق مخرجات nas_chunk.

انقل هذا الملف من جذر الحزمة إلى الموديول الداخلي `nas_chunk/` ليتم
استيراده كجزء من الحزمة الحقيقية بدلاً من النسخة الطالعة في الجذر.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"

ANSWER_SYSTEM_PROMPT = """أنت محرك إجابات (Answer Engine) لنظام RAG احترافي وسريع.
هتستلم:
- سؤال المستخدم.
- مجموعة من المقاطع (context chunks) مسترجعة من قاعدة معرفة، كل مقطع ليه رقم [1], [2], ...

القواعد اللي لازم تلتزم بيها بدون استثناء:
1. جاوب فقط بناءً على المعلومات الموجودة في الـ context تحت. ممنوع تختلق أو تفترض أي معلومة مش موجودة فيه.
2. لو السياق مفيهوش إجابة كافية للسؤال، قول بصراحة إن المعلومة مش متوفرة في المصادر المتاحة، وميتفلسفش.
3. جاوب بنفس لغة سؤال المستخدم (لو سأل بالعربي جاوب بالعربي، لو بالإنجليزي جاوب بالإنجليزي).
4. خلي الإجابة مختصرة، مباشرة، ومنظمة (نقط أو فقرات قصيرة لو الموضوع فيه أكتر من جزئية).
5. بعد كل معلومة أساسية حط رقم المصدر اللي جبتها منه بالشكل ده: [1] أو [2] حسب رقم المقطع في الـ context.
6. متكررش أرقام السياق نفسها زي ما هي، ومتشرحش إنك "هترجع بناءً على السياق" — جاوب مباشرة.

رجّع الإجابة كنص عادي فقط (مش JSON).
"""


# ---------------------------------------------------------------------------
# أدوات مساعدة عامة
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini API error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini connection error: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini returned invalid JSON") from exc


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _chunk_embedding_text(chunk: Dict[str, Any]) -> str:
    parts = [
        chunk.get("title", ""),
        chunk.get("summary", ""),
        chunk.get("content", ""),
    ]
    keywords = chunk.get("keywords") or []
    questions = chunk.get("questions") or []
    if keywords:
        parts.append("كلمات مفتاحية: " + "، ".join(keywords))
    if questions:
        parts.append("أسئلة محتملة: " + " | ".join(questions))
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# عميل الـ Embeddings
# ---------------------------------------------------------------------------


class GeminiEmbeddingClient:
    """غلاف بسيط فوق Gemini Embeddings API (models/embedContent)."""

    def __init__(
        self,
        api_key: str,
        model: str = EMBEDDING_MODEL,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        if not api_key:
            raise ValueError("لازم تدي api_key بتاع Gemini.")
        self.api_key = api_key
        self.model_name = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def embed(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(self.model_name)}:embedContent"
            f"?key={urllib.parse.quote(self.api_key)}"
        )
        payload = {
            "model": f"models/{self.model_name}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = _post_json(endpoint, payload)
                values = result.get("embedding", {}).get("values")
                if values:
                    return values
                last_error = RuntimeError(f"Gemini embedding response غريب: {result}")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"فشل عمل embedding بعد {self.max_retries} محاولات: {last_error}")

    def embed_batch(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        max_workers: int = 8,
        progress_callback: Optional[Any] = None,
    ) -> List[List[float]]:
        results: List[Optional[List[float]]] = [None] * len(texts)
        done_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.embed, text, task_type): idx
                for idx, text in enumerate(texts)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()
                done_count += 1
                if progress_callback:
                    progress_callback(done_count, len(texts))

        return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# فهرس الاسترجاع (Index)
# ---------------------------------------------------------------------------


class RAGIndex:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        api_key: str,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self.chunks = chunks
        self.embedding_client = GeminiEmbeddingClient(api_key, model=embedding_model)
        self._embeddings: Optional[List[List[float]]] = None

    @classmethod
    def from_json_file(
        cls, json_path: str, api_key: str, embedding_model: str = EMBEDDING_MODEL
    ) -> "RAGIndex":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = data.get("chunks", [])
        return cls(chunks, api_key, embedding_model=embedding_model)

    def build(
        self,
        cache_path: Optional[str] = None,
        max_workers: int = 8,
        force_rebuild: bool = False,
        progress_callback: Optional[Any] = None,
    ) -> "RAGIndex":
        if not force_rebuild and cache_path and os.path.exists(cache_path):
            if self._load_cache(cache_path):
                return self

        texts = [_chunk_embedding_text(chunk) for chunk in self.chunks]
        self._embeddings = self.embedding_client.embed_batch(
            texts, task_type="RETRIEVAL_DOCUMENT",
            max_workers=max_workers, progress_callback=progress_callback,
        )

        if cache_path:
            self._save_cache(cache_path)

        return self

    def _save_cache(self, cache_path: str) -> None:
        cache = {
            "model": self.embedding_client.model_name,
            "chunk_ids": [c.get("id") for c in self.chunks],
            "embeddings": self._embeddings,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)

    def _load_cache(self, cache_path: str) -> bool:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False

        cached_ids = cache.get("chunk_ids") or []
        current_ids = [c.get("id") for c in self.chunks]
        if cached_ids != current_ids or cache.get("model") != self.embedding_client.model_name:
            return False

        self._embeddings = cache.get("embeddings")
        return bool(self._embeddings)

    def search(
        self, query: str, top_k: int = 5, keyword_boost: float = 0.08
    ) -> List[Tuple[Dict[str, Any], float]]:
        if self._embeddings is None:
            raise RuntimeError("لازم تعمل index.build() الأول قبل البحث.")

        query_embedding = self.embedding_client.embed(query, task_type="RETRIEVAL_QUERY")
        query_terms = set(re.findall(r"\w+", query.lower()))

        scored: List[Tuple[Dict[str, Any], float]] = []
        for chunk, embedding in zip(self.chunks, self._embeddings):
            score = cosine_similarity(query_embedding, embedding)

            haystack_terms = set(
                re.findall(
                    r"\w+",
                    " ".join(chunk.get("keywords") or []).lower()
                    + " "
                    + " ".join(chunk.get("questions") or []).lower(),
                )
            )
            overlap = len(query_terms & haystack_terms)
            if overlap:
                score += keyword_boost * min(overlap, 3)

            scored.append((chunk, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# محرك الإجابة (Generation)
# ---------------------------------------------------------------------------


class RAGEngine:
    def __init__(
        self,
        index: RAGIndex,
        api_key: str,
        llm_model: str = LLM_MODEL,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.index = index
        self.api_key = api_key
        self.llm_model = llm_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _build_context(self, results: List[Tuple[Dict[str, Any], float]]) -> str:
        blocks = []
        for i, (chunk, score) in enumerate(results, start=1):
            title = chunk.get("title") or f"مقطع {chunk.get('chunk_index')}"
            content = chunk.get("content", "")
            blocks.append(f"[{i}] {title}\n{content}")
        return "\n\n".join(blocks)

    def _call_llm(self, question: str, context: str, temperature: float) -> str:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(self.llm_model)}:generateContent"
            f"?key={urllib.parse.quote(self.api_key)}"
        )
        user_prompt = f"السياق (context):\n{context}\n\nسؤال المستخدم:\n{question}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "systemInstruction": {"role": "system", "parts": [{"text": ANSWER_SYSTEM_PROMPT}]},
            "generationConfig": {"temperature": temperature},
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = _post_json(endpoint, payload, timeout=60)
                candidates = result.get("candidates") or []
                if not candidates:
                    raise RuntimeError(f"Gemini رجع بدون candidates: {result}")
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                if not text:
                    raise RuntimeError(f"Gemini رجع رد فاضي: {result}")
                return text.strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"فشل استدعاء Gemini LLM بعد {self.max_retries} محاولات: {last_error}")

    def answer(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.2,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        start = time.time()

        results = self.index.search(question, top_k=top_k)
        results = [r for r in results if r[1] >= min_score] or results[:1]

        context = self._build_context(results)
        answer_text = self._call_llm(question, context, temperature)

        sources = [
            {
                "ref": f"[{i}]",
                "chunk_id": chunk.get("id"),
                "title": chunk.get("title"),
                "score": round(score, 4),
            }
            for i, (chunk, score) in enumerate(results, start=1)
        ]

        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "elapsed_seconds": round(time.time() - start, 2),
        }


# ---------------------------------------------------------------------------
# دالة تسهيل سريعة (all-in-one)
# ---------------------------------------------------------------------------

_INDEX_CACHE: Dict[str, RAGIndex] = {}


def build_index(
    json_path: str,
    api_key: str,
    cache_path: Optional[str] = None,
    embedding_model: str = EMBEDDING_MODEL,
    max_workers: int = 8,
    force_rebuild: bool = False,
) -> RAGIndex:
    index = RAGIndex.from_json_file(json_path, api_key, embedding_model=embedding_model)
    index.build(cache_path=cache_path, max_workers=max_workers, force_rebuild=force_rebuild)
    return index


def ask(
    question: str,
    json_path: str,
    api_key: str,
    cache_path: Optional[str] = None,
    top_k: int = 5,
    llm_model: str = LLM_MODEL,
    embedding_model: str = EMBEDDING_MODEL,
    temperature: float = 0.2,
    reuse_index: bool = True,
) -> Dict[str, Any]:
    cache_key = f"{json_path}:{cache_path}:{embedding_model}"
    if reuse_index and cache_key in _INDEX_CACHE:
        index = _INDEX_CACHE[cache_key]
    else:
        index = build_index(
            json_path, api_key, cache_path=cache_path, embedding_model=embedding_model
        )
        if reuse_index:
            _INDEX_CACHE[cache_key] = index

    engine = RAGEngine(index, api_key, llm_model=llm_model)
    return engine.answer(question, top_k=top_k, temperature=temperature)
