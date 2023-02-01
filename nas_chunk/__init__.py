"""
nas_chunk — مكتبة بسيطة وسريعة لتحويل أي نص طويل إلى ملف JSON
منظم وجاهز لأنظمة RAG، باستخدام Gemini.

الاستخدام الأساسي:

    from nas_chunk import text_to_rag_json

    result = text_to_rag_json(
        text=my_long_text,
        api_key="YOUR_GEMINI_API_KEY",
        output_path="output.json",
    )
"""

from .orchestrator import text_to_rag_json
from .splitter import split_text
from .gemini_client import GeminiJSONConverter

import json
from typing import Any, Dict, List, Optional


def ask(
    question: str,
    json_path: str,
    top_k: int = 3,
    api_key: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> Dict[str, Any]:
    """بحث محلي خفيف داخل ناتج `text_to_rag_json` ويُرجع إجابة مع المصادر.

    الدالة بسيطة وتعتمد على تطابق كلمات السؤال مع `title`/`summary`/`content`.
    تقبل `api_key` و `cache_path` لمطابقة توقيع الاستخدام القديم لكنهما
    غير مستخدمين هنا (يمكن توسيعهم لاحقاً لإضافة بحث دلالي).

    ترجع dict يحوي `question`, `matches`, `answer`, `sources`.
    """
    if not question or not json_path:
        raise ValueError("لازم تحدد `question` و `json_path`.")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"فشل في قراءة JSON من {json_path}: {exc}") from exc

    chunks = data.get("chunks") or []

    q_tokens = [t for t in question.lower().split() if t]

    def score_chunk(chunk: Dict[str, Any]) -> int:
        text = " ".join(
            [
                str(chunk.get("title", "")),
                str(chunk.get("summary", "")),
                str(chunk.get("content", "")),
            ]
        ).lower()
        s = 0
        for t in q_tokens:
            if t in text:
                s += 2
        for qq in chunk.get("questions", []) or []:
            if any(t in str(qq).lower() for t in q_tokens):
                s += 3
        return s

    scored: List[Dict[str, Any]] = []
    for c in chunks:
        if not c:
            continue
        sc = score_chunk(c)
        scored.append({"score": sc, "chunk": c})

    scored.sort(key=lambda x: x["score"], reverse=True)

    matches = [{"score": s["score"], **s["chunk"]} for s in scored if s["score"] > 0][:top_k]

    # إجابة مبسطة: جمع الـ summaries أو محتوى أفضل النتائج
    if matches:
        parts = []
        for m in matches:
            if m.get("summary"):
                parts.append(m.get("summary"))
            else:
                parts.append(m.get("content", ""))
        answer = "\n\n".join(parts)
        sources = [m.get("id") for m in matches if m.get("id")]
    else:
        answer = ""
        sources = []

    return {"question": question, "matches": matches, "answer": answer, "sources": sources}

__all__ = [
    "text_to_rag_json",
    "split_text",
    "GeminiJSONConverter",
    "ask",
]

__version__ = "0.1.0"
