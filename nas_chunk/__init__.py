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


def ask(question: str, json_path: str, top_k: int = 3) -> Dict[str, Any]:
    """بحث بسيط داخل ناتج `text_to_rag_json` عن أفضل الـ chunks للإجابة.

    هذه دالة مساعدة خفيفة للبحث المحلي تُسهل الاستخدام من ناتج الـ RAG.

    المعاملات
    ----------
    question: نص السؤال.
    json_path: مسار ملف JSON الناتج من `text_to_rag_json`.
    top_k: عدد النتائج المرجعة.

    ترجع
    -----
    dict مع `question` و `matches` مرتبة حسب درجة تشابه بسيطة.
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
        # فحص أسئلة مُقترحة داخل الـ chunk لڤترة أقوى
        for qq in chunk.get("questions", []) or []:
            if any(t in str(qq).lower() for t in q_tokens):
                s += 3
        return s

    scored: List[Dict[str, Any]] = []
    for c in chunks:
        sc = score_chunk(c or {})
        if sc > 0:
            scored.append({"score": sc, "chunk": c})

    scored.sort(key=lambda x: x["score"], reverse=True)

    matches = [
        {"score": s["score"], **s["chunk"]} for s in scored[:top_k]
    ]

    return {"question": question, "matches": matches}

__all__ = [
    "text_to_rag_json",
    "split_text",
    "GeminiJSONConverter",
    "ask",
]

__version__ = "0.1.0"
