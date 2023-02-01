"""
المحرك الرئيسي للمكتبة: بياخد نص طويل + Gemini API key، يقسمه لأجزاء،
يبعت كل جزء لـ Gemini بالتوازي (عشان السرعة)، ويطلع ملف JSON واحد
منظم وجاهز للاستخدام في RAG.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .gemini_client import GeminiJSONConverter
from .splitter import split_text


def text_to_rag_json(
    text: str,
    api_key: str,
    output_path: Optional[str] = None,
    *,
    model: str = "gemini-2.0-flash",
    max_chars: int = 4000,
    overlap_chars: int = 300,
    max_workers: int = 8,
    max_retries: int = 3,
    temperature: float = 0.2,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    الدالة الرئيسية للمكتبة.

    المعاملات
    ----------
    text : النص الكامل اللي عايز تحوله.
    api_key : مفتاح Gemini API.
    output_path : لو حددته، هيتكتب ملف JSON على المسار ده. لو None هيرجع الديكشنري بس.
    model : اسم موديل Gemini (افتراضي gemini-2.0-flash عشان السرعة).
    max_chars : أقصى حجم لكل chunk بالحروف.
    overlap_chars : قد إيه تداخل بين كل chunk والتاني.
    max_workers : عدد الـ threads اللي هتشتغل بالتوازي (بيزود السرعة جداً).
    max_retries : عدد محاولات إعادة الطلب لو Gemini رجع رد مش JSON صحيح.
    temperature : درجة إبداع الموديل (قليلة = أدق وأثبت للـ RAG).
    progress_callback : دالة اختيارية بتتنادى بعد كل chunk خالص، بشكل (done, total).

    ترجع
    ----
    dict فيها:
        {
          "chunks": [...],
          "meta": {...}
        }
    """
    if not text or not text.strip():
        raise ValueError("النص فاضي، مفيش حاجة نقسمها.")

    raw_chunks = split_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
    total = len(raw_chunks)

    converter = GeminiJSONConverter(
        api_key=api_key,
        model=model,
        max_retries=max_retries,
        temperature=temperature,
    )

    results: List[Optional[Dict[str, Any]]] = [None] * total
    start_time = time.time()
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(converter.convert_chunk, chunk, idx): idx
            for idx, chunk in enumerate(raw_chunks)
        }

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result()
            done_count += 1
            if progress_callback:
                progress_callback(done_count, total)

    elapsed = time.time() - start_time

    output: Dict[str, Any] = {
        "chunks": results,
        "meta": {
            "chunk_count": total,
            "model": model,
            "source_char_count": len(text),
            "max_chars_per_chunk": max_chars,
            "overlap_chars": overlap_chars,
            "processing_time_seconds": round(elapsed, 2),
            "failed_chunks": sum(
                1
                for r in results
                if r and r.get("metadata", {}).get("status") != "ok"
            ),
        },
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    return output
