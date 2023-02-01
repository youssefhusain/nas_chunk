"""
غلاف بسيط فوق Gemini API لتحويل كل قطعة نص (chunk) إلى JSON منظم
جاهز للاستخدام في أنظمة RAG (retrieval-augmented generation).
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


SYSTEM_PROMPT = """You are a text-processing engine for a RAG (Retrieval Augmented Generation) pipeline.
You will receive ONE chunk of text (possibly Arabic, possibly mixed language).
Your job is to transform it into a SINGLE JSON object (no markdown, no code fences, no extra text) with EXACTLY this schema:

{
  "title": "short descriptive title for this chunk (max 10 words)",
  "summary": "2-3 sentence summary of the chunk, in the SAME language as the original text",
  "content": "the cleaned, well-formatted version of the original text (keep all facts, keep original language, remove noise/broken formatting)",
  "keywords": ["list", "of", "5-10", "key", "terms", "in original language"],
  "questions": ["3-6 realistic questions a user might ask that THIS chunk directly answers, in original language"]
}

Rules:
- Output ONLY valid JSON. No ```json fences. No explanations before or after.
- Keep "content" faithful to the source — do not invent facts, do not omit important info.
- Keep the original language of the text (do not translate).
- If the chunk is very short or fragmented, still produce your best-effort JSON with the same schema.
"""


class GeminiJSONConverter:
    """يحول أي قطعة نص إلى JSON منظم باستخدام Gemini."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.2,
    ) -> None:
        if not api_key:
            raise ValueError("لازم تدي api_key بتاع Gemini.")

        self.api_key = api_key
        self.model_name = self._resolve_model_name(model)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature

    @staticmethod
    def _resolve_model_name(model: str) -> str:
        """يرجع اسم موديل مدعوم بدل الأسماء القديمة التي اختفت."""
        normalized = (model or "").strip()
        if not normalized:
            return "gemini-2.5-flash"

        mapping = {
            "gemini-2.0-flash": "gemini-2.5-flash",
            "gemini-2.0-flash-latest": "gemini-2.5-flash",
            "gemini-2.0-flash-thinking-exp": "gemini-2.5-flash",
        }
        return mapping.get(normalized, normalized)

    def _generate_content(self, chunk_text: str) -> str:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(self.model_name)}:generateContent"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": chunk_text}]}],
            "systemInstruction": {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
            },
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint + f"?key={urllib.parse.quote(self.api_key)}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Gemini API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini connection error: {exc.reason}") from exc

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini returned invalid JSON") from exc

        candidates = result.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {body}")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError(f"Gemini returned empty content: {body}")

        text_parts = []
        for part in parts:
            if isinstance(part, dict):
                text_parts.append(part.get("text", ""))

        assembled = "".join(text_parts)
        if not assembled:
            raise RuntimeError(f"Gemini returned no textual content: {body}")

        return assembled

    @staticmethod
    def _extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
        """يحاول يطلع JSON صحيح من رد الموديل حتى لو فيه نص زيادة حواليه."""
        raw_text = raw_text.strip()
        # حاول أول حاجة parse مباشر
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # حاول تشيل أي code fence
        cleaned = re.sub(r"^```(json)?|```$", "", raw_text, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # حاول تلاقي أول { ... } متوازن في النص
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def convert_chunk(self, chunk_text: str, chunk_index: int) -> Dict[str, Any]:
        """يحول قطعة نص واحدة لـ JSON. بيعمل retry تلقائي لو الرد مش JSON صحيح."""
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_text = self._generate_content(chunk_text)
                parsed = self._extract_json(raw_text)

                if parsed is not None:
                    return self._normalize(parsed, chunk_text, chunk_index)

                last_error = ValueError("الرد مش JSON صحيح")

            except Exception as exc:  # noqa: BLE001
                last_error = exc

            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)

        # لو كل المحاولات فشلت، نرجع fallback بسيط بدل ما نوقف كل العملية
        return self._fallback(chunk_text, chunk_index, last_error)

    @staticmethod
    def _normalize(
        parsed: Dict[str, Any], original_chunk: str, chunk_index: int
    ) -> Dict[str, Any]:
        return {
            "id": f"chunk_{chunk_index:05d}",
            "chunk_index": chunk_index,
            "title": parsed.get("title", ""),
            "summary": parsed.get("summary", ""),
            "content": parsed.get("content", original_chunk),
            "keywords": parsed.get("keywords", []) or [],
            "questions": parsed.get("questions", []) or [],
            "metadata": {
                "char_count": len(parsed.get("content", original_chunk)),
                "source_char_count": len(original_chunk),
                "status": "ok",
            },
        }

    @staticmethod
    def _fallback(
        original_chunk: str, chunk_index: int, error: Optional[Exception]
    ) -> Dict[str, Any]:
        return {
            "id": f"chunk_{chunk_index:05d}",
            "chunk_index": chunk_index,
            "title": "",
            "summary": "",
            "content": original_chunk,
            "keywords": [],
            "questions": [],
            "metadata": {
                "char_count": len(original_chunk),
                "source_char_count": len(original_chunk),
                "status": "fallback_raw_text",
                "error": str(error) if error else None,
            },
        }
