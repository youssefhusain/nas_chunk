"""
تقسيم النصوص الطويلة إلى أجزاء (chunks) متوازنة الحجم مع تداخل بسيط
بين كل جزء والتالي عشان السياق مايتقطعش عند حدود القسمة.
"""

from __future__ import annotations

import re
from typing import List


def _split_into_sentences(text: str) -> List[str]:
    """تقسيم بدائي للنص إلى جمل (يدعم العربي والإنجليزي)."""
    # نقسم بعد علامات الترقيم (. ! ? ، ؟ !) مع الحفاظ عليها
    pattern = r"(?<=[\.\!\?\؟\؛])\s+"
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def split_text(
    text: str,
    max_chars: int = 4000,
    overlap_chars: int = 300,
) -> List[str]:
    """
    يقسم النص لأجزاء بحجم أقصى max_chars حرف، مع تداخل overlap_chars
    حرف بين كل جزء والي بعده عشان نحافظ على السياق.

    القسمة بتحصل على حدود جمل كامل قد الإمكان (مش بتقطع في نص الجملة).
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = _split_into_sentences(text)
    if not sentences:
        sentences = [text]

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        # لو الجملة نفسها أطول من max_chars قسّمها بالقوة
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i : i + max_chars].strip())
            continue

        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current.strip())
            # ابدأ الجزء الجديد بتداخل من نهاية الجزء اللي فات
            if overlap_chars > 0 and chunks:
                tail = chunks[-1][-overlap_chars:]
                current = f"{tail} {sentence}".strip()
            else:
                current = sentence

    if current:
        chunks.append(current.strip())

    return chunks
