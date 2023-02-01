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

__all__ = [
    "text_to_rag_json",
    "split_text",
    "GeminiJSONConverter",
]

__version__ = "0.1.0"
