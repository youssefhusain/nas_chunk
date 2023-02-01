# nas_chunk

مكتبة بايثون بسيطة وسريعة، بتاخد أي نص طويل (docs, PDF text, صفحات ويب...)
وبتحوله لملف **JSON منظم جاهز لأنظمة RAG**، باستخدام **Gemini**.

## الفكرة

1. بتقسم النص الكبير لأجزاء (chunks) متوازنة، مع تداخل بسيط بينهم عشان مايضيعش السياق.
2. بتبعت كل chunk لـ Gemini **بالتوازي** (multi-threading) عشان السرعة.
3. Gemini بيرجع لكل chunk:
   - عنوان مختصر
   - ملخص
   - نص منظف
   - كلمات مفتاحية
   - أسئلة محتملة يجاوب عليها الـ chunk ده (بتساعد الـ retrieval يطلع أدق نتايج)
4. النتيجة كلها بتتجمع في ملف JSON واحد جاهز تحطه في أي vector DB / retriever.

## التثبيت

```bash
pip install -r requirements.txt
# أو لو نزلتها كباكدج
pip install .
```

## الاستخدام

```python
from nas_chunk import text_to_rag_json

with open("my_document.txt", "r", encoding="utf-8") as f:
    text = f.read()

result = text_to_rag_json(
    text=text,
    api_key="YOUR_GEMINI_API_KEY",
    output_path="output.json",   # هيتكتب الملف هنا
    max_chars=4000,              # حجم كل chunk (اختياري)
    overlap_chars=300,           # تداخل بين الأجزاء (اختياري)
    max_workers=8,               # عدد الطلبات المتوازية (اختياري - زوّدها للسرعة)
    model="gemini-2.0-flash",    # موديل سريع، تقدر تغيره لأي موديل تاني
)

print(result["meta"])
```

## شكل ملف الـ JSON الناتج

```json
{
  "chunks": [
    {
      "id": "chunk_00000",
      "chunk_index": 0,
      "title": "...",
      "summary": "...",
      "content": "...",
      "keywords": ["...", "..."],
      "questions": ["...", "..."],
      "metadata": {
        "char_count": 1234,
        "source_char_count": 1300,
        "status": "ok"
      }
    }
  ],
  "meta": {
    "chunk_count": 12,
    "model": "gemini-2.0-flash",
    "source_char_count": 48000,
    "max_chars_per_chunk": 4000,
    "overlap_chars": 300,
    "processing_time_seconds": 9.3,
    "failed_chunks": 0
  }
}
```

## ملاحظات لأداء أفضل في RAG

- استخدم `content` كنص أساسي للـ embedding.
- استخدم `questions` و `keywords` كـ metadata إضافية أو حتى اعملهم embeddings منفصلة، بيزودوا دقة الاسترجاع كتير لأنهم بيغطوا صيغ أسئلة المستخدم الحقيقية.
- لو عندك نص كبير جداً (مئات الآلاف من الحروف)، زوّد `max_workers` (لكن خلي بالك من rate limits بتاعة Gemini).
- لو حصل خطأ في أي chunk، هترجعله بره الـ JSON بحالة `"status": "fallback_raw_text"` بدل ما توقف كل العملية.
