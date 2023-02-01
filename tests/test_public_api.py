import os

import nas_chunk


def test_build_and_query_rag_uses_single_entrypoint(tmp_path, monkeypatch):
    captured = {}

    def fake_text_to_rag_json(**kwargs):
        captured["text_to_rag_json"] = kwargs
        return {"chunks": [], "meta": {"chunk_count": 0}}

    def fake_ask(**kwargs):
        captured["ask"] = kwargs
        return {"answer": "done", "sources": []}

    monkeypatch.setattr(nas_chunk, "text_to_rag_json", fake_text_to_rag_json)
    monkeypatch.setattr(nas_chunk, "ask", fake_ask)

    output_path = tmp_path / "output.json"
    result = nas_chunk.build_and_query_rag(
        text="hello world",
        question="what is this?",
        api_key="fake-key",
        output_path=str(output_path),
    )

    assert result["answer"]["answer"] == "done"
    assert captured["text_to_rag_json"]["text"] == "hello world"
    assert captured["ask"]["question"] == "what is this?"
    assert captured["ask"]["json_path"] == str(output_path)
    assert os.path.exists(output_path)
