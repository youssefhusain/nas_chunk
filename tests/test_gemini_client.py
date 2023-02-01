from nas_chunk.gemini_client import GeminiJSONConverter


def test_deprecated_model_names_are_mapped_to_supported_models():
    assert GeminiJSONConverter._resolve_model_name("gemini-2.0-flash") == "gemini-2.5-flash"
    assert GeminiJSONConverter._resolve_model_name("gemini-2.0-flash-latest") == "gemini-2.5-flash"
    assert GeminiJSONConverter._resolve_model_name("gemini-2.5-flash") == "gemini-2.5-flash"
