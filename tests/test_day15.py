"""
Day 15 tests — validates LlamaService boundaries and error handling.
Some tests require Ollama running; others test logic without it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.llama_service import (
    LlamaConfig,
    check_ollama_connection,
    validate_response,
    LlamaService,
    get_llama_service,
)


def test_config_has_required_boundaries():
    config = LlamaConfig()
    assert config.timeout_seconds > 0
    assert config.max_tokens > 0
    assert config.max_retries >= 0
    assert 0.0 <= config.temperature <= 1.0


def test_validate_response_rejects_none():
    is_valid, reason = validate_response(None)
    assert is_valid is False
    assert "None" in reason


def test_validate_response_rejects_empty():
    is_valid, reason = validate_response("")
    assert is_valid is False


def test_validate_response_rejects_too_short():
    is_valid, reason = validate_response("Hi", min_length=5)
    assert is_valid is False
    assert "short" in reason.lower()


def test_validate_response_accepts_normal_text():
    is_valid, reason = validate_response(
        "Your Tapal Tea stock is running low and should be reordered soon."
    )
    assert is_valid is True
    assert reason == "Valid"


def test_validate_response_detects_repetition():
    repeated_text = "the the the " * 20
    is_valid, reason = validate_response(repeated_text)
    assert is_valid is False
    assert "repetition" in reason.lower()


def test_connection_check_returns_structured_dict():
    # Should never raise, always returns a dict with these keys
    status = check_ollama_connection()
    assert "connected" in status
    assert "phi3_ready" in status
    assert isinstance(status["connected"], bool)


def test_llama_service_singleton():
    service1 = get_llama_service()
    service2 = get_llama_service()
    # Should be the SAME object (singleton pattern)
    assert service1 is service2


def test_llama_service_generate_returns_structured_response():
    # Tests the RETURN SHAPE works even if Ollama is offline —
    # generate() should never raise, always return a dict
    service = get_llama_service()
    result  = service.generate("Test prompt")

    assert "success"      in result
    assert "text"         in result
    assert "duration_sec" in result
    assert "attempts"     in result
    assert "error"        in result
    assert isinstance(result["success"], bool)


def test_llama_service_is_ready_returns_bool():
    service = get_llama_service()
    assert isinstance(service.is_ready(), bool)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"✅ {name} passed")