"""
Day 11 tests — validates chains work without testing Ollama
(Ollama may not be running in CI environments).
Tests prompt structure and chain composition only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from rag.prompts import (
    get_base_prompt,
    get_analytics_prompt,
    get_inventory_prompt,
    get_rag_prompt,
)
from rag.chains import route_question


def test_base_prompt_has_question_variable():
    prompt = get_base_prompt()
    assert isinstance(prompt, ChatPromptTemplate)
    assert "question" in prompt.input_variables


def test_analytics_prompt_variables():
    prompt = get_analytics_prompt()
    vars   = prompt.input_variables
    assert "question"      in vars
    assert "store_context" in vars


def test_inventory_prompt_variables():
    prompt = get_inventory_prompt()
    vars   = prompt.input_variables
    assert "question"          in vars
    assert "inventory_context" in vars
    assert "alerts_context"    in vars


def test_rag_prompt_variables():
    prompt = get_rag_prompt()
    vars   = prompt.input_variables
    assert "question" in vars
    assert "context"  in vars


def test_router_inventory_keywords():
    assert route_question("What should I restock?")       == "inventory"
    assert route_question("Which items are dead stock?")  == "inventory"
    assert route_question("low stock alert")              == "inventory"


def test_router_analytics_keywords():
    assert route_question("What is my revenue this week?") == "analytics"
    assert route_question("Which category has best margin") == "analytics"
    assert route_question("show me profit")                 == "analytics"


def test_router_base_fallback():
    assert route_question("Hello, how are you?")   == "base"
    assert route_question("Tell me about Tapal")   == "base"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"✅ {name} passed")