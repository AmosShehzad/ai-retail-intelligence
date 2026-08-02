"""
Day 15 (updated): Local + Cloud LLM Interface

Supports TWO providers, switched by LLM_PROVIDER env var:
- "ollama" → local phi3 via Ollama (development)
- "groq"   → cloud Llama 3.1 via Groq API (deployment)

Same class, same methods (generate/agenerate), same return shape.
Everything downstream (pipeline.py, chains.py) works unchanged
regardless of which provider is active.
"""

import os
import time
import asyncio
import logging
import httpx
from typing import Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Which provider to use ──────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LlamaConfig:
    """
    Runtime boundary settings. Same config works for both providers —
    Groq just ignores fields it doesn't need (like base_url).
    """
    # Ollama-specific
    model_name: str = "phi3"
    base_url  : str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Groq-specific
    groq_model: str = "llama-3.1-8b-instant"

    # Shared settings
    temperature: float = 0.1
    max_tokens : int   = 256
    timeout_seconds: int = 360
    max_retries: int = 1
    min_response_length: int = 5


DEFAULT_CONFIG = LlamaConfig()


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: CONNECTION HEALTH CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_ollama_connection(base_url: str = None) -> Dict[str, Any]:
    """Verifies Ollama server is running and phi3 is available."""
    if base_url is None:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()

        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        phi3_available = any("phi3" in name for name in model_names)

        return {
            "connected"       : True,
            "server_reachable": True,
            "models_available": model_names,
            "phi3_ready"      : phi3_available,
        }
    except httpx.ConnectError:
        log.error("Cannot connect to Ollama at %s — is 'ollama serve' running?", base_url)
        return {
            "connected": False, "server_reachable": False, "phi3_ready": False,
            "error": "Ollama server not reachable. Run 'ollama serve' first.",
        }
    except httpx.TimeoutException:
        log.error("Ollama connection timed out at %s", base_url)
        return {
            "connected": False, "server_reachable": False, "phi3_ready": False,
            "error": "Ollama connection timed out.",
        }
    except Exception as e:
        log.error("Unexpected error checking Ollama: %s", str(e))
        return {
            "connected": False, "server_reachable": False, "phi3_ready": False,
            "error": str(e),
        }


def check_groq_connection(api_key: str = None) -> Dict[str, Any]:
    """
    Verifies the Groq API key is valid by making a lightweight test call.
    Groq is cloud-hosted — "connection" just means the API key works.
    """
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return {
            "connected": False, "server_reachable": False, "phi3_ready": False,
            "error": "GROQ_API_KEY not set in environment.",
        }

    try:
        response = httpx.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        models = [m["id"] for m in response.json().get("data", [])]

        return {
            "connected"       : True,
            "server_reachable": True,
            "models_available": models,
            "phi3_ready"      : True,  # reused field name meaning "ready"
        }
    except Exception as e:
        log.error("Groq connection check failed: %s", str(e))
        return {
            "connected": False, "server_reachable": False, "phi3_ready": False,
            "error": str(e),
        }


def check_llm_connection(config: LlamaConfig) -> Dict[str, Any]:
    """Routes to the correct provider's health check."""
    if LLM_PROVIDER == "groq":
        return check_groq_connection()
    else:
        return check_ollama_connection(config.base_url)


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: RESPONSE VALIDATION (provider-agnostic)
# ══════════════════════════════════════════════════════════════════════════════

def validate_response(text: Optional[str], min_length: int = 5) -> tuple[bool, str]:
    """
    Checks whether a generated response is usable.
    Catches: None/empty, too short, repetition artifacts.
    """
    if text is None:
        return False, "Response is None"

    text = text.strip()

    if len(text) == 0:
        return False, "Response is empty string"

    if len(text) < min_length:
        return False, f"Response too short ({len(text)} chars, min {min_length})"

    words = text.split()
    if len(words) > 15:
        first_three = " ".join(words[:3])
        if text.count(first_three) > 5:
            return False, "Response contains repetition artifacts"

    return True, "Valid"


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: THE LLAMA SERVICE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class LlamaService:
    """
    Production wrapper around either Ollama (local) or Groq (cloud).
    Same public interface regardless of provider:
    - .is_ready()
    - .generate(prompt, system_prompt)
    - .agenerate(prompt, system_prompt)
    """

    def __init__(self, config: LlamaConfig = DEFAULT_CONFIG):
        self.config = config
        self.provider = LLM_PROVIDER

        if self.provider == "groq":
            self._init_groq(config)
        else:
            self._init_ollama(config)

    def _init_ollama(self, config: LlamaConfig):
        """Initializes the local Ollama/phi3 client."""
        from langchain_ollama import ChatOllama

        self.llm = ChatOllama(
            model      = config.model_name,
            base_url   = config.base_url,
            temperature= config.temperature,
            num_predict= config.max_tokens,
        )

        self.connection_status = check_ollama_connection(config.base_url)

        if not self.connection_status["connected"]:
            log.warning("LlamaService (Ollama) initialized but not reachable.")
        elif not self.connection_status["phi3_ready"]:
            log.warning("Ollama running but phi3 not found. Run: ollama pull phi3")
        else:
            log.info(
                "LlamaService ready | provider=ollama | model=%s | timeout=%ds",
                config.model_name, config.timeout_seconds
            )

    def _init_groq(self, config: LlamaConfig):
        """Initializes the cloud Groq client."""
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")

        self.llm = ChatGroq(
            model      = config.groq_model,
            temperature= config.temperature,
            max_tokens = config.max_tokens,
            api_key    = api_key,
            timeout    = config.timeout_seconds,
        )

        self.connection_status = check_groq_connection(api_key)

        if not self.connection_status["connected"]:
            log.warning("LlamaService (Groq) initialized but connection failed: %s",
                       self.connection_status.get("error"))
        else:
            log.info(
                "LlamaService ready | provider=groq | model=%s",
                config.groq_model
            )

    def is_ready(self) -> bool:
        return (
            self.connection_status.get("connected", False) and
            self.connection_status.get("phi3_ready", False)
        )

    def generate(self, prompt_text: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Sync generation. Works identically for both providers because
        both ChatOllama and ChatGroq implement the same LangChain
        chat model interface (.invoke(messages)).
        """
        if not self.is_ready():
            log.error("generate() called but LlamaService is not ready")
            return {
                "success": False, "text": None, "duration_sec": 0.0,
                "attempts": 0,
                "error": f"{self.provider} is not reachable or not configured.",
            }

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt_text))

        last_error = None

        for attempt in range(1, self.config.max_retries + 2):
            start_time = time.time()

            try:
                response = self._invoke_with_timeout(messages)
                duration = round(time.time() - start_time, 2)
                response_text = response.content if hasattr(response, "content") else str(response)

                is_valid, reason = validate_response(response_text, self.config.min_response_length)

                if is_valid:
                    log.info(
                        "Generation success | provider=%s | attempt=%d/%d | duration=%.2fs | length=%d chars",
                        self.provider, attempt, self.config.max_retries + 1, duration, len(response_text)
                    )
                    return {
                        "success": True, "text": response_text.strip(),
                        "duration_sec": duration, "attempts": attempt, "error": None,
                    }
                else:
                    log.warning("Generation attempt %d failed validation: %s. Retrying...", attempt, reason)
                    last_error = reason

            except TimeoutError:
                duration = round(time.time() - start_time, 2)
                log.warning("Generation attempt %d timed out after %.2fs. Retrying...", attempt, duration)
                last_error = f"Timeout after {self.config.timeout_seconds}s"

            except Exception as e:
                import concurrent.futures
                if isinstance(e, concurrent.futures.TimeoutError):
                    log.warning("Generation attempt %d futures timeout. Retrying...", attempt)
                    last_error = f"Timeout after {self.config.timeout_seconds}s"
                else:
                    duration = round(time.time() - start_time, 2)
                    log.error("Generation attempt %d raised exception: %s", attempt, str(e))
                    last_error = str(e)

        log.error(
            "Generation FAILED after %d attempts. Last error: %s",
            self.config.max_retries + 1, last_error
        )
        return {
            "success": False, "text": None, "duration_sec": 0.0,
            "attempts": self.config.max_retries + 1, "error": last_error,
        }

    def _invoke_with_timeout(self, messages):
        """
        Enforces the configured timeout on a single LLM call
        using a thread-based watchdog. Works for both Ollama and Groq
        since both use the same .invoke() interface.
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.llm.invoke, messages)
            try:
                return future.result(timeout=self.config.timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"{self.provider} did not respond within {self.config.timeout_seconds}s"
                )

    async def agenerate(self, prompt_text: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Async generation — used by FastAPI's /rag/ask endpoint.
        Both ChatOllama and ChatGroq support .ainvoke() natively.
        """
        if not self.is_ready():
            return {
                "success": False, "text": None, "duration_sec": 0.0,
                "attempts": 0,
                "error": f"{self.provider} is not reachable or not configured.",
            }

        # Build messages BEFORE calling the LLM (this was broken before)
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt_text))

        last_error = None

        for attempt in range(1, self.config.max_retries + 2):
            start_time = time.time()
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(messages),
                    timeout=self.config.timeout_seconds,
                )

                duration = round(time.time() - start_time, 2)
                response_text = response.content if hasattr(response, "content") else str(response)

                is_valid, reason = validate_response(response_text, self.config.min_response_length)

                if is_valid:
                    log.info(
                        "Async generation success | provider=%s | attempt=%d | duration=%.2fs",
                        self.provider, attempt, duration
                    )
                    return {
                        "success": True, "text": response_text.strip(),
                        "duration_sec": duration, "attempts": attempt, "error": None,
                    }
                else:
                    last_error = reason
                    log.warning("Async attempt %d failed validation: %s", attempt, reason)

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.config.timeout_seconds}s"
                log.warning("Async generation attempt %d timed out", attempt)

            except Exception as e:
                last_error = str(e)
                log.error("Async generation attempt %d failed: %s", attempt, str(e))

        return {
            "success": False, "text": None, "duration_sec": 0.0,
            "attempts": self.config.max_retries + 1, "error": last_error,
        }


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: SINGLETON INSTANCE
# ══════════════════════════════════════════════════════════════════════════════

_llama_service_instance: Optional[LlamaService] = None


def get_llama_service(force_recreate: bool = False) -> LlamaService:
    """
    Returns the shared LlamaService instance, creating it on first call.
    """
    global _llama_service_instance

    if _llama_service_instance is None or force_recreate:
        log.info("Creating new LlamaService instance | provider=%s", LLM_PROVIDER)
        _llama_service_instance = LlamaService(DEFAULT_CONFIG)

    return _llama_service_instance


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — manual testing
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"LLM SERVICE TEST — provider = {LLM_PROVIDER}")
    print("=" * 60)

    service = get_llama_service()
    print(f"\nReady: {service.is_ready()}")

    if not service.is_ready():
        print(f"Error: {service.connection_status.get('error')}")
        exit(1)

    print("\n[1] Testing generation with simple prompt...")
    result = service.generate(
        prompt_text="What is a profit margin? Answer in one sentence.",
        system_prompt="You are a helpful retail business assistant.",
    )
    print(f"    Success  : {result['success']}")
    print(f"    Duration : {result['duration_sec']}s")
    print(f"    Response : {result.get('text')}")

    print("\n[2] Testing with retail persona...")
    result2 = service.generate(
        prompt_text="My Tapal Tea stock is at 5 units, selling 3 per day. Should I reorder?",
        system_prompt=(
            "You are an AI Retail Intelligence Assistant for a Pakistani "
            "kiryana store. Give short, practical, 2-sentence answers."
        ),
    )
    print(f"    Success  : {result2['success']}")
    print(f"    Duration : {result2['duration_sec']}s")
    print(f"    Response : {result2.get('text')}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)