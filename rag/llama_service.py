"""
Day 15: Local Quantized Ollama System Interface

What this file does:
1. Wraps raw ChatOllama in a production-grade service class
2. Enforces runtime boundaries: timeout, token limits, retries
3. Validates Ollama is actually reachable BEFORE attempting calls
4. Provides consistent error handling so FastAPI endpoints never
   crash from raw Ollama exceptions
5. Tracks response timing for observability (feeds Day 18 LangSmith)

Why a class instead of just functions:
This service needs to hold STATE across calls — connection status,
timeout config, retry count. A class is the right tool for that.
Day 16's RAG pipeline creates ONE LlamaService instance at startup
and reuses it for every request (instead of recreating ChatOllama
every time, which is wasteful).
"""

import time
import logging
import httpx
from typing import Optional, Dict, Any
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: CONFIGURATION — the "runtime limits and response boundaries"
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LlamaConfig:
    """
    Holds all runtime boundary settings in one place.

    @dataclass auto-generates __init__ so we just declare fields
    with defaults — no need to write boilerplate constructor code.

    Why centralize config here instead of scattering values
    across the codebase: if you need to tune performance later
    (e.g. reduce timeout because Ollama is slow on your machine),
    you change ONE place, not every file that calls Ollama.
    """

    # Which Ollama model to use
    model_name: str = "phi3"

    # Where Ollama server is listening
    base_url: str = "http://localhost:11434"

    # TEMPERATURE: controls randomness of output
    # 0.0 = fully deterministic (same input → same output every time)
    # 1.0 = highly creative/random
    # For a RETAIL ANALYTICS assistant, low temperature is correct —
    # you want consistent, factual answers, not creative variation
    temperature: float = 0.1

    # MAX TOKENS: hard cap on response length
    # 1 token ≈ 0.75 words in English
    # 400 tokens ≈ 300 words ≈ a focused 3-4 sentence business answer
    # Prevents phi3 from writing essays when the owner needs a quick answer
    max_tokens: int = 256

    # TIMEOUT: max seconds to wait for Ollama to respond
    # Without this, a hung request blocks your FastAPI endpoint forever
    # 180s is generous for local CPU inference of phi3 (8B params)
    timeout_seconds: int = 180

    # RETRY COUNT: how many times to retry a failed/empty call
    # Local LLMs occasionally return empty strings due to sampling
    # quirks — one retry usually fixes it without real cost (no API fees)
    max_retries: int = 1

    # MIN RESPONSE LENGTH: minimum characters to consider a response valid
    # Catches cases where Ollama returns "" or just whitespace
    min_response_length: int = 5


# Default config instance — used unless explicitly overridden
DEFAULT_CONFIG = LlamaConfig()


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: CONNECTION HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_ollama_connection(base_url: str = "http://localhost:11434") -> Dict[str, Any]:
    """
    Verifies Ollama server is running and phi3 model is available
    BEFORE attempting any actual generation call.

    Why check first instead of just trying and catching errors:
    A clear "Ollama is offline" message is much more useful for
    debugging than a generic connection timeout exception buried
    in a stack trace. This also lets FastAPI's /health endpoint
    (Day 8) report accurate service status.

    Returns a dict with status info — never raises an exception,
    so callers can check .get("connected") safely without try/except.
    """
    try:
        # httpx is a lightweight HTTP client — Ollama exposes a
        # simple REST API at /api/tags that lists installed models
        response = httpx.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()  # raises if status code is 4xx/5xx

        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]

        # Check if phi3 specifically is pulled and available
        # (Ollama might be running but without the model downloaded)
        phi3_available = any("phi3" in name for name in model_names)

        return {
            "connected"       : True,
            "server_reachable": True,
            "models_available": model_names,
            "phi3_ready"    : phi3_available,
        }

    except httpx.ConnectError:
        # Ollama server isn't running at all
        log.error("Cannot connect to Ollama at %s — is 'ollama serve' running?", base_url)
        return {
            "connected"       : False,
            "server_reachable": False,
            "phi3_ready"    : False,
            "error"           : "Ollama server not reachable. Run 'ollama serve' first.",
        }

    except httpx.TimeoutException:
        log.error("Ollama connection timed out at %s", base_url)
        return {
            "connected"       : False,
            "server_reachable": False,
            "phi3_ready"    : False,
            "error"           : "Ollama connection timed out.",
        }

    except Exception as e:
        log.error("Unexpected error checking Ollama: %s", str(e))
        return {
            "connected"       : False,
            "server_reachable": False,
            "phi3_ready"    : False,
            "error"           : str(e),
        }


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: RESPONSE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_response(
    text     : Optional[str],
    min_length: int = 5
) -> tuple[bool, str]:
    """
    Checks whether a generated response is usable.

    Returns (is_valid, reason) tuple — explicit about WHY a
    response failed validation, which makes debugging easier
    than a plain True/False.

    Catches three failure modes specific to local LLMs:
    1. None or empty string (Ollama sometimes returns nothing)
    2. Too short to be meaningful (just punctuation, whitespace)
    3. Repetition artifacts (local models can loop on short prompts,
       e.g. "the the the the...")
    """
    if text is None:
        return False, "Response is None"

    text = text.strip()

    if len(text) == 0:
        return False, "Response is empty string"

    if len(text) < min_length:
        return False, f"Response too short ({len(text)} chars, min {min_length})"

    # Detect repetition artifacts — if the same 3-word phrase
    # repeats more than 5 times, the model is likely stuck looping
    words = text.split()
    if len(words) > 15:
        first_three = " ".join(words[:3])
        repeat_count = text.count(first_three)
        if repeat_count > 5:
            return False, "Response contains repetition artifacts"

    return True, "Valid"


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: THE LLAMA SERVICE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class LlamaService:
    """
    Production-grade wrapper around Ollama/phi3.

    This is the SINGLE entry point for all LLM calls in the project
    from Day 16 onward. Instead of every chain creating its own raw
    ChatOllama(), they call LlamaService.generate() which guarantees:
    - Connection was checked
    - Timeout is enforced
    - Failed/empty responses are retried automatically
    - Response timing is logged (useful for Day 18 LangSmith)
    """

    def __init__(self, config: LlamaConfig = DEFAULT_CONFIG):
        """
        Initializes the service with a config and creates the
        underlying ChatOllama client ONCE (not per-request).

        Why create ChatOllama once in __init__ instead of every call:
        Recreating the client object on every request is wasteful —
        it's a lightweight object but there's no reason to rebuild it
        294 times if your retriever returns 294 chains worth of calls.
        """
        self.config = config

        # langchain_ollama.ChatOllama is the actual client that
        # talks to the Ollama HTTP server under the hood
        self.llm = ChatOllama(
            model      = config.model_name,
            base_url   = config.base_url,
            temperature= config.temperature,
            num_predict= config.max_tokens,   # max output tokens
            # request_timeout controls how long ChatOllama waits
            # for the HTTP response from Ollama before giving up
        )

        # Run a connection check at initialization time so we know
        # immediately (at server startup) if Ollama isn't ready —
        # rather than discovering it on the first user request
        self.connection_status = check_ollama_connection(config.base_url)

        if not self.connection_status["connected"]:
            log.warning(
                "LlamaService initialized but Ollama is NOT reachable. "
                "Generation calls will fail until Ollama is started."
            )
        elif not self.connection_status["phi3_ready"]:
            log.warning(
                "Ollama is running but phi3 model not found. "
                "Run: ollama pull phi3"
            )
        else:
            log.info(
                "LlamaService ready | model=%s | timeout=%ds | max_tokens=%d",
                config.model_name, config.timeout_seconds, config.max_tokens
            )

    def is_ready(self) -> bool:
        """
        Quick check: can this service actually generate responses
        right now? Used by FastAPI's /health endpoint (Day 8) and
        Day 16's RAG endpoint before attempting generation.
        """
        return (
            self.connection_status.get("connected", False) and
            self.connection_status.get("phi3_ready", False)
        )

    def generate(
        self,
        prompt_text: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates a response from phi3 with full safety boundaries:
        - Retries on failure/empty response (up to max_retries)
        - Times out if Ollama hangs (timeout_seconds)
        - Validates the response before returning it
        - Always returns a structured dict — NEVER raises an exception
          to the caller (so FastAPI endpoints never crash from this)

        Returns a dict shape that's consistent whether generation
        succeeded or failed:
        {
            "success"       : bool,
            "text"          : str or None,
            "duration_sec"  : float,
            "attempts"      : int,
            "error"         : str or None,
        }
        """
        # Guard clause: don't even attempt if Ollama isn't ready
        if not self.is_ready():
            log.error("generate() called but LlamaService is not ready")
            return {
                "success"     : False,
                "text"        : None,
                "duration_sec": 0.0,
                "attempts"    : 0,
                "error"       : "Ollama is not reachable or phi3 model not installed.",
            }

        # Build the message list for ChatOllama
        # SystemMessage sets the AI's role/persona (from Day 11 prompts)
        # HumanMessage is the actual user-facing prompt text
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt_text))

        last_error = None

        # RETRY LOOP — attempts generation up to (max_retries + 1) times
        # range(1, N+1) so attempt numbers in logs read naturally (1, 2, 3...)
        for attempt in range(1, self.config.max_retries + 2):
            start_time = time.time()

            try:
                # The actual LLM call, with a manual timeout wrapper
                # using a simple time-check approach (ChatOllama doesn't
                # have a built-in per-call timeout parameter in older
                # versions, so we enforce it ourselves)
                response = self._invoke_with_timeout(messages)

                duration = round(time.time() - start_time, 2)

                # Extract plain text from the response object
                response_text = response.content if hasattr(response, "content") else str(response)

                # Validate the response isn't empty/broken
                is_valid, reason = validate_response(
                    response_text, self.config.min_response_length
                )

                if is_valid:
                    log.info(
                        "Generation success | attempt=%d/%d | duration=%.2fs | length=%d chars",
                        attempt, self.config.max_retries + 1, duration, len(response_text)
                    )
                    return {
                        "success"     : True,
                        "text"        : response_text.strip(),
                        "duration_sec": duration,
                        "attempts"    : attempt,
                        "error"       : None,
                    }
                else:
                    # Response came back but failed validation — retry
                    log.warning(
                        "Generation attempt %d failed validation: %s. Retrying...",
                        attempt, reason
                    )
                    last_error = reason

            except TimeoutError as e:
                duration = round(time.time() - start_time, 2)
                log.warning(
                    "Generation attempt %d timed out after %.2fs. Retrying...",
                    attempt, duration
                )
                last_error = f"Timeout after {self.config.timeout_seconds}s"

            except Exception as e:
                duration = round(time.time() - start_time, 2)
                log.error(
                    "Generation attempt %d raised exception: %s",
                    attempt, str(e)
                )
                last_error = str(e)

        # All retries exhausted — return a clean failure response
        # instead of raising, so FastAPI can return a proper error JSON
        log.error(
            "Generation FAILED after %d attempts. Last error: %s",
            self.config.max_retries + 1, last_error
        )
        return {
            "success"     : False,
            "text"        : None,
            "duration_sec": 0.0,
            "attempts"    : self.config.max_retries + 1,
            "error"       : last_error,
        }

    def _invoke_with_timeout(self, messages):
        """
        Internal helper that enforces the configured timeout
        on a single Ollama call using a thread-based watchdog.

        Why this approach: ChatOllama's invoke() is a blocking call.
        To enforce OUR timeout (not rely on the HTTP client's default,
        which may be much longer or unlimited), we run the call in a
        separate thread and check if it completes within our deadline.
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.llm.invoke, messages)
            try:
                # .result(timeout=...) raises concurrent.futures.TimeoutError
                # if the call doesn't finish within timeout_seconds
                return future.result(timeout=self.config.timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Ollama did not respond within {self.config.timeout_seconds}s"
                )

    async def agenerate(
        self,
        prompt_text: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Async version of generate() — used by FastAPI's async endpoints
        (Day 16's /rag/ask uses this, not the sync generate()).

        Why a separate async method instead of just making generate()
        async: keeps the sync version usable in scripts/tests (Day 11's
        evaluate_baseline.py) without needing an event loop, while
        still providing proper async support for production FastAPI use.
        """
        if not self.is_ready():
            return {
                "success"     : False,
                "text"        : None,
                "duration_sec": 0.0,
                "attempts"    : 0,
                "error"       : "Ollama is not reachable or phi3 model not installed.",
            }

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt_text))

        last_error = None

        for attempt in range(1, self.config.max_retries + 2):
            start_time = time.time()
            try:
                # ainvoke() is ChatOllama's native async method
                import asyncio
                response = await asyncio.wait_for(
                    self.llm.ainvoke(messages),
                    timeout=self.config.timeout_seconds,
                )

                duration = round(time.time() - start_time, 2)
                response_text = response.content if hasattr(response, "content") else str(response)

                is_valid, reason = validate_response(
                    response_text, self.config.min_response_length
                )

                if is_valid:
                    log.info(
                        "Async generation success | attempt=%d | duration=%.2fs",
                        attempt, duration
                    )
                    return {
                        "success"     : True,
                        "text"        : response_text.strip(),
                        "duration_sec": duration,
                        "attempts"    : attempt,
                        "error"       : None,
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
            "success"     : False,
            "text"        : None,
            "duration_sec": 0.0,
            "attempts"    : self.config.max_retries + 1,
            "error"       : last_error,
        }


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: SINGLETON INSTANCE (shared across the whole app)
# ══════════════════════════════════════════════════════════════════════════════

# A module-level singleton — created ONCE when this file is first imported.
# Day 16's RAG pipeline and FastAPI's lifespan (Day 8) both reuse THIS
# instance rather than creating a new LlamaService per request.
#
# Why a singleton: ChatOllama connections and the startup health check
# are relatively expensive to redo on every single API call. Creating
# one instance and reusing it is the correct production pattern.
_llama_service_instance: Optional[LlamaService] = None


def get_llama_service(force_recreate: bool = False) -> LlamaService:
    """
    Returns the shared LlamaService instance, creating it on first call.

    force_recreate=True: rebuilds the service (useful if Ollama was
    restarted and you need a fresh connection check without restarting
    the whole FastAPI app).
    """
    global _llama_service_instance

    if _llama_service_instance is None or force_recreate:
        log.info("Creating new LlamaService instance...")
        _llama_service_instance = LlamaService(DEFAULT_CONFIG)

    return _llama_service_instance


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — manual testing
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Running this file directly tests the full service:
    1. Connection check
    2. A normal generation call
    3. A call with a system prompt (like Day 11's persona)
    4. Verifies timeout/retry boundaries are configured correctly
    """
    print("\n" + "=" * 60)
    print("DAY 15 — LLAMA SERVICE TEST")
    print("=" * 60)

    # Step 1: Connection check
    print("\n[1] Checking Ollama connection...")
    status = check_ollama_connection()
    print(f"    Connected     : {status['connected']}")
    print(f"    phi3 ready  : {status.get('phi3_ready', False)}")
    if not status["connected"]:
        print(f"    Error: {status.get('error')}")
        print("\n⚠️  Start Ollama with 'ollama serve' and try again.")
        exit(1)

    # Step 2: Create service
    print("\n[2] Initializing LlamaService...")
    service = get_llama_service()
    print(f"    Ready: {service.is_ready()}")
    print(f"    Config: model={service.config.model_name}, "
          f"timeout={service.config.timeout_seconds}s, "
          f"max_tokens={service.config.max_tokens}, "
          f"retries={service.config.max_retries}")

    # Step 3: Test generation
    print("\n[3] Testing generation with simple prompt...")
    result = service.generate(
        prompt_text="What is a profit margin? Answer in one sentence.",
        system_prompt="You are a helpful retail business assistant.",
    )
    print(f"    Success     : {result['success']}")
    print(f"    Duration    : {result['duration_sec']}s")
    print(f"    Attempts    : {result['attempts']}")
    if result["success"]:
        print(f"    Response    : {result['text']}")
    else:
        print(f"    Error       : {result['error']}")

    # Step 4: Test with retail persona (Day 11 style)
    print("\n[4] Testing with retail assistant persona...")
    result2 = service.generate(
        prompt_text="My Tapal Tea stock is at 5 units, selling 3 per day. Should I reorder?",
        system_prompt=(
            "You are an AI Retail Intelligence Assistant for a Pakistani "
            "kiryana store. Give short, practical, 2-sentence answers."
        ),
    )
    print(f"    Success     : {result2['success']}")
    print(f"    Duration    : {result2['duration_sec']}s")
    if result2["success"]:
        print(f"    Response    : {result2['text']}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)