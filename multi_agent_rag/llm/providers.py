"""
Unified LLM provider layer.

One function — chat(system, user) — backed by any of four providers,
selected with the LLM_PROVIDER env var:

  anthropic  Claude via the Anthropic API (adaptive thinking supported)
  ollama     local open-source models served by Ollama       (OpenAI-compatible /v1)
  lmstudio   local open-source models served by LM Studio    (OpenAI-compatible /v1)
  openai     any other OpenAI-compatible endpoint: vLLM, llama.cpp server,
             Together, Groq, OpenRouter, or OpenAI itself

Every call goes through retry with exponential backoff on transient errors
(connection failures, rate limits, 5xx). Configuration errors (missing key,
server not running, model not pulled) fail fast with an actionable message.
"""
from __future__ import annotations

import time
from typing import Any

import config
from utils import get_logger

log = get_logger("LLM")


class LLMError(RuntimeError):
    """Non-retryable LLM failure (bad config, auth, unknown model)."""


class LLMConnectionError(LLMError):
    """The provider endpoint is unreachable."""


# ── Provider implementations ────────────────────────────────────────────────

class AnthropicProvider:
    name = "anthropic"

    def __init__(self):
        if not config.ANTHROPIC_API_KEY:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Either export it, put it in .env, "
                "or switch to a local model with LLM_PROVIDER=ollama / lmstudio."
            )
        import anthropic
        self._retryable = (
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        )
        self.client = anthropic.Anthropic(
            api_key=config.ANTHROPIC_API_KEY,
            timeout=config.LLM_TIMEOUT,
            max_retries=0,  # we handle retries ourselves
        )
        self.model = config.CLAUDE_MODEL

    def chat(self, system: str, user_msg: str, thinking: bool, max_tokens: int) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_msg}],
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        with self.client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
        parts = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(parts).strip()

    def is_retryable(self, exc: Exception) -> bool:
        return isinstance(exc, self._retryable)

    def check(self) -> dict:
        return {"provider": self.name, "model": self.model, "ok": True,
                "detail": "API key present (validity checked on first call)"}


class OpenAICompatProvider:
    """Covers ollama, lmstudio, and any generic OpenAI-compatible endpoint."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str):
        self.name = name
        self.base_url = base_url
        self.model = model
        import openai
        self._openai = openai
        self._retryable = (
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.APITimeoutError,
        )
        # Local servers ignore the key but the client requires a non-empty one
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=config.LLM_TIMEOUT,
            max_retries=0,
        )
        if not self.model:
            self.model = self._first_available_model()

    def _first_available_model(self) -> str:
        """LM Studio serves whatever model the user loaded; pick the first."""
        try:
            models = list(self.client.models.list())
        except Exception as e:
            raise LLMConnectionError(
                f"Cannot reach {self.name} at {self.base_url} — is the server "
                f"running? ({e})"
            ) from e
        if not models:
            raise LLMError(
                f"{self.name} at {self.base_url} reports no loaded models. "
                "Load a model first (LM Studio: load in the UI and enable the "
                "local server; Ollama: `ollama pull <model>`)."
            )
        log.info("[%s] auto-selected model: %s", self.name, models[0].id)
        return models[0].id

    def chat(self, system: str, user_msg: str, thinking: bool, max_tokens: int) -> str:
        # `thinking` is Anthropic-specific; OpenAI-compatible servers ignore it.
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=config.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        )
        content = resp.choices[0].message.content or ""
        return content.strip()

    def is_retryable(self, exc: Exception) -> bool:
        return isinstance(exc, self._retryable)

    def check(self) -> dict:
        try:
            models = [m.id for m in self.client.models.list()]
        except Exception as e:
            return {"provider": self.name, "model": self.model, "ok": False,
                    "detail": f"unreachable at {self.base_url}: {e}"}
        ok = self.model in models or not models
        detail = f"reachable, {len(models)} model(s) available"
        if not ok:
            detail += f" — configured model '{self.model}' not in {models[:10]}"
        return {"provider": self.name, "model": self.model, "ok": ok, "detail": detail}


# ── Factory + retry wrapper ─────────────────────────────────────────────────

_provider = None


def get_provider():
    """Lazily build the configured provider (singleton)."""
    global _provider
    if _provider is not None:
        return _provider

    p = config.LLM_PROVIDER
    if p == "anthropic":
        _provider = AnthropicProvider()
    elif p == "ollama":
        _provider = OpenAICompatProvider("ollama", config.OLLAMA_BASE_URL,
                                         "ollama", config.OLLAMA_MODEL)
    elif p == "lmstudio":
        _provider = OpenAICompatProvider("lmstudio", config.LMSTUDIO_BASE_URL,
                                         "lm-studio", config.LMSTUDIO_MODEL)
    elif p == "openai":
        _provider = OpenAICompatProvider("openai", config.OPENAI_BASE_URL,
                                         config.OPENAI_API_KEY, config.OPENAI_MODEL)
    else:
        raise LLMError(
            f"Unknown LLM_PROVIDER '{p}'. Choose: anthropic | ollama | lmstudio | openai"
        )
    log.info("LLM provider ready: %s (model=%s)", _provider.name, _provider.model)
    return _provider


def chat(system: str, user_msg: str, thinking: bool = True,
         max_tokens: int | None = None) -> str:
    """Call the configured LLM with retry + exponential backoff."""
    provider = get_provider()
    max_tokens = max_tokens or config.MAX_TOKENS
    last_exc: Exception | None = None

    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        try:
            return provider.chat(system, user_msg, thinking, max_tokens)
        except Exception as e:  # noqa: BLE001 — classified below
            last_exc = e
            if not provider.is_retryable(e) or attempt == config.LLM_MAX_RETRIES:
                break
            delay = config.LLM_RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning("[%s] transient error (attempt %d/%d): %s — retrying in %.1fs",
                        provider.name, attempt, config.LLM_MAX_RETRIES, e, delay)
            time.sleep(delay)

    raise LLMError(
        f"LLM call failed via {provider.name} (model={provider.model}): {last_exc}"
    ) from last_exc


def check_provider() -> dict:
    """Health-check the configured provider without running the pipeline."""
    try:
        return get_provider().check()
    except LLMError as e:
        return {"provider": config.LLM_PROVIDER, "model": "?", "ok": False, "detail": str(e)}
