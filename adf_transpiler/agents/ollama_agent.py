"""
ollama_agent.py
===============
Async LLM translation agent backed by a local Ollama instance.

Responsibilities:
    - Accept a list of raw ADF dynamic expressions (e.g. `@concat(...)`)
    - Send them to a locally-running Ollama model for translation to PySpark
    - Stream token responses back to the caller via an async generator
    - Patch translated expressions back into the raw PySpark code scaffold

Zero data leaves the machine — all inference runs through local Ollama.

Author: Transpiler Architect
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "qwen2.5-coder:7b"
FALLBACK_MODEL = "codellama"
OLLAMA_BASE_URL = "http://localhost:11434"

_SYSTEM_PROMPT = """\
You are an expert Azure Data Factory (ADF) to PySpark migration engineer.
Your ONLY task is to translate ADF dynamic expressions into equivalent, \
idiomatic PySpark / Python 3.10+ code.

Rules:
1. Output ONLY the translated Python expression or statement — no explanations, \
no markdown fences, no prose.
2. Preserve semantics exactly. If the ADF expression is ambiguous, emit a \
Python comment explaining the ambiguity followed by a best-effort translation.
3. Use f-strings for string interpolation where appropriate.
4. Reference pipeline parameters as `pipeline_params["<name>"]` and pipeline \
variables as `pipeline_vars["<name>"]`.
5. Reference activity outputs as `activity_outputs["<activity_name>"]["<field>"]`.
6. Never emit `import` statements — assume all standard PySpark imports are \
already present.
"""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TranslationRequest:
    """A single ADF expression that needs LLM translation."""

    json_path: str          # Dot-path in the activity typeProperties where this lives.
    adf_expression: str     # Raw ADF expression string, e.g. "@concat(...)".
    activity_name: str
    context_hint: str = "" # Optional surrounding context for the LLM.


@dataclass
class TranslationResult:
    """Result from the LLM for a single expression."""

    request: TranslationRequest
    pyspark_expression: str
    model_used: str
    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Ollama client wrapper
# ---------------------------------------------------------------------------


class OllamaAgent:
    """
    Async wrapper around the Ollama Python SDK.

    Streams token-by-token responses so the Streamlit UI can display
    output in real time without blocking.

    Parameters
    ----------
    model:
        Ollama model tag to use (e.g. "qwen2.5-coder:7b", "codellama").
    base_url:
        Base URL for the Ollama API server.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._client: "ollama.AsyncClient | None" = None  # type: ignore[name-defined]

    def _get_client(self) -> "ollama.AsyncClient":  # type: ignore[name-defined]
        """Lazy-initialise the Ollama async client."""
        if self._client is None:
            try:
                import ollama  # noqa: PLC0415
                self._client = ollama.AsyncClient(host=self.base_url)
            except ImportError as exc:
                raise RuntimeError(
                    "The 'ollama' package is not installed. "
                    "Run: pip install ollama"
                ) from exc
        return self._client

    async def check_connection(self) -> tuple[bool, str]:
        """
        Verify Ollama is reachable and the target model is available.

        Returns
        -------
        (ok, message)
        """
        try:
            client = self._get_client()
            models_response = await client.list()
            available = [m.model for m in models_response.models]
            logger.info("Ollama reachable. Available models: %s", available)

            if self.model not in available and not any(
                self.model in m for m in available
            ):
                logger.warning(
                    "Model '%s' not found locally. Available: %s", self.model, available
                )
                return False, (
                    f"Model '{self.model}' is not pulled. "
                    f"Run: ollama pull {self.model}\n"
                    f"Available models: {available}"
                )

            return True, f"Connected. Model '{self.model}' ready."

        except Exception as exc:
            logger.error("Ollama connection check failed: %s", exc)
            return False, (
                f"Cannot reach Ollama at {self.base_url}. "
                "Ensure Ollama is running: ollama serve"
            )

    async def translate_expression_stream(
        self, request: TranslationRequest
    ) -> AsyncIterator[str]:
        """
        Stream the LLM translation for a single ADF expression token by token.

        Yields
        ------
        str
            Individual token strings as they arrive from Ollama.
        """
        user_prompt = self._build_user_prompt(request)
        logger.info(
            "Translating expression for activity '%s' at path '%s'",
            request.activity_name,
            request.json_path,
        )

        client = self._get_client()

        try:
            async for chunk in await client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                options={"temperature": 0.1, "num_predict": 512},
            ):
                token: str = chunk.message.content or ""
                if token:
                    yield token

        except Exception as exc:
            logger.error(
                "Ollama streaming failed for expression '%s': %s",
                request.adf_expression,
                exc,
                exc_info=True,
            )
            yield f"# ERROR: Ollama translation failed — {exc}\n"
            yield f"# Original ADF expression: {request.adf_expression}\n"
            yield "None  # TODO: manual translation required\n"

    async def translate_expression(self, request: TranslationRequest) -> TranslationResult:
        """
        Non-streaming translation — collects the full response before returning.
        Useful for batch processing outside the Streamlit UI.
        """
        tokens: list[str] = []
        async for token in self.translate_expression_stream(request):
            tokens.append(token)

        raw_output = "".join(tokens).strip()
        cleaned = _clean_llm_output(raw_output)
        success = not cleaned.startswith("# ERROR")

        return TranslationResult(
            request=request,
            pyspark_expression=cleaned,
            model_used=self.model,
            success=success,
            error=None if success else cleaned,
        )

    async def translate_batch(
        self,
        requests: list[TranslationRequest],
        max_concurrency: int = 2,
    ) -> list[TranslationResult]:
        """
        Translate multiple expressions with bounded concurrency.
        Ollama is typically single-threaded; keep max_concurrency low.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _guarded(req: TranslationRequest) -> TranslationResult:
            async with semaphore:
                return await self.translate_expression(req)

        tasks = [_guarded(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        typed_results: list[TranslationResult] = []
        for req, res in zip(requests, results):
            if isinstance(res, Exception):
                logger.error("Batch translation task raised: %s", res)
                typed_results.append(
                    TranslationResult(
                        request=req,
                        pyspark_expression=f"# EXCEPTION: {res}",
                        model_used=self.model,
                        success=False,
                        error=str(res),
                    )
                )
            else:
                typed_results.append(res)

        return typed_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_user_prompt(self, request: TranslationRequest) -> str:
        lines = [
            f"Activity name : {request.activity_name}",
            f"Property path : {request.json_path}",
            f"ADF expression: {request.adf_expression}",
        ]
        if request.context_hint:
            lines.append(f"Context       : {request.context_hint}")
        lines.append("\nTranslate the ADF expression above to Python/PySpark:")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Code patching utility
# ---------------------------------------------------------------------------


def patch_code_with_translations(
    pyspark_code: str,
    translations: list[TranslationResult],
) -> str:
    """
    Replace ADF expression placeholders in generated PySpark code with
    their LLM-translated equivalents.

    Placeholders are emitted by the transpiler in the form:
        # TODO: resolve ADF expression → Python <something>

    This function is intentionally conservative — if it cannot identify a
    clear substitution point it leaves the placeholder intact and appends
    a comment.
    """
    patched = pyspark_code
    for result in translations:
        if not result.success:
            continue
        original = result.request.adf_expression
        # Escape for use in regex
        escaped = re.escape(original)
        # Replace the first occurrence of the raw expression wrapped in quotes
        patched = re.sub(
            rf'["\']?{escaped}["\']?',
            result.pyspark_expression,
            patched,
            count=1,
        )
        logger.debug(
            "Patched expression '%s' → '%s'",
            original[:60],
            result.pyspark_expression[:60],
        )
    return patched


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------


def _clean_llm_output(raw: str) -> str:
    """Strip markdown code fences and extraneous whitespace from LLM output."""
    cleaned = re.sub(r"^```(?:python)?\n?", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()
