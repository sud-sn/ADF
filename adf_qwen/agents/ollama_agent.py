"""
ollama_agent.py
===============
Async LLM translation agent backed by a local Ollama instance.

Optimised for Qwen2.5-Coder as the primary inference model.

Responsibilities:
    - Accept raw ADF dynamic expressions (e.g. `@concat(...)`)
    - Dispatch them to a locally-running Ollama model via the Qwen2.5-Coder
      chat template format for maximum translation fidelity
    - Stream token-by-token responses back to the caller
    - Clean Qwen-specific output artefacts before returning
    - Patch translated expressions back into the PySpark code scaffold

Model hierarchy (all local, zero data egress):
    Primary  : qwen2.5-coder:7b   (fast, fits in 8 GB VRAM)
    Upgraded : qwen2.5-coder:14b  (higher accuracy, needs ~16 GB VRAM)
    Heavy    : qwen2.5-coder:32b  (best quality, needs ~24 GB VRAM)
    Fallback : qwen2.5-coder:7b-instruct (instruct-tuned variant)

Author: Transpiler Architect
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model profiles — Qwen2.5-Coder specific inference parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelProfile:
    """
    Encapsulates model-specific Ollama inference parameters.

    Qwen2.5-Coder performs best with very low temperature (near-deterministic
    for code tasks), a mild repetition penalty to suppress looping, and
    explicit context length to handle long ADF expressions.
    """

    model_tag: str
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    num_predict: int          # max output tokens
    num_ctx: int              # context window (tokens)
    description: str = ""

    def as_options(self) -> dict:
        return {
            "temperature":    self.temperature,
            "top_p":          self.top_p,
            "top_k":          self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "num_predict":    self.num_predict,
            "num_ctx":        self.num_ctx,
        }


# Registry of known model profiles.
# Callers can look up a profile by model tag prefix match.
MODEL_PROFILES: dict[str, ModelProfile] = {
    "qwen2.5-coder:7b": ModelProfile(
        model_tag="qwen2.5-coder:7b",
        temperature=0.05,       # near-deterministic for code generation
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,     # suppress repetitive completions
        num_predict=512,
        num_ctx=8192,
        description="Qwen2.5-Coder 7B — fast, fits in 8 GB VRAM",
    ),
    "qwen2.5-coder:7b-instruct": ModelProfile(
        model_tag="qwen2.5-coder:7b-instruct",
        temperature=0.05,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        num_predict=512,
        num_ctx=8192,
        description="Qwen2.5-Coder 7B Instruct — instruct-tuned variant",
    ),
    "qwen2.5-coder:14b": ModelProfile(
        model_tag="qwen2.5-coder:14b",
        temperature=0.05,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.05,    # 14B is less prone to repetition
        num_predict=768,
        num_ctx=16384,
        description="Qwen2.5-Coder 14B — higher accuracy, needs ~16 GB VRAM",
    ),
    "qwen2.5-coder:14b-instruct": ModelProfile(
        model_tag="qwen2.5-coder:14b-instruct",
        temperature=0.05,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.05,
        num_predict=768,
        num_ctx=16384,
        description="Qwen2.5-Coder 14B Instruct",
    ),
    "qwen2.5-coder:32b": ModelProfile(
        model_tag="qwen2.5-coder:32b",
        temperature=0.05,
        top_p=0.95,
        top_k=50,
        repeat_penalty=1.05,
        num_predict=1024,
        num_ctx=32768,
        description="Qwen2.5-Coder 32B — best quality, needs ~24 GB VRAM",
    ),
    # Generic fallback for any unrecognised model tag
    "_default": ModelProfile(
        model_tag="_default",
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        num_predict=512,
        num_ctx=4096,
        description="Generic fallback profile",
    ),
}


def get_model_profile(model_tag: str) -> ModelProfile:
    """
    Resolve the best matching ModelProfile for a given model tag.
    Falls back to _default if no exact or prefix match is found.
    """
    if model_tag in MODEL_PROFILES:
        return MODEL_PROFILES[model_tag]
    # Prefix match — e.g. "qwen2.5-coder:7b-q4_K_M" → "qwen2.5-coder:7b"
    for key in MODEL_PROFILES:
        if key != "_default" and model_tag.startswith(key.split(":")[0]):
            logger.debug("Model '%s' matched profile '%s' by prefix.", model_tag, key)
            return MODEL_PROFILES[key]
    logger.warning("No profile found for model '%s'. Using _default.", model_tag)
    return MODEL_PROFILES["_default"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = "qwen2.5-coder:7b"
FALLBACK_MODEL  = "qwen2.5-coder:7b-instruct"   # instruct variant as fallback
OLLAMA_BASE_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# System prompt — tuned for Qwen2.5-Coder chat template
#
# Qwen2.5-Coder uses the ChatML format internally:
#   <|im_start|>system\n…<|im_end|>
#   <|im_start|>user\n…<|im_end|>
#   <|im_start|>assistant\n…<|im_end|>
#
# Ollama handles the template wrapping automatically via its modelfile,
# so we only need to provide clean role content here.
#
# Key tuning decisions for Qwen2.5-Coder:
#   - Explicit "output ONLY code" constraint — Qwen tends to over-explain
#   - Forbidden list for common Qwen verbosity patterns
#   - Short, direct sentences — Qwen follows imperative style well
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a specialist Azure Data Factory (ADF) to PySpark migration engineer.
Your ONLY task: translate ADF dynamic expressions into Python 3.10+ / PySpark code.

OUTPUT RULES — strictly enforced:
1. Output ONLY the translated Python expression or code fragment.
2. Do NOT include any explanation, commentary, preamble, or prose.
3. Do NOT wrap output in markdown code fences (no ```python or ```).
4. Do NOT output "Here is the translation" or similar phrases.
5. Do NOT add import statements — all PySpark imports are already present.

TRANSLATION RULES:
- Preserve semantics exactly. Never guess — if ambiguous, emit a short inline
  Python comment then provide a best-effort translation on the next line.
- Use Python f-strings for string interpolation (not .format() or % style).
- Map ADF pipeline parameters  → pipeline_params["<name>"]
- Map ADF pipeline variables   → pipeline_vars["<name>"]
- Map ADF activity outputs     → activity_outputs["<activity_name>"]["<field>"]
- Map @item()                  → item  (loop variable in ForEach context)
- Map @pipeline().RunId        → pipeline_params["run_id"]
- Map @utcNow()                → datetime.utcnow().isoformat()
- Map @guid()                  → str(uuid.uuid4())

COMMON ADF FUNCTIONS:
- @concat(a, b, ...)           → f"{a}{b}..."  or  "".join([a, b, ...])
- @string(x)                   → str(x)
- @int(x)                      → int(x)
- @bool(x)                     → bool(x)
- @equals(a, b)                → a == b
- @greater(a, b)               → a > b
- @less(a, b)                  → a < b
- @if(cond, a, b)              → a if cond else b
- @and(a, b)                   → a and b
- @or(a, b)                    → a or b
- @not(a)                      → not a
- @length(x)                   → len(x)
- @empty(x)                    → not x
- @contains(s, sub)            → sub in s
- @startsWith(s, prefix)       → s.startswith(prefix)
- @endsWith(s, suffix)         → s.endswith(suffix)
- @replace(s, old, new)        → s.replace(old, new)
- @split(s, delim)             → s.split(delim)
- @trim(s)                     → s.strip()
- @toUpper(s)                  → s.upper()
- @toLower(s)                  → s.lower()
- @first(arr)                  → arr[0]
- @last(arr)                   → arr[-1]
- @take(arr, n)                → arr[:n]
- @skip(arr, n)                → arr[n:]
- @union(a, b)                 → list(set(a) | set(b))
- @intersection(a, b)          → list(set(a) & set(b))
- @addDays(date, n)            → date + timedelta(days=n)
- @formatDateTime(dt, fmt)     → dt.strftime(fmt)
"""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TranslationRequest:
    """A single ADF expression queued for LLM translation."""

    json_path: str          # Dot-path in the activity typeProperties.
    adf_expression: str     # Raw ADF expression, e.g. "@concat(...)".
    activity_name: str
    context_hint: str = ""  # Optional surrounding context for the LLM.


@dataclass
class TranslationResult:
    """Outcome of a single LLM translation call."""

    request: TranslationRequest
    pyspark_expression: str
    model_used: str
    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Qwen2.5-Coder specific output cleaning
# ---------------------------------------------------------------------------


# Patterns that Qwen2.5-Coder may emit despite the system prompt constraints.
_QWEN_ARTEFACT_PATTERNS: list[re.Pattern] = [
    re.compile(r"<\|im_(start|end)\|>(\w+)?\n?"),   # ChatML token leakage
    re.compile(r"^(Sure|Certainly|Of course)[,!].*\n", re.MULTILINE),  # preamble
    re.compile(r"^Here('s| is) the (translation|equivalent|code)[:\s]*\n", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^```(?:python)?\n?", re.MULTILINE),  # opening fence
    re.compile(r"\n?```$", re.MULTILINE),             # closing fence
    re.compile(r"^In Python[,:].*\n", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Note:.*$", re.MULTILINE | re.IGNORECASE),  # trailing notes
]


def _clean_qwen_output(raw: str) -> str:
    """
    Strip Qwen2.5-Coder-specific artefacts from raw model output.

    Applies patterns in order, then strips leading/trailing whitespace.
    More conservative than a single regex — easier to debug and extend.
    """
    cleaned = raw
    for pattern in _QWEN_ARTEFACT_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Ollama client wrapper
# ---------------------------------------------------------------------------


class OllamaAgent:
    """
    Async wrapper around the Ollama Python SDK, tuned for Qwen2.5-Coder.

    Streams token-by-token responses so the Streamlit UI can display
    output in real time without blocking.

    Parameters
    ----------
    model:
        Ollama model tag. Defaults to qwen2.5-coder:7b.
    base_url:
        Ollama REST API base URL.
    timeout:
        Per-request timeout in seconds. Increase for larger model variants.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = 180,          # Qwen 7B is slower than CodeLlama
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._profile: ModelProfile = get_model_profile(model)
        self._client: "ollama.AsyncClient | None" = None  # type: ignore[name-defined]
        logger.info(
            "OllamaAgent initialised: model=%s profile=%s",
            model,
            self._profile.description,
        )

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def _get_client(self) -> "ollama.AsyncClient":  # type: ignore[name-defined]
        if self._client is None:
            try:
                import ollama  # noqa: PLC0415
                self._client = ollama.AsyncClient(host=self.base_url)
            except ImportError as exc:
                raise RuntimeError(
                    "The 'ollama' package is not installed. Run: pip install ollama"
                ) from exc
        return self._client

    # ------------------------------------------------------------------
    # Connection health check
    # ------------------------------------------------------------------

    async def check_connection(self) -> tuple[bool, str]:
        """
        Verify Ollama is reachable and the target model is pulled locally.

        Returns
        -------
        (ok, human_readable_message)
        """
        try:
            client = self._get_client()
            models_response = await client.list()
            available = [m.model for m in models_response.models]
            logger.info("Ollama reachable. Available models: %s", available)

            # Exact match first, then prefix match for quantised variants
            # e.g. "qwen2.5-coder:7b-q4_K_M" satisfies a "qwen2.5-coder:7b" request
            model_found = self.model in available or any(
                m.startswith(self.model.split(":")[0]) for m in available
            )

            if not model_found:
                return False, (
                    f"Model '{self.model}' is not pulled.\n"
                    f"Run:  ollama pull {self.model}\n\n"
                    f"Available: {', '.join(available) or 'none'}"
                )

            return True, f"✅ Connected — '{self.model}' ready. ({self._profile.description})"

        except Exception as exc:
            logger.error("Ollama connection check failed: %s", exc)
            return False, (
                f"Cannot reach Ollama at {self.base_url}.\n"
                "Ensure Ollama is running:  ollama serve"
            )

    # ------------------------------------------------------------------
    # Streaming translation
    # ------------------------------------------------------------------

    async def translate_expression_stream(
        self, request: TranslationRequest
    ) -> AsyncIterator[str]:
        """
        Stream the LLM translation for a single ADF expression, token by token.

        Uses the model profile's inference parameters for Qwen2.5-Coder.

        Yields
        ------
        str
            Raw token strings as they arrive from Ollama.
        """
        user_prompt = self._build_user_prompt(request)
        logger.info(
            "Translating [%s] path='%s' expr='%s'",
            request.activity_name,
            request.json_path,
            request.adf_expression[:80],
        )

        client = self._get_client()

        try:
            async for chunk in await client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                stream=True,
                options=self._profile.as_options(),
            ):
                token: str = chunk.message.content or ""
                if token:
                    yield token

        except Exception as exc:
            logger.error(
                "Ollama streaming failed for '%s': %s",
                request.adf_expression[:60],
                exc,
                exc_info=True,
            )
            # Yield a valid Python stub so the code file remains syntactically
            # correct even when translation fails.
            yield f"# TRANSLATION_ERROR: {exc}\n"
            yield f"# Original ADF expression: {request.adf_expression}\n"
            yield "None  # TODO: manual translation required\n"

    # ------------------------------------------------------------------
    # Non-streaming translation (batch / test mode)
    # ------------------------------------------------------------------

    async def translate_expression(self, request: TranslationRequest) -> TranslationResult:
        """
        Collect the full streaming response before returning.
        Intended for batch processing and unit tests.
        """
        tokens: list[str] = []
        async for token in self.translate_expression_stream(request):
            tokens.append(token)

        raw_output = "".join(tokens)
        cleaned = _clean_qwen_output(raw_output)
        success = not cleaned.startswith("# TRANSLATION_ERROR")

        return TranslationResult(
            request=request,
            pyspark_expression=cleaned,
            model_used=self.model,
            success=success,
            error=None if success else cleaned,
        )

    # ------------------------------------------------------------------
    # Batch translation with bounded concurrency
    # ------------------------------------------------------------------

    async def translate_batch(
        self,
        requests: list[TranslationRequest],
        max_concurrency: int = 1,    # Qwen 7B is memory-heavy — default to serial
    ) -> list[TranslationResult]:
        """
        Translate multiple expressions with semaphore-bounded concurrency.

        Note: Qwen2.5-Coder 7B is memory-intensive. Keep max_concurrency=1
        unless you have a dedicated GPU with >16 GB VRAM. Increase to 2 only
        for the 14B/32B variants on high-memory systems.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _guarded(req: TranslationRequest) -> TranslationResult:
            async with semaphore:
                return await self.translate_expression(req)

        tasks = [_guarded(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        typed: list[TranslationResult] = []
        for req, res in zip(requests, results):
            if isinstance(res, Exception):
                logger.error("Batch task raised for '%s': %s", req.activity_name, res)
                typed.append(TranslationResult(
                    request=req,
                    pyspark_expression=f"# EXCEPTION: {res}",
                    model_used=self.model,
                    success=False,
                    error=str(res),
                ))
            else:
                typed.append(res)
        return typed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_user_prompt(self, request: TranslationRequest) -> str:
        """
        Build a structured user prompt that Qwen2.5-Coder responds to well.

        Qwen2.5-Coder follows labelled key-value prompts reliably when the
        instruction is explicit and single-sentence.
        """
        lines = [
            f"Activity : {request.activity_name}",
            f"Path     : {request.json_path}",
            f"ADF      : {request.adf_expression}",
        ]
        if request.context_hint:
            lines.append(f"Context  : {request.context_hint}")
        lines.append(
            "\nOutput the Python/PySpark equivalent of the ADF expression above. "
            "No explanation. Code only."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Code patching
# ---------------------------------------------------------------------------


def patch_code_with_translations(
    pyspark_code: str,
    translations: list[TranslationResult],
) -> str:
    """
    Substitute ADF expression placeholders in generated PySpark code with
    their Qwen-translated Python equivalents.

    Intentionally conservative — leaves the placeholder intact and appends
    a warning comment if a substitution point cannot be found unambiguously.
    """
    patched = pyspark_code
    for result in translations:
        if not result.success:
            continue
        original = result.request.adf_expression
        escaped = re.escape(original)
        new_code, n_subs = re.subn(
            rf'["\']?{escaped}["\']?',
            result.pyspark_expression,
            patched,
            count=1,
        )
        if n_subs:
            patched = new_code
            logger.debug("Patched '%s' → '%s'", original[:60], result.pyspark_expression[:60])
        else:
            logger.warning("Could not find substitution point for: %s", original[:60])
    return patched
