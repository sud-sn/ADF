"""
ui_agent.py
===========
UI Workflow Orchestration Agent.

This module is the single authoritative coordinator between the Streamlit
frontend and all backend subsystems (parser, transpiler, Ollama LLM agent).

Responsibilities:
    - Own and enforce the pipeline workflow as a formal finite state machine.
    - Provide a clean, typed public API that the Streamlit UI delegates to.
    - Abstract all async/sync bridging so the UI layer never touches asyncio.
    - Manage retry logic with exponential back-off for Ollama calls.
    - Emit structured workflow events for UI progress rendering.
    - Encapsulate all session-state reads/writes behind a typed facade so the
      UI never accesses `st.session_state` keys by raw string.

Design principles applied:
    - Single Responsibility: UI knows nothing about parsing or LLM mechanics.
    - Open/Closed: New pipeline phases can be added by extending WorkflowPhase
      and adding a handler; existing logic is untouched.
    - Dependency Inversion: Agent depends on abstract interfaces, not concrete
      Streamlit widgets.
    - Fail-fast with graceful degradation: hard errors surface immediately;
      soft errors (bad ADF expression) produce partial output with clear markers.

Author: Transpiler Architect
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Generator, Iterator

from core.adf_parser import ParsedPipeline, parse_pipeline_json
from core.dataflow_parser import ParsedDataFlow, parse_dataflow_json, detect_json_type
from core.transpiler import ActivityTranspiler, TranspilerResult
from core.dataflow_transpiler import DataFlowTranspiler, DataFlowTranspileResult
from agents.ollama_agent import (
    OllamaAgent,
    TranslationRequest,
    TranslationResult,
    _clean_qwen_output,
    get_system_prompt,
    patch_code_with_translations,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workflow state machine
# ---------------------------------------------------------------------------


class WorkflowPhase(Enum):
    """
    Ordered phases of the transpilation pipeline.

    Transitions are strictly enforced — no phase can be entered unless all
    prerequisite phases have completed successfully.
    """
    IDLE = auto()
    PARSING = auto()
    PARSED = auto()
    TRANSPILING = auto()
    TRANSPILED = auto()
    CONNECTING_OLLAMA = auto()
    TRANSLATING = auto()
    DONE = auto()
    ERROR = auto()


# Allowed forward transitions (phase → set of valid next phases)
_VALID_TRANSITIONS: dict[WorkflowPhase, set[WorkflowPhase]] = {
    WorkflowPhase.IDLE:              {WorkflowPhase.PARSING, WorkflowPhase.ERROR},
    WorkflowPhase.PARSING:           {WorkflowPhase.PARSED, WorkflowPhase.ERROR},
    WorkflowPhase.PARSED:            {WorkflowPhase.TRANSPILING, WorkflowPhase.PARSING, WorkflowPhase.ERROR},
    WorkflowPhase.TRANSPILING:       {WorkflowPhase.TRANSPILED, WorkflowPhase.ERROR},
    WorkflowPhase.TRANSPILED:        {WorkflowPhase.CONNECTING_OLLAMA, WorkflowPhase.PARSING, WorkflowPhase.DONE},
    WorkflowPhase.CONNECTING_OLLAMA: {WorkflowPhase.TRANSLATING, WorkflowPhase.TRANSPILED, WorkflowPhase.ERROR},
    WorkflowPhase.TRANSLATING:       {WorkflowPhase.DONE, WorkflowPhase.ERROR},
    WorkflowPhase.DONE:              {WorkflowPhase.IDLE, WorkflowPhase.PARSING},
    WorkflowPhase.ERROR:             {WorkflowPhase.IDLE, WorkflowPhase.PARSING},
}


# ---------------------------------------------------------------------------
# Event system
# ---------------------------------------------------------------------------


class EventKind(Enum):
    PHASE_CHANGED = "phase_changed"
    PROGRESS = "progress"
    EXPRESSION_TRANSLATED = "expression_translated"
    CODE_PATCHED = "code_patched"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class WorkflowEvent:
    """Immutable event emitted by UIAgent for the UI layer to consume."""
    kind: EventKind
    message: str
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


# Callback signature: receives a WorkflowEvent, returns nothing.
EventCallback = Callable[[WorkflowEvent], None]


# ---------------------------------------------------------------------------
# Session state facade
# ---------------------------------------------------------------------------


@dataclass
class WorkflowState:
    """
    Typed container for all mutable workflow state.

    The UI agent owns this object exclusively. Streamlit's `st.session_state`
    stores a single instance of this class under a fixed key, ensuring that
    all state mutations go through this typed dataclass rather than raw dict
    access scattered across the UI layer.
    """
    phase: WorkflowPhase = WorkflowPhase.IDLE
    mode: str = "pipeline"   # "pipeline" or "dataflow"
    parsed_pipeline: ParsedPipeline | None = None
    parsed_dataflow: ParsedDataFlow | None = None
    transpiler_result: TranspilerResult | None = None
    dataflow_result: DataFlowTranspileResult | None = None
    final_code: str | None = None
    translation_log: list[TranslationLogEntry] = field(default_factory=list)
    error_message: str | None = None
    ollama_model: str = "qwen2.5-coder:7b"   # Primary: Qwen2.5-Coder 7B
    ollama_url: str = "http://localhost:11434"
    ollama_connected: bool = False
    total_expressions: int = 0
    translated_expressions: int = 0

    @property
    def progress_pct(self) -> float:
        if self.total_expressions == 0:
            return 0.0
        return self.translated_expressions / self.total_expressions

    @property
    def is_busy(self) -> bool:
        return self.phase in {
            WorkflowPhase.PARSING,
            WorkflowPhase.TRANSPILING,
            WorkflowPhase.CONNECTING_OLLAMA,
            WorkflowPhase.TRANSLATING,
        }

    @property
    def can_transpile(self) -> bool:
        if self.mode == "dataflow":
            return self.phase == WorkflowPhase.PARSED and self.parsed_dataflow is not None
        return self.phase == WorkflowPhase.PARSED and self.parsed_pipeline is not None

    @property
    def can_translate(self) -> bool:
        if self.mode == "dataflow":
            return (
                self.phase == WorkflowPhase.TRANSPILED
                and self.dataflow_result is not None
                and self.dataflow_result.has_pending_llm_work
            )
        return (
            self.phase == WorkflowPhase.TRANSPILED
            and self.transpiler_result is not None
            and self.transpiler_result.has_pending_llm_work
        )

    @property
    def can_download(self) -> bool:
        return self.final_code is not None and self.phase in {
            WorkflowPhase.TRANSPILED,
            WorkflowPhase.TRANSLATING,
            WorkflowPhase.DONE,
        }


@dataclass
class TranslationLogEntry:
    """Single entry in the Ollama translation audit log."""
    activity_name: str
    json_path: str
    adf_expression: str
    pyspark_expression: str
    model: str
    success: bool
    duration_ms: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


@dataclass
class RetryPolicy:
    """Exponential back-off retry configuration for Ollama calls."""
    max_attempts: int = 3
    base_delay_s: float = 1.0
    backoff_factor: float = 2.0
    max_delay_s: float = 30.0

    def delays(self) -> Iterator[float]:
        delay = self.base_delay_s
        for _ in range(self.max_attempts - 1):
            yield delay
            delay = min(delay * self.backoff_factor, self.max_delay_s)


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------


class UIAgent:
    """
    Workflow orchestration agent for the ADF-to-PySpark Transpiler UI.

    Lifecycle
    ---------
    A single instance should be stored in `st.session_state` for the duration
    of a Streamlit session. The UI layer creates it once via `UIAgent.from_session_state()`
    and calls public methods to drive each workflow step.

    Thread/async safety
    -------------------
    Streamlit runs each user interaction as a synchronous script re-run.
    All async work (Ollama calls) is bridged via `asyncio.run()` inside
    dedicated methods. The agent never exposes coroutines to the UI layer.

    Example
    -------
    ::

        agent = UIAgent.get_or_create(st.session_state)
        if uploaded_file:
            agent.parse(uploaded_file.read().decode())
        if agent.state.can_transpile:
            agent.transpile()
        if agent.state.can_translate:
            for event in agent.translate_streaming():
                render_event(event)
    """

    SESSION_KEY = "_ui_agent_state"

    def __init__(
        self,
        state: WorkflowState,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._state = state
        self._retry = retry_policy or RetryPolicy()
        self._callbacks: list[EventCallback] = []

    # ------------------------------------------------------------------
    # Factory / session integration
    # ------------------------------------------------------------------

    @classmethod
    def get_or_create(cls, session: "dict") -> "UIAgent":
        """
        Retrieve an existing UIAgent from Streamlit's session_state or
        create a fresh one if none exists.

        Parameters
        ----------
        session:
            Pass `st.session_state` directly.
        """
        if cls.SESSION_KEY not in session:
            logger.info("Initialising new UIAgent workflow state.")
            session[cls.SESSION_KEY] = WorkflowState()
        return cls(state=session[cls.SESSION_KEY])

    @classmethod
    def reset(cls, session: "dict") -> "UIAgent":
        """Hard-reset the workflow. Discards all parsed/transpiled state."""
        logger.info("UIAgent: hard reset triggered.")
        session[cls.SESSION_KEY] = WorkflowState(
            ollama_model=session.get(cls.SESSION_KEY, WorkflowState()).ollama_model,
            ollama_url=session.get(cls.SESSION_KEY, WorkflowState()).ollama_url,
        )
        return cls(state=session[cls.SESSION_KEY])

    # ------------------------------------------------------------------
    # Public state property (read-only for UI)
    # ------------------------------------------------------------------

    @property
    def state(self) -> WorkflowState:
        return self._state

    # ------------------------------------------------------------------
    # Event subscription
    # ------------------------------------------------------------------

    def subscribe(self, callback: EventCallback) -> None:
        """Register a UI callback to receive WorkflowEvents."""
        self._callbacks.append(callback)

    def _emit(self, kind: EventKind, message: str, **payload) -> None:
        event = WorkflowEvent(kind=kind, message=message, payload=payload)
        logger.debug("UIAgent event: %s — %s", kind.value, message)
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as exc:
                logger.warning("Event callback raised: %s", exc)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_ollama(self, model: str, url: str) -> None:
        """Update Ollama connection settings (safe to call at any phase)."""
        self._state.ollama_model = model
        self._state.ollama_url = url
        self._state.ollama_connected = False  # invalidate previous check
        logger.info("Ollama config updated: model=%s url=%s", model, url)

    # ------------------------------------------------------------------
    # Phase 1: Parse
    # ------------------------------------------------------------------

    def parse(self, raw_json: str) -> bool:
        """
        Validate and parse a raw ADF pipeline JSON string.

        Parameters
        ----------
        raw_json:
            UTF-8 string contents of a pipeline.json export.

        Returns
        -------
        bool
            True on success; False on failure (error stored in state.error_message).
        """
        self._transition(WorkflowPhase.PARSING)
        self._emit(EventKind.INFO, "Validating and parsing ADF pipeline JSON…")

        try:
            parsed = parse_pipeline_json(raw_json)
        except ValueError as exc:
            return self._fail(f"Pipeline parse failed: {exc}")
        except Exception as exc:
            logger.exception("Unexpected parse error")
            return self._fail(f"Unexpected error during parsing: {exc}")

        self._state.parsed_pipeline = parsed
        self._state.transpiler_result = None
        self._state.final_code = None
        self._state.translation_log = []
        self._state.error_message = None

        self._transition(WorkflowPhase.PARSED)
        self._emit(
            EventKind.INFO,
            f"Pipeline '{parsed.pipeline.name}' parsed — "
            f"{parsed.activity_count} activities.",
            pipeline_name=parsed.pipeline.name,
            activity_count=parsed.activity_count,
        )
        return True

    # ------------------------------------------------------------------
    # Phase 1b: Parse Data Flow
    # ------------------------------------------------------------------

    def parse_dataflow(self, raw_json: str) -> bool:
        """
        Validate and parse a raw ADF Data Flow definition JSON string.

        Parameters
        ----------
        raw_json:
            UTF-8 string contents of a data flow definition JSON export.

        Returns
        -------
        bool
            True on success; False on failure.
        """
        self._transition(WorkflowPhase.PARSING)
        self._state.mode = "dataflow"
        self._emit(EventKind.INFO, "Validating and parsing ADF Data Flow definition JSON...")

        try:
            parsed = parse_dataflow_json(raw_json)
        except ValueError as exc:
            return self._fail(f"Data Flow parse failed: {exc}")
        except Exception as exc:
            logger.exception("Unexpected data flow parse error")
            return self._fail(f"Unexpected error during data flow parsing: {exc}")

        self._state.parsed_dataflow = parsed
        self._state.parsed_pipeline = None
        self._state.dataflow_result = None
        self._state.transpiler_result = None
        self._state.final_code = None
        self._state.translation_log = []
        self._state.error_message = None

        self._transition(WorkflowPhase.PARSED)
        self._emit(
            EventKind.INFO,
            f"Data Flow '{parsed.name}' parsed - "
            f"{len(parsed.sources)} sources, {len(parsed.sinks)} sinks, "
            f"{len(parsed.transformations)} transformations.",
            dataflow_name=parsed.name,
            source_count=len(parsed.sources),
            sink_count=len(parsed.sinks),
            transformation_count=len(parsed.transformations),
        )
        return True

    # ------------------------------------------------------------------
    # Phase 1c: Auto-detect and parse
    # ------------------------------------------------------------------

    def auto_parse(self, raw_json: str) -> bool:
        """
        Auto-detect JSON type and parse accordingly.

        Returns True on success, False on failure.
        """
        try:
            json_type = detect_json_type(raw_json)
        except ValueError as exc:
            return self._fail(f"JSON type detection failed: {exc}")

        self._state.mode = json_type
        if json_type == "dataflow":
            return self.parse_dataflow(raw_json)
        return self.parse(raw_json)

    # ------------------------------------------------------------------
    # Phase 2: Transpile
    # ------------------------------------------------------------------

    def transpile(self) -> bool:
        """
        Run the deterministic PySpark template engine against the parsed pipeline.

        Must be called after a successful `parse()`.

        Returns
        -------
        bool
            True on success; False on failure.
        """
        if not self._state.can_transpile:
            return self._fail(
                "Cannot transpile: pipeline has not been successfully parsed.",
                kind=EventKind.WARNING,
            )

        self._transition(WorkflowPhase.TRANSPILING)
        self._emit(EventKind.INFO, "Running deterministic PySpark template mapping…")

        try:
            transpiler = ActivityTranspiler()
            result = transpiler.transpile(self._state.parsed_pipeline)  # type: ignore[arg-type]
        except Exception as exc:
            logger.exception("Transpilation raised unexpectedly")
            return self._fail(f"Transpilation engine error: {exc}")

        self._state.transpiler_result = result
        self._state.final_code = result.full_code
        self._state.total_expressions = result.total_pending_expressions
        self._state.translated_expressions = 0

        self._transition(WorkflowPhase.TRANSPILED)
        self._emit(
            EventKind.INFO,
            f"Transpilation complete. "
            f"Pending ADF expressions for Ollama: {result.total_pending_expressions}.",
            pending_expressions=result.total_pending_expressions,
            has_pending=result.has_pending_llm_work,
        )
        return True

    # ------------------------------------------------------------------
    # Phase 2b: Transpile Data Flow
    # ------------------------------------------------------------------

    def transpile_dataflow(self) -> bool:
        """
        Run the data flow transpiler against the parsed data flow definition.

        Must be called after a successful ``parse_dataflow()``.

        Returns
        -------
        bool
            True on success; False on failure.
        """
        if not self._state.can_transpile or self._state.mode != "dataflow":
            return self._fail(
                "Cannot transpile: data flow has not been successfully parsed.",
                kind=EventKind.WARNING,
            )

        self._transition(WorkflowPhase.TRANSPILING)
        self._emit(EventKind.INFO, "Running Data Flow PySpark transpiler...")

        try:
            transpiler = DataFlowTranspiler()
            result = transpiler.transpile(self._state.parsed_dataflow)  # type: ignore[arg-type]
        except Exception as exc:
            logger.exception("Data flow transpilation raised unexpectedly")
            return self._fail(f"Data flow transpilation error: {exc}")

        self._state.dataflow_result = result
        self._state.transpiler_result = None  # Clear pipeline result
        self._state.final_code = result.pyspark_code
        self._state.total_expressions = result.total_pending_expressions
        self._state.translated_expressions = 0

        self._transition(WorkflowPhase.TRANSPILED)
        self._emit(
            EventKind.INFO,
            f"Data Flow transpilation complete. "
            f"Pending DFS expressions for Ollama: {result.total_pending_expressions}.",
            pending_expressions=result.total_pending_expressions,
            has_pending=result.has_pending_llm_work,
        )
        return True

    # ------------------------------------------------------------------
    # Phase 3: Ollama connection check
    # ------------------------------------------------------------------

    def check_ollama_connection(self) -> tuple[bool, str]:
        """
        Verify Ollama is reachable and the configured model is available.

        Returns
        -------
        (success, human_readable_message)
        """
        self._transition(WorkflowPhase.CONNECTING_OLLAMA)
        agent = self._build_ollama_agent()

        try:
            ok, msg = asyncio.run(agent.check_connection())
        except Exception as exc:
            msg = f"Connection check raised: {exc}"
            ok = False

        self._state.ollama_connected = ok

        if ok:
            self._emit(EventKind.INFO, msg)
            self._transition(WorkflowPhase.TRANSPILED)  # back to ready state
        else:
            self._emit(EventKind.WARNING, msg)
            self._transition(WorkflowPhase.TRANSPILED)  # allow retry

        return ok, msg

    # ------------------------------------------------------------------
    # Phase 4: LLM translation (streaming)
    # ------------------------------------------------------------------

    def translate_streaming(self) -> Generator[WorkflowEvent, None, None]:
        """
        Drive ADF-expression → PySpark translation via Ollama, yielding
        a WorkflowEvent for every meaningful state change.

        This is a synchronous generator — Streamlit calls `next()` in a loop
        and renders each event as it arrives. The generator itself bridges to
        async Ollama calls internally.

        Yields
        ------
        WorkflowEvent
            PROGRESS events during translation, EXPRESSION_TRANSLATED on each
            completion, CODE_PATCHED when the code buffer is updated, and
            PHASE_CHANGED at start/end.
        """
        if not self._state.can_translate:
            yield self._make_event(
                EventKind.WARNING,
                "Translation prerequisites not met.",
            )
            return

        self._transition(WorkflowPhase.TRANSLATING)
        yield self._make_event(EventKind.PHASE_CHANGED, "Starting Ollama translation.")

        result = self._state.transpiler_result
        assert result is not None  # guaranteed by can_translate guard above

        requests = self._build_translation_requests(result)
        self._state.total_expressions = len(requests)
        self._state.translated_expressions = 0

        if not requests:
            self._transition(WorkflowPhase.DONE)
            yield self._make_event(EventKind.INFO, "No ADF expressions require translation.")
            return

        agent = self._build_ollama_agent()
        current_code = result.full_code

        for i, req in enumerate(requests):
            yield self._make_event(
                EventKind.PROGRESS,
                f"[{i + 1}/{len(requests)}] Translating expression in "
                f"'{req.activity_name}' at path '{req.json_path}'…",
                current=i,
                total=len(requests),
                activity_name=req.activity_name,
                json_path=req.json_path,
            )

            translation, duration_ms = self._translate_with_retry(agent, req)

            log_entry = TranslationLogEntry(
                activity_name=req.activity_name,
                json_path=req.json_path,
                adf_expression=req.adf_expression,
                pyspark_expression=translation.pyspark_expression,
                model=translation.model_used,
                success=translation.success,
                duration_ms=duration_ms,
                error=translation.error,
            )
            self._state.translation_log.append(log_entry)

            if translation.success:
                current_code = patch_code_with_translations(current_code, [translation])
                self._state.final_code = current_code
                self._state.translated_expressions += 1

                yield self._make_event(
                    EventKind.EXPRESSION_TRANSLATED,
                    f"✅ Translated '{req.adf_expression[:60]}…'",
                    adf_expression=req.adf_expression,
                    pyspark_expression=translation.pyspark_expression,
                    duration_ms=duration_ms,
                )
                yield self._make_event(
                    EventKind.CODE_PATCHED,
                    "Code buffer updated.",
                    code=current_code,
                )
            else:
                yield self._make_event(
                    EventKind.WARNING,
                    f"⚠ Translation failed for '{req.adf_expression[:60]}': {translation.error}",
                    adf_expression=req.adf_expression,
                    error=translation.error,
                )

        self._state.final_code = current_code
        self._transition(WorkflowPhase.DONE)
        yield self._make_event(
            EventKind.PHASE_CHANGED,
            f"Translation complete. "
            f"{self._state.translated_expressions}/{self._state.total_expressions} expressions translated.",
            translated=self._state.translated_expressions,
            total=self._state.total_expressions,
        )

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def get_final_code(self) -> str | None:
        """Return the current code buffer (may be partial if translation is in progress)."""
        return self._state.final_code

    def get_download_filename(self) -> str:
        if self._state.mode == "dataflow" and self._state.parsed_dataflow:
            name = self._state.parsed_dataflow.name
        elif self._state.parsed_pipeline:
            name = self._state.parsed_pipeline.pipeline.name
        else:
            name = "output"
        return f"{name}_spark.py"

    def get_translation_log_text(self) -> str:
        """Render the translation log as a human-readable string for display."""
        if not self._state.translation_log:
            return "No translations recorded yet."

        lines: list[str] = ["=" * 72, "  ADF → PySpark Translation Audit Log", "=" * 72]
        for i, entry in enumerate(self._state.translation_log, 1):
            status = "✅ OK" if entry.success else f"❌ FAILED: {entry.error}"
            lines += [
                f"\n[{i:02d}] Activity : {entry.activity_name}",
                f"     Path     : {entry.json_path}",
                f"     ADF      : {entry.adf_expression}",
                f"     PySpark  : {entry.pyspark_expression}",
                f"     Model    : {entry.model}  |  {entry.duration_ms:.0f}ms  |  {status}",
            ]
        lines.append("\n" + "=" * 72)
        return "\n".join(lines)

    def get_dag_summary(self) -> list[dict] | None:
        """
        Return a serialisable DAG summary for the UI to render.
        Each dict represents one top-level activity or transformation node.
        """
        # Data Flow mode
        if self._state.mode == "dataflow" and self._state.parsed_dataflow:
            df = self._state.parsed_dataflow
            summary: list[dict] = []
            for src in df.sources:
                summary.append({
                    "name": src.name, "type": "Source",
                    "depends_on": [], "children": [],
                    "type_properties": {
                        "dataset": src.dataset_ref,
                        "query": src.query or "N/A",
                        "columns": [f"{c.name}: {c.data_type}" for c in src.columns],
                    },
                })
            for t in df.transformations:
                summary.append({
                    "name": t.name, "type": t.transform_type.value.title(),
                    "depends_on": [{"activity": inp, "conditions": ["Data"]} for inp in t.inputs],
                    "children": [],
                    "type_properties": {
                        "transform_type": t.transform_type.value,
                        "description": t.description,
                        "expressions": t.expressions or {},
                        "condition": t.condition or "",
                        "join_type": t.join_type or "",
                        "alter_row_policy": t.alter_row_policy or "",
                    },
                })
            for sink in df.sinks:
                summary.append({
                    "name": sink.name, "type": "Sink",
                    "depends_on": [], "children": [],
                    "type_properties": {
                        "dataset": sink.dataset_ref,
                        "keys": sink.keys,
                        "column_mappings": [
                            f"{m.source_column} -> {m.sink_column}" for m in sink.column_mappings
                        ],
                        "upsertable": sink.upsertable,
                        "format": sink.format,
                    },
                })
            return summary

        # Pipeline mode
        if not self._state.parsed_pipeline:
            return None

        summary = []
        for activity in self._state.parsed_pipeline.pipeline.resolved_activities:
            node: dict = {
                "name": activity.name,
                "type": activity.type,
                "depends_on": [
                    {"activity": d.activity, "conditions": [c.value for c in d.dependency_conditions]}
                    for d in activity.depends_on
                ],
                "type_properties": activity.type_properties,
                "children": [],
            }

            # Import locally to avoid circular reference at module level
            from core.adf_parser import ForEachActivity, IfConditionActivity
            if isinstance(activity, ForEachActivity):
                node["children"] = [
                    {"name": a.name, "type": a.type, "branch": "foreach_inner"}
                    for a in activity.inner_activities
                ]
            elif isinstance(activity, IfConditionActivity):
                node["children"] = [
                    {"name": a.name, "type": a.type, "branch": "true"}
                    for a in activity.true_branch_activities
                ] + [
                    {"name": a.name, "type": a.type, "branch": "false"}
                    for a in activity.false_branch_activities
                ]

            summary.append(node)
        return summary

    # ------------------------------------------------------------------
    # Internal — state machine
    # ------------------------------------------------------------------

    def _transition(self, target: WorkflowPhase) -> None:
        current = self._state.phase
        allowed = _VALID_TRANSITIONS.get(current, set())

        if target not in allowed:
            raise RuntimeError(
                f"Invalid workflow transition: {current.name} → {target.name}. "
                f"Allowed targets: {[p.name for p in allowed]}"
            )

        logger.info("Workflow: %s → %s", current.name, target.name)
        self._state.phase = target
        self._emit(
            EventKind.PHASE_CHANGED,
            f"Phase: {target.name}",
            previous=current.name,
            current=target.name,
        )

    def _fail(
        self,
        message: str,
        kind: EventKind = EventKind.ERROR,
    ) -> bool:
        logger.error("UIAgent failure: %s", message)
        self._state.error_message = message
        if kind == EventKind.ERROR:
            self._state.phase = WorkflowPhase.ERROR
        self._emit(kind, message)
        return False

    # ------------------------------------------------------------------
    # Internal — Ollama
    # ------------------------------------------------------------------

    def _build_ollama_agent(self) -> OllamaAgent:
        return OllamaAgent(
            model=self._state.ollama_model,
            base_url=self._state.ollama_url,
        )

    def _build_translation_requests(
        self, result: TranspilerResult | None = None
    ) -> list[TranslationRequest]:
        requests: list[TranslationRequest] = []

        # Data Flow mode: build from dataflow_result
        if self._state.mode == "dataflow" and self._state.dataflow_result:
            df_result = self._state.dataflow_result
            for json_path, expr in df_result.pending_llm_expressions:
                requests.append(
                    TranslationRequest(
                        json_path=json_path,
                        adf_expression=expr,
                        activity_name=df_result.dataflow_name,
                        context_hint=f"Data Flow Script expression in '{json_path}'",
                    )
                )
            return requests

        # Pipeline mode: build from transpiler activity results
        if result is None:
            return requests
        for ar in result.activity_results:
            for json_path, expr in ar.pending_llm_expressions:
                requests.append(
                    TranslationRequest(
                        json_path=json_path,
                        adf_expression=expr,
                        activity_name=ar.activity_name,
                        context_hint=f"Activity type: {ar.activity_type}",
                    )
                )
        return requests

    def _translate_with_retry(
        self,
        agent: OllamaAgent,
        request: TranslationRequest,
    ) -> tuple[TranslationResult, float]:
        """
        Attempt translation with exponential back-off retry.

        Returns
        -------
        (TranslationResult, elapsed_ms)
        """
        last_exc: Exception | None = None
        attempts = 0
        t0 = time.monotonic()

        for delay in [0.0, *self._retry.delays()]:
            if delay > 0:
                logger.warning(
                    "Retrying translation (attempt %d/%d) after %.1fs back-off…",
                    attempts + 1,
                    self._retry.max_attempts,
                    delay,
                )
                time.sleep(delay)

            attempts += 1
            try:
                tr = asyncio.run(agent.translate_expression(request))
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.info(
                    "Translation succeeded on attempt %d (%.0fms): %s",
                    attempts,
                    elapsed_ms,
                    request.adf_expression[:50],
                )
                return tr, elapsed_ms

            except Exception as exc:
                last_exc = exc
                logger.warning("Translation attempt %d failed: %s", attempts, exc)

        # All retries exhausted — return a failed result rather than raising
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.error(
            "All %d translation attempts failed for expression: %s",
            self._retry.max_attempts,
            request.adf_expression[:80],
        )
        return (
            TranslationResult(
                request=request,
                pyspark_expression=(
                    f"# TRANSLATION_FAILED after {self._retry.max_attempts} attempts\n"
                    f"# Original ADF expression: {request.adf_expression}\n"
                    f"# Last error: {last_exc}\n"
                    f"None  # TODO: manual translation required"
                ),
                model_used=agent.model,
                success=False,
                error=str(last_exc),
            ),
            elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Internal — event helpers
    # ------------------------------------------------------------------

    def _make_event(
        self, kind: EventKind, message: str, **payload
    ) -> WorkflowEvent:
        event = WorkflowEvent(kind=kind, message=message, payload=payload)
        self._emit(kind, message, **payload)
        return event
