"""
transpiler.py
=============
Deterministic template-mapping engine.

Converts strongly-typed ADF activity objects (produced by adf_parser.py) into
PySpark boilerplate code using static Jinja2-style string templates.

- Simple / well-understood activities (Copy, Lookup, SetVariable, Wait) are
  handled entirely here — no LLM required.
- Complex ADF dynamic expressions (`@concat(...)`, `@pipeline().parameters.*`)
  embedded inside activity properties are flagged as `pending_llm_expressions`
  so the Ollama agent can translate them in a follow-up pass.

Author: Transpiler Architect
"""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

from .adf_parser import (
    ActivityBase,
    ActivityType,
    CopyActivity,
    ForEachActivity,
    GenericActivity,
    IfConditionActivity,
    ParsedPipeline,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ADF expression detection
# ---------------------------------------------------------------------------

_ADF_EXPR_RE = re.compile(r"@[a-zA-Z\(]")


def _is_adf_expression(value: Any) -> bool:
    """Return True if *value* looks like an ADF dynamic expression string."""
    return isinstance(value, str) and bool(_ADF_EXPR_RE.search(value))


def _extract_expressions(obj: Any, path: str = "") -> list[tuple[str, str]]:
    """
    Recursively walk a dict/list and collect (json_path, expression) pairs
    for every ADF expression string found.
    """
    found: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.extend(_extract_expressions(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(_extract_expressions(v, f"{path}[{i}]"))
    elif _is_adf_expression(obj):
        found.append((path, obj))
    return found


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ActivityTranspileResult:
    """Transpilation result for a single activity."""

    activity_name: str
    activity_type: str
    pyspark_code: str
    pending_llm_expressions: list[tuple[str, str]] = field(default_factory=list)
    is_deterministic: bool = True
    error: str | None = None


@dataclass
class TranspilerResult:
    """Aggregate result for a full pipeline."""

    pipeline_name: str
    header_code: str
    activity_results: list[ActivityTranspileResult] = field(default_factory=list)
    footer_code: str = ""

    @property
    def full_code(self) -> str:
        """Assemble the complete PySpark script, with LLM placeholders in place."""
        parts = [self.header_code]
        for r in self.activity_results:
            parts.append(f"\n# {'=' * 70}")
            parts.append(f"# Activity: {r.activity_name}  [{r.activity_type}]")
            if not r.is_deterministic:
                parts.append("# ⚠ Contains ADF expressions — pending Ollama translation.")
            parts.append(f"# {'=' * 70}")
            parts.append(r.pyspark_code)
        parts.append(self.footer_code)
        return "\n".join(parts)

    @property
    def has_pending_llm_work(self) -> bool:
        return any(r.pending_llm_expressions for r in self.activity_results)

    @property
    def total_pending_expressions(self) -> int:
        return sum(len(r.pending_llm_expressions) for r in self.activity_results)


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------


def _safe_var(name: str) -> str:
    """Convert an ADF activity name to a valid Python identifier."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()


def _indent(code: str, spaces: int = 4) -> str:
    return textwrap.indent(textwrap.dedent(code), " " * spaces)


# ---------------------------------------------------------------------------
# Per-activity transpilers
# ---------------------------------------------------------------------------


def _transpile_copy(activity: CopyActivity) -> ActivityTranspileResult:
    """Map a CopyActivity to PySpark DataFrame read/write boilerplate."""
    var = _safe_var(activity.name)
    props = activity.parsed_type_properties

    source_type = props.source.type if props else "UnknownSource"
    sink_type = props.sink.type if props else "UnknownSink"

    source_format = _infer_spark_format(source_type)
    sink_format = _infer_spark_format(sink_type)

    expressions = _extract_expressions(activity.type_properties)

    code = f'''\
# --- CopyActivity: {activity.name} ---
# Source connector : {source_type}
# Sink connector   : {sink_type}

logger.info("Starting copy: {activity.name}")

df_{var} = (
    spark.read
    .format("{source_format}")
    # TODO: set .option("path", ...) or configure linked service credentials
    .load()
)

logger.info("Read complete. Row count: %d", df_{var}.count())

(
    df_{var}.write
    .format("{sink_format}")
    .mode("overwrite")  # ADF sink writeBehavior: {getattr(props.sink, "write_behavior", "N/A") if props else "N/A"}
    # TODO: set .option("path", ...) or configure sink credentials
    .save()
)

logger.info("Write complete: {activity.name}")
'''

    return ActivityTranspileResult(
        activity_name=activity.name,
        activity_type=activity.type,
        pyspark_code=code,
        pending_llm_expressions=expressions,
        is_deterministic=not bool(expressions),
    )


def _transpile_foreach(activity: ForEachActivity) -> ActivityTranspileResult:
    """Map a ForEachActivity to a Python for-loop over a Spark-collected list."""
    var = _safe_var(activity.name)
    props = activity.parsed_type_properties
    is_sequential = props.is_sequential if props else True
    items_expr = str(props.items) if props else "@unknown"
    expressions = _extract_expressions(activity.type_properties)

    parallel_note = (
        "# Sequential execution enforced."
        if is_sequential
        else f"# ADF batchCount={props.batch_count if props else 'N/A'}. "
             "# Consider spark.sparkContext.parallelize() for true parallelism."
    )

    code = f'''\
# --- ForEachActivity: {activity.name} ---
# Items expression (requires Ollama translation): {items_expr}
{parallel_note}

logger.info("Starting ForEach: {activity.name}")

# PLACEHOLDER: Replace with actual resolved collection after Ollama translation.
items_{var}: list = []  # TODO: resolve ADF expression → Python list

for item_{var} in items_{var}:
    logger.debug("ForEach [{activity.name}] processing item: %s", item_{var})
    # --- Inner activities translated below ---
    # {"; ".join(a.name for a in activity.inner_activities) or "none"}
    pass  # TODO: implement inner activity logic

logger.info("ForEach complete: {activity.name}")
'''

    return ActivityTranspileResult(
        activity_name=activity.name,
        activity_type=activity.type,
        pyspark_code=code,
        pending_llm_expressions=expressions,
        is_deterministic=False,
    )


def _transpile_if_condition(activity: IfConditionActivity) -> ActivityTranspileResult:
    """Map an IfConditionActivity to a Python if/else block."""
    var = _safe_var(activity.name)
    props = activity.parsed_type_properties
    expr_raw = str(props.expression) if props else "@unknown"
    expressions = _extract_expressions(activity.type_properties)

    true_names = [a.name for a in activity.true_branch_activities]
    false_names = [a.name for a in activity.false_branch_activities]

    code = f'''\
# --- IfConditionActivity: {activity.name} ---
# ADF expression (requires Ollama translation): {expr_raw}

logger.info("Evaluating IfCondition: {activity.name}")

# PLACEHOLDER: Replace with Ollama-translated Python boolean expression.
condition_{var}: bool = False  # TODO: resolve ADF expression → Python bool

if condition_{var}:
    logger.info("[{activity.name}] TRUE branch executing: {true_names}")
    # TODO: implement true branch → {", ".join(true_names) or "empty"}
    pass
else:
    logger.info("[{activity.name}] FALSE branch executing: {false_names}")
    # TODO: implement false branch → {", ".join(false_names) or "empty"}
    pass
'''

    return ActivityTranspileResult(
        activity_name=activity.name,
        activity_type=activity.type,
        pyspark_code=code,
        pending_llm_expressions=expressions,
        is_deterministic=False,
    )


def _transpile_set_variable(activity: GenericActivity) -> ActivityTranspileResult:
    var = _safe_var(activity.name)
    var_name = activity.type_properties.get("variableName", "unknown_var")
    value = activity.type_properties.get("value", "")
    expressions = _extract_expressions(activity.type_properties)

    code = f'''\
# --- SetVariable: {activity.name} ---
{var}_value = {repr(value)}  # TODO: translate ADF expression if needed
pipeline_vars["{var_name}"] = {var}_value
logger.debug("SetVariable: {var_name} = %s", {var}_value)
'''
    return ActivityTranspileResult(
        activity_name=activity.name,
        activity_type=activity.type,
        pyspark_code=code,
        pending_llm_expressions=expressions,
        is_deterministic=not bool(expressions),
    )


def _transpile_wait(activity: GenericActivity) -> ActivityTranspileResult:
    wait_time = activity.type_properties.get("waitTimeInSeconds", 30)
    code = f'''\
# --- Wait: {activity.name} ---
import time
logger.info("Waiting {wait_time}s for: {activity.name}")
time.sleep({wait_time})
'''
    return ActivityTranspileResult(
        activity_name=activity.name,
        activity_type=activity.type,
        pyspark_code=code,
        is_deterministic=True,
    )


def _transpile_generic(activity: ActivityBase) -> ActivityTranspileResult:
    expressions = _extract_expressions(activity.type_properties)
    code = f'''\
# --- {activity.type}: {activity.name} ---
# ⚠ No dedicated template for type "{activity.type}".
# Raw typeProperties preserved for manual implementation or Ollama translation.
# typeProperties: {activity.type_properties}
pass
'''
    return ActivityTranspileResult(
        activity_name=activity.name,
        activity_type=activity.type,
        pyspark_code=code,
        pending_llm_expressions=expressions,
        is_deterministic=False,
    )


def _infer_spark_format(connector_type: str) -> str:
    mapping = {
        "AzureBlobSource": "parquet",
        "AzureBlobSink": "parquet",
        "AzureSqlSource": "jdbc",
        "AzureSqlSink": "jdbc",
        "AzureDataLakeStoreSource": "parquet",
        "AzureDataLakeStoreSink": "parquet",
        "DelimitedTextSource": "csv",
        "DelimitedTextSink": "csv",
        "JsonSource": "json",
        "JsonSink": "json",
        "ParquetSource": "parquet",
        "ParquetSink": "parquet",
        "OrcSource": "orc",
        "OrcSink": "orc",
        "AvroSource": "avro",
        "AvroSink": "avro",
    }
    return mapping.get(connector_type, "parquet")


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_SPECIFIC_HANDLERS = {
    ActivityType.SET_VARIABLE.value: _transpile_set_variable,
    ActivityType.WAIT.value: _transpile_wait,
}


# ---------------------------------------------------------------------------
# Main transpiler class
# ---------------------------------------------------------------------------


class ActivityTranspiler:
    """
    Converts a `ParsedPipeline` into a `TranspilerResult` containing
    ready-to-use PySpark code with ADF-expression placeholders.

    Usage
    -----
    ::

        parsed = parse_pipeline_json(raw_json)
        result = ActivityTranspiler().transpile(parsed)
        print(result.full_code)
    """

    def transpile(self, parsed: ParsedPipeline) -> TranspilerResult:
        logger.info("Transpiling pipeline '%s'.", parsed.pipeline.name)

        result = TranspilerResult(
            pipeline_name=parsed.pipeline.name,
            header_code=self._build_header(parsed),
            footer_code=self._build_footer(parsed),
        )

        for activity in parsed.pipeline.resolved_activities:
            try:
                activity_result = self._dispatch(activity)
            except Exception as exc:
                logger.error(
                    "Transpilation failed for activity '%s': %s", activity.name, exc, exc_info=True
                )
                activity_result = ActivityTranspileResult(
                    activity_name=activity.name,
                    activity_type=activity.type,
                    pyspark_code=f"# ERROR transpiling '{activity.name}': {exc}\npass",
                    error=str(exc),
                    is_deterministic=False,
                )

            result.activity_results.append(activity_result)

        logger.info(
            "Transpilation complete. Pending LLM expressions: %d",
            result.total_pending_expressions,
        )
        return result

    def _dispatch(self, activity: ActivityBase) -> ActivityTranspileResult:
        if isinstance(activity, CopyActivity):
            return _transpile_copy(activity)
        if isinstance(activity, ForEachActivity):
            return _transpile_foreach(activity)
        if isinstance(activity, IfConditionActivity):
            return _transpile_if_condition(activity)
        if activity.type in _SPECIFIC_HANDLERS:
            return _SPECIFIC_HANDLERS[activity.type](activity)  # type: ignore[arg-type]
        return _transpile_generic(activity)

    def _build_header(self, parsed: ParsedPipeline) -> str:
        param_lines = "\n".join(
            f'    "{k}": "{v.default_value}",'
            for k, v in parsed.pipeline.properties.parameters.items()
        )
        var_lines = "\n".join(
            f'    "{k}": "{v.default_value}",'
            for k, v in parsed.pipeline.properties.variables.items()
        )
        return f'''\
"""
Auto-generated PySpark translation of ADF Pipeline: {parsed.pipeline.name}
Description : {parsed.pipeline.properties.description or "N/A"}
Activities  : {parsed.activity_count}
Generated by: ADF-to-PySpark Transpiler v1.0

⚠ Sections marked with TODO require:
  (a) Linked-service credential injection, OR
  (b) Ollama LLM translation of ADF expressions.
"""

import logging
import time
from pyspark.sql import SparkSession, DataFrame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("{parsed.pipeline.name}")

spark = (
    SparkSession.builder
    .appName("{parsed.pipeline.name}")
    .getOrCreate()
)

# Pipeline parameters (override at runtime as needed)
pipeline_params: dict = {{
{param_lines}
}}

# Pipeline variables (mutable state)
pipeline_vars: dict = {{
{var_lines}
}}

logger.info("Pipeline '{parsed.pipeline.name}' starting.")
'''

    def _build_footer(self, parsed: ParsedPipeline) -> str:
        return f'''
# ---------------------------------------------------------------------------
logger.info("Pipeline '{parsed.pipeline.name}' completed successfully.")
spark.stop()
'''
