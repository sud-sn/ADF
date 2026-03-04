"""
dataflow_transpiler.py
======================
Deterministic PySpark code generator for ADF Mapping Data Flow definitions.

Converts the typed object graph produced by ``dataflow_parser.py`` into
production-ready PySpark code.  Complex DFS expressions (``iif``, ``isNull``,
nested function calls) are translated inline where possible; truly complex
ones are flagged for Ollama LLM translation.

Author: Transpiler Architect
"""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass, field

from .dataflow_parser import (
    DFSColumn,
    DFSParameter,
    DFSSink,
    DFSSource,
    DFSTransformType,
    DFSTransformation,
    ParsedDataFlow,
    SinkColumnMapping,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_var(name: str) -> str:
    """Convert a DFS stream name to a valid Python identifier."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()


def _indent(code: str, spaces: int = 4) -> str:
    return textwrap.indent(textwrap.dedent(code), " " * spaces)


# ---------------------------------------------------------------------------
# DFS expression → PySpark expression translator (deterministic)
# ---------------------------------------------------------------------------

_DFS_EXPR_RE = re.compile(
    r"\b(iif|isNull|trim|equals|toString|toInteger|toDecimal|toFloat|toLong|"
    r"toShort|toBoolean|toDate|toTimestamp|concat|length|upper|lower|"
    r"substring|replace|left|right|round|ceil|floor|abs|power|sqrt|"
    r"currentDate|currentTimestamp|year|month|dayOfMonth|hour|minute|second|"
    r"addDays|addMonths|daysBetween|monthsBetween|coalesce|negate|"
    r"like|rlike|in|between|split|regexReplace|regexExtract|"
    r"md5|sha2|crc32|ascii|encode|decode)\s*\("
)


def _has_complex_expression(expr: str) -> bool:
    """Return True if the expression contains DFS function calls that need translation."""
    return bool(_DFS_EXPR_RE.search(expr))


def _translate_simple_expr(expr: str) -> str:
    """
    Attempt deterministic DFS → PySpark expression translation for simple cases.

    Returns the translated expression, or the original if too complex.
    """
    result = expr.strip()

    # $paramName → dataflow_params["paramName"]
    result = re.sub(r"\$(\w+)", r'dataflow_params["\1"]', result)

    # true() / false() → lit(True) / lit(False)
    result = re.sub(r"\btrue\(\)", "lit(True)", result)
    result = re.sub(r"\bfalse\(\)", "lit(False)", result)

    # Simple function mappings (only for non-nested single calls)
    simple_mappings = {
        r"\btrim\((\w+)\)": r'trim(col("\1"))',
        r"\bisNull\((\w+)\)": r'col("\1").isNull()',
        r"\bupper\((\w+)\)": r'upper(col("\1"))',
        r"\blower\((\w+)\)": r'lower(col("\1"))',
        r"\blength\((\w+)\)": r'length(col("\1"))',
        r"\babs\((\w+)\)": r'abs(col("\1"))',
        r"\bceil\((\w+)\)": r'ceil(col("\1"))',
        r"\bfloor\((\w+)\)": r'floor(col("\1"))',
        r"\bround\((\w+)\)": r'round(col("\1"))',
        r"\bsqrt\((\w+)\)": r'sqrt(col("\1"))',
        r"\bcurrentDate\(\)": "current_date()",
        r"\bcurrentTimestamp\(\)": "current_timestamp()",
        r"\byear\((\w+)\)": r'year(col("\1"))',
        r"\bmonth\((\w+)\)": r'month(col("\1"))',
        r"\bdayOfMonth\((\w+)\)": r'dayofmonth(col("\1"))',
    }
    for pattern, replacement in simple_mappings.items():
        result = re.sub(pattern, replacement, result)

    return result


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DataFlowTranspileResult:
    """Full transpilation result for a data flow definition."""
    dataflow_name: str
    pyspark_code: str
    pending_llm_expressions: list[tuple[str, str]] = field(default_factory=list)
    source_count: int = 0
    sink_count: int = 0
    transformation_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def has_pending_llm_work(self) -> bool:
        return bool(self.pending_llm_expressions)

    @property
    def total_pending_expressions(self) -> int:
        return len(self.pending_llm_expressions)


# ---------------------------------------------------------------------------
# Per-transformation transpilers
# ---------------------------------------------------------------------------


def _transpile_source(source: DFSSource) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark read code for a DFS source."""
    var = _safe_var(source.name)
    pending: list[tuple[str, str]] = []

    # Determine read format and options
    if source.query:
        code = f'''\
# --- Source: {source.name} ---
# Dataset: {source.dataset_ref}
# Description: {source.description}
df_{var} = (
    spark.read
    .format("jdbc")
    .option("query", """{source.query}""")
    .option("isolationLevel", "{source.isolation_level}")
    # TODO: Configure JDBC connection: .option("url", ...), .option("user", ...), .option("password", ...)
    .load()
)'''
    else:
        code = f'''\
# --- Source: {source.name} ---
# Dataset: {source.dataset_ref}
# Description: {source.description}
df_{var} = (
    spark.read
    .format("parquet")  # TODO: adjust format based on dataset '{source.dataset_ref}'
    # TODO: Configure source path/connection
    .load()
)'''

    # Add schema enforcement if columns are defined
    if source.columns:
        schema_lines = ", ".join(
            f'StructField("{c.name}", {_dfs_to_spark_type(c.data_type)}, True)'
            for c in source.columns
        )
        code += f"""

# Enforce source schema
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DecimalType, TimestampType, LongType, DoubleType, BooleanType
_schema_{var} = StructType([{schema_lines}])
df_{var} = spark.createDataFrame(df_{var}.rdd, schema=_schema_{var})"""

    code += f'\nlogger.info("Source [{source.name}] loaded. Rows: %d", df_{var}.count())\n'
    return code, pending


def _transpile_derive(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark withColumn code for derived columns.

    Uses pure PySpark DSL (col, when, lit, trim, etc.) — never wraps
    PySpark functions inside expr() strings because expr() expects
    raw Spark SQL, not Python DSL code.
    """
    var = _safe_var(t.name)
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"
    pending: list[tuple[str, str]] = []

    lines = [
        f"# --- DerivedColumn: {t.name} ---",
    ]
    if t.description:
        lines.append(f"# {t.description}")

    lines.append(f"df_{var} = df_{input_var}")

    for col_name, dfs_expr in t.expressions.items():
        if _has_complex_expression(dfs_expr):
            # Complex expression → send to LLM for translation
            pending.append((f"{t.name}.{col_name}", dfs_expr))
            # Generate a clean placeholder — LLM will replace this
            lines.append(f'# DFS: {col_name} = {dfs_expr}')
            lines.append(f'df_{var} = df_{var}.withColumn("{col_name}", lit(None))  # __LLM_PENDING__: {t.name}.{col_name}')
        else:
            # Simple expression — translate to PySpark DSL directly
            pyspark_expr = _translate_to_dsl(dfs_expr)
            lines.append(f'df_{var} = df_{var}.withColumn("{col_name}", {pyspark_expr})')

    lines.append("")
    return "\n".join(lines), pending


def _translate_to_dsl(dfs_expr: str) -> str:
    """
    Translate a simple DFS expression to pure PySpark DSL code.

    Only handles non-nested cases. Complex/nested expressions should
    be routed to the LLM via _has_complex_expression() check.
    """
    expr_str = dfs_expr.strip()

    # $paramName → lit(str(dataflow_params.get("paramName", "")))
    if re.match(r"^\$(\w+)$", expr_str):
        param = re.match(r"^\$(\w+)$", expr_str).group(1)
        return f'lit(str(dataflow_params.get("{param}", "")))'

    # Simple column reference (bare identifier)
    if re.match(r"^\w+$", expr_str):
        return f'col("{expr_str}")'

    # String literal
    if re.match(r"^'.*'$", expr_str):
        return f"lit({expr_str})"

    # Numeric literal
    if re.match(r"^-?\d+(\.\d+)?$", expr_str):
        return f"lit({expr_str})"

    # true() / false()
    expr_str = re.sub(r"\btrue\(\)", "lit(True)", expr_str)
    expr_str = re.sub(r"\bfalse\(\)", "lit(False)", expr_str)

    # $paramName in compound expressions
    expr_str = re.sub(r"\$(\w+)", r'str(dataflow_params.get("\1", ""))', expr_str)

    # Apply simple function translations
    expr_str = _translate_simple_expr(expr_str)

    return expr_str



def _transpile_join(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark join code."""
    var = _safe_var(t.name)
    pending: list[tuple[str, str]] = []

    left_var = _safe_var(t.inputs[0]) if len(t.inputs) > 0 else "unknown_left"
    right_var = _safe_var(t.inputs[1]) if len(t.inputs) > 1 else "unknown_right"

    # Translate join condition
    condition = _translate_simple_expr(t.condition) if t.condition else "True"
    if _has_complex_expression(t.condition):
        pending.append((f"{t.name}.join_condition", t.condition))

    # Map DFS join types to PySpark
    join_type_map = {
        "left": "left",
        "right": "right",
        "inner": "inner",
        "full": "full",
        "cross": "cross",
        "leftanti": "left_anti",
        "leftsemi": "left_semi",
    }
    spark_join = join_type_map.get(t.join_type.lower().replace("'", ""), "left")

    code = f'''\
# --- Join: {t.name} ---
# Type: {t.join_type} | Match: {t.match_type}
# Condition: {t.condition}
df_{var} = df_{left_var}.join(
    df_{right_var},
    on=({condition}),
    how="{spark_join}",
)
logger.info("Join [{t.name}] complete. Rows: %d", df_{var}.count())
'''
    return code, pending


def _transpile_filter(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark filter code."""
    var = _safe_var(t.name)
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"
    pending: list[tuple[str, str]] = []

    condition = _translate_simple_expr(t.condition)
    if _has_complex_expression(t.condition):
        pending.append((f"{t.name}.filter_condition", t.condition))

    code = f'''\
# --- Filter: {t.name} ---
# Condition: {t.condition}
df_{var} = df_{input_var}.filter({repr(condition)})
logger.info("Filter [{t.name}] complete. Rows: %d", df_{var}.count())
'''
    return code, pending


def _transpile_aggregate(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark groupBy/agg code."""
    var = _safe_var(t.name)
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"
    pending: list[tuple[str, str]] = []

    group_cols = ", ".join(f'"{c}"' for c in t.group_by) if t.group_by else ""

    agg_exprs: list[str] = []
    for col_name, expr_str in t.expressions.items():
        if _has_complex_expression(expr_str):
            pending.append((f"{t.name}.{col_name}", expr_str))
            agg_exprs.append(f'    # DFS: {col_name} = {expr_str}')
            agg_exprs.append(f'    lit(None).alias("{col_name}")  # __LLM_PENDING__: {t.name}.{col_name}')
        else:
            translated = _translate_to_dsl(expr_str)
            agg_exprs.append(f'    {translated}.alias("{col_name}")')

    # Join the aggregation expressions. If we have multiple lines (from DFS comments), join with newline.
    if agg_exprs:
        # Some are comments, some are real code. Let's just join them nicely.
        # But wait, agg() takes positional arguments, so only the code lines need commas.
        # It's cleaner to list them and add commas to the non-comment lines.
        formatted_exprs = []
        for i, line in enumerate(agg_exprs):
            if line.strip().startswith("#"):
                formatted_exprs.append(line)
            else:
                # Is it the last code line?
                is_last = i == len(agg_exprs) - 1 or all(x.strip().startswith("#") for x in agg_exprs[i+1:])
                formatted_exprs.append(line + ("" if is_last else ","))
        agg_body = "\n".join(formatted_exprs)
    else:
        agg_body = '    lit(1).alias("_placeholder")'

    code = f'''\
# --- Aggregate: {t.name} ---
df_{var} = (
    df_{input_var}
    .groupBy({group_cols})
    .agg(
{agg_body}
    )
)
logger.info("Aggregate [{t.name}] complete. Rows: %d", df_{var}.count())
'''
    return code, pending


def _transpile_sort(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark orderBy code."""
    var = _safe_var(t.name)
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"

    order_exprs = []
    for col_name, direction in t.sort_orders:
        if direction == "desc":
            order_exprs.append(f'col("{col_name}").desc()')
        else:
            order_exprs.append(f'col("{col_name}").asc()')

    orders = ", ".join(order_exprs) if order_exprs else '"_unknown"'

    code = f'''\
# --- Sort: {t.name} ---
df_{var} = df_{input_var}.orderBy({orders})
'''
    return code, []


def _transpile_select(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark select/rename code."""
    var = _safe_var(t.name)
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"

    if t.expressions:
        rename_lines = []
        for sink_col, src_col in t.expressions.items():
            if sink_col != src_col:
                rename_lines.append(f'    .withColumnRenamed("{src_col}", "{sink_col}")')
            else:
                rename_lines.append(f'    # Keep: {sink_col}')
        renames = "\n".join(rename_lines)
        code = f'''\
# --- Select: {t.name} ---
df_{var} = (
    df_{input_var}
{renames}
)
'''
    else:
        code = f'''\
# --- Select: {t.name} ---
df_{var} = df_{input_var}  # TODO: specify column selection
'''
    return code, []


def _transpile_alter_row(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark alter row policy code."""
    var = _safe_var(t.name)
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"
    pending: list[tuple[str, str]] = []

    policy_map = {
        "upsertIf": "upsert",
        "insertIf": "insert",
        "deleteIf": "delete",
        "updateIf": "update",
    }
    policy_action = policy_map.get(t.alter_row_policy, "upsert")

    condition = _translate_simple_expr(t.alter_row_condition)
    if _has_complex_expression(t.alter_row_condition):
        pending.append((f"{t.name}.alter_row_condition", t.alter_row_condition))

    code = f'''\
# --- AlterRow: {t.name} ---
# Policy: {t.alter_row_policy}({t.alter_row_condition})
# Action: {policy_action} rows where condition is true
# PySpark does not have a direct alter row equivalent.
# The policy is applied at the sink level (upsert/insert/update/delete).
df_{var} = df_{input_var}  # Rows marked for: {policy_action}
_alter_row_policy_{_safe_var(t.name)} = "{policy_action}"
_alter_row_condition_{_safe_var(t.name)} = {repr(condition)}
'''
    return code, pending


def _transpile_union(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark union code."""
    var = _safe_var(t.name)
    input_vars = [f"df_{_safe_var(i)}" for i in t.inputs]

    if len(input_vars) >= 2:
        first = input_vars[0]
        unions = "\n".join(f"    .unionByName({v}, allowMissingColumns=True)" for v in input_vars[1:])
        code = f'''\
# --- Union: {t.name} ---
df_{var} = (
    {first}
{unions}
)
logger.info("Union [{t.name}] complete. Rows: %d", df_{var}.count())
'''
    else:
        code = f'''\
# --- Union: {t.name} ---
df_{var} = df_{_safe_var(t.inputs[0]) if t.inputs else "unknown"}  # Single input union
'''
    return code, []


def _transpile_exists(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark exists (semi/anti join) code."""
    var = _safe_var(t.name)
    pending: list[tuple[str, str]] = []

    left_var = _safe_var(t.inputs[0]) if len(t.inputs) > 0 else "unknown"
    right_var = _safe_var(t.inputs[1]) if len(t.inputs) > 1 else "unknown"

    # Check for negate option
    negate = "negate" in t.body and "true" in t.body.lower()
    join_how = "left_anti" if negate else "left_semi"

    condition = _translate_simple_expr(t.condition) if t.condition else "True"
    if t.condition and _has_complex_expression(t.condition):
        pending.append((f"{t.name}.exists_condition", t.condition))

    code = f'''\
# --- Exists: {t.name} ---
# {"NOT EXISTS (anti join)" if negate else "EXISTS (semi join)"}
df_{var} = df_{left_var}.join(df_{right_var}, on=({condition}), how="{join_how}")
'''
    return code, pending


def _transpile_lookup(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark lookup join code."""
    var = _safe_var(t.name)
    pending: list[tuple[str, str]] = []

    left_var = _safe_var(t.inputs[0]) if len(t.inputs) > 0 else "unknown"
    right_var = _safe_var(t.inputs[1]) if len(t.inputs) > 1 else "unknown"

    condition = _translate_simple_expr(t.condition) if t.condition else "True"
    if t.condition and _has_complex_expression(t.condition):
        pending.append((f"{t.name}.lookup_condition", t.condition))

    code = f'''\
# --- Lookup: {t.name} ---
df_{var} = df_{left_var}.join(df_{right_var}, on=({condition}), how="left")
logger.info("Lookup [{t.name}] complete. Rows: %d", df_{var}.count())
'''
    return code, pending


def _transpile_window(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark window function code."""
    var = _safe_var(t.name)
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"
    pending: list[tuple[str, str]] = []

    code = f'''\
# --- Window: {t.name} ---
from pyspark.sql.window import Window
# TODO: Configure window specification from DFS script
# _window_{var} = Window.partitionBy(...).orderBy(...)
df_{var} = df_{input_var}  # TODO: apply window functions
'''

    for col_name, expr_str in t.expressions.items():
        pending.append((f"{t.name}.{col_name}", expr_str))

    return code, pending


def _transpile_conditional_split(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark conditional split (multiple filter branches)."""
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"
    pending: list[tuple[str, str]] = []

    lines = [f"# --- ConditionalSplit: {t.name} ---"]

    for output_name, cond in t.split_conditions.items():
        out_var = _safe_var(output_name)
        translated = _translate_simple_expr(cond)
        if _has_complex_expression(cond):
            pending.append((f"{t.name}.{output_name}", cond))
        lines.append(f'df_{out_var} = df_{input_var}.filter({repr(translated)})')

    lines.append("")
    return "\n".join(lines), pending


def _transpile_surrogate_key(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark monotonically_increasing_id code."""
    var = _safe_var(t.name)
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"

    col_name = t.columns[0] if t.columns else "surrogate_key"
    code = f'''\
# --- SurrogateKey: {t.name} ---
from pyspark.sql.functions import monotonically_increasing_id
df_{var} = df_{input_var}.withColumn("{col_name}", monotonically_increasing_id())
'''
    return code, []


def _transpile_pivot(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark pivot code."""
    var = _safe_var(t.name)
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"
    pending: list[tuple[str, str]] = []

    if t.body:
        pending.append((f"{t.name}.pivot", t.body))

    code = f'''\
# --- Pivot: {t.name} ---
# TODO: Configure pivot from DFS: {t.body[:80]}...
df_{var} = df_{input_var}  # TODO: .groupBy(...).pivot(...).agg(...)
'''
    return code, pending


def _transpile_rank(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark rank code."""
    var = _safe_var(t.name)
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"

    col_name = t.columns[0] if t.columns else "rank"
    code = f'''\
# --- Rank: {t.name} ---
from pyspark.sql.window import Window
from pyspark.sql.functions import rank as spark_rank
# TODO: Configure window for rank
# _window_{var} = Window.orderBy(...)
# df_{var} = df_{input_var}.withColumn("{col_name}", spark_rank().over(_window_{var}))
df_{var} = df_{input_var}  # TODO: apply rank
'''
    return code, []


def _transpile_sink(sink: DFSSink) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark write code for a DFS sink."""
    var = _safe_var(sink.name)
    # Use the sink's input stream variable, not its own name
    input_var = _safe_var(sink.inputs[0]) if sink.inputs else var
    pending: list[tuple[str, str]] = []

    # Build column mapping select
    select_lines = ""
    if sink.column_mappings:
        select_exprs = []
        for mapping in sink.column_mappings:
            if mapping.sink_column != mapping.source_column:
                select_exprs.append(f'col("{mapping.source_column}").alias("{mapping.sink_column}")')
            else:
                select_exprs.append(f'col("{mapping.sink_column}")')
        select_str = ", ".join(select_exprs)
        select_lines = f"\n    .select({select_str})"

    # Partition configuration
    partition_line = ""
    if sink.partition_by and sink.partition_count:
        partition_line = f'\n    .repartition({sink.partition_count})  # partitionBy: {sink.partition_by}'

    # Error handling
    error_lines = ""
    if sink.output_rejected_data:
        error_lines = f"""
# Error handling: outputRejectedData=True
# Rejected data path: {sink.rejected_data_path or 'N/A'}"""
        if sink.rejected_data_path:
            pending.append((f"{sink.name}.rejected_data_path", sink.rejected_data_path))

    # Determine whether to use standard write or delta merge
    if sink.upsertable and sink.keys:
        keys_str = ", ".join(f'"{k}"' for k in sink.keys)
        # Using string interpolation to generate actual code, so use raw strings inside the f-string correctly
        # The condition will look like: f"target.{k} = source.{k}" and we join them with AND
        merge_conditions_str = " AND ".join(f"target.{k} = source.{k}" for k in sink.keys)

        write_code = f'''\
# Upsert mode: keys=[{keys_str}]
# Using Delta Lake merge for true upsert
from delta.tables import DeltaTable

# TODO: Configure target path for dataset '{sink.dataset_ref}'
_target_path_{var} = "<target_path>"

try:
    delta_table_{var} = DeltaTable.forPath(spark, _target_path_{var})
    (
        delta_table_{var}.alias("target").merge(
            (
                df_{input_var}{select_lines}{partition_line}
            ).alias("source"),
            "{merge_conditions_str}"
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
except Exception as e:
    logger.warning("Delta table merge failed (table might not exist yet). Falling back to overwrite. Error: %s", e)
    (
        df_{input_var}{select_lines}{partition_line}
        .write
        .format("delta")
        .mode("overwrite")
        .save(_target_path_{var})
    )'''
    else:
        # Standard append/overwrite
        write_mode = "overwrite"  # Default fallback
        write_code = f'''\
(
    df_{input_var}{select_lines}{partition_line}
    .write
    .format("{"delta" if sink.upsertable else "parquet"}")  # TODO: adjust format based on dataset '{sink.dataset_ref}'
    .mode("{write_mode}")
    # TODO: Configure sink path/connection for '{sink.dataset_ref}'
    .save()
)'''

    code = f'''\
# --- Sink: {sink.name} ---
# Dataset: {sink.dataset_ref}
# Description: {sink.description}
# Format: {sink.format} | Batch size: {sink.batch_size or "default"}{error_lines}
{write_code}
logger.info("Sink [{sink.name}] write complete.")
'''
    return code, pending



def _transpile_generic_transform(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Fallback transpiler for unhandled transformation types."""
    var = _safe_var(t.name)
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"
    pending: list[tuple[str, str]] = []

    if t.body:
        pending.append((f"{t.name}.body", t.body))

    code = f'''\
# --- {t.transform_type.value}: {t.name} ---
# No dedicated handler for transform type '{t.transform_type.value}'.
# Raw DFS: {t.raw_line[:100]}...
df_{var} = df_{input_var}  # TODO: implement {t.transform_type.value} logic
'''
    return code, pending


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------


def _dfs_to_spark_type(dfs_type: str) -> str:
    """Map a DFS data type to PySpark SQL type constructor."""
    t = dfs_type.strip().lower()
    if t == "string":
        return "StringType()"
    if t == "integer" or t == "int":
        return "IntegerType()"
    if t == "long":
        return "LongType()"
    if t == "double" or t == "float":
        return "DoubleType()"
    if t == "boolean":
        return "BooleanType()"
    if t == "timestamp":
        return "TimestampType()"
    if t == "date":
        return "DateType()"
    if t.startswith("decimal"):
        # decimal(p,s) → DecimalType(p,s)
        m = re.match(r"decimal\((\d+),\s*(\d+)\)", dfs_type, re.IGNORECASE)
        if m:
            return f"DecimalType({m.group(1)}, {m.group(2)})"
        return "DecimalType(38, 18)"
    if t == "short":
        return "ShortType()"
    if t == "byte":
        return "ByteType()"
    if t == "binary":
        return "BinaryType()"
    return "StringType()"  # default fallback


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------


_TRANSFORM_HANDLERS = {
    DFSTransformType.DERIVE:            _transpile_derive,
    DFSTransformType.JOIN:              _transpile_join,
    DFSTransformType.FILTER:            _transpile_filter,
    DFSTransformType.AGGREGATE:         _transpile_aggregate,
    DFSTransformType.SORT:              _transpile_sort,
    DFSTransformType.SELECT:            _transpile_select,
    DFSTransformType.ALTER_ROW:         _transpile_alter_row,
    DFSTransformType.UNION:             _transpile_union,
    DFSTransformType.EXISTS:            _transpile_exists,
    DFSTransformType.LOOKUP:            _transpile_lookup,
    DFSTransformType.WINDOW:            _transpile_window,
    DFSTransformType.CONDITIONAL_SPLIT: _transpile_conditional_split,
    DFSTransformType.SURROGATE_KEY:     _transpile_surrogate_key,
    DFSTransformType.PIVOT:             _transpile_pivot,
    DFSTransformType.RANK:              _transpile_rank,
}


# ---------------------------------------------------------------------------
# Main transpiler
# ---------------------------------------------------------------------------


class DataFlowTranspiler:
    """
    Converts a ``ParsedDataFlow`` into PySpark code.

    Usage
    -----
    ::

        parsed = parse_dataflow_json(raw_json)
        result = DataFlowTranspiler().transpile(parsed)
        print(result.pyspark_code)
    """

    def transpile(self, parsed: ParsedDataFlow) -> DataFlowTranspileResult:
        logger.info("Transpiling data flow '%s'.", parsed.name)

        all_pending: list[tuple[str, str]] = []
        warnings: list[str] = []
        code_parts: list[str] = []

        # ---- Header ----
        code_parts.append(self._build_header(parsed))

        # ---- Parameters ----
        if parsed.parameters:
            code_parts.append(self._build_params(parsed))

        # ---- Build name→object lookup ----
        source_map = {s.name: s for s in parsed.sources}
        sink_map = {s.name: s for s in parsed.sinks}
        transform_map = {t.name: t for t in parsed.transformations}

        # ---- Process in topological order ----
        for name in parsed.transformation_order:
            code_parts.append(f"\n# {'=' * 70}")

            if name in source_map:
                code, pending = _transpile_source(source_map[name])
                code_parts.append(code)
                all_pending.extend(pending)

            elif name in sink_map:
                code, pending = _transpile_sink(sink_map[name])
                code_parts.append(code)
                all_pending.extend(pending)

            elif name in transform_map:
                t = transform_map[name]
                handler = _TRANSFORM_HANDLERS.get(t.transform_type, _transpile_generic_transform)
                try:
                    code, pending = handler(t)
                    code_parts.append(code)
                    all_pending.extend(pending)
                except Exception as exc:
                    logger.error("Transpilation failed for '%s': %s", name, exc, exc_info=True)
                    code_parts.append(f"# ERROR transpiling '{name}': {exc}\npass\n")
                    warnings.append(f"Failed to transpile '{name}': {exc}")

            else:
                warnings.append(f"Unknown node '{name}' in execution order — skipped.")

        # ---- Footer ----
        code_parts.append(self._build_footer(parsed))

        result = DataFlowTranspileResult(
            dataflow_name=parsed.name,
            pyspark_code="\n".join(code_parts),
            pending_llm_expressions=all_pending,
            source_count=len(parsed.sources),
            sink_count=len(parsed.sinks),
            transformation_count=len(parsed.transformations),
            warnings=warnings,
        )

        logger.info(
            "Data Flow transpilation complete. Pending LLM: %d, Warnings: %d",
            result.total_pending_expressions,
            len(warnings),
        )
        return result

    def _build_header(self, parsed: ParsedDataFlow) -> str:
        return f'''\
"""
Auto-generated PySpark translation of ADF Mapping Data Flow: {parsed.name}
Folder      : {parsed.folder or "N/A"}
Sources     : {len(parsed.sources)}
Sinks       : {len(parsed.sinks)}
Transforms  : {len(parsed.transformations)}
Generated by: ADF-to-PySpark Transpiler v1.0 (Data Flow Mode)

Sections marked with TODO require manual configuration or Ollama LLM translation.
"""

import logging
import time
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, lit, expr, when, trim, upper, lower, length, concat,
    current_date, current_timestamp, year, month, dayofmonth,
    hour, minute, second, round, ceil, floor, abs, sqrt,
    coalesce, isnull,
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType,
    DoubleType, DecimalType, BooleanType, TimestampType, DateType,
    ShortType, ByteType, BinaryType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("{parsed.name}")

spark = (
    SparkSession.builder
    .appName("{parsed.name}")
    .getOrCreate()
)

logger.info("Data Flow '{parsed.name}' starting.")
'''

    def _build_params(self, parsed: ParsedDataFlow) -> str:
        param_lines = "\n".join(
            f'    "{p.name}": {repr(p.default_value) if p.default_value else "None"},  # type: {p.data_type}'
            for p in parsed.parameters
        )
        return f'''\

# Data Flow parameters (override at runtime)
dataflow_params: dict = {{
{param_lines}
}}
'''

    def _build_footer(self, parsed: ParsedDataFlow) -> str:
        return f'''
# ---------------------------------------------------------------------------
logger.info("Data Flow '{parsed.name}' completed successfully.")
spark.stop()
'''
