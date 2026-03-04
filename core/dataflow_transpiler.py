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
    """Convert a DFS stream name to a valid Python identifier.

    Strips whitespace first, replaces non-alnum chars with ``_``,
    collapses consecutive underscores, and strips leading/trailing ``_``
    to avoid unreferenceable names like ``df___insertrows``.
    """
    cleaned = name.strip()
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", cleaned).lower()
    cleaned = re.sub(r"_+", "_", cleaned)       # collapse __ → _
    cleaned = cleaned.strip("_")                  # strip leading/trailing _
    return cleaned or "unnamed"


def _indent(code: str, spaces: int = 4) -> str:
    return textwrap.indent(textwrap.dedent(code), " " * spaces)


# ---------------------------------------------------------------------------
# DFS expression → PySpark expression translator (deterministic)
#
# Handles ALL single-level DFS function calls deterministically.
# Only *nested* function calls (function inside another function's args)
# are sent to the LLM.  Examples:
#   trim(X)                   → deterministic (single-level)
#   equals(A, B)              → deterministic (single-level, two-arg)
#   toString(X)               → deterministic (single-level cast)
#   addDays(X, 5)             → deterministic (single-level, two-arg)
#   iif(A == '0', '1', '2')  → deterministic (single-level, no nested funcs)
#   iif(isNull(X), '0', X)   → LLM (nested: isNull inside iif)
# ---------------------------------------------------------------------------

# All known DFS function names (used for detection).
_DFS_FUNC_NAMES = frozenset({
    "iif", "isNull", "trim", "equals", "toString", "toInteger", "toDecimal",
    "toFloat", "toLong", "toShort", "toBoolean", "toDate", "toTimestamp",
    "concat", "length", "upper", "lower", "substring", "replace", "left",
    "right", "round", "ceil", "floor", "abs", "power", "sqrt",
    "currentDate", "currentTimestamp", "year", "month", "dayOfMonth",
    "hour", "minute", "second", "addDays", "addMonths", "daysBetween",
    "monthsBetween", "coalesce", "negate", "like", "rlike", "between",
    "split", "regexReplace", "regexExtract", "md5", "sha2", "crc32",
    "ascii", "encode", "decode", "byName", "byPosition", "not",
    "true", "false", "case",
})

_DFS_FUNC_RE = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in sorted(_DFS_FUNC_NAMES, key=len, reverse=True)) + r")\s*\("
)


def _find_matching_paren(expr: str, open_pos: int) -> int:
    """Return index of the closing ')' matching the '(' at *open_pos*, or -1."""
    depth = 0
    i = open_pos
    in_str = False
    str_ch = ""
    while i < len(expr):
        ch = expr[i]
        if in_str:
            if ch == str_ch:
                in_str = False
            i += 1
            continue
        if ch in ("'", '"'):
            in_str = True
            str_ch = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _has_complex_expression(expr: str) -> bool:
    """
    Return True only if expression has *nested* DFS function calls.

    Single-level calls like ``trim(X)``, ``equals(A, B)``, ``toString(Y)``
    return False — the deterministic translator handles those.

    Nested calls like ``iif(isNull(X), ...)`` return True → sent to LLM.
    """
    calls = list(_DFS_FUNC_RE.finditer(expr))
    if not calls:
        return False

    for match in calls:
        # Position of the opening '('
        paren_start = match.end() - 1
        paren_end = _find_matching_paren(expr, paren_start)
        if paren_end == -1:
            return True  # Unbalanced parens → treat as complex

        inner = expr[paren_start + 1:paren_end]
        # If the arguments themselves contain another DFS function call → nested → complex
        if _DFS_FUNC_RE.search(inner):
            return True

    return False  # Only single-level function calls → deterministic


# ---------------------------------------------------------------------------
# Arg splitting / function call extraction
# ---------------------------------------------------------------------------


def _split_args(inner: str) -> list[str]:
    """Split comma-separated function args, respecting nested parens and quotes."""
    args: list[str] = []
    depth = 0
    buf = ""
    in_str = False
    str_ch = ""

    for ch in inner:
        if in_str:
            buf += ch
            if ch == str_ch:
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            str_ch = ch
            buf += ch
        elif ch == "(":
            depth += 1
            buf += ch
        elif ch == ")":
            depth -= 1
            buf += ch
        elif ch == "," and depth == 0:
            args.append(buf.strip())
            buf = ""
        else:
            buf += ch

    if buf.strip():
        args.append(buf.strip())
    return args


def _try_extract_func(expr: str) -> tuple[str, list[str]] | None:
    """
    If *expr* is a single function call ``func(arg1, arg2, ...)``,
    return ``(func_name, [raw_arg_strings])``.  Otherwise return ``None``.
    """
    expr = expr.strip()
    m = re.match(r"^(\w+)\s*\(", expr)
    if not m:
        return None

    func_name = m.group(1)
    paren_start = m.end() - 1
    paren_end = _find_matching_paren(expr, paren_start)
    if paren_end == -1:
        return None
    # Must consume the entire expression (nothing after the closing paren)
    if expr[paren_end + 1:].strip():
        return None

    inner = expr[paren_start + 1:paren_end]
    if not inner.strip():
        return func_name, []
    return func_name, _split_args(inner)


# ---------------------------------------------------------------------------
# Deterministic single-level function → PySpark DSL
# ---------------------------------------------------------------------------


def _translate_func_call(func_name: str, raw_args: list[str]) -> str | None:
    """
    Translate a single-level DFS function call to PySpark DSL.

    Each raw arg is translated via ``_translate_to_dsl`` (wraps bare idents
    in ``col()``, literals in ``lit()``, params in ``dataflow_params[...]``).

    Returns the PySpark DSL string, or ``None`` if not handled.
    """
    # Translate each arg to PySpark DSL
    args = [_translate_to_dsl(a) for a in raw_args]
    n = len(args)

    # --- Null / Boolean / Comparison ---
    if func_name == "isNull" and n == 1:
        return f"{args[0]}.isNull()"
    if func_name == "equals" and n == 2:
        return f"({args[0]} == {args[1]})"
    if func_name == "not" and n == 1:
        return f"~({args[0]})"
    if func_name == "negate" and n == 1:
        return f"-({args[0]})"
    if func_name == "true" and n == 0:
        return "lit(True)"
    if func_name == "false" and n == 0:
        return "lit(False)"
    if func_name == "coalesce" and n >= 1:
        return f"coalesce({', '.join(args)})"
    if func_name == "between" and n == 3:
        return f"{args[0]}.between({args[1]}, {args[2]})"

    # --- iif (conditional — only when args are non-nested) ---
    if func_name == "iif" and n == 3:
        cond = _translate_join_condition(raw_args[0].strip())
        true_val = _translate_to_dsl(raw_args[1].strip())
        false_val = _translate_to_dsl(raw_args[2].strip())
        return f"when({cond}, {true_val}).otherwise({false_val})"

    # --- String functions ---
    if func_name == "trim" and n == 1:
        return f"trim({args[0]})"
    if func_name == "upper" and n == 1:
        return f"upper({args[0]})"
    if func_name == "lower" and n == 1:
        return f"lower({args[0]})"
    if func_name == "length" and n == 1:
        return f"length({args[0]})"
    if func_name == "concat" and n >= 1:
        return f"concat({', '.join(args)})"
    if func_name == "substring" and n >= 2:
        pos = raw_args[1].strip()
        ln = raw_args[2].strip() if n >= 3 else "1"
        return f"substring({args[0]}, {pos}, {ln})"
    if func_name == "replace" and n == 3:
        return f"regexp_replace({args[0]}, {args[1]}, {args[2]})"
    if func_name == "regexReplace" and n == 3:
        return f"regexp_replace({args[0]}, {args[1]}, {args[2]})"
    if func_name == "regexExtract" and n >= 2:
        idx = raw_args[2].strip() if n >= 3 else "0"
        return f"regexp_extract({args[0]}, {args[1]}, {idx})"
    if func_name == "left" and n == 2:
        return f"{args[0]}.substr(1, {raw_args[1].strip()})"
    if func_name == "right" and n == 2:
        ln = raw_args[1].strip()
        return f"{args[0]}.substr(length({args[0]}) - {ln} + 1, {ln})"
    if func_name == "like" and n == 2:
        return f"{args[0]}.like({args[1]})"
    if func_name == "rlike" and n == 2:
        return f"{args[0]}.rlike({args[1]})"
    if func_name == "split" and n == 2:
        return f"pyspark_split({args[0]}, {args[1]})"

    # --- Cast functions (PySpark type objects, never bare strings) ---
    if func_name == "toString" and n == 1:
        return f"{args[0]}.cast(StringType())"
    if func_name == "toInteger" and n == 1:
        return f"{args[0]}.cast(IntegerType())"
    if func_name == "toLong" and n == 1:
        return f"{args[0]}.cast(LongType())"
    if func_name == "toFloat" and n == 1:
        return f"{args[0]}.cast(FloatType())"
    if func_name == "toShort" and n == 1:
        return f"{args[0]}.cast(ShortType())"
    if func_name == "toBoolean" and n == 1:
        return f"{args[0]}.cast(BooleanType())"
    if func_name == "toDecimal":
        if n == 1:
            return f"{args[0]}.cast(DecimalType(18, 2))"
        if n == 3:
            p = raw_args[1].strip()
            s = raw_args[2].strip()
            return f"{args[0]}.cast(DecimalType({p}, {s}))"
    if func_name == "toDate":
        if n == 1:
            return f"{args[0]}.cast(DateType())"
        if n == 2:
            return f"to_date({args[0]}, {args[1]})"
    if func_name == "toTimestamp":
        if n == 1:
            return f"{args[0]}.cast(TimestampType())"
        if n == 2:
            return f"to_timestamp({args[0]}, {args[1]})"

    # --- Math functions ---
    if func_name == "abs" and n == 1:
        return f"abs({args[0]})"
    if func_name == "ceil" and n == 1:
        return f"ceil({args[0]})"
    if func_name == "floor" and n == 1:
        return f"floor({args[0]})"
    if func_name == "round" and n >= 1:
        if n == 2:
            return f"round({args[0]}, {raw_args[1].strip()})"
        return f"round({args[0]})"
    if func_name == "sqrt" and n == 1:
        return f"sqrt({args[0]})"
    if func_name == "power" and n == 2:
        return f"pow({args[0]}, {args[1]})"

    # --- Date/Time functions ---
    if func_name == "currentDate" and n == 0:
        return "current_date()"
    if func_name == "currentTimestamp" and n == 0:
        return "current_timestamp()"
    if func_name == "year" and n == 1:
        return f"year({args[0]})"
    if func_name == "month" and n == 1:
        return f"month({args[0]})"
    if func_name == "dayOfMonth" and n == 1:
        return f"dayofmonth({args[0]})"
    if func_name == "hour" and n == 1:
        return f"hour({args[0]})"
    if func_name == "minute" and n == 1:
        return f"minute({args[0]})"
    if func_name == "second" and n == 1:
        return f"second({args[0]})"
    if func_name == "addDays" and n == 2:
        return f"date_add({args[0]}, {raw_args[1].strip()})"
    if func_name == "addMonths" and n == 2:
        return f"add_months({args[0]}, {raw_args[1].strip()})"
    if func_name == "daysBetween" and n == 2:
        return f"datediff({args[1]}, {args[0]})"
    if func_name == "monthsBetween" and n == 2:
        return f"months_between({args[1]}, {args[0]})"

    # --- Hash / Encoding ---
    if func_name == "md5" and n == 1:
        return f"md5({args[0]})"
    if func_name == "sha2" and n == 2:
        return f"sha2({args[0]}, {raw_args[1].strip()})"
    if func_name == "crc32" and n == 1:
        return f"crc32({args[0]})"

    # --- Reference helpers ---
    if func_name == "byName" and n == 1:
        # byName("col") → col("col")
        raw = raw_args[0].strip().strip("'\"")
        return f'col("{raw}")'
    if func_name == "byPosition" and n == 1:
        return f"col({raw_args[0].strip()})"

    # Unknown function → not handled deterministically
    return None


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
        # Use repr() to safely escape any internal quotes in the SQL query
        safe_query = repr(source.query)
        code = f'''\
# --- Source: {source.name} ---
# Dataset: {source.dataset_ref}
# Description: {source.description}
df_{var} = (
    spark.read
    .format("jdbc")
    .option("query", {safe_query})
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
_schema_{var} = StructType([{schema_lines}])
df_{var} = spark.createDataFrame(df_{var}.rdd, schema=_schema_{var})"""

    code += f'\nlogger.info("Source [{source.name}] loaded. Rows: %d", df_{var}.count())\n'
    return code, pending


def _transpile_derive(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark withColumn code for derived columns.

    Uses pure PySpark DSL (col, when, lit, trim, etc.) — never wraps
    PySpark functions inside expr() strings because expr() expects
    raw Spark SQL, not Python DSL code.

    Chains all .withColumn() calls in a single expression to avoid
    repeated DataFrame re-assignment overhead.
    """
    var = _safe_var(t.name)
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"
    pending: list[tuple[str, str]] = []

    lines = [
        f"# --- DerivedColumn: {t.name} ---",
    ]
    if t.description:
        lines.append(f"# {t.description}")

    # Collect all withColumn calls for chaining
    with_col_calls: list[str] = []
    for col_name, dfs_expr in t.expressions.items():
        if _has_complex_expression(dfs_expr):
            # Complex expression → send to LLM for translation
            pending.append((f"{t.name}.{col_name}", dfs_expr))
            with_col_calls.append(f'    # DFS: {col_name} = {dfs_expr}')
            with_col_calls.append(f'    .withColumn("{col_name}", lit(None))  # __LLM_PENDING__: {t.name}.{col_name}')
        else:
            # Simple expression — translate to PySpark DSL directly
            pyspark_expr = _translate_to_dsl(dfs_expr)
            with_col_calls.append(f'    .withColumn("{col_name}", {pyspark_expr})')

    if with_col_calls:
        chained = "\n".join(with_col_calls)
        lines.append(f"df_{var} = (")
        lines.append(f"    df_{input_var}")
        lines.append(chained)
        lines.append(")")
    else:
        lines.append(f"df_{var} = df_{input_var}")

    lines.append("")
    return "\n".join(lines), pending


def _wrap_identifier(token: str) -> str:
    """Wrap a bare identifier in col(), leaving literals and already-wrapped tokens alone."""
    token = token.strip()
    if not token:
        return token
    # Already wrapped in col() or a function call
    if token.startswith("col(") or token.startswith("lit(") or "(" in token:
        return token
    # String literal
    if re.match(r"^'.*'$", token) or re.match(r'^".*"$', token):
        return f"lit({token})"
    # Numeric literal (must check BEFORE bare identifier — digits match \w+)
    if re.match(r"^-?\d+(\.\d+)?$", token):
        return f"lit({token})"
    # Boolean keywords
    if token in ("True", "False", "None"):
        return f"lit({token})"
    # Stream.Column dotted reference (e.g. ForTrim.CUNO) → col("CUNO")
    dot_m = re.match(r"^(\w+)\.(\w+)$", token)
    if dot_m:
        return f'col("{dot_m.group(2)}")'
    # Bare identifier (but NOT pure digits)
    if re.match(r"^\w+$", token) and not re.match(r"^\d+$", token):
        return f'col("{token}")'
    return token


def _translate_comparison(clause: str) -> str:
    """
    Translate a single comparison clause like ``A == B`` or ``trim(X) == Y``
    into PySpark DSL, wrapping bare identifiers in col().
    """
    clause = clause.strip()
    # Match: left OP right  (where OP is ==, !=, >=, <=, >, <)
    m = re.match(r"^(.+?)\s*(==|!=|>=|<=|>|<)\s*(.+)$", clause)
    if m:
        left = _translate_to_dsl(m.group(1).strip())
        op = m.group(2)
        right = _translate_to_dsl(m.group(3).strip())
        return f"({left} {op} {right})"
    # No comparison operator — just translate the expression
    return _translate_to_dsl(clause)


def _translate_join_condition(raw_condition: str) -> str:
    """
    Translate a full DFS join condition into PySpark DSL.

    Handles ``&&`` (AND) and ``||`` (OR) operators, converting them to
    the PySpark bitwise ``&`` and ``|`` operators with proper parentheses.
    Example: ``A == B && C == D`` → ``(col("A") == col("B")) & (col("C") == col("D"))``
    """
    cond = raw_condition.strip()
    if not cond:
        return "True"

    # Split by || first (lower precedence), then && (higher precedence)
    or_parts = re.split(r"\s*\|\|\s*", cond)
    translated_or: list[str] = []
    for or_part in or_parts:
        and_parts = re.split(r"\s*&&\s*", or_part)
        translated_and = [_translate_comparison(ap) for ap in and_parts]
        if len(translated_and) > 1:
            translated_or.append("(" + " & ".join(translated_and) + ")")
        else:
            translated_or.append(translated_and[0])

    if len(translated_or) > 1:
        return " | ".join(translated_or)
    return translated_or[0]


def _translate_math_expr(expr_str: str) -> str:
    """
    Translate a DFS math expression (e.g. ``(NEPR/COFS)*ORQT``) into PySpark DSL,
    wrapping bare identifiers in ``col()``.
    """
    # Tokenise around math operators and parens, preserving them
    tokens = re.split(r"(\s*[+\-*/()]+\s*)", expr_str)
    result_parts: list[str] = []
    for token in tokens:
        stripped = token.strip()
        if not stripped or re.match(r"^[+\-*/()]+$", stripped):
            result_parts.append(token)  # keep operator/paren with spacing
        else:
            result_parts.append(_wrap_identifier(stripped))
    return "".join(result_parts)


def _translate_to_dsl(dfs_expr: str) -> str:
    """
    Translate a DFS expression to pure PySpark DSL code.

    Handles:
    - Bare identifiers → col("...")
    - String / numeric / boolean literals → lit(...)
    - Parameter references ($name) → lit(str(dataflow_params.get(...)))
    - Single-level DFS function calls → PySpark equivalents
    - Math expressions → col("A") / col("B") * col("C")
    - Boolean operators && / || → & / | with parenthesised clauses
    """
    expr_str = dfs_expr.strip()

    # $paramName → lit(str(dataflow_params.get("paramName", "")))
    if re.match(r"^\$(\w+)$", expr_str):
        param = re.match(r"^\$(\w+)$", expr_str).group(1)
        return f'lit(str(dataflow_params.get("{param}", "")))'

    # String literal
    if re.match(r"^'.*'$", expr_str) or re.match(r'^".*"$', expr_str):
        return f"lit({expr_str})"

    # Numeric literal (must check BEFORE bare identifier — digits match \w+)
    if re.match(r"^-?\d+(\.\d+)?$", expr_str):
        return f"lit({expr_str})"

    # Stream.Column dotted reference (e.g. ForTrim.CUNO) → col("CUNO")
    # In ADF DFS, "StreamName.ColumnName" means column from a specific stream.
    # In PySpark we just reference the column directly.
    dot_m = re.match(r"^(\w+)\.(\w+)$", expr_str)
    if dot_m:
        col_name = dot_m.group(2)
        return f'col("{col_name}")'

    # Simple column reference (bare identifier — but NOT pure digits)
    if re.match(r"^\w+$", expr_str) and not re.match(r"^\d+$", expr_str):
        return f'col("{expr_str}")'

    # Try to translate as a single function call (covers all ~50 DFS functions)
    func_match = _try_extract_func(expr_str)
    if func_match:
        func_name, raw_args = func_match
        result = _translate_func_call(func_name, raw_args)
        if result is not None:
            return result

    # true() / false() in compound expressions
    expr_str = re.sub(r"\btrue\(\)", "lit(True)", expr_str)
    expr_str = re.sub(r"\bfalse\(\)", "lit(False)", expr_str)

    # $paramName in compound expressions → lit(str(dataflow_params.get("paramName", "")))
    expr_str = re.sub(r"\$(\w+)", r'lit(str(dataflow_params.get("\1", "")))', expr_str)

    # Handle math expressions (contains +, -, *, /)
    if re.search(r"[+\-*/]", expr_str) and not expr_str.startswith("col("):
        expr_str = _translate_math_expr(expr_str)

    return expr_str



def _transpile_join(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark join code with proper col() wrapping and & operators."""
    var = _safe_var(t.name)
    pending: list[tuple[str, str]] = []

    left_var = _safe_var(t.inputs[0]) if len(t.inputs) > 0 else "unknown_left"
    right_var = _safe_var(t.inputs[1]) if len(t.inputs) > 1 else "unknown_right"

    # Translate join condition using the new robust translator
    if t.condition and _has_complex_expression(t.condition):
        pending.append((f"{t.name}.join_condition", t.condition))
        condition = _translate_join_condition(t.condition)
    elif t.condition:
        condition = _translate_join_condition(t.condition)
    else:
        condition = "True"

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
# Condition (DFS): {t.condition}
# broadcast() wraps the right (smaller/dimension) table to avoid shuffle.
# Remove broadcast() if the right table is large (>100MB).
df_{var} = df_{left_var}.join(
    broadcast(df_{right_var}),
    on=({condition}),
    how="{spark_join}",
)
logger.info("Join [{t.name}] complete. Rows: %d", df_{var}.count())
'''
    return code, pending


def _transpile_filter(t: DFSTransformation) -> tuple[str, list[tuple[str, str]]]:
    """Generate PySpark filter code using pure DSL — never SQL strings."""
    var = _safe_var(t.name)
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"
    pending: list[tuple[str, str]] = []

    if _has_complex_expression(t.condition):
        # Complex → send to LLM; emit a placeholder that is valid PySpark
        pending.append((f"{t.name}.filter_condition", t.condition))
        condition_code = f"lit(True)  # __LLM_PENDING__: {t.name}.filter_condition"
    else:
        # Deterministic translation → pure PySpark DSL expression
        condition_code = _translate_join_condition(t.condition)

    code = f'''\
# --- Filter: {t.name} ---
# DFS condition: {t.condition}
df_{var} = df_{input_var}.filter({condition_code})
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
    """Generate PySpark alter row policy code using pure DSL."""
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

    if _has_complex_expression(t.alter_row_condition):
        pending.append((f"{t.name}.alter_row_condition", t.alter_row_condition))
        condition_code = f"lit(True)  # __LLM_PENDING__: {t.name}.alter_row_condition"
    else:
        condition_code = _translate_join_condition(t.alter_row_condition)

    code = f'''\
# --- AlterRow: {t.name} ---
# Policy: {t.alter_row_policy}({t.alter_row_condition})
# Action: {policy_action} rows where condition is true
# PySpark does not have a direct alter row equivalent.
# The policy is applied at the sink level (upsert/insert/update/delete).
df_{var} = df_{input_var}  # Rows marked for: {policy_action}
_alter_row_policy_{_safe_var(t.name)} = "{policy_action}"
_alter_row_condition_{_safe_var(t.name)} = {condition_code}
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

    if t.condition and _has_complex_expression(t.condition):
        pending.append((f"{t.name}.exists_condition", t.condition))
        condition = _translate_join_condition(t.condition)
    elif t.condition:
        condition = _translate_join_condition(t.condition)
    else:
        condition = "True"

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

    if t.condition and _has_complex_expression(t.condition):
        pending.append((f"{t.name}.lookup_condition", t.condition))
        condition = _translate_join_condition(t.condition)
    elif t.condition:
        condition = _translate_join_condition(t.condition)
    else:
        condition = "True"

    code = f'''\
# --- Lookup: {t.name} ---
# TIP: Consider broadcast() on the lookup DataFrame for performance.
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
    """Generate PySpark conditional split (multiple filter branches).

    Each output stream from the split becomes its own filtered DataFrame.
    Uses pure PySpark DSL conditions — never SQL expression strings.
    """
    input_var = _safe_var(t.inputs[0]) if t.inputs else "unknown"
    pending: list[tuple[str, str]] = []

    lines = [f"# --- ConditionalSplit: {t.name} ---"]

    for output_name, cond in t.split_conditions.items():
        out_var = _safe_var(output_name)
        if _has_complex_expression(cond):
            pending.append((f"{t.name}.{output_name}", cond))
            translated = f"lit(True)  # __LLM_PENDING__: {t.name}.{output_name}"
        else:
            translated = _translate_join_condition(cond)
        lines.append(f'# DFS condition: {cond}')
        lines.append(f'df_{out_var} = df_{input_var}.filter({translated})')
        lines.append(f'logger.info("ConditionalSplit [{t.name}] -> {output_name}: %d rows", df_{out_var}.count())')

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
    if t == "double":
        return "DoubleType()"
    if t == "float":
        return "FloatType()"
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

        # Collect secondary split output names — these are virtual topo-sort
        # nodes whose DataFrames are created by the parent split transpiler.
        split_output_names: set[str] = set()
        for t in parsed.transformations:
            if t.transform_type.value == "split" and t.split_conditions:
                for out_name in t.split_conditions:
                    if out_name != t.name:
                        split_output_names.add(out_name)

        # ---- Process in topological order ----
        for name in parsed.transformation_order:
            # Skip virtual split output nodes — their DataFrames are emitted
            # by the parent conditional split handler.
            if name in split_output_names:
                continue

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
    col, lit, when, trim, upper, lower, length, concat,
    current_date, current_timestamp, year, month, dayofmonth,
    hour, minute, second, round, ceil, floor, abs, sqrt, pow,
    coalesce, isnull, date_add, add_months, datediff,
    months_between, to_date, to_timestamp, substring,
    regexp_replace, regexp_extract, md5, sha2,
    split as pyspark_split, monotonically_increasing_id,
    broadcast,
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType,
    DoubleType, FloatType, DecimalType, BooleanType, TimestampType,
    DateType, ShortType, ByteType, BinaryType,
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
