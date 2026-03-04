"""
dataflow_parser.py
==================
Parser for ADF Mapping Data Flow definition JSON files.

Handles the Data Flow Script (DFS) embedded in ``scriptLines``, converting it
to a strongly-typed Python object graph that the ``dataflow_transpiler`` can
consume.

Key responsibilities:
    - Validate and extract Data Flow definition JSON (sources, sinks, transformations)
    - Parse DFS scriptLines into a typed transformation DAG
    - Discriminate between Pipeline JSON and Data Flow JSON via ``detect_json_type()``

Author: Transpiler Architect
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON type detection
# ---------------------------------------------------------------------------


def detect_json_type(raw_json: str) -> str:
    """
    Auto-detect whether the JSON is an ADF Pipeline or a Data Flow definition.

    Returns
    -------
    ``"pipeline"`` or ``"dataflow"``

    Raises
    ------
    ValueError
        If the JSON cannot be parsed or the document type cannot be determined.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object.")

    props = data.get("properties", {})

    # Data Flow: properties.type == "MappingDataFlow" and has scriptLines/sources
    if props.get("type") == "MappingDataFlow":
        return "dataflow"

    tp = props.get("typeProperties", {})
    if isinstance(tp, dict) and ("scriptLines" in tp or "sources" in tp):
        return "dataflow"

    # Pipeline: has properties.activities array
    if "activities" in props and isinstance(props["activities"], list):
        return "pipeline"

    # Fallback heuristics
    if "type" in data and "factories/pipelines" in data.get("type", ""):
        return "pipeline"

    raise ValueError(
        "Cannot determine JSON type. Expected either a Pipeline JSON "
        "(with properties.activities) or a Data Flow JSON "
        "(with properties.typeProperties.scriptLines)."
    )


# ---------------------------------------------------------------------------
# DFS transformation types
# ---------------------------------------------------------------------------


class DFSTransformType(str, Enum):
    SOURCE = "source"
    SINK = "sink"
    DERIVE = "derive"
    JOIN = "join"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    SELECT = "select"
    SORT = "sort"
    ALTER_ROW = "alterRow"
    UNION = "union"
    EXISTS = "exists"
    LOOKUP = "lookup"
    WINDOW = "window"
    CONDITIONAL_SPLIT = "split"
    PIVOT = "pivot"
    UNPIVOT = "unpivot"
    FLATTEN = "foldDown"
    SURROGATE_KEY = "keyGenerate"
    RANK = "rank"
    STRINGIFY = "stringify"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class DFSParameter:
    """Data flow parameter from the ``parameters{}`` block."""
    name: str
    data_type: str
    default_value: str | None = None


@dataclass
class DFSColumn:
    """Column definition from output/input schema blocks."""
    name: str
    data_type: str


@dataclass
class DFSSource:
    """Parsed source from both JSON metadata and DFS script."""
    name: str
    dataset_ref: str = ""
    description: str = ""
    columns: list[DFSColumn] = field(default_factory=list)
    query: str | None = None
    format: str | None = None
    allow_schema_drift: bool = True
    validate_schema: bool = False
    isolation_level: str = "READ_UNCOMMITTED"
    raw_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class SinkColumnMapping:
    """Column mapping in a sink's ``mapColumn()`` block."""
    sink_column: str
    source_column: str


@dataclass
class DFSSink:
    """Parsed sink from both JSON metadata and DFS script."""
    name: str
    dataset_ref: str = ""
    description: str = ""
    inputs: list[str] = field(default_factory=list)
    input_columns: list[DFSColumn] = field(default_factory=list)
    column_mappings: list[SinkColumnMapping] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    format: str = "table"
    upsertable: bool = False
    insertable: bool = False
    updateable: bool = False
    deletable: bool = False
    batch_size: int | None = None
    partition_by: str | None = None
    partition_count: int | None = None
    error_handling: str | None = None
    output_rejected_data: bool = False
    rejected_data_path: str | None = None
    raw_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class DFSTransformation:
    """A single transformation step in the DFS DAG."""
    name: str
    transform_type: DFSTransformType
    inputs: list[str] = field(default_factory=list)
    description: str = ""
    body: str = ""           # raw DFS body (everything inside the parens)
    raw_line: str = ""       # original full DFS statement

    # Type-specific parsed fields
    expressions: dict[str, str] = field(default_factory=dict)  # col → expr
    condition: str = ""
    join_type: str = ""
    match_type: str = ""
    columns: list[str] = field(default_factory=list)
    sort_orders: list[tuple[str, str]] = field(default_factory=list)  # (col, asc/desc)
    group_by: list[str] = field(default_factory=list)
    alter_row_policy: str = ""  # upsertIf, insertIf, deleteIf, updateIf
    alter_row_condition: str = ""
    split_conditions: dict[str, str] = field(default_factory=dict)  # output → condition


@dataclass
class ParsedDataFlow:
    """
    Top-level result from ``parse_dataflow_json()``.
    Consumers depend on this interface, not on internal models.
    """
    name: str
    folder: str = ""
    parameters: list[DFSParameter] = field(default_factory=list)
    sources: list[DFSSource] = field(default_factory=list)
    sinks: list[DFSSink] = field(default_factory=list)
    transformations: list[DFSTransformation] = field(default_factory=list)
    script: str = ""              # full joined script
    transformation_order: list[str] = field(default_factory=list)  # topological
    raw_json: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DFS script tokeniser / statement splitter
# ---------------------------------------------------------------------------


def _join_script_lines(lines: list[str]) -> str:
    """Join scriptLines array into a single DFS script string."""
    return "\n".join(lines)


def _split_statements(script: str) -> list[str]:
    """
    Split a DFS script into individual transformation statements.

    Each statement ends with ``~> OutputName``.  We split on this pattern
    while keeping the arrow + name attached to the preceding statement.
    """
    # Remove the parameters block first (handled separately)
    cleaned = re.sub(r"parameters\s*\{[^}]*\}", "", script, flags=re.DOTALL).strip()

    # Split on the ~> pattern — each statement ends with ~> name
    parts = re.split(r"(~>\s*\w+)", cleaned)

    statements: list[str] = []
    i = 0
    while i < len(parts):
        stmt = parts[i].strip()
        # Accumulate all ~> parts that follow (for multi-output like split: ~> out1, out2)
        while i + 1 < len(parts) and parts[i + 1].strip().startswith("~>"):
            stmt = stmt + " " + parts[i + 1].strip()
            i += 2
            # If next part is a comma-continued name list (no keyword), append it
            if i < len(parts) and not re.search(r"\b(source|sink|derive|join|filter|aggregate|select|sort|alterRow|union|exists|lookup|window|split|pivot|unpivot|foldDown|keyGenerate|rank|stringify)\s*\(", parts[i]):
                # Could be continuation of output names
                break
            break
        else:
            i += 1
        if stmt and "~>" in stmt:
            statements.append(stmt)

    return statements


# ---------------------------------------------------------------------------
# DFS parameter parser
# ---------------------------------------------------------------------------


def _parse_parameters(script: str) -> list[DFSParameter]:
    """Extract parameters from the ``parameters{...}`` block."""
    m = re.search(r"parameters\s*\{([^}]*)\}", script, re.DOTALL)
    if not m:
        return []

    params: list[DFSParameter] = []
    for line in m.group(1).strip().split("\n"):
        line = line.strip().rstrip(",")
        if not line:
            continue
        # Pattern: name as type [= default]
        pm = re.match(r"(\w+)\s+as\s+(\w+)(?:\s*=\s*(.+))?", line)
        if pm:
            params.append(DFSParameter(
                name=pm.group(1),
                data_type=pm.group(2),
                default_value=pm.group(3).strip("'\" ") if pm.group(3) else None,
            ))
    return params


# ---------------------------------------------------------------------------
# DFS statement parser
# ---------------------------------------------------------------------------


_TRANSFORM_KEYWORDS: dict[str, DFSTransformType] = {
    "source":       DFSTransformType.SOURCE,
    "sink":         DFSTransformType.SINK,
    "derive":       DFSTransformType.DERIVE,
    "join":         DFSTransformType.JOIN,
    "filter":       DFSTransformType.FILTER,
    "aggregate":    DFSTransformType.AGGREGATE,
    "select":       DFSTransformType.SELECT,
    "sort":         DFSTransformType.SORT,
    "alterRow":     DFSTransformType.ALTER_ROW,
    "union":        DFSTransformType.UNION,
    "exists":       DFSTransformType.EXISTS,
    "lookup":       DFSTransformType.LOOKUP,
    "window":       DFSTransformType.WINDOW,
    "split":        DFSTransformType.CONDITIONAL_SPLIT,
    "pivot":        DFSTransformType.PIVOT,
    "unpivot":      DFSTransformType.UNPIVOT,
    "foldDown":     DFSTransformType.FLATTEN,
    "keyGenerate":  DFSTransformType.SURROGATE_KEY,
    "rank":         DFSTransformType.RANK,
    "stringify":    DFSTransformType.STRINGIFY,
}


def _extract_output_name(stmt: str) -> str:
    """Extract the output stream name from ``... ~> OutputName`` or ``... ~> Name1, Name2``.

    For multi-output statements (like conditional split), returns the first name.
    Use ``_extract_output_names()`` to get all output names.
    """
    names = _extract_output_names(stmt)
    return names[0] if names else ""


def _extract_output_names(stmt: str) -> list[str]:
    """Extract all output stream names from ``... ~> Name1, Name2, ...``."""
    m = re.search(r"~>\s*(.+?)\s*$", stmt)
    if not m:
        return []
    raw = m.group(1)
    return [n.strip() for n in raw.split(",") if n.strip()]


def _extract_body(stmt: str, keyword: str) -> str:
    """
    Extract the body (contents within outermost parentheses) of a
    transformation call.  Handles nested parentheses.
    """
    # Find the keyword followed by (
    pattern = rf"\b{re.escape(keyword)}\s*\("
    m = re.search(pattern, stmt)
    if not m:
        return ""

    start = m.end() - 1  # position of opening (
    depth = 0
    i = start
    while i < len(stmt):
        if stmt[i] == "(":
            depth += 1
        elif stmt[i] == ")":
            depth -= 1
            if depth == 0:
                return stmt[start + 1:i]
        i += 1
    # If unbalanced, return everything after the opening paren
    return stmt[start + 1:]


def _extract_inputs(stmt: str, keyword: str) -> list[str]:
    """
    Extract input stream name(s) that precede the transformation keyword.

    Pattern: ``InputA, InputB keyword(...)`` → [InputA, InputB]
    Source statements have no inputs.
    """
    pattern = rf"^(.*?)\b{re.escape(keyword)}\s*\("
    m = re.match(pattern, stmt, re.DOTALL)
    if not m:
        return []
    prefix = m.group(1).strip()
    if not prefix:
        return []
    return [s.strip() for s in prefix.split(",") if s.strip()]


def _parse_output_columns(body: str) -> list[DFSColumn]:
    """Parse ``output(col as type, ...)`` blocks using depth-aware matching."""
    # Find the start of output( using regex
    m = re.search(r"output\s*\(", body)
    if not m:
        return []

    # Depth-aware extraction: find the matching closing paren
    start = m.end() - 1  # position of opening (
    depth = 0
    i = start
    while i < len(body):
        if body[i] == "(":
            depth += 1
        elif body[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    inner = body[start + 1:i]

    cols: list[DFSColumn] = []
    # Split by comma at depth 0 (respecting nested parens in types)
    buf = ""
    d = 0
    for ch in inner:
        if ch == "(":
            d += 1
            buf += ch
        elif ch == ")":
            d -= 1
            buf += ch
        elif ch == "," and d == 0:
            line = buf.strip()
            cm = re.match(r"(\w+)\s+as\s+([\w()., ]+)", line)
            if cm:
                cols.append(DFSColumn(name=cm.group(1), data_type=cm.group(2).strip()))
            buf = ""
        else:
            buf += ch
    # Don't forget the last item
    if buf.strip():
        line = buf.strip()
        cm = re.match(r"(\w+)\s+as\s+([\w()., ]+)", line)
        if cm:
            cols.append(DFSColumn(name=cm.group(1), data_type=cm.group(2).strip()))
    return cols


def _parse_key_value_options(body: str) -> dict[str, str]:
    """
    Extract key: value pairs from the body, handling quoted string values
    and boolean/numeric values.
    """
    opts: dict[str, str] = {}
    # Match patterns like: key: 'value' or key: value or key: true
    for m in re.finditer(r"(\w+)\s*:\s*(?:'([^']*)'|\"([^\"]*)\"|(\w+))", body):
        key = m.group(1)
        val = m.group(2) or m.group(3) or m.group(4) or ""
        opts[key] = val
    return opts


def _parse_map_column(body: str) -> list[SinkColumnMapping]:
    """Parse ``mapColumn(sink_col = source_col, ...)`` blocks."""
    m = re.search(r"mapColumn\s*\((.*?)\)", body, re.DOTALL)
    if not m:
        return []
    mappings: list[SinkColumnMapping] = []
    for line in m.group(1).split(","):
        line = line.strip()
        if "=" in line:
            parts = line.split("=", 1)
            mappings.append(SinkColumnMapping(
                sink_column=parts[0].strip(),
                source_column=parts[1].strip(),
            ))
        elif line:
            # Same name mapping
            mappings.append(SinkColumnMapping(
                sink_column=line.strip(),
                source_column=line.strip(),
            ))
    return mappings


def _parse_keys(body: str) -> list[str]:
    """Parse ``keys:['col1','col2']`` from sink body."""
    m = re.search(r"keys\s*:\s*\[(.*?)\]", body)
    if not m:
        return []
    return [k.strip().strip("'\"") for k in m.group(1).split(",") if k.strip()]


def _parse_partition_by(body: str) -> tuple[str | None, int | None]:
    """Parse ``partitionBy('roundRobin', 30)`` from body."""
    m = re.search(r"partitionBy\s*\(\s*'(\w+)'\s*,?\s*(\d+)?\s*\)", body)
    if not m:
        return None, None
    return m.group(1), int(m.group(2)) if m.group(2) else None


def _parse_derive_expressions(body: str) -> dict[str, str]:
    """
    Parse derived column expressions: ``col = expr, col2 = expr2``.

    Handles nested function calls in expressions by tracking paren depth.
    """
    exprs: dict[str, str] = {}
    # Remove leading/trailing whitespace
    body = body.strip()
    if not body:
        return exprs

    # Split by comma at top-level (depth 0) — respecting nested parens
    current_key = ""
    current_val = ""
    depth = 0
    in_string = False
    string_char = ""
    parts: list[str] = []
    buf = ""

    for ch in body:
        if in_string:
            buf += ch
            if ch == string_char:
                in_string = False
            continue

        if ch in ("'", '"'):
            in_string = True
            string_char = ch
            buf += ch
        elif ch == "(":
            depth += 1
            buf += ch
        elif ch == ")":
            depth -= 1
            buf += ch
        elif ch == "," and depth == 0:
            parts.append(buf.strip())
            buf = ""
        else:
            buf += ch

    if buf.strip():
        parts.append(buf.strip())

    for part in parts:
        # Match: column_name = expression
        m = re.match(r"(\w+)\s*=\s*(.+)", part.strip(), re.DOTALL)
        if m:
            exprs[m.group(1).strip()] = m.group(2).strip()

    return exprs


def _parse_join_details(body: str) -> tuple[str, str, str]:
    """
    Parse join body for condition, joinType, matchType.

    Returns (condition, joinType, matchType).
    """
    opts = _parse_key_value_options(body)
    join_type = opts.get("joinType", "inner")
    match_type = opts.get("matchType", "exact")

    # The condition is everything before the first comma-separated option
    # e.g. "trim(DIVI) == trim(PFT_CTR_CD),\n     joinType:'left'..."
    condition = ""
    # Find the first key: value pair and take everything before it
    first_opt_match = re.search(r",\s*\w+\s*:", body)
    if first_opt_match:
        condition = body[:first_opt_match.start()].strip()
    else:
        condition = body.strip()

    return condition, join_type, match_type


def _parse_alter_row(body: str) -> tuple[str, str]:
    """
    Parse alterRow body for policy and condition.

    Returns (policy, condition). Policy is one of: upsertIf, insertIf, deleteIf, updateIf.
    """
    for policy in ("upsertIf", "insertIf", "deleteIf", "updateIf"):
        m = re.search(rf"{policy}\s*\((.+)\)", body, re.DOTALL)
        if m:
            return policy, m.group(1).strip()
    return "upsertIf", "true()"


def _parse_sort_orders(body: str) -> list[tuple[str, str]]:
    """Parse sort body for column + direction pairs."""
    orders: list[tuple[str, str]] = []
    for m in re.finditer(r"(asc|desc)\s*\(\s*(\w+)\s*(?:,\s*\w+:\s*\w+)*\s*\)", body):
        orders.append((m.group(2), m.group(1)))
    # Fallback: plain column names
    if not orders:
        for col in body.split(","):
            col = col.strip()
            if col:
                orders.append((col, "asc"))
    return orders


def _parse_statement(stmt: str, json_sources: dict, json_sinks: dict,
                     json_transforms: dict) -> DFSTransformation | DFSSource | DFSSink | None:
    """
    Parse a single DFS statement into a typed object.

    Parameters
    ----------
    stmt : str
        Full DFS statement including ``~> OutputName``.
    json_sources : dict
        Source metadata from JSON ``sources`` array, keyed by name.
    json_sinks : dict
        Sink metadata from JSON ``sinks`` array, keyed by name.
    json_transforms : dict
        Transformation metadata from JSON ``transformations`` array, keyed by name.
    """
    output_name = _extract_output_name(stmt)
    if not output_name:
        logger.warning("Could not extract output name from: %s", stmt[:80])
        return None

    # Detect the transformation type
    detected_type = DFSTransformType.UNKNOWN
    keyword = ""
    for kw, tt in _TRANSFORM_KEYWORDS.items():
        # Match keyword followed by ( — avoid partial matches
        if re.search(rf"\b{re.escape(kw)}\s*\(", stmt):
            detected_type = tt
            keyword = kw
            break

    if detected_type == DFSTransformType.UNKNOWN:
        logger.warning("Unknown transformation in: %s", stmt[:80])
        return DFSTransformation(
            name=output_name,
            transform_type=DFSTransformType.UNKNOWN,
            raw_line=stmt,
        )

    body = _extract_body(stmt, keyword)
    inputs = _extract_inputs(stmt, keyword)

    # ---- SOURCE ----
    if detected_type == DFSTransformType.SOURCE:
        columns = _parse_output_columns(body)
        opts = _parse_key_value_options(body)
        json_meta = json_sources.get(output_name, {})
        return DFSSource(
            name=output_name,
            dataset_ref=json_meta.get("dataset", {}).get("referenceName", ""),
            description=json_meta.get("description", ""),
            columns=columns,
            query=opts.get("query"),
            format=opts.get("format"),
            allow_schema_drift=opts.get("allowSchemaDrift", "true").lower() == "true",
            validate_schema=opts.get("validateSchema", "false").lower() == "true",
            isolation_level=opts.get("isolationLevel", "READ_UNCOMMITTED"),
            raw_options=opts,
        )

    # ---- SINK ----
    if detected_type == DFSTransformType.SINK:
        input_columns = _parse_output_columns(body)  # input() block uses same syntax
        col_mappings = _parse_map_column(body)
        keys = _parse_keys(body)
        opts = _parse_key_value_options(body)
        part_type, part_count = _parse_partition_by(body)
        json_meta = json_sinks.get(output_name, {})

        # Extract rejected data path expression
        rejected_path = None
        rp_match = re.search(r"rejectedData_folderPath\s*:\s*\((.+?)\)\s*,", body, re.DOTALL)
        if rp_match:
            rejected_path = rp_match.group(1).strip()

        return DFSSink(
            name=output_name,
            dataset_ref=json_meta.get("dataset", {}).get("referenceName", ""),
            description=json_meta.get("description", ""),
            inputs=inputs,
            input_columns=input_columns,
            column_mappings=col_mappings,
            keys=keys,
            format=opts.get("format", "table"),
            upsertable=opts.get("upsertable", "false").lower() == "true",
            insertable=opts.get("insertable", "false").lower() == "true",
            updateable=opts.get("updateable", "false").lower() == "true",
            deletable=opts.get("deletable", "false").lower() == "true",
            batch_size=int(opts["batchSize"]) if "batchSize" in opts else None,
            partition_by=part_type,
            partition_count=part_count,
            error_handling=opts.get("errorHandlingOption"),
            output_rejected_data=opts.get("outputRejectedData", "false").lower() == "true",
            rejected_data_path=rejected_path,
            raw_options=opts,
        )

    # ---- DERIVE ----
    if detected_type == DFSTransformType.DERIVE:
        expressions = _parse_derive_expressions(body)
        desc = json_transforms.get(output_name, {}).get("description", "")
        return DFSTransformation(
            name=output_name,
            transform_type=detected_type,
            inputs=inputs,
            description=desc,
            body=body,
            raw_line=stmt,
            expressions=expressions,
        )

    # ---- JOIN ----
    if detected_type == DFSTransformType.JOIN:
        condition, join_type, match_type = _parse_join_details(body)
        desc = json_transforms.get(output_name, {}).get("description", "")
        return DFSTransformation(
            name=output_name,
            transform_type=detected_type,
            inputs=inputs,
            description=desc,
            body=body,
            raw_line=stmt,
            condition=condition,
            join_type=join_type,
            match_type=match_type,
        )

    # ---- ALTER ROW ----
    if detected_type == DFSTransformType.ALTER_ROW:
        policy, cond = _parse_alter_row(body)
        desc = json_transforms.get(output_name, {}).get("description", "")
        return DFSTransformation(
            name=output_name,
            transform_type=detected_type,
            inputs=inputs,
            description=desc,
            body=body,
            raw_line=stmt,
            alter_row_policy=policy,
            alter_row_condition=cond,
        )

    # ---- FILTER ----
    if detected_type == DFSTransformType.FILTER:
        desc = json_transforms.get(output_name, {}).get("description", "")
        return DFSTransformation(
            name=output_name,
            transform_type=detected_type,
            inputs=inputs,
            description=desc,
            body=body,
            raw_line=stmt,
            condition=body.strip(),
        )

    # ---- AGGREGATE ----
    if detected_type == DFSTransformType.AGGREGATE:
        # groupBy(col1, col2), agg_col = func(...)
        group_by: list[str] = []
        gb_match = re.search(r"groupBy\s*\((.*?)\)", body)
        if gb_match:
            group_by = [c.strip() for c in gb_match.group(1).split(",") if c.strip()]
        # Expressions are the aggregation columns
        # Remove groupBy block, parse remaining as derive expressions
        agg_body = re.sub(r"groupBy\s*\(.*?\)\s*,?\s*", "", body, flags=re.DOTALL)
        expressions = _parse_derive_expressions(agg_body)
        desc = json_transforms.get(output_name, {}).get("description", "")
        return DFSTransformation(
            name=output_name,
            transform_type=detected_type,
            inputs=inputs,
            description=desc,
            body=body,
            raw_line=stmt,
            expressions=expressions,
            group_by=group_by,
        )

    # ---- SORT ----
    if detected_type == DFSTransformType.SORT:
        orders = _parse_sort_orders(body)
        desc = json_transforms.get(output_name, {}).get("description", "")
        return DFSTransformation(
            name=output_name,
            transform_type=detected_type,
            inputs=inputs,
            description=desc,
            body=body,
            raw_line=stmt,
            sort_orders=orders,
        )

    # ---- SELECT ----
    if detected_type == DFSTransformType.SELECT:
        col_mappings = _parse_map_column(body)
        columns = [cm.source_column for cm in col_mappings] if col_mappings else []
        desc = json_transforms.get(output_name, {}).get("description", "")
        return DFSTransformation(
            name=output_name,
            transform_type=detected_type,
            inputs=inputs,
            description=desc,
            body=body,
            raw_line=stmt,
            columns=columns,
            expressions={cm.sink_column: cm.source_column for cm in col_mappings} if col_mappings else {},
        )

    # ---- CONDITIONAL SPLIT ----
    if detected_type == DFSTransformType.CONDITIONAL_SPLIT:
        # DFS split syntax:  input split(cond1, cond2, disjoint: false) ~> out1, out2
        # The conditions are comma-separated in the body, and the output stream names
        # are comma-separated after ~>. We zip them together.
        output_names = _extract_output_names(stmt)

        # Parse conditions from the body (split at top-level commas, skip known options)
        cond_parts: list[str] = []
        depth = 0
        buf = ""
        for ch in body:
            if ch in ("(",):
                depth += 1
                buf += ch
            elif ch == ")":
                depth -= 1
                buf += ch
            elif ch == "," and depth == 0:
                cond_parts.append(buf.strip())
                buf = ""
            else:
                buf += ch
        if buf.strip():
            cond_parts.append(buf.strip())

        # Filter out known options like 'disjoint: false'
        conditions = [
            p for p in cond_parts
            if p and not re.match(r"^\w+\s*:\s*", p)
        ]

        # Zip conditions with output names
        split_conds: dict[str, str] = {}
        for idx, out_name in enumerate(output_names):
            if idx < len(conditions):
                split_conds[out_name] = conditions[idx]
            else:
                # Default branch (no explicit condition) — catches everything else
                split_conds[out_name] = "true()"

        desc = json_transforms.get(output_name, {}).get("description", "")
        return DFSTransformation(
            name=output_name,
            transform_type=detected_type,
            inputs=inputs,
            description=desc,
            body=body,
            raw_line=stmt,
            split_conditions=split_conds,
        )

    # ---- SURROGATE KEY ----
    if detected_type == DFSTransformType.SURROGATE_KEY:
        columns_parsed = _parse_output_columns(body)
        desc = json_transforms.get(output_name, {}).get("description", "")
        return DFSTransformation(
            name=output_name,
            transform_type=detected_type,
            inputs=inputs,
            description=desc,
            body=body,
            raw_line=stmt,
            columns=[c.name for c in columns_parsed],
        )

    # ---- Generic fallback ----
    desc = json_transforms.get(output_name, {}).get("description", "")
    return DFSTransformation(
        name=output_name,
        transform_type=detected_type,
        inputs=inputs,
        description=desc,
        body=body,
        raw_line=stmt,
    )


# ---------------------------------------------------------------------------
# Topological sort for transformation DAG
# ---------------------------------------------------------------------------


def _topological_sort(
    sources: list[DFSSource],
    sinks: list[DFSSink],
    transformations: list[DFSTransformation],
) -> list[str]:
    """
    Compute a valid execution order for transformations.

    Sources → intermediate transforms → sinks.
    """
    # Build adjacency and in-degree maps
    all_names: dict[str, list[str]] = {}  # name → inputs

    for s in sources:
        all_names[s.name] = []
    for t in transformations:
        all_names[t.name] = t.inputs
    for s in sinks:
        all_names[s.name] = s.inputs

    # Figure out sink inputs from transformations: if a transformation feeds a sink,
    # we need to check which transformation writes to the sink
    # In DFS, the sink statement has an input: "input_stream sink(...) ~> sink_name"
    # We stored that as a transformation's inputs in _extract_inputs

    in_degree: dict[str, int] = {name: 0 for name in all_names}
    graph: dict[str, list[str]] = {name: [] for name in all_names}

    for name, inputs in all_names.items():
        for inp in inputs:
            if inp in graph:
                graph[inp].append(name)
                in_degree[name] = in_degree.get(name, 0) + 1

    # Kahn's algorithm
    queue = [n for n in all_names if in_degree.get(n, 0) == 0]
    order: list[str] = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Add any nodes not reached (disconnected)
    for name in all_names:
        if name not in order:
            order.append(name)

    return order


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_dataflow_json(raw_json: str) -> ParsedDataFlow:
    """
    Parse an ADF Mapping Data Flow definition JSON string.

    Parameters
    ----------
    raw_json : str
        Complete JSON contents of a Data Flow definition export.

    Returns
    -------
    ParsedDataFlow
        Strongly-typed parsed result with sources, sinks, transformations,
        and execution order.

    Raises
    ------
    ValueError
        If the JSON is invalid or not a recognised Data Flow definition.
    """
    logger.info("Beginning ADF Data Flow JSON ingestion.")

    try:
        data: dict[str, Any] = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON: line {exc.lineno}, col {exc.colno}: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object.")

    name = data.get("name", "unnamed_dataflow")
    props = data.get("properties", {})
    folder = props.get("folder", {}).get("name", "")
    tp = props.get("typeProperties", {})

    # ---- Extract JSON metadata ----
    json_sources_raw = tp.get("sources", [])
    json_sinks_raw = tp.get("sinks", [])
    json_transforms_raw = tp.get("transformations", [])
    script_lines = tp.get("scriptLines", [])

    if not script_lines and not json_sources_raw:
        raise ValueError(
            "Data Flow definition must contain 'typeProperties.scriptLines' or "
            "'typeProperties.sources'."
        )

    # Index JSON metadata by name for cross-referencing
    json_sources = {s["name"]: s for s in json_sources_raw if "name" in s}
    json_sinks = {s["name"]: s for s in json_sinks_raw if "name" in s}
    json_transforms = {t["name"]: t for t in json_transforms_raw if "name" in t}

    # ---- Parse DFS script ----
    script = _join_script_lines(script_lines) if script_lines else ""
    parameters = _parse_parameters(script) if script else []
    statements = _split_statements(script) if script else []

    logger.info("Data Flow '%s': %d scriptLine statements found.", name, len(statements))

    # ---- Parse each statement ----
    sources: list[DFSSource] = []
    sinks: list[DFSSink] = []
    transformations: list[DFSTransformation] = []

    for stmt in statements:
        parsed = _parse_statement(stmt, json_sources, json_sinks, json_transforms)
        if parsed is None:
            continue
        if isinstance(parsed, DFSSource):
            sources.append(parsed)
        elif isinstance(parsed, DFSSink):
            sinks.append(parsed)
        elif isinstance(parsed, DFSTransformation):
            transformations.append(parsed)

    # If no scriptLines, create sources/sinks from JSON metadata only
    if not script and json_sources_raw:
        for s in json_sources_raw:
            sources.append(DFSSource(
                name=s.get("name", ""),
                dataset_ref=s.get("dataset", {}).get("referenceName", ""),
                description=s.get("description", ""),
            ))
    if not script and json_sinks_raw:
        for s in json_sinks_raw:
            sinks.append(DFSSink(
                name=s.get("name", ""),
                dataset_ref=s.get("dataset", {}).get("referenceName", ""),
                description=s.get("description", ""),
            ))

    # ---- Compute execution order ----
    order = _topological_sort(sources, sinks, transformations)

    logger.info(
        "Data Flow '%s' parsed: %d sources, %d sinks, %d transformations.",
        name, len(sources), len(sinks), len(transformations),
    )

    return ParsedDataFlow(
        name=name,
        folder=folder,
        parameters=parameters,
        sources=sources,
        sinks=sinks,
        transformations=transformations,
        script=script,
        transformation_order=order,
        raw_json=data,
    )
