"""core — deterministic parsing and transpilation engine."""
from .adf_parser import parse_pipeline_json, ParsedPipeline
from .transpiler import ActivityTranspiler, TranspilerResult
from .dataflow_parser import parse_dataflow_json, ParsedDataFlow, detect_json_type
from .dataflow_transpiler import DataFlowTranspiler, DataFlowTranspileResult

__all__ = [
    "parse_pipeline_json", "ParsedPipeline",
    "ActivityTranspiler", "TranspilerResult",
    "parse_dataflow_json", "ParsedDataFlow", "detect_json_type",
    "DataFlowTranspiler", "DataFlowTranspileResult",
]
