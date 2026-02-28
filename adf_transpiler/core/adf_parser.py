"""
adf_parser.py
=============
Deterministic ingestion and parsing engine for Azure Data Factory Pipeline JSON definitions.

Responsibilities:
    - Strict schema validation via Pydantic v2 models.
    - Activity-type discrimination and subclass routing.
    - Clean, strongly-typed Python object graph returned to callers.

Author: Transpiler Architect
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Annotated, Any, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ActivityType(str, Enum):
    COPY = "Copy"
    FOR_EACH = "ForEach"
    IF_CONDITION = "IfCondition"
    LOOKUP = "Lookup"
    EXECUTE_PIPELINE = "ExecutePipeline"
    SET_VARIABLE = "SetVariable"
    APPEND_VARIABLE = "AppendVariable"
    WEB = "Web"
    WAIT = "Wait"
    DELETE = "Delete"
    GET_METADATA = "GetMetadata"
    DATABRICKS_NOTEBOOK = "DatabricksNotebook"
    MAPPING_DATA_FLOW = "MappingDataFlow"
    UNKNOWN = "Unknown"


class DependencyCondition(str, Enum):
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    SKIPPED = "Skipped"
    COMPLETED = "Completed"


# ---------------------------------------------------------------------------
# Shared / Primitive models
# ---------------------------------------------------------------------------


class DependsOn(BaseModel):
    """Represents a single upstream dependency edge in the activity DAG."""

    activity: str = Field(..., description="Name of the upstream activity.")
    dependency_conditions: list[DependencyCondition] = Field(
        default_factory=lambda: [DependencyCondition.SUCCEEDED],
        alias="dependencyConditions",
    )

    model_config = {"populate_by_name": True}


class DatasetReference(BaseModel):
    reference_name: str = Field(..., alias="referenceName")
    type: str = Field(default="DatasetReference")
    parameters: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class LinkedServiceReference(BaseModel):
    reference_name: str = Field(..., alias="referenceName")
    type: str = Field(default="LinkedServiceReference")
    parameters: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class PipelineReference(BaseModel):
    reference_name: str = Field(..., alias="referenceName")
    type: str = Field(default="PipelineReference")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Base Activity
# ---------------------------------------------------------------------------


class ActivityBase(BaseModel):
    """
    Lowest-common-denominator model for any ADF activity.
    Concrete subclasses extend this with type-specific `type_properties`.
    """

    name: str = Field(..., min_length=1, description="Unique activity name within the pipeline.")
    type: str = Field(..., description="ADF activity type discriminator string.")
    depends_on: list[DependsOn] = Field(default_factory=list, alias="dependsOn")
    description: str | None = Field(default=None)
    user_properties: list[dict[str, Any]] = Field(default_factory=list, alias="userProperties")
    # Raw type-specific property bag; subclasses parse this further.
    type_properties: dict[str, Any] = Field(default_factory=dict, alias="typeProperties")

    model_config = {"populate_by_name": True, "extra": "allow"}

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Activity name must not be blank or whitespace.")
        return v


# ---------------------------------------------------------------------------
# Copy Activity
# ---------------------------------------------------------------------------


class CopySource(BaseModel):
    type: str = Field(..., description="ADF source connector type, e.g. 'AzureBlobSource'.")
    additional_columns: Any | None = Field(default=None, alias="additionalColumns")
    query: str | None = Field(default=None)
    # Capture all remaining source-specific settings without strict enforcement.
    model_config = {"populate_by_name": True, "extra": "allow"}


class CopySink(BaseModel):
    type: str = Field(..., description="ADF sink connector type, e.g. 'AzureSqlSink'.")
    write_behavior: str | None = Field(default=None, alias="writeBehavior")
    upsert_settings: dict[str, Any] | None = Field(default=None, alias="upsertSettings")
    model_config = {"populate_by_name": True, "extra": "allow"}


class CopyTranslator(BaseModel):
    type: str = Field(default="TabularTranslator")
    column_mappings: str | None = Field(default=None, alias="columnMappings")
    model_config = {"populate_by_name": True, "extra": "allow"}


class CopyTypeProperties(BaseModel):
    source: CopySource
    sink: CopySink
    enable_staging: bool = Field(default=False, alias="enableStaging")
    translator: CopyTranslator | None = Field(default=None)
    model_config = {"populate_by_name": True, "extra": "allow"}


class CopyActivityInputOutput(BaseModel):
    reference_name: str = Field(..., alias="referenceName")
    type: str = Field(default="DatasetReference")
    parameters: dict[str, Any] = Field(default_factory=dict)
    model_config = {"populate_by_name": True}


class CopyActivity(ActivityBase):
    """Strongly-typed model for ADF Copy activities."""

    type: str = Field(default=ActivityType.COPY.value)
    inputs: list[CopyActivityInputOutput] = Field(default_factory=list)
    outputs: list[CopyActivityInputOutput] = Field(default_factory=list)
    parsed_type_properties: CopyTypeProperties | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def parse_type_properties(self) -> "CopyActivity":
        if self.type_properties:
            try:
                self.parsed_type_properties = CopyTypeProperties.model_validate(
                    self.type_properties
                )
            except Exception as exc:
                logger.warning(
                    "CopyActivity '%s': typeProperties validation failed — %s", self.name, exc
                )
        return self


# ---------------------------------------------------------------------------
# ForEach Activity
# ---------------------------------------------------------------------------


class ForEachTypeProperties(BaseModel):
    is_sequential: bool = Field(default=False, alias="isSequential")
    batch_count: int | None = Field(default=None, alias="batchCount", ge=1, le=50)
    items: dict[str, Any] = Field(
        ..., description="ADF expression object describing the iterable collection."
    )
    # Activities nested inside ForEach — kept as raw dicts; parsed by the
    # discriminator function to avoid circular import issues.
    activities: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class ForEachActivity(ActivityBase):
    """Strongly-typed model for ADF ForEach activities."""

    type: str = Field(default=ActivityType.FOR_EACH.value)
    parsed_type_properties: ForEachTypeProperties | None = Field(default=None, exclude=True)
    # Resolved after top-level parse; populated by the pipeline parser.
    inner_activities: list[AnyActivity] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def parse_type_properties(self) -> "ForEachActivity":
        if self.type_properties:
            try:
                self.parsed_type_properties = ForEachTypeProperties.model_validate(
                    self.type_properties
                )
            except Exception as exc:
                logger.warning(
                    "ForEachActivity '%s': typeProperties validation failed — %s", self.name, exc
                )
        return self


# ---------------------------------------------------------------------------
# IfCondition Activity
# ---------------------------------------------------------------------------


class IfConditionTypeProperties(BaseModel):
    expression: dict[str, Any] = Field(
        ..., description="ADF expression object evaluating to a boolean."
    )
    if_true_activities: list[dict[str, Any]] = Field(
        default_factory=list, alias="ifTrueActivities"
    )
    if_false_activities: list[dict[str, Any]] = Field(
        default_factory=list, alias="ifFalseActivities"
    )

    model_config = {"populate_by_name": True}


class IfConditionActivity(ActivityBase):
    """Strongly-typed model for ADF IfCondition activities."""

    type: str = Field(default=ActivityType.IF_CONDITION.value)
    parsed_type_properties: IfConditionTypeProperties | None = Field(default=None, exclude=True)
    # Resolved by the pipeline parser after initial parse.
    true_branch_activities: list[AnyActivity] = Field(default_factory=list, exclude=True)
    false_branch_activities: list[AnyActivity] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def parse_type_properties(self) -> "IfConditionActivity":
        if self.type_properties:
            try:
                self.parsed_type_properties = IfConditionTypeProperties.model_validate(
                    self.type_properties
                )
            except Exception as exc:
                logger.warning(
                    "IfConditionActivity '%s': typeProperties validation failed — %s",
                    self.name,
                    exc,
                )
        return self


# ---------------------------------------------------------------------------
# Generic / fallback Activity
# ---------------------------------------------------------------------------


class GenericActivity(ActivityBase):
    """
    Catch-all model for activity types not yet given a dedicated subclass.
    Preserves the raw typeProperties bag for downstream consumers.
    """


# ---------------------------------------------------------------------------
# Discriminated union type alias
# ---------------------------------------------------------------------------

AnyActivity = Annotated[
    Union[CopyActivity, ForEachActivity, IfConditionActivity, GenericActivity],
    Field(discriminator=None),  # Manual dispatch — see _dispatch_activity below.
]

# Rebuild forward refs now that AnyActivity is defined.
ForEachActivity.model_rebuild()
IfConditionActivity.model_rebuild()


# ---------------------------------------------------------------------------
# Activity dispatcher
# ---------------------------------------------------------------------------

_ACTIVITY_TYPE_MAP: dict[str, type[ActivityBase]] = {
    ActivityType.COPY.value: CopyActivity,
    ActivityType.FOR_EACH.value: ForEachActivity,
    ActivityType.IF_CONDITION.value: IfConditionActivity,
}


def _dispatch_activity(raw: dict[str, Any]) -> AnyActivity:
    """
    Route a raw activity dict to its strongly-typed Pydantic subclass.

    Falls back to GenericActivity for unrecognised types so that the
    pipeline parse never hard-fails on unknown activity kinds.
    """
    activity_type: str = raw.get("type", "")
    model_cls = _ACTIVITY_TYPE_MAP.get(activity_type, GenericActivity)

    logger.debug("Dispatching activity '%s' → %s", raw.get("name"), model_cls.__name__)
    return model_cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Pipeline-level models
# ---------------------------------------------------------------------------


class PipelineVariable(BaseModel):
    type: str = Field(..., description="Variable type: String | Bool | Array")
    default_value: Any | None = Field(default=None, alias="defaultValue")
    model_config = {"populate_by_name": True}


class PipelineParameter(BaseModel):
    type: str
    default_value: Any | None = Field(default=None, alias="defaultValue")
    model_config = {"populate_by_name": True}


class PipelineFolder(BaseModel):
    name: str


class PipelineProperties(BaseModel):
    """
    Maps to the `properties` block of a standard ADF pipeline JSON export.
    """

    description: str | None = Field(default=None)
    activities: list[dict[str, Any]] = Field(default_factory=list)
    parameters: dict[str, PipelineParameter] = Field(default_factory=dict)
    variables: dict[str, PipelineVariable] = Field(default_factory=dict)
    annotations: list[Any] = Field(default_factory=list)
    folder: PipelineFolder | None = Field(default=None)
    concurrency: int | None = Field(default=None, ge=1)

    model_config = {"populate_by_name": True, "extra": "allow"}


class Pipeline(BaseModel):
    """
    Top-level ADF Pipeline document model.

    After validation, `resolved_activities` contains the fully-typed,
    recursively-resolved activity graph.
    """

    name: str = Field(..., min_length=1)
    type: str = Field(default="Microsoft.DataFactory/factories/pipelines")
    properties: PipelineProperties

    # Populated by _resolve_activities; not part of raw JSON.
    resolved_activities: list[AnyActivity] = Field(default_factory=list, exclude=True)

    model_config = {"populate_by_name": True, "extra": "allow"}

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Pipeline name must not be blank or whitespace.")
        return v


# ---------------------------------------------------------------------------
# Recursive activity resolution
# ---------------------------------------------------------------------------


def _resolve_activity_list(raw_activities: list[dict[str, Any]]) -> list[AnyActivity]:
    """
    Recursively resolve a list of raw activity dicts into typed objects,
    descending into ForEach / IfCondition branches.
    """
    resolved: list[AnyActivity] = []

    for raw in raw_activities:
        try:
            activity = _dispatch_activity(raw)
        except Exception as exc:
            logger.error(
                "Failed to parse activity '%s' (type=%s): %s",
                raw.get("name", "<unnamed>"),
                raw.get("type", "<unknown>"),
                exc,
                exc_info=True,
            )
            # Degrade gracefully: attempt GenericActivity before giving up.
            try:
                activity = GenericActivity.model_validate(raw)
                logger.warning(
                    "Activity '%s' fell back to GenericActivity after parse error.",
                    raw.get("name"),
                )
            except Exception as fallback_exc:
                logger.critical(
                    "Activity '%s' could not be parsed even as GenericActivity: %s",
                    raw.get("name"),
                    fallback_exc,
                )
                continue

        # --- Recurse into container activities ---
        if isinstance(activity, ForEachActivity) and activity.parsed_type_properties:
            activity.inner_activities = _resolve_activity_list(
                activity.parsed_type_properties.activities
            )

        elif isinstance(activity, IfConditionActivity) and activity.parsed_type_properties:
            activity.true_branch_activities = _resolve_activity_list(
                activity.parsed_type_properties.if_true_activities
            )
            activity.false_branch_activities = _resolve_activity_list(
                activity.parsed_type_properties.if_false_activities
            )

        resolved.append(activity)

    return resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ParsedPipeline(BaseModel):
    """
    The contract object returned by `parse_pipeline_json`.
    Consumers should depend on this interface, not on internal models.
    """

    pipeline: Pipeline
    activity_count: int
    activity_types_found: list[str]
    has_nested_activities: bool

    model_config = {"arbitrary_types_allowed": True}


def parse_pipeline_json(raw_json: str) -> ParsedPipeline:
    """
    Ingest and validate a raw ADF Pipeline JSON string.

    Parameters
    ----------
    raw_json:
        The complete contents of an ADF pipeline export JSON file.

    Returns
    -------
    ParsedPipeline
        A strongly-typed result envelope containing the validated Pipeline
        object and derived metadata.

    Raises
    ------
    ValueError
        If the JSON is syntactically invalid or structurally incompatible
        with the expected ADF pipeline schema.
    """
    logger.info("Beginning ADF pipeline JSON ingestion.")

    # ---- Phase 1: JSON syntax validation ----
    try:
        raw_dict: dict[str, Any] = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.error("JSON syntax error at line %d, col %d: %s", exc.lineno, exc.colno, exc.msg)
        raise ValueError(
            f"Malformed JSON input — line {exc.lineno}, col {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(raw_dict, dict):
        raise ValueError("Top-level JSON value must be an object (dict), not a primitive or array.")

    # ---- Phase 2: Pydantic schema validation ----
    try:
        pipeline = Pipeline.model_validate(raw_dict)
    except Exception as exc:
        # Pydantic v2 raises ValidationError; catch broadly so callers get a
        # clean ValueError regardless of pydantic version internals.
        logger.error("Pipeline schema validation failed: %s", exc)
        raise ValueError(f"ADF pipeline schema validation failed: {exc}") from exc

    logger.info("Pipeline '%s' passed schema validation.", pipeline.name)

    # ---- Phase 3: Recursive activity resolution ----
    pipeline.resolved_activities = _resolve_activity_list(pipeline.properties.activities)

    activity_count = len(pipeline.resolved_activities)
    types_found = sorted({a.type for a in pipeline.resolved_activities})
    has_nested = any(
        isinstance(a, (ForEachActivity, IfConditionActivity))
        for a in pipeline.resolved_activities
    )

    logger.info(
        "Pipeline '%s' resolved: %d top-level activities, types=%s, nested=%s",
        pipeline.name,
        activity_count,
        types_found,
        has_nested,
    )

    return ParsedPipeline(
        pipeline=pipeline,
        activity_count=activity_count,
        activity_types_found=types_found,
        has_nested_activities=has_nested,
    )
