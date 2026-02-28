"""agents — LLM and workflow orchestration agents."""
from .ollama_agent import OllamaAgent, TranslationRequest, TranslationResult, patch_code_with_translations
from .ui_agent import UIAgent, WorkflowPhase, WorkflowState, WorkflowEvent, EventKind, TranslationLogEntry

__all__ = [
    "OllamaAgent", "TranslationRequest", "TranslationResult", "patch_code_with_translations",
    "UIAgent", "WorkflowPhase", "WorkflowState", "WorkflowEvent", "EventKind", "TranslationLogEntry",
]
