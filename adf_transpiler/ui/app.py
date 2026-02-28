"""
app.py
======
Streamlit frontend for the ADF-to-PySpark Local Converter.

Architecture flow implemented here:
    1. File upload  →  adf_parser.parse_pipeline_json()
    2. Deterministic transpilation  →  core.transpiler.ActivityTranspiler
    3. ADF expression streaming  →  agents.ollama_agent.OllamaAgent
    4. Live code display with syntax highlighting

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import streamlit as st

# Make local packages importable when running from project root.
sys.path.insert(0, str(Path(__file__).parent))

from core.adf_parser import parse_pipeline_json, ParsedPipeline
from core.transpiler import ActivityTranspiler, TranspilerResult
from agents.ollama_agent import (
    OllamaAgent,
    TranslationRequest,
    patch_code_with_translations,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ADF → PySpark Transpiler",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults = {
        "parsed_pipeline": None,
        "transpiler_result": None,
        "final_code": None,
        "ollama_model": "codellama",
        "ollama_url": "http://localhost:11434",
        "translation_log": [],
        "phase": "idle",  # idle | parsed | transpiled | translating | done
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")
    st.divider()

    st.subheader("🤖 Ollama Settings")
    st.session_state.ollama_model = st.selectbox(
        "Model",
        options=["codellama", "llama3", "llama3:8b", "mistral", "deepseek-coder"],
        index=0,
        help="Must be pulled locally via `ollama pull <model>`",
    )
    st.session_state.ollama_url = st.text_input(
        "Ollama API URL",
        value=st.session_state.ollama_url,
        help="Default: http://localhost:11434",
    )

    if st.button("🔌 Test Ollama Connection", use_container_width=True):
        with st.spinner("Checking connection…"):
            agent = OllamaAgent(
                model=st.session_state.ollama_model,
                base_url=st.session_state.ollama_url,
            )
            ok, msg = asyncio.run(agent.check_connection())
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    st.divider()
    st.subheader("ℹ️ Data Privacy")
    st.info(
        "All LLM inference runs **locally** via Ollama. "
        "No pipeline metadata leaves your machine.",
        icon="🔒",
    )

    st.divider()
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("⚡ ADF → PySpark Transpiler")
st.caption("Local, deterministic, zero data-leakage conversion of Azure Data Factory pipelines.")

tab_upload, tab_parse, tab_code, tab_log = st.tabs(
    ["📂 Upload", "🔍 Pipeline Analysis", "💻 Generated Code", "📋 Translation Log"]
)

# ---------------------------------------------------------------------------
# Tab 1: Upload
# ---------------------------------------------------------------------------

with tab_upload:
    st.subheader("Upload ADF Pipeline JSON")
    st.markdown(
        "Export your pipeline from ADF Studio: **Author → Pipelines → ⋯ → Download**"
    )

    uploaded_file = st.file_uploader(
        "Drop your `pipeline.json` here",
        type=["json"],
        help="Standard ADF pipeline export format.",
        label_visibility="collapsed",
    )

    col_parse, col_sample = st.columns([1, 1])

    with col_parse:
        parse_clicked = st.button(
            "🔍 Parse Pipeline",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

    with col_sample:
        load_sample = st.button(
            "📄 Load Sample Pipeline",
            use_container_width=True,
            help="Load the bundled sample pipeline for a quick demo.",
        )

    # ---- Load sample ----
    if load_sample:
        sample_path = Path(__file__).parent / "sample_data" / "sample_pipeline.json"
        if sample_path.exists():
            raw_json = sample_path.read_text()
            st.session_state.phase = "idle"
            _parse_and_store(raw_json)
        else:
            st.error("Sample file not found. Ensure `sample_data/sample_pipeline.json` exists.")

    # ---- Parse uploaded file ----
    if parse_clicked and uploaded_file is not None:
        raw_json = uploaded_file.read().decode("utf-8")
        _parse_and_store(raw_json)


def _parse_and_store(raw_json: str) -> None:
    """Parse JSON, store result in session state, update phase."""
    with st.spinner("Parsing and validating pipeline schema…"):
        try:
            parsed: ParsedPipeline = parse_pipeline_json(raw_json)
            st.session_state.parsed_pipeline = parsed
            st.session_state.transpiler_result = None
            st.session_state.final_code = None
            st.session_state.translation_log = []
            st.session_state.phase = "parsed"
            st.success(f"✅ Pipeline **{parsed.pipeline.name}** parsed successfully.")
        except ValueError as exc:
            st.error(f"❌ Parse Error: {exc}")
            logger.error("Parse failed: %s", exc)


# ---------------------------------------------------------------------------
# Tab 2: Pipeline Analysis
# ---------------------------------------------------------------------------

with tab_parse:
    parsed: ParsedPipeline | None = st.session_state.parsed_pipeline

    if parsed is None:
        st.info("Upload and parse a pipeline JSON file first.")
    else:
        st.subheader(f"Pipeline: `{parsed.pipeline.name}`")

        desc = parsed.pipeline.properties.description
        if desc:
            st.caption(desc)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Activities", parsed.activity_count)
        col2.metric("Parameters", len(parsed.pipeline.properties.parameters))
        col3.metric("Variables", len(parsed.pipeline.properties.variables))
        col4.metric("Has Nested", "Yes" if parsed.has_nested_activities else "No")

        st.divider()
        st.subheader("Activity Graph (DAG)")

        for activity in parsed.pipeline.resolved_activities:
            with st.expander(f"**{activity.name}** `[{activity.type}]`", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Depends On:**")
                    if activity.depends_on:
                        for dep in activity.depends_on:
                            st.markdown(
                                f"- `{dep.activity}` → {', '.join(c.value for c in dep.dependency_conditions)}"
                            )
                    else:
                        st.markdown("_No dependencies (entry point)_")
                with col_b:
                    st.markdown("**Type Properties (raw):**")
                    st.json(activity.type_properties, expanded=False)

        st.divider()
        if st.button("🚀 Run Deterministic Transpilation", type="primary", use_container_width=True):
            with st.spinner("Generating PySpark boilerplate…"):
                transpiler = ActivityTranspiler()
                result: TranspilerResult = transpiler.transpile(parsed)
                st.session_state.transpiler_result = result
                st.session_state.final_code = result.full_code
                st.session_state.phase = "transpiled"

            st.success(
                f"✅ Transpilation complete. "
                f"Pending Ollama expressions: **{result.total_pending_expressions}**"
            )
            st.rerun()

# ---------------------------------------------------------------------------
# Tab 3: Generated Code
# ---------------------------------------------------------------------------

with tab_code:
    result: TranspilerResult | None = st.session_state.transpiler_result

    if result is None:
        st.info("Run transpilation from the **Pipeline Analysis** tab first.")
    else:
        st.subheader(f"Generated PySpark — `{result.pipeline_name}`")

        col_dl, col_ollama, _ = st.columns([1, 1, 2])

        with col_dl:
            st.download_button(
                "⬇️ Download .py",
                data=st.session_state.final_code or "",
                file_name=f"{result.pipeline_name}_spark.py",
                mime="text/x-python",
                use_container_width=True,
            )

        with col_ollama:
            needs_llm = result.has_pending_llm_work
            translate_clicked = st.button(
                f"🤖 Translate {result.total_pending_expressions} ADF Expressions (Ollama)",
                type="primary" if needs_llm else "secondary",
                use_container_width=True,
                disabled=not needs_llm,
            )

        # ---- Streaming Ollama translation ----
        if translate_clicked:
            agent = OllamaAgent(
                model=st.session_state.ollama_model,
                base_url=st.session_state.ollama_url,
            )

            ok, conn_msg = asyncio.run(agent.check_connection())
            if not ok:
                st.error(f"Ollama not available: {conn_msg}")
            else:
                _run_ollama_translation(agent, result)

        # ---- Code display ----
        st.code(
            st.session_state.final_code or result.full_code,
            language="python",
            line_numbers=True,
        )


def _run_ollama_translation(agent: OllamaAgent, result: TranspilerResult) -> None:
    """Stream Ollama translations and patch them into the code in real time."""
    st.session_state.phase = "translating"
    log: list[str] = []
    all_requests: list[TranslationRequest] = []

    for ar in result.activity_results:
        for json_path, expr in ar.pending_llm_expressions:
            all_requests.append(
                TranslationRequest(
                    json_path=json_path,
                    adf_expression=expr,
                    activity_name=ar.activity_name,
                )
            )

    progress = st.progress(0, text="Starting Ollama translation…")
    code_placeholder = st.empty()
    current_code = result.full_code
    all_results = []

    for i, req in enumerate(all_requests):
        progress.progress(
            i / len(all_requests),
            text=f"Translating [{req.activity_name}] {req.json_path} …",
        )

        log.append(f"\n--- [{req.activity_name}] {req.adf_expression[:80]} ---")
        token_buffer: list[str] = []

        async def _stream() -> str:
            async for token in agent.translate_expression_stream(req):
                token_buffer.append(token)
            return "".join(token_buffer)

        translated = asyncio.run(_stream())
        log.append(f"→ {translated}")

        from agents.ollama_agent import TranslationResult, _clean_llm_output
        tr = TranslationResult(
            request=req,
            pyspark_expression=_clean_llm_output(translated),
            model_used=agent.model,
            success=True,
        )
        all_results.append(tr)

        # Patch incrementally so user sees progress
        current_code = patch_code_with_translations(current_code, [tr])
        code_placeholder.code(current_code, language="python", line_numbers=True)

    progress.progress(1.0, text="✅ All expressions translated.")
    st.session_state.final_code = current_code
    st.session_state.translation_log = log
    st.session_state.phase = "done"
    st.rerun()


# ---------------------------------------------------------------------------
# Tab 4: Translation Log
# ---------------------------------------------------------------------------

with tab_log:
    log = st.session_state.translation_log
    if not log:
        st.info("No Ollama translations have run yet.")
    else:
        st.subheader("Ollama Translation Log")
        st.code("\n".join(log), language="text")
