"""
app.py
======
Streamlit frontend — pure rendering layer.
All workflow logic, state transitions, error handling, and LLM coordination
are fully delegated to agents.ui_agent.UIAgent.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.ui_agent import EventKind, UIAgent, WorkflowPhase, WorkflowState

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

# ---------------------------------------------------------------------------
# Page config (must be FIRST Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ADF → PySpark Transpiler",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

agent: UIAgent = UIAgent.get_or_create(st.session_state)
state: WorkflowState = agent.state

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
_PHASE_COLORS = {
    WorkflowPhase.IDLE: "gray", WorkflowPhase.PARSING: "blue",
    WorkflowPhase.PARSED: "green", WorkflowPhase.TRANSPILING: "blue",
    WorkflowPhase.TRANSPILED: "green", WorkflowPhase.CONNECTING_OLLAMA: "blue",
    WorkflowPhase.TRANSLATING: "orange", WorkflowPhase.DONE: "green",
    WorkflowPhase.ERROR: "red",
}

with st.sidebar:
    st.markdown("## ⚡ ADF Transpiler")
    st.divider()
    color = _PHASE_COLORS.get(state.phase, "gray")
    st.markdown(f"**Status:** :{color}[{state.phase.name}]")
    if state.error_message:
        st.error(state.error_message, icon="🚨")
    st.divider()

    st.subheader("🤖 Ollama Settings")
    _models = [
        "qwen2.5-coder:7b",            # ← default — fast, 8 GB VRAM
        "qwen2.5-coder:7b-instruct",   # instruct-tuned variant
        "qwen2.5-coder:14b",           # higher accuracy, 16 GB VRAM
        "qwen2.5-coder:14b-instruct",
        "qwen2.5-coder:32b",           # best quality, 24 GB VRAM
        "codellama",                   # fallback options
        "llama3",
        "llama3:8b",
        "deepseek-coder:6.7b",
    ]
    _idx = _models.index(state.ollama_model) if state.ollama_model in _models else 0
    selected_model = st.selectbox("Model", options=_models, index=_idx if _idx >= 0 else 0, disabled=state.is_busy, help="Primary: qwen2.5-coder:7b. Pull with: `ollama pull qwen2.5-coder:7b`")
    ollama_url_input = st.text_input("Ollama API URL", value=state.ollama_url, disabled=state.is_busy)

    if selected_model != state.ollama_model or ollama_url_input != state.ollama_url:
        agent.configure_ollama(selected_model, ollama_url_input)

    c_test, c_status = st.columns([2, 1])
    with c_test:
        if st.button("🔌 Test Connection", use_container_width=True, disabled=state.is_busy):
            with st.spinner("Pinging Ollama…"):
                ok, msg = agent.check_ollama_connection()
            (st.success if ok else st.error)(msg, icon="✅" if ok else "❌")
    with c_status:
        st.markdown("🟢 Ready" if state.ollama_connected else "🔴 Unknown")

    st.divider()
    st.info("🔒 **Zero data leakage.**\n\nAll LLM inference runs locally via Ollama. No pipeline metadata leaves your machine.")
    st.divider()
    if st.button("🔄 Reset Session", use_container_width=True, disabled=state.is_busy):
        UIAgent.reset(st.session_state)
        st.rerun()

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_upload, tab_analysis, tab_code, tab_log = st.tabs([
    "📂 1 · Upload", "🔍 2 · Pipeline Analysis", "💻 3 · Generated Code", "📋 4 · Translation Log"
])

# ===========================================================================
# TAB 1 — Upload
# ===========================================================================
with tab_upload:
    st.subheader("Upload ADF Pipeline JSON")
    st.markdown("Export from ADF Studio: **Author → Pipelines → ··· → Download**")

    uploaded_file = st.file_uploader("Drop `pipeline.json` here", type=["json"], label_visibility="collapsed", disabled=state.is_busy)
    c_parse, c_sample, _ = st.columns([1, 1, 2])

    with c_parse:
        if st.button("🔍 Parse Pipeline", type="primary", use_container_width=True, disabled=uploaded_file is None or state.is_busy):
            with st.spinner("Validating schema…"):
                ok = agent.parse(uploaded_file.read().decode("utf-8"))
            if ok:
                st.success(f"✅ **{state.parsed_pipeline.pipeline.name}** — {state.parsed_pipeline.activity_count} activities.")
                st.rerun()
            else:
                st.error(state.error_message)

    with c_sample:
        sample_path = Path(__file__).parent / "sample_data" / "sample_pipeline.json"
        if st.button("📄 Load Sample", use_container_width=True, disabled=state.is_busy):
            if sample_path.exists():
                with st.spinner("Loading sample…"):
                    ok = agent.parse(sample_path.read_text())
                if ok:
                    st.success("Sample pipeline loaded.")
                    st.rerun()
            else:
                st.error("sample_data/sample_pipeline.json not found.")

    if state.phase == WorkflowPhase.IDLE:
        st.divider()
        st.markdown("""
**Supported activities and output strategy:**

| Activity | Engine | PySpark Output |
|---|---|---|
| `Copy` | Deterministic | `spark.read / df.write` scaffold |
| `ForEach` | Deterministic + Ollama | `for item in items:` loop |
| `IfCondition` | Deterministic + Ollama | `if condition: / else:` |
| `SetVariable` | Deterministic | `pipeline_vars["x"] = …` |
| `Wait` | Deterministic | `time.sleep(n)` |
| All others | Partial + Ollama | Stub with raw typeProperties |
""")

# ===========================================================================
# TAB 2 — Pipeline Analysis
# ===========================================================================
with tab_analysis:
    if state.parsed_pipeline is None:
        st.info("⬅️ Upload and parse a pipeline in Tab 1 first.")
    else:
        parsed = state.parsed_pipeline
        pl = parsed.pipeline

        st.subheader(f"Pipeline: `{pl.name}`")
        if pl.properties.description:
            st.caption(pl.properties.description)
        if pl.properties.folder:
            st.caption(f"📁 {pl.properties.folder.name}")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Activities", parsed.activity_count)
        c2.metric("Parameters", len(pl.properties.parameters))
        c3.metric("Variables", len(pl.properties.variables))
        c4.metric("Nested", "Yes" if parsed.has_nested_activities else "No")
        c5.metric("Pending LLM", state.transpiler_result.total_pending_expressions if state.transpiler_result else "—")

        st.divider()

        if pl.properties.parameters:
            with st.expander("⚙️ Parameters"):
                st.table([{"Name": k, "Type": v.type, "Default": str(v.default_value)} for k, v in pl.properties.parameters.items()])

        if pl.properties.variables:
            with st.expander("🔀 Variables"):
                st.table([{"Name": k, "Type": v.type, "Default": str(v.default_value)} for k, v in pl.properties.variables.items()])

        st.subheader("Activity DAG")
        _icons = {"Copy": "📋", "ForEach": "🔁", "IfCondition": "🔀", "Lookup": "🔍", "SetVariable": "📝", "Wait": "⏱️", "ExecutePipeline": "▶️", "DatabricksNotebook": "📓", "MappingDataFlow": "🌊"}

        for node in (agent.get_dag_summary() or []):
            icon = _icons.get(node["type"], "⚙️")
            with st.expander(f"{icon} **{node['name']}** `[{node['type']}]`"):
                cl, cr = st.columns(2)
                with cl:
                    st.markdown("**Dependencies:**")
                    if node["depends_on"]:
                        for dep in node["depends_on"]:
                            st.markdown(f"- `{dep['activity']}` — {', '.join(dep['conditions'])}")
                    else:
                        st.markdown("_Entry point_")
                    if node["children"]:
                        st.markdown("**Nested:**")
                        for ch in node["children"]:
                            st.markdown(f"- `{ch['name']}` [{ch['type']}] `{ch.get('branch','')}`")
                with cr:
                    st.markdown("**Type Properties:**")
                    st.json(node["type_properties"], expanded=False)

        st.divider()
        if st.button("🚀 Generate PySpark Code", type="primary", use_container_width=True, disabled=state.is_busy or not state.can_transpile):
            with st.spinner("Running template engine…"):
                ok = agent.transpile()
            if ok:
                st.success(f"✅ Code generated. Pending Ollama expressions: **{state.transpiler_result.total_pending_expressions}**")
                st.rerun()
            else:
                st.error(state.error_message)

# ===========================================================================
# TAB 3 — Generated Code
# ===========================================================================
with tab_code:
    if state.transpiler_result is None:
        st.info("⬅️ Run Generate PySpark Code in Tab 2 first.")
    else:
        result = state.transpiler_result
        st.subheader(f"Generated PySpark — `{result.pipeline_name}`")

        c_dl, c_llm, c_stat = st.columns([1, 2, 1])
        with c_dl:
            st.download_button("⬇️ Download .py", data=agent.get_final_code() or "", file_name=agent.get_download_filename(), mime="text/x-python", use_container_width=True, disabled=not state.can_download)
        with c_llm:
            pending = result.total_pending_expressions
            label = f"🤖 Translate {pending} ADF Expressions via Ollama" if pending > 0 else "✅ Fully Deterministic — No LLM Needed"
            translate_clicked = st.button(label, type="primary" if state.can_translate else "secondary", use_container_width=True, disabled=not state.can_translate or state.is_busy)
        with c_stat:
            if state.total_expressions > 0:
                st.metric("Translated", f"{state.translated_expressions}/{state.total_expressions}", delta=f"{int(state.progress_pct*100)}%")

        if translate_clicked:
            with st.spinner("Checking Ollama connection…"):
                ok, conn_msg = agent.check_ollama_connection()
            if not ok:
                st.error(f"❌ Ollama unavailable: {conn_msg}")
            else:
                st.info(f"🤖 Translating via `{state.ollama_model}`…")
                progress_bar = st.progress(0, text="Preparing…")
                code_placeholder = st.empty()

                for event in agent.translate_streaming():
                    if event.kind == EventKind.PROGRESS:
                        pct = event.payload.get("current", 0) / max(event.payload.get("total", 1), 1)
                        progress_bar.progress(pct, text=event.message)
                    elif event.kind == EventKind.CODE_PATCHED:
                        code_placeholder.code(event.payload.get("code", ""), language="python", line_numbers=True)
                    elif event.kind == EventKind.PHASE_CHANGED and state.phase == WorkflowPhase.DONE:
                        progress_bar.progress(1.0, text="✅ Translation complete!")
                    elif event.kind == EventKind.WARNING:
                        st.warning(event.message)
                    elif event.kind == EventKind.ERROR:
                        st.error(event.message)
                st.rerun()

        st.code(agent.get_final_code() or result.full_code, language="python", line_numbers=True)

        st.divider()
        st.subheader("Activity Status")
        st.dataframe([{
            "Activity": ar.activity_name, "Type": ar.activity_type,
            "Deterministic": "✅" if ar.is_deterministic else "⚠️ Needs LLM",
            "Pending Expressions": len(ar.pending_llm_expressions),
            "Error": ar.error or "",
        } for ar in result.activity_results], use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 4 — Translation Log
# ===========================================================================
with tab_log:
    st.subheader("Ollama Translation Audit Log")
    if not state.translation_log:
        st.info("No Ollama translations have been executed yet.")
    else:
        total = len(state.translation_log)
        succeeded = sum(1 for e in state.translation_log if e.success)
        avg_ms = sum(e.duration_ms for e in state.translation_log) / total

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("Succeeded", succeeded)
        c3.metric("Avg Duration", f"{avg_ms:.0f}ms")

        st.divider()
        st.code(agent.get_translation_log_text(), language="text")

        st.divider()
        st.subheader("Detail View")
        st.dataframe([{
            "Activity": e.activity_name,
            "Path": e.json_path,
            "ADF Expression": (e.adf_expression[:60] + "…") if len(e.adf_expression) > 60 else e.adf_expression,
            "PySpark": (e.pyspark_expression[:60] + "…") if len(e.pyspark_expression) > 60 else e.pyspark_expression,
            "Model": e.model, "ms": f"{e.duration_ms:.0f}", "Status": "✅" if e.success else "❌",
        } for e in state.translation_log], use_container_width=True, hide_index=True)
