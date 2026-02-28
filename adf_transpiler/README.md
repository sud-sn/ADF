# ADF-to-PySpark Local Converter

> **Deterministic parser + LLM-assisted transpiler** that converts Azure Data Factory pipeline JSONs into production-ready PySpark code — 100% locally, zero data leakage.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI  (app.py)                   │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
             ▼                                    ▼
┌────────────────────────┐           ┌────────────────────────────┐
│  core/adf_parser.py    │           │  agents/ollama_agent.py     │
│  ─ Pydantic models     │──────────▶│  ─ AsyncOllamaAgent         │
│  ─ DAG resolution      │           │  ─ ADF expr → PySpark       │
│  ─ parse_pipeline_json │           │  ─ Streaming responses      │
└────────────┬───────────┘           └────────────────────────────┘
             │
             ▼
┌────────────────────────┐
│  core/transpiler.py    │
│  ─ ActivityTranspiler  │
│  ─ Template mapping    │
│  ─ PySpark codegen     │
└────────────────────────┘
```

---

## Project Structure

```
adf_transpiler/
├── app.py                        ← Streamlit entry point (run this)
├── requirements.txt
├── README.md
├── .gitignore
│
├── core/
│   ├── __init__.py
│   ├── adf_parser.py             ← Pydantic models + pipeline parser
│   └── transpiler.py             ← Deterministic PySpark template engine
│
├── agents/
│   ├── __init__.py
│   └── ollama_agent.py           ← Async Ollama LLM agent
│
├── ui/
│   └── app.py                    ← Full Streamlit UI implementation
│
└── sample_data/
    └── sample_pipeline.json      ← Realistic ADF pipeline for demo/testing
```

---

## Quick Start

### 1. Prerequisites

| Requirement              | Version | Notes                                                    |
| ------------------------ | ------- | -------------------------------------------------------- |
| Python                   | 3.10+   | Required for `match`/`case` and new type hints           |
| Ollama                   | Latest  | [https://ollama.ai/download](https://ollama.ai/download) |
| `qwen2.5-coder:7b` model | —       | `ollama pull qwen2.5-coder:7b`                           |

### 2. Install Python dependencies

```bash
cd adf_transpiler
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start Ollama

```bash
# In a separate terminal
ollama serve

# Pull the model (first time only)
ollama pull qwen2.5-coder:7b
# Or for alternatives:
ollama pull codellama
ollama pull llama3
```

### 4. Run the app

```bash
streamlit run app.py
```

The browser opens at `http://localhost:8501`.

---

## Usage

1. **Upload** your ADF pipeline JSON (exported from ADF Studio).
2. **Parse** — the deterministic engine validates the schema and builds the activity DAG.
3. **Transpile** — generates PySpark boilerplate for all activities.
4. **Translate** — Ollama translates embedded ADF expressions (`@concat(...)`, etc.) to Python.
5. **Download** the final `.py` file.

---

## Supported Activity Types

| ADF Type          | Transpiler Support | Template Output                                  |
| ----------------- | ------------------ | ------------------------------------------------ |
| `Copy`            | ✅ Full            | `spark.read.format(...).load()` + `write.save()` |
| `ForEach`         | ✅ Full            | Python `for` loop with inner activity stubs      |
| `IfCondition`     | ✅ Full            | Python `if/else` block                           |
| `SetVariable`     | ✅ Full            | `pipeline_vars["x"] = ...`                       |
| `Wait`            | ✅ Full            | `time.sleep(n)`                                  |
| `Lookup`          | ⚠ Generic          | Raw typeProperties preserved                     |
| `ExecutePipeline` | ⚠ Generic          | Raw typeProperties preserved                     |
| All others        | ⚠ Generic          | Raw typeProperties preserved with TODO           |

---

## ADF Expression Translation (Ollama)

ADF dynamic expressions like:

```
@concat('SELECT * FROM customers WHERE dt > ''', pipeline().parameters.watermark_date, '''')
```

Are sent to your local Ollama model and translated to:

```python
f"SELECT * FROM customers WHERE dt > '{pipeline_params['watermark_date']}'"
```

**No data leaves your machine.**

---

## Development

```bash
# Lint
ruff check .

# Type check
mypy core/ agents/

# Format
black .

# Tests (add your own in tests/)
pytest
```

---

## Configuration

| Setting      | Default                  | Description                                    |
| ------------ | ------------------------ | ---------------------------------------------- |
| Ollama Model | `qwen2.5-coder:7b`       | Change in sidebar — must be pulled locally     |
| Ollama URL   | `http://localhost:11434` | Change if Ollama runs on a different port/host |

---

## Extending

- **New activity type**: Add a subclass in `core/adf_parser.py` and a handler in `core/transpiler.py`.
- **New template**: Add a `_transpile_<type>()` function and register it in `_SPECIFIC_HANDLERS`.
- **Different LLM**: Swap the Ollama model in the sidebar — the agent is model-agnostic.
