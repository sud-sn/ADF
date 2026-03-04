# ADF-to-PySpark Transpiler Service Guide

This document explains how to start and stop the required services (Streamlit UI and Ollama AI engine) for the ADF-to-PySpark transpiler on your local Windows machine.

---

## 🟢 Starting the Services

To run the application, you need to start two separate services. You can do this by opening **two separate PowerShell windows**.

### Terminal 1: Start Ollama (The AI Engine)
Ollama handles translating the complex ADF expressions into PySpark code.

1. Open PowerShell.
2. Run the following command:
   ```powershell
   ollama serve
   ```
   *(Keep this terminal open in the background. If you see a message saying "bind: Only one usage of each socket address", Ollama is already running, which is fine!)*

### Terminal 2: Start Streamlit (The Web UI)
Streamlit hosts the user interface where you upload JSON files and see the results.

1. Open a **new** PowerShell window.
2. Navigate to the project directory:
   ```powershell
   cd c:\Users\sutharsans\Documents\Repo\ADF-master
   ```
3. Start the app:
   ```powershell
   streamlit run app.py
   ```
4. A browser window will automatically open at `http://localhost:8501`.

*(Note: The first translation might be slightly slower as Ollama loads the `qwen2.5-coder:7b` model into memory.)*

---

## 🔴 Stopping the Services

When you are done testing, it is highly recommended to stop both services to free up your computer's RAM and CPU.

### Method 1: The Quick Stop Command
You can forcibly stop both Streamlit and Ollama from any PowerShell window:

```powershell
Get-Process | Where-Object { $_.MainWindowTitle -match "Streamlit" -or $_.Name -match "python" -or $_.Name -match "ollama" } | Stop-Process -Force
```

### Method 2: Manual Shutdown
1. Go to the terminal running **Streamlit** and press `Ctrl + C`. This stops the UI.
2. Go to the terminal running **Ollama** and press `Ctrl + C`.
3. Check your Windows System Tray (the small icons near the clock on the bottom right of your screen). If you see the Ollama icon (a cute little llama face), right-click it and select **Quit Ollama**.
