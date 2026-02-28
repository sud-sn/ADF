# 🚀 Testing the ADF-to-PySpark Transpiler Locally

To test this in real-time on your system, you need to run two things simultaneously:

1. **Ollama**: The local AI service that runs the model.
2. **Streamlit**: The Python web framework that runs the UI.

Here are the exact step-by-step instructions to get everything running.

---

## Step 1: Install and Start Ollama

Ollama is a separate program that runs in the background to serve the AI models locally.

1. Download and install **Ollama for Windows** from [ollama.com/download](https://ollama.com/download).
2. Once installed, open your terminal (Command Prompt or PowerShell) and pull the default model you'd like to use. For example:
   ```bash
   ollama pull codellama
   ```
   _(This may take a few minutes as `codellama` is a few gigabytes in size)._
3. Verify Ollama is running correctly by verifying the daemon is running in your system tray or by executing:
   ```bash
   ollama serve
   ```

---

## Step 2: Set up the Python Environment

Open a **new** terminal window (Command Prompt or PowerShell), navigate to your project folder (`c:\Users\susan\Documents\Repo\ADF\adf_transpiler`), and set up the Python environment.

1. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment:**

   ```bash
   # On Windows:
   .venv\Scripts\activate
   ```

3. **Install the required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Step 3: Run the Application

With Ollama running in the background and your Python environment activated, you can now start the Streamlit UI.

1. In the same terminal where you activated your `.venv`, run:
   ```bash
   streamlit run app.py
   ```
2. This will start a local web server and automatically open your default web browser to `http://localhost:8501`.

---

## Step 4: Test the Flow

Your environment is now fully set up. To test the flow:

1. Open the UI in your browser.
2. In the Sidebar, click **"🔌 Test Connection"** to ensure the Streamlit app can talk to your local Ollama instance.
3. On the main page (Tab 1), click **"📄 Load Sample"**. This will use the bundled `sample_pipeline.json` file in the project.
4. Go to **"🔍 2 · Pipeline Analysis"** tab and click **"🚀 Generate PySpark Code"**.
5. Go to **"💻 3 · Generated Code"** tab and click **"🤖 Translate X ADF Expressions (Ollama)"**.

You should see the LLM actively replacing the expression placeholders in the code block in real-time. Once you have verified the sample works, you can drop in any of your own exported Azure Data Factory JSON files and test them!
