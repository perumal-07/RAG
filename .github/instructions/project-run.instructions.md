---
name: "Project Run Instructions"
description: "Use when running, testing, or troubleshooting this document retrieval application. Covers the FastAPI backend, Streamlit frontend, Ollama models, document uploads, and pytest commands."
applyTo: "**/*.py"
---

# Project run instructions

- Run commands from the repository root.
- Use the project virtual environment:
  `.\.venv\Scripts\Activate.ps1`
- Start the API with:
  `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`
- Start the UI in a second terminal with:
  `python -m streamlit run app/frontend.py`
- Confirm the API is available at `http://127.0.0.1:8000/health` before using the UI.
- Keep Ollama running with `nomic-embed-text` and `llama3.2:3b` installed.
- Run tests with:
  `python -m pytest -v`
- Supported uploads are PDF, DOCX, TXT, Markdown, CSV, JSON, HTML, and RTF.
- Keep uploaded documents under the 10 MB API limit.
- Do not commit credentials, local model data, or generated ChromaDB files.
