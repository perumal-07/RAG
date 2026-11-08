# Retrieval augmented generation

A document question-answering application that combines a FastAPI backend,
Streamlit frontend, Ollama models, LangChain, and ChromaDB.

Upload documents, index their contents, and ask questions about the
information they contain.

## Features

- Upload and index PDF, DOCX, TXT, Markdown, CSV, JSON, HTML, and RTF files.
- Ask questions through a Streamlit chat interface.
- Retrieve relevant document chunks from ChromaDB.
- Generate answers with an Ollama chat model.
- Use the FastAPI endpoints directly or through the frontend.

## Requirements

- Python 3.13 or a compatible supported Python version.
- Ollama installed and running.
- Ollama models:

  ```powershell
  ollama pull nomic-embed-text
  ollama pull llama3.2:3b
  ```

## Installation

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands with
`.venv\Scripts\python.exe` instead.

## Run the application

Use two terminals from the project directory.

### Terminal 1: FastAPI backend

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend URLs:

- API root: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- API documentation: http://127.0.0.1:8000/docs

### Terminal 2: Streamlit frontend

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app/frontend.py
```

Open the Streamlit URL shown in the terminal, normally
http://localhost:8501.


### Unsupported document

Use one of these extensions:
`.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.json`, `.html`, `.htm`, or `.rtf`.
