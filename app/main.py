
from pathlib import Path
from uuid import uuid4
import logging
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

# Import our RAG function
from app.services.rag_chain import answer_question

# Import document ingestion
from app.ingestion.ingest import (
    SUPPORTED_EXTENSIONS,
    ingest_single_document,
)


# =====================================
# 1. CONFIGURATION
# =====================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]

DOCUMENT_FOLDER = BASE_DIR / "data" / "knowledge_base"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# =====================================
# 2. CREATE FASTAPI APP
# =====================================

app = FastAPI(
    title="Retrieval augmented generation",
    description="Retrieval augmented generation backend using Ollama",
    version="1.0.0"
)


# =====================================
# 3. REQUEST AND RESPONSE MODELS
# =====================================

class ChatRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Customer question"
    )


class ChatResponse(BaseModel):

    answer: str

    sources: list[str]


class UploadResponse(BaseModel):

    filename: str

    chunks_indexed: int

    message: str


# =====================================
# 4. API ROOT
# =====================================

@app.get("/")
def root():

    return {
        "message": "Retrieval augmented generation API is running.",
        "health": "/health",
        "docs": "/docs"
    }


# =====================================
# 5. HEALTH CHECK API
# =====================================

@app.get("/health")
def health_check():

    return {
        "status": "success",
        "message": "Retrieval augmented generation API is running!"
    }


# =====================================
# 6. QUESTION API
# =====================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty."
        )

    try:

        logger.info("Question received.")

        # Call our existing RAG pipeline
        answer, sources = answer_question(question)

        return ChatResponse(
            answer=answer,
            sources=sources
        )

    except Exception as error:

        logger.exception("Chat API failed.")

        raise HTTPException(
            status_code=503,
            detail="Unable to process your question. Check the document service."
        ) from error


# =====================================
# 7. DOCUMENT UPLOAD API
# =====================================

@app.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):

    original_name = file.filename or ""

    # Validate extension
    safe_original_name = original_name.replace("\\", "/").split("/")[-1]

    extension = Path(safe_original_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported_types = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type. Use: {supported_types}."
        )

    # Check file size
    file.file.seek(0, 2)

    file_size = file.file.tell()

    file.file.seek(0)

    if file_size == 0:

        raise HTTPException(
            status_code=400,
            detail="Uploaded document is empty."
        )

    if file_size > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail="Documents must be 10 MB or smaller."
        )

    if extension == ".pdf":
        header = file.file.read(5)
        file.file.seek(0)
        if header != b"%PDF-":
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file."
            )

    # Create knowledge-base directory
    DOCUMENT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Generate a unique safe filename
    new_filename = f"{uuid4().hex}_{safe_original_name}"

    destination = DOCUMENT_FOLDER / new_filename

    try:

        # Save document
        with destination.open("wb") as output:

            shutil.copyfileobj(file.file, output)

        logger.info("Document saved: %s", new_filename)

        # Process and store embeddings
        chunk_count = ingest_single_document(destination)

        return UploadResponse(
            filename=new_filename,
            chunks_indexed=chunk_count,
            message="Document uploaded and indexed successfully!"
        )

    except Exception as error:

        logger.exception("Document ingestion failed.")

        raise HTTPException(
            status_code=500,
            detail="Document was saved but ingestion failed. Check server logs."
        ) from error

    finally:

        file.file.close()
