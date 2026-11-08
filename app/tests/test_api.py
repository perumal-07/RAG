
from fastapi.testclient import TestClient

from app import main as api


# Create test client
client = TestClient(api.app)


# =====================================
# TEST 1: HEALTH CHECK
# =====================================

def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"


# =====================================
# TEST 2: CHAT API
# =====================================

def test_chat_api(monkeypatch):

    # Mock the RAG function.
    # This test does not require Ollama.

    def fake_answer_question(question):

        return (
            "Customers can return products within 30 days.",
            ["[Source 1] sample_document.txt - Page unknown"]
        )

    monkeypatch.setattr(
        api,
        "answer_question",
        fake_answer_question
    )

    response = client.post(
        "/chat",
        json={
            "question": "What is the return policy?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "30 days" in data["answer"]

    assert len(data["sources"]) > 0


# =====================================
# TEST 3: EMPTY QUESTION
# =====================================

def test_empty_question():

    response = client.post(
        "/chat",
        json={
            "question": " "
        }
    )

    assert response.status_code == 422


# =====================================
# TEST 4: UNSUPPORTED FILE TYPE
# =====================================

def test_invalid_file_upload():

    response = client.post(
        "/upload",
        files={
            "file": (
                "notes.exe",
                b"Sample text",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


# =====================================
# TEST 5: INVALID PDF CONTENT
# =====================================

def test_invalid_pdf_content():

    response = client.post(
        "/upload",
        files={
            "file": (
                "sample.pdf",
                b"This is not a PDF",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 400


# =====================================
# TEST 6: SUCCESSFUL DOCUMENT UPLOAD
# =====================================

def test_successful_document_upload(monkeypatch, tmp_path):

    # Use a temporary folder instead of real knowledge base
    monkeypatch.setattr(api, "DOCUMENT_FOLDER", tmp_path)

    # Mock ingestion so Ollama is not required
    monkeypatch.setattr(
        api,
        "ingest_single_document",
        lambda path: 3
    )

    response = client.post(
        "/upload",
        files={
            "file": (
                "test.pdf",
                b"%PDF-1.4\nFake test PDF content",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["chunks_indexed"] == 3

    assert data["message"] == "Document uploaded and indexed successfully!"

    # Verify the file was saved
    saved_files = list(tmp_path.glob("*.pdf"))

    assert len(saved_files) == 1


def test_successful_text_upload(monkeypatch, tmp_path):

    monkeypatch.setattr(api, "DOCUMENT_FOLDER", tmp_path)
    monkeypatch.setattr(api, "ingest_single_document", lambda path: 2)

    response = client.post(
        "/upload",
        files={
            "file": (
                "notes.txt",
                b"Example document content.",
                "text/plain"
            )
        }
    )

    assert response.status_code == 200
    assert response.json()["chunks_indexed"] == 2
    assert len(list(tmp_path.glob("*.txt"))) == 1
