
from pathlib import Path
import hashlib
import csv
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from docx import Document as WordDocument

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ollama embedding model
from langchain_ollama import OllamaEmbeddings

# Chroma vector database
from langchain_chroma import Chroma


# ---------------------------------
# 1. PROJECT PATHS
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DOCUMENT_FOLDER = BASE_DIR / "data" / "knowledge_base"
CHROMA_FOLDER = BASE_DIR / "chroma_db"

COLLECTION_NAME = "knowledge_base"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".html",
    ".htm",
    ".rtf",
}


# ---------------------------------
# 2. LOAD DOCUMENTS
# ---------------------------------

def load_documents():

    print("\nLoading documents...")

    document_files = sorted(
        path for path in DOCUMENT_FOLDER.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not document_files:
        raise FileNotFoundError(
            f"No supported documents found in: {DOCUMENT_FOLDER}"
        )

    all_documents = []

    for document_file in document_files:

        print(f"Reading: {document_file.name}")

        documents = load_document(document_file)
        all_documents.extend(documents)

        print(f"Sections extracted: {len(documents)}")

    if not all_documents:
        raise ValueError("No readable text found in the documents.")

    print(f"\nTotal sections loaded: {len(all_documents)}")

    return all_documents


def load_document(document_path: Path):

    extension = document_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type: {extension or 'unknown'}"
        )

    if extension == ".pdf":
        documents = PyPDFLoader(str(document_path)).load()
    elif extension == ".docx":
        word_document = WordDocument(str(document_path))
        content = "\n".join(
            paragraph.text
            for paragraph in word_document.paragraphs
            if paragraph.text.strip()
        )
        documents = [Document(page_content=content)]
    elif extension == ".csv":
        rows = list(csv.reader(document_path.read_text(encoding="utf-8-sig").splitlines()))
        content = "\n".join(" | ".join(row) for row in rows)
        documents = [Document(page_content=content)]
    elif extension == ".json":
        content = json.dumps(
            json.loads(document_path.read_text(encoding="utf-8")),
            indent=2,
            ensure_ascii=False,
        )
        documents = [Document(page_content=content)]
    else:
        content = document_path.read_text(encoding="utf-8")
        documents = [Document(page_content=content)]

    documents = [
        document for document in documents
        if document.page_content.strip()
    ]

    for document in documents:
        document.metadata["source_file"] = document_path.name

    return documents


# ---------------------------------
# 3. SPLIT DOCUMENTS INTO CHUNKS
# ---------------------------------

def split_documents(documents):

    print("\nSplitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


# ---------------------------------
# 4. GENERATE EMBEDDINGS
# ---------------------------------

def create_embedding_model():

    print("\nInitializing Ollama embedding model...")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    print("Embedding model initialized.")

    return embeddings


# ---------------------------------
# 5. STORE VECTORS IN CHROMADB
# ---------------------------------

def store_in_chromadb(chunks, embeddings):

    print("\nConnecting to ChromaDB...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_FOLDER)
    )

    # Stable IDs help prevent duplicate records
    # when the same document is ingested again.
    chunk_ids = []

    for chunk in chunks:

        identity = (
            f"{chunk.metadata.get('source_file', '')}|"
            f"{chunk.metadata.get('page', '')}|"
            f"{chunk.page_content}"
        )

        chunk_id = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        chunk_ids.append(chunk_id)

    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids
    )

    print(f"Successfully stored {len(chunks)} chunks!")

    return vector_store


# ---------------------------------
# 6. TEST THE STORED KNOWLEDGE
# ---------------------------------

def test_vector_database(vector_store):

    print("\nTesting ChromaDB retrieval...")

    question = "What is the return policy?"

    results = vector_store.similarity_search(
        question,
        k=2
    )

    print(f"\nQuestion: {question}")

    print("\nRetrieved information:")

    for result in results:

        print("\n-----------------------------")

        print(result.page_content)

        print("Source:", result.metadata.get("source_file"))

        print("Page:", result.metadata.get("page"))


# ---------------------------------
# 7. MAIN FUNCTION
# ---------------------------------

def main():

    try:

        documents = load_documents()

        chunks = split_documents(documents)

        embeddings = create_embedding_model()

        vector_store = store_in_chromadb(
            chunks,
            embeddings
        )

        test_vector_database(vector_store)

        print("\n================================")
        print("DOCUMENT INGESTION COMPLETED!")
        print("================================")

    except Exception as error:

        print("\nERROR:", error)

        print(
            "\nCheck that Ollama is running and "
            "the embedding model is installed."
        )



# =====================================
# INGEST A SINGLE UPLOADED DOCUMENT
# =====================================

def ingest_single_document(document_path: Path):

    print(f"\nIngesting uploaded document: {document_path.name}")

    documents = load_document(document_path)
    if not documents:
        raise ValueError("No readable text found in document.")

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No document chunks were created.")

    embeddings = create_embedding_model()

    store_in_chromadb(chunks, embeddings)

    print(f"Successfully indexed {len(chunks)} chunks.")

    return len(chunks)
if __name__ == "__main__":
    main()
