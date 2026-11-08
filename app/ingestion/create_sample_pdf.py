
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Find the main project folder
BASE_DIR = Path(__file__).resolve().parents[2]

# PDF output location
PDF_PATH = BASE_DIR / "data" / "knowledge_base" / "sample_knowledge_base.pdf"

# Create the folder if it doesn't exist
PDF_PATH.parent.mkdir(parents=True, exist_ok=True)

# Sample knowledge-base content
lines = [
    "EXAMPLE KNOWLEDGE BASE",
    "",
    "1. DOCUMENT OVERVIEW",
    "This document contains example information for testing retrieval.",
    "Upload your own documents to build a custom knowledge base.",
    "",
    "2. DOCUMENT TYPES",
    "The application supports PDF, DOCX, TXT, Markdown, CSV, JSON, HTML and RTF files.",
    "Documents are split into searchable sections before indexing.",
    "",
    "3. QUESTION ANSWERING",
    "Ask questions about the information contained in your uploaded documents."
]

# Generate PDF
pdf = canvas.Canvas(str(PDF_PATH), pagesize=letter)

text = pdf.beginText(50, 750)
text.setFont("Helvetica", 11)
text.setLeading(18)

for line in lines:
    text.textLine(line)

pdf.drawText(text)
pdf.save()

print("Sample knowledge-base document created successfully!")
print(f"File location: {PDF_PATH}")
