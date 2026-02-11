import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def ingest_pdfs(pdf_folder="data/raw/pdfs", output_folder="data/raw/processed_pdfs"):
    """
    Ingest all PDFs from a folder and save as .txt files.
    """
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file)
            text = extract_text_from_pdf(pdf_path)
            out_file = os.path.join(output_folder, file.replace(".pdf", ".txt"))
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Ingested {file} → {out_file}")

if __name__ == "__main__":
    ingest_pdfs()
