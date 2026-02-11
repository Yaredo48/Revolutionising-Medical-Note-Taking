import os
from PyPDF2 import PdfReader

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def ingest_pdfs():
    pdf_folder = os.path.join(BASE_DIR, "data/raw/pdfs")
    output_folder = os.path.join(BASE_DIR, "data/raw/processed_pdfs")

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
