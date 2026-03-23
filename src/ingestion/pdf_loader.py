import fitz  # PyMuPDF
import os
from .text_cleaner import clean_text

def extract_text_from_pdf(pdf_path):
    """
    Extracts clean text from a PDF using PyMuPDF
    """
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"

    return text


def load_pdfs_from_folder(folder_path):
    """
    Loads all PDFs from a folder and returns a list of documents
    """
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            text = extract_text_from_pdf(full_path)

            documents.append({
                "file_name": file,
                "text": text
            })

    return documents


def load_and_clean_documents(folder_path):
    documents = load_pdfs_from_folder(folder_path)

    cleaned_docs = []

    for doc in documents:
        cleaned_text = clean_text(doc["text"])

        cleaned_docs.append({
            "file_name": doc["file_name"],
            "text": cleaned_text
        })

    return cleaned_docs