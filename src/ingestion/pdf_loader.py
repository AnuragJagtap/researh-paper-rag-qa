import os
import json
from PyPDF2 import PdfReader
from tqdm import tqdm


class PDFLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def extract_text_from_pdf(self, file_path):
        """
        Extract text page-by-page with metadata
        """
        reader = PdfReader(file_path)
        pages_data = []

        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    pages_data.append({
                        "page": page_num + 1,
                        "text": text
                    })
            except Exception as e:
                print(f"[ERROR] Page {page_num} in {file_path}: {e}")

        return pages_data

    def clean_text(self, text):
        """
        Basic cleaning (can be extended later)
        """
        text = text.replace("\n", " ")
        text = " ".join(text.split())  # remove extra spaces
        return text

    def load_all_pdfs(self):
        """
        Main function: loads all PDFs and returns structured data
        """
        all_documents = []

        files = [f for f in os.listdir(self.data_path) if f.endswith(".pdf")]

        print(f"[INFO] Found {len(files)} PDF files.")

        for file in tqdm(files):
            file_path = os.path.join(self.data_path, file)

            pages = self.extract_text_from_pdf(file_path)

            for page in pages:
                cleaned_text = self.clean_text(page["text"])

                all_documents.append({
                    "source": file,
                    "page": page["page"],
                    "text": cleaned_text
                })

        return all_documents

    def save_to_json(self, documents, output_path):
        """
        Save processed data
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2)

        print(f"[INFO] Saved processed data to {output_path}")
    


if __name__ == "__main__":
    loader = PDFLoader("data/raw")
    documents = loader.load_all_pdfs()
    loader.save_to_json(documents, "data/processed/raw_texts.json")
    