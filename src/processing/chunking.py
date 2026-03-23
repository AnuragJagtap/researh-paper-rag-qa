import json
import os


class TextChunker:
    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        """
        Split text into chunks with overlap
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            chunks.append(chunk)

            start += self.chunk_size - self.overlap

        return chunks

    def process_documents(self, documents):
        """
        Convert documents into chunks with metadata
        """
        all_chunks = []

        for doc in documents:
            text = doc["text"]
            source = doc["source"]
            page = doc["page"]

            chunks = self.chunk_text(text)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "chunk_id": f"{source}_{page}_{i}",
                    "source": source,
                    "page": page,
                    "text": chunk
                })

        return all_chunks

    def save_chunks(self, chunks, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        print(f"[INFO] Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    input_path = "data/processed/raw_texts.json"
    output_path = "data/processed/chunks.json"

    if not os.path.exists(input_path):
        raise FileNotFoundError("Run PDF loader first!")

    with open(input_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    chunker = TextChunker(chunk_size=500, overlap=100)
    chunks = chunker.process_documents(documents)
    chunker.save_chunks(chunks, output_path)