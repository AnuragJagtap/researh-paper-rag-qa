from src.ingestion.pdf_loader import load_and_clean_documents
from src.ingestion.chunker import chunk_text

docs = load_and_clean_documents("data/raw")

all_chunks = []

for doc in docs:
    chunks = chunk_text(doc["text"])
    
    for chunk in chunks:
        all_chunks.append({
            "file_name": doc["file_name"],
            "chunk": chunk
        })

print(f"Total chunks created: {len(all_chunks)}")

print("\nSample Chunk:\n")
print(all_chunks[0]["chunk"])