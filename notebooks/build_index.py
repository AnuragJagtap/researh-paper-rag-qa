from src.ingestion.pdf_loader import load_and_clean_documents
from src.ingestion.chunker import chunk_text
from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSVectorStore

def main():
    print("🔷 Building FAISS Index...")

    docs = load_and_clean_documents("data/raw")

    all_chunks = []
    metadata = []

    for doc in docs:
        chunks = chunk_text(doc["text"])

        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({"source": doc["file_name"]})

    print(f"Total chunks: {len(all_chunks)}")

    embedder = Embedder()
    embeddings = embedder.encode(all_chunks)

    dimension = embeddings.shape[1]
    vector_store = FAISSVectorStore(dimension)

    vector_store.add_embeddings(embeddings, all_chunks, metadata)

    # 🔥 Save index
    vector_store.save()

    print("✅ Index built and saved successfully!")


if __name__ == "__main__":
    main()