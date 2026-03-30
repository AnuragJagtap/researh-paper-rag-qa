import os

from src.ingestion.pdf_loader import load_and_clean_documents
from src.ingestion.chunker import chunk_text
from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSVectorStore


def main():
    print("\n🔷 Step 1: Loading documents...")
    docs = load_and_clean_documents("data/raw")
    print(f"✅ Loaded {len(docs)} documents")

    print("\n🔷 Step 2: Chunking documents...")
    all_chunks = []
    metadata = []

    for doc in docs:
        chunks = chunk_text(doc["text"])

        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({
                "source": doc["file_name"]
            })

    print(f"✅ Total chunks created: {len(all_chunks)}")

    print("\n🔷 Step 3: Generating embeddings...")
    embedder = Embedder()
    embeddings = embedder.encode(all_chunks)
    print("✅ Embeddings generated")

    print("\n🔷 Step 4: Creating FAISS index...")
    dimension = embeddings.shape[1]
    vector_store = FAISSVectorStore(dimension)

    vector_store.add_embeddings(embeddings, all_chunks, metadata)
    print("✅ FAISS index created")

    print("\n🔷 Step 5: Testing retrieval...")

    while True:
        query = input("\n💬 Enter your question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        query_embedding = embedder.encode([query])
        results = vector_store.search(query_embedding, k=3)

        print("\n🔍 Top Results:\n")

        for i, res in enumerate(results):
            print(f"--- Result {i+1} ---")
            print(f"📄 Source: {res['metadata']['source']}")
            print(f"🧠 Text: {res['text'][:400]}")
            print()


if __name__ == "__main__":
    main()