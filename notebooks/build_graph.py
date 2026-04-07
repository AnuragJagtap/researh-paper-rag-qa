from src.ingestion.pdf_loader import load_and_clean_documents
from src.ingestion.chunker import chunk_text
from src.graph.entity_extractor import extract_triplets
from src.graph.graph_store import GraphStore


def main():
    docs = load_and_clean_documents("data/raw")

    graph_store = GraphStore()

    for doc in docs:
        chunks = chunk_text(doc["text"])

        for chunk in chunks:
            triplets = extract_triplets(chunk)
            graph_store.add_triplets(triplets)

    print("✅ Graph built successfully!")

    return graph_store


if __name__ == "__main__":
    main()