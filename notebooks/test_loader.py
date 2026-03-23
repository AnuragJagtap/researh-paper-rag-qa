from src.ingestion.pdf_loader import load_and_clean_documents

docs = load_and_clean_documents("data/raw")

print(f"Loaded {len(docs)} documents")

print("\nSample Output:\n")
print(docs[0]["text"][:1000])