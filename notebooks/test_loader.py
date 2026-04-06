from src.vectorstore.faiss_store import FAISSVectorStore
from src.embedding.embedder import Embedder

print("\n🔷 Loading FAISS index...")

embedder = Embedder()

# dimension must match your model (MiniLM = 384)
vector_store = FAISSVectorStore(dimension=384)
vector_store.load()

print("✅ Index loaded successfully")