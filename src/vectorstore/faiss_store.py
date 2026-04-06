import faiss
import numpy as np
import pickle
import os


class FAISSVectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.data = []

    def add_embeddings(self, embeddings, texts, metadata=None):
        self.index.add(np.array(embeddings))

        for i, text in enumerate(texts):
            self.data.append({
                "text": text,
                "metadata": metadata[i] if metadata else {}
            })

    def search(self, query_embedding, k=3):
        D, I = self.index.search(np.array(query_embedding), k)
        return [self.data[i] for i in I[0]]

    # 🔥 NEW: Save index
    def save(self, path="data/embeddings"):
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "faiss.index"))

        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.data, f)

    # 🔥 NEW: Load index
    def load(self, path="data/embeddings"):
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))

        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            self.data = pickle.load(f)