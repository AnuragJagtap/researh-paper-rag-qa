import faiss
import numpy as np


class FAISSVectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.data = []  # store metadata + text

    def add_embeddings(self, embeddings, texts, metadata=None):
        self.index.add(np.array(embeddings))

        for i, text in enumerate(texts):
            self.data.append({
                "text": text,
                "metadata": metadata[i] if metadata else {}
            })

    def search(self, query_embedding, k=3):
        D, I = self.index.search(np.array(query_embedding), k)

        results = [self.data[i] for i in I[0]]
        return results