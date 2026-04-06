import faiss
import numpy as np
import pickle
import os
from rank_bm25 import BM25Okapi


class FAISSVectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.data = []
        self.bm25 = None
        self.tokenized_corpus = []

    def add_embeddings(self, embeddings, texts, metadata=None):
        self.index.add(np.array(embeddings))

        for i, text in enumerate(texts):
            self.data.append({
                "text": text,
                "metadata": metadata[i] if metadata else {}
            })

        # 🔥 Build BM25 corpus
        self.tokenized_corpus = [doc["text"].lower().split() for doc in self.data]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query_embedding, query_text, k=3, alpha=0.9):
        """
        Hybrid search:
        alpha = weight for semantic search
        (1 - alpha) = weight for BM25
        """

        # 🔹 FAISS search
        D, I = self.index.search(np.array(query_embedding), k * 3)

        faiss_scores = {i: D[0][idx] for idx, i in enumerate(I[0])}

        # 🔹 BM25 search
        tokenized_query = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # 🔹 Combine scores
        combined_scores = {}

        for i in range(len(self.data)):
            faiss_score = faiss_scores.get(i, float('inf'))  # lower is better
            bm25_score = bm25_scores[i]

            # Normalize FAISS (invert distance)
            faiss_score = 1 / (1 + faiss_score)

            score = alpha * faiss_score + (1 - alpha) * bm25_score
            combined_scores[i] = score

        # 🔹 Get top-k
        sorted_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:k]

        return [self.data[i] for i in sorted_indices]

    def save(self, path="data/embeddings"):
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "faiss.index"))

        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.data, f)

    def load(self, path="data/embeddings"):
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))

        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            self.data = pickle.load(f)

        # 🔥 Rebuild BM25 after loading
        self.tokenized_corpus = [doc["text"].lower().split() for doc in self.data]
        self.bm25 = BM25Okapi(self.tokenized_corpus)