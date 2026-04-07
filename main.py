from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSVectorStore
from src.generation.generator import Generator
from src.utils.text_preprocessing import clean_query
from src.graph.graph_store import GraphStore
from textblob import TextBlob


def main():
    print("\n🚀 Research Paper Graph RAG System\n")

    # 🔹 Load components
    print("🔷 Loading system...")

    embedder = Embedder()

    vector_store = FAISSVectorStore(dimension=384)
    vector_store.load()

    generator = Generator()

    # 🔹 Initialize graph (assumes already built in memory or extend later to load)
    graph_store = GraphStore()

    print("✅ System ready!\n")

    print("💬 Ask your questions (type 'exit' to quit)\n")

    while True:
        query = input("🔍 Question: ")

        if query.lower() == "exit":
            print("\n👋 Exiting system. Goodbye!")
            break

        if not query.strip():
            print("⚠️ Please enter a valid question.\n")
            continue

        # 🔹 Step 1: Clean query
        cleaned_query = clean_query(query)

        # 🔹 Step 2: Spell correction
        corrected_query = str(TextBlob(cleaned_query).correct())

        # 🔹 Show interpretation
        if corrected_query != query:
            print(f"\n🔧 Interpreted query: {corrected_query}")

        # 🔹 Step 3: Embed query
        query_embedding = embedder.encode([corrected_query])

        # 🔹 Step 4: Hybrid retrieval (FAISS + BM25)
        vector_results = vector_store.search(query_embedding, corrected_query, k=5)

        # 🔹 Step 5: Graph retrieval
        graph_results = graph_store.query(corrected_query)

        # 🔹 Step 6: Combine contexts
        combined_results = vector_results.copy()

        for gr in graph_results[:2]:  # limit graph results
            combined_results.append({
                "text": gr,
                "metadata": {"source": "graph"}
            })

        # 🔹 Step 7: Generate answer
        answer = generator.generate_answer(corrected_query, combined_results)

        # 🔹 Output
        print("\n" + "=" * 60)
        print("🧠 ANSWER")
        print("=" * 60)
        print(answer)

        # 🔹 Conditional sources
        if not any(phrase in answer.lower() for phrase in [
            "i don't know",
            "i do not know",
            "not enough information"
        ]):
            print("\n📚 SOURCES")
            for res in combined_results:
                print(f"- {res['metadata']['source']}")

        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()