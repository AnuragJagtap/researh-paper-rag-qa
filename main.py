from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSVectorStore
from src.generation.generator import Generator
from src.utils.text_preprocessing import clean_query
from textblob import TextBlob


def main():
    print("\n🚀 Research Paper RAG System (Hybrid Search)\n")

    # 🔹 Load components
    print("🔷 Loading system...")

    embedder = Embedder()

    vector_store = FAISSVectorStore(dimension=384)
    vector_store.load()

    generator = Generator()

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

        # 🔹 Show interpretation (only if changed)
        if corrected_query != query:
            print(f"\n🔧 Interpreted query: {corrected_query}")

        # 🔹 Step 3: Embed query
        query_embedding = embedder.encode([corrected_query])

        # 🔹 Step 4: Hybrid retrieval
        results = vector_store.search(query_embedding, corrected_query, k=3)

        # 🔹 Step 5: Generate answer
        answer = generator.generate_answer(corrected_query, results)

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
            for res in results:
                print(f"- {res['metadata']['source']}")

        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()