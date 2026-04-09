import sys
import os
import streamlit as st

# 🔥 Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSVectorStore
from src.generation.generator import Generator
from src.utils.text_preprocessing import clean_query
from src.graph.graph_store import GraphStore
from textblob import TextBlob


# 🔷 Page config
st.set_page_config(
    page_title="Research Paper RAG QA",
    layout="wide"
)

st.title("📄 Research Paper Graph RAG QA System")
st.markdown("Ask questions based on research papers")

# 🔷 Load system (cached)
@st.cache_resource
def load_system():
    embedder = Embedder()

    vector_store = FAISSVectorStore(dimension=384)
    vector_store.load()

    generator = Generator()

    # 🔹 Load graph (if available)
    graph_store = GraphStore()
    try:
        graph_store.load()
    except:
        pass

    return embedder, vector_store, generator, graph_store


embedder, vector_store, generator, graph_store = load_system()

# 🔷 Input
query = st.text_input("🔍 Ask a question:")

if query:
    with st.spinner("Processing..."):

        # 🔹 Step 1: Clean query
        cleaned_query = clean_query(query)

        # 🔹 Step 2: Spell correction
        corrected_query = str(TextBlob(cleaned_query).correct())

        # 🔹 Step 3: Embed query
        query_embedding = embedder.encode([corrected_query])

        # 🔹 Step 4: Hybrid retrieval
        vector_results = vector_store.search(query_embedding, corrected_query, k=5)

        # 🔹 Step 5: Graph retrieval
        graph_results = graph_store.query(corrected_query)

        # 🔹 Step 6: Combine results
        combined_results = vector_results.copy()

        for gr in graph_results[:2]:
            combined_results.append({
                "text": gr,
                "metadata": {"source": "graph"}
            })

        # 🔹 Step 7: Generate answer
        answer, source_map = generator.generate_answer(corrected_query, combined_results)

    # 🔷 Show interpreted query
    if corrected_query != query:
        st.markdown(f"🔧 **Interpreted query:** `{corrected_query}`")

    # 🔷 Display answer
    st.subheader("🧠 Answer")
    st.write(answer)

    # 🔷 Show sources only if confident
    if not any(phrase in answer.lower() for phrase in [
        "i don't know",
        "i do not know",
        "not enough information"
    ]):
        st.subheader("📚 Sources")

        # 🔹 Sorted sources (correct numbering)
        sorted_sources = sorted(source_map.items(), key=lambda x: x[1])

        for source, source_id in sorted_sources:
            st.markdown(f"[Source {source_id}] **{source}**")