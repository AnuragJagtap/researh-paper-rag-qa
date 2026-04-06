import sys
import os
import streamlit as st

# 🔥 Fix import path for Streamlit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSVectorStore
from src.generation.generator import Generator
from src.utils.text_preprocessing import clean_query
from textblob import TextBlob


# 🔷 Page Config
st.set_page_config(
    page_title="Research Paper RAG QA",
    layout="wide"
)

st.title("📄 Research Paper RAG QA System")
st.markdown("Ask questions based on research papers")

# 🔷 Load system (cached)
@st.cache_resource
def load_system():
    embedder = Embedder()

    vector_store = FAISSVectorStore(dimension=384)
    vector_store.load()

    generator = Generator()

    return embedder, vector_store, generator


embedder, vector_store, generator = load_system()

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

        # 🔹 Step 4: Hybrid search
        results = vector_store.search(query_embedding, corrected_query, k=3)

        # 🔹 Step 5: Generate answer
        answer = generator.generate_answer(corrected_query, results)

    # 🔷 Show query interpretation (only if changed)
    if corrected_query != query:
        st.markdown(f"🔧 **Interpreted query:** `{corrected_query}`")

    # 🔷 Display Answer
    st.subheader("🧠 Answer")
    st.write(answer)

    # 🔷 Show sources only if answer is confident
    if not any(phrase in answer.lower() for phrase in [
        "i don't know",
        "i do not know",
        "not enough information"
    ]):
        st.subheader("📚 Sources")
        for res in results:
            st.markdown(f"- **{res['metadata']['source']}**")