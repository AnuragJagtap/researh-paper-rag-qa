# 📄 Research Paper QA System (RAG)

# 🚀 Overview
A **production-grade Retrieval-Augmented Generation (RAG) system that enables users to query research papers using natural language and receive detailed, citation-backed answers**.

This system combines:

- **Semantic Search (FAISS)**
- **Keyword Search (BM25)**
- **Graph-based Retrieval (Knowledge Graph)**
- **LLM-based Answer Generation (OpenRouter)**

# 🧠 Features
- Hybrid Retrieval (FAISS + BM25)
- Graph RAG (Entity-Relationship extraction)
- Citation-aware answers ([Source 1])
- Query correction & normalization
- Persistent vector storage (FAISS)
- Streamlit UI (interactive app)
- Source deduplication & alignment
- Robust to typos and noisy queries

# System Architecture

User Query

↓

Query Preprocessing (cleaning + spell correction)

↓

Embedding Model

↓

Hybrid Retrieval

  ├── FAISS (semantic)

  ├── BM25 (keyword)

  └── Graph (entity relations)

↓

Context Fusion

↓

LLM (OpenRouter)

↓

Answer + Citations

# 📂 Project Structure

researh-paper-rag-qa/

│

├── app/

  │   └── streamlit_app.py       # Streamlit UI
  
│

├── src/

  │   ├── ingestion/            # PDF loading & chunking
  
  │   ├── embedding/            # Sentence Transformers
  
  │   ├── vectorstore/          # FAISS + BM25 hybrid search
  
  │   ├── generation/           # LLM + prompt engineering
  
  │   ├── graph/                # Graph RAG (triplets)
  
  │   └── utils/                # preprocessing
  
│

├── notebooks/

  │   ├── build_index.py        # Build FAISS index
  
  │   └── build_graph.py        # Build knowledge graph
  
│

├── data/

  │   ├── raw/                  # Input PDFs

  │   └── embeddings/           # Saved FAISS index

│

├── main.py                   # CLI interface

├── requirements.txt

└── README.md


# ⚙️ Installation

## 1. Clone the repository
     git clone https://github.com/your-username/research-paper-rag.git
     cd research-paper-rag
   
## 2. Create Virtual Enviornment
    python -m venv .venv
    .venv\Scripts\activate   # Windows

## 3. Install Dependencies
    pip install -r requirements.txt
## 4. Download SpaCy Model
    python -m spacy download en_core_web_sm
## 5. Set API Key (OpenRouter)
  Create .env
    OPENROUTER_API_KEY=your_api_key_here
    
# Build Index and Graph

## Step 1: Build FAISS index (one-time)
    python -m notebooks.build_index

## Step 2: Build Knowledge Graph
    python -m notebooks.build_graph

# Run The System 

## CLI Mode
    python main.py

## StreamLit UI 
    streamlit run app/streamlit_app.py

# 💬 Example Output

🧠**Answer**

**Definition / Overview**: The Transformer architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. [Source 1]. It is a type of neural network architecture used for natural language processing tasks.

**Key Explanation**: The Transformer architecture is composed of stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, as shown in Figure 1 [Source 2]. The encoder takes in a sequence of symbols as input and generates a continuous representation of the input sequence. The decoder then generates an output sequence of symbols one element at a time, consuming the previously generated symbols as additional input when generating the next [Source 2].

**Important Points**: The Transformer architecture is used in the BERT model, which is a pre-trained architecture that is fine-tuned for downstream tasks [Source 1]. The architecture has been shown to perform well on various natural language processing tasks, including language translation, text classification, and question answering. Additionally, the Transformer architecture has been compared to other architectures, such as LSTM, and has been found to outperform them in many cases [Source 3].

📚 **Sources**

[Source 1] BERT_Pretraining.pdf

[Source 2] Attention_is_all_you_need.pdf

[Source 3] language_understanding_paper.pdf


# 🔍 Key Technologies

- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector DB**: FAISS
- **Keyword Search**: BM25
- **Graph**: NetworkX
- **LLM**: OpenRouter (LLaMA / Mistral)
- **UI**: Streamlit
- **NLP**: SpaCy, TextBlob

---

# 🧪 Key Improvements Over Basic RAG

| Feature | Basic RAG | This Project |
|--------|----------|-------------|
| Retrieval | Semantic only | Hybrid (Semantic + BM25) |
| Structure | Flat text | Graph-based knowledge |
| Query handling | Raw | Cleaned + corrected |
| Output | Plain | Citation-aware |
| Storage | Recomputed | Persistent |
| UI | None | Streamlit |

---

# 📈 Future Improvements

- 🔹 Neo4j integration (graph database)
- 🔹 LLM-based triplet extraction
- 🔹 Re-ranking (cross-encoder)
- 🔹 Multi-document QA
- 🔹 Deployment (Streamlit Cloud / Render)

---

# 🎯 What I Learned

- Designing **end-to-end RAG pipelines**
- Hybrid retrieval strategies
- Graph-based knowledge integration
- Prompt engineering for grounded answers
- Building production-like NLP systems

---

# 🤝 Contributing

Feel free to fork, improve, and raise PRs!

---

# ⭐ If You Like This Project

Give it a ⭐ on GitHub—it helps!

---

# 🔥 Resume-Ready Line

> Built a production-grade Hybrid Graph RAG system combining FAISS, BM25, and knowledge graphs with LLM-based answer generation and citation alignment.
