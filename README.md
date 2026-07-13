# 🏢 EstateGPT -- Enterprise Real Estate AI Assistant

An enterprise-grade Retrieval-Augmented Generation (RAG) application
that helps customers, sales teams, and support executives instantly
retrieve accurate information from real estate documents.

------------------------------------------------------------------------

## ✨ Overview

EstateGPT combines semantic search, keyword search, and Large Language
Models to answer questions from your real estate knowledge base. Every
answer is grounded in uploaded documents and includes citations, helping
users verify the information instead of relying on AI-generated guesses.

### Key Highlights

-   🔍 Hybrid Retrieval (FAISS + BM25 + Reciprocal Rank Fusion)
-   🤖 Llama 3.3 via Groq (or any OpenAI-compatible API)
-   💬 Multi-turn conversational memory
-   📄 Source citations for every answer
-   📚 PDF, DOCX, HTML, Markdown, TXT and CSV support
-   ⚡ Streaming responses
-   🌙 Light & Dark mode
-   🔐 Secure login system
-   🐳 Docker ready

------------------------------------------------------------------------

# 🚀 Features

-   Enterprise Streamlit interface
-   Hybrid semantic + keyword retrieval
-   Automatic document chunking
-   Metadata-aware indexing
-   Session-based conversation memory
-   Source attribution and citation cards
-   Hallucination reduction using retrieval gating
-   Extractive mode when no LLM API key is configured
-   Fast FAISS indexing
-   Offline unit tests

------------------------------------------------------------------------

# 🏗 Architecture

Knowledge Base ↓ Document Loader ↓ Chunking & Metadata ↓ Embeddings ↓
FAISS Index + BM25 ↓ Hybrid Retriever ↓ LLM ↓ Grounded Response +
Citations

------------------------------------------------------------------------

# 🛠 Tech Stack

  Category       Technology
  -------------- -----------------------
  Language       Python 3.11
  UI             Streamlit
  Framework      LangChain
  Vector Store   FAISS
  Embeddings     Sentence Transformers
  Retrieval      BM25 + RRF
  LLM            Groq Llama 3.3
  Testing        PyTest
  Container      Docker

------------------------------------------------------------------------

# 📂 Project Structure

``` text
RealEstateRAG/
│
├── app.py
├── src/
├── tests/
├── docs/
├── data/
├── Dockerfile
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

# ⚙ Installation

``` bash
git clone <repository-url>
cd RealEstateRAG

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

------------------------------------------------------------------------

# ▶️ Run

``` bash
streamlit run app.py
```

Default Login

    Username : demo
    Password : realestate2026

------------------------------------------------------------------------

# 🧠 How Retrieval Works

1.  Load documents.
2.  Split into chunks.
3.  Generate embeddings.
4.  Build FAISS index.
5.  Search using FAISS + BM25.
6.  Merge results using Reciprocal Rank Fusion.
7.  Generate a grounded answer with citations.

------------------------------------------------------------------------

# 📌 Example Questions

-   What is the payment plan for Skyline Horizon Towers?
-   Which projects cost under ₹1 crore?
-   Compare the possession dates of Meridian projects.
-   What is Urban Nest's cancellation policy?
-   P52100034899

------------------------------------------------------------------------

# 🔒 Hallucination Protection

The assistant only answers when relevant documents are retrieved. If no
supporting evidence is found, it refuses the request instead of
inventing an answer.

------------------------------------------------------------------------

# 🧪 Testing

``` bash
pytest
ruff check src app.py tests
```

------------------------------------------------------------------------

# 🐳 Docker

``` bash
docker build -t estategpt .
docker run -p 8501:8501 --env-file .env estategpt
```

------------------------------------------------------------------------

# 📈 Future Improvements

-   User management
-   Analytics dashboard
-   Admin panel
-   OCR support
-   Voice interaction
-   Multi-language support
-   Cloud storage integration

------------------------------------------------------------------------

# 📄 License

This project is intended for educational, research, and demonstration
purposes. Add your preferred open-source license before public
distribution.
