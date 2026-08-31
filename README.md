# FinBot - Your Banking Assistant 🤖💳

FinBot is a modern, 100% secure, and offline banking intelligence assistant designed to answer user queries accurately regarding bank accounts, loans, credit cards, and financial services.

## 🏗️ Project Architecture

The project follows a decoupled client-server architecture:
* **Backend:** Built with **FastAPI** and Python, utilizing `sentence-transformers` (`all-MiniLM-L6-v2`) and cosine similarity for efficient semantic vector search against a pre-computed knowledge base.
* **Frontend:** A responsive, enterprise-grade dark-themed UI built using **HTML5**, **Tailwind CSS**, and modern **JavaScript (Fetch API)**.

---

## 🛠️ Tech Stack
Python / FastAPI (REST API & Backend Logic)

Sentence Transformers & Scikit-Learn (Semantic Matching & Cosine Similarity)

Tailwind CSS (Responsive UI & Styling)


## 📂 Project Structure

```text
Banking ChatBot/
├── Backend/
│   ├── app.py                      # FastAPI server and vector search endpoint
│   ├── generate_embeddings.py      # Script to process dataset and build vector embeddings
│   ├── banking_knowledge_base.pkl  # Pre-computed vector database and metadata
│   └── requirements.txt            # Python dependencies[cite: 1]
└── frontend/
    └── index.html                  # Modern Tailwind CSS chat interface
