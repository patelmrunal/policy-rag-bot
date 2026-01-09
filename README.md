📄 PolicyBot – Retrieval Augmented Generation (RAG) System

A CLI-based Retrieval-Augmented Generation (RAG) assistant that answers questions strictly based on company policy documents (PDFs), with strong hallucination control and clear source attribution.

🚀 Objective

- The goal of this project is to demonstrate:
- Effective prompt engineering
- A correct and minimal RAG pipeline
- Strong hallucination avoidance
- Clear reasoning and evaluation of LLM outputs

The assistant retrieves relevant policy content and generates grounded answers only from retrieved documents.

🧠 Architecture Overview

User Query
   ↓
FAISS Vector Store (Semantic Retrieval)
   ↓
Retrieved Policy Chunks
   ↓
Prompt + Context
   ↓
LLM (Llama 3.1 via Groq)
   ↓
Grounded Answer + Sources


📁 Project Structure

project-root/
│
├── data/
│   └── policies.pdf              # Company policy documents
│
├── src/
│   ├── app.py                    # CLI entry point
│   ├── ingestion.py              # PDF loading & vector DB creation
│   ├── rag_pipeline.py           # RAG chain & prompt logic
│   ├── vectorstore/              # FAISS index (generated)
│   └── __pycache__/
│
├── .env                          # API keys
├── requirements.txt
└── README.md


⚙️ Setup Instructions

1️⃣ Clone the Repository

git clone <repo-url>
cd project-root

2️⃣ Create Virtual Environment

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Set Environment Variables

Create a .env file:
GROQ_API_KEY=your_groq_api_key_here

📚 Data Preparation
PDF Loading

Uses PyPDFLoader to load policy PDFs.

Chunking Strategy
chunk_size = 400
chunk_overlap = 50

Why this chunk size?

- Policy documents contain structured paragraphs.
- 400 tokens preserve semantic meaning.
- 50-token overlap prevents context loss across chunk boundaries.

▶️ How to Run
Step 1: Create Vector Database
python src/ingestion.py

Step 2: Start the Bot
python src/app.py

Example
User: What is the refund policy?
Bot: Customers can request a refund within 7 days...
[Sources Used]
- policies.pdf


What I’m Most Proud Of

- Strong hallucination control
- Clean, minimal RAG pipeline
- Clear prompt design
- Accurate grounding with sources

One Thing I’d Improve Next

- Add automated evaluation & confidence scoring
- Compare multiple prompt versions quantitatively

🧑‍💻 Tech Stack

Python
LangChain
FAISS
HuggingFace Embeddings
Groq (Llama 3.1)