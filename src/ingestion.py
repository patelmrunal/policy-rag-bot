import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Paths
DATA_PATH = "../data/policies.pdf"
DB_FAISS_PATH = "vectorstore/db_faiss"


def create_vector_db():
    print(f"--- Loading PDF from {DATA_PATH} ---")
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()

    print("--- Splitting Text into Chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks.")

    print("--- Generating Embeddings (This uses CPU, might take a moment) ---")
    # Using a free, local model (no API cost)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("--- Creating FAISS Vector Store ---")
    db = FAISS.from_documents(texts, embeddings)

    # Save locally
    db.save_local(DB_FAISS_PATH)
    print(f"--- Vector DB saved to {DB_FAISS_PATH} ---")


if __name__ == "__main__":
    # Create directory if it doesn't exist
    os.makedirs("vectorstore", exist_ok=True)
    create_vector_db()
