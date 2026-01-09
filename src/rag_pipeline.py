import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


# load environment variables
load_dotenv()

# Setup Groq LLM (Llama 3 8B)
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

DB_FAISS_PATH = "vectorstore/db_faiss"

# Modern System Prompt
system_prompt = (
    "You are a helpful Policy Assistant for TechNova Solutions. "
    "Use the following pieces of retrieved context to answer the user's question. "
    "If the answer is not in the context, say 'I don't know the answer to that based on the policy documents.' "
    "Use bold headings for different policy sections."
    "Please use bullet points for lists"
    "Do not try to make up an answer."
    "\n\n"
    "Context: {context}"
)

# Use ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


def qa_bot():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS DB
    try:
        db = FAISS.load_local(
            DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error: Could not load FAISS DB. Error: {e}")
        return None

    # Create the Modern Chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        db.as_retriever(search_kwargs={"k": 3}), document_chain
    )

    return retrieval_chain


# Example Usage
if __name__ == "__main__":
    bot = qa_bot()
    if bot:
        query = "What is the policy on remote work?"
        response = bot.invoke({"input": query})

        print("\n--- Answer ---")
        print(response["answer"])
