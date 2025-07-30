# ai_agent.py
import os
import warnings
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check for GROQ API Key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ GROQ_API_KEY not found in .env file. Please set it.")

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Use local Hugging Face embeddings 
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

# GROQ for LLM 
from langchain_groq import ChatGroq

# Chains & QA
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Memory & History
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -------------------------------
# Configuration
# -------------------------------
LLM_MODEL = "llama-3.1-8b-instant"  # Fast & capable
EMBEDDING_MODEL = "hkunlp/instructor-large"  # Local, no server needed
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FAISS_INDEX_PATH = "faiss_index"
DB_PATH = "sqlite:///chat_memory.db"
DOCS_DIR = "docs"

# Create directories
Path(DOCS_DIR).mkdir(exist_ok=True)
Path(FAISS_INDEX_PATH).mkdir(exist_ok=True)

# Initialize LLM (GROQ)
llm = ChatGroq(
    model=LLM_MODEL,
    temperature=0.3,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    streaming=False,  # Prevents input lag
)

# -------------------------------
# Document Loading
# -------------------------------
def load_documents(directory: str) -> List:
    documents = []
    for file_path in Path(directory).rglob("*.*"):
        ext = file_path.suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(str(file_path))
            elif ext == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                continue
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path.name
            documents.extend(docs)
        except Exception as e:
            print(f"⚠️ Failed to load {file_path.name}: {e}")
    return documents

# -------------------------------
# Vectorstore with Hugging Face Embeddings
# -------------------------------
def create_vectorstore(force_rebuild: bool = False):
    index_path = Path(FAISS_INDEX_PATH) / "index.faiss"
    if index_path.exists() and not force_rebuild:
        print("🔁 Loading existing vector store...")
        embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore

    print("🛠️ Processing documents...")
    docs = load_documents(DOCS_DIR)
    if not docs:
        print("🚫 No documents found in 'docs/' folder.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    print(f"✅ Creating vector store with {len(chunks)} chunks...")
    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("📌 Vector store saved.")
    return vectorstore

# -------------------------------
# Persistent Memory
# -------------------------------
def get_session_history(session_id: str):
    return SQLChatMessageHistory(session_id, DB_PATH)

# -------------------------------
# RAG Prompt
# -------------------------------
CUSTOM_PROMPT_TEMPLATE = """Use only the following context to answer the question.
If you don't know the answer, say: 'This information is not covered in the document I have.'

Context: {context}
Question: {question}
Helpful Answer:
"""

custom_prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

# -------------------------------
# Global Variables
# -------------------------------
vectorstore = None
retriever = None
long_term_chain = None

def initialize_chains():
    global vectorstore, retriever, long_term_chain
    vectorstore = create_vectorstore(force_rebuild=False)
    if vectorstore is None:
        return False
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    long_term_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
    )
    return True

# -------------------------------
# Short-Term Memory Chain
# -------------------------------
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm
conversational_rag_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# -------------------------------
# CLI Interface
# -------------------------------
def display_menu():
    print("\n" + "=" * 60)
    print("🧠 AI Context Awareness Agent")
    print("=" * 60)
    print("1️⃣  Chat & Remember (Personal Memory)")
    print("2️⃣  Ask About Documents")
    print("3️⃣  View Chat History")
    print("4️⃣  Rebuild Document Index")
    print("5️⃣  Exit")

def get_session_id():
    return input("👤 Enter your user ID (e.g., 'alice'): ").strip() or "default"

def main():
    global vectorstore, retriever, long_term_chain

    print("🚀 Initializing AI Agent...")

    if not initialize_chains():
        print("❌ Failed to initialize. No documents loaded.")
        return

    session_id = get_session_id()

    while True:
        display_menu()
        try:
            choice = input("\n➡️ Choose (1-5): ").strip()

            if choice == "1":
                query = input("💬 You: ").strip()
                if not query:
                    continue
                config = {"configurable": {"session_id": session_id}}
                response = conversational_rag_chain.invoke({"input": query}, config=config)
                print(f"🤖 AI: {response.content}")

            elif choice == "2":
                query = input("📄 Ask about docs: ").strip()
                if not query:
                    continue
                result = long_term_chain.invoke({"query": query})
                answer = result["result"].strip()
                sources = list(set([doc.metadata["source"] for doc in result["source_documents"]]))
                print(f"📄 AI: {answer}")
                if sources:
                    print(f"📌 Sources: {', '.join(sources)}")

            elif choice == "3":
                history = get_session_history(session_id)
                print(f"\n📜 Chat History ({session_id}):")
                for msg in history.messages:
                    prefix = "👤 You: " if isinstance(msg, HumanMessage) else "🤖 AI: "
                    print(f"{prefix}{msg.content[:100]}...")

            elif choice == "4":
                print("🔄 Rebuilding document index...")
                if not initialize_chains():
                    print("❌ Failed to rebuild index.")
                else:
                    print("✅ Index rebuilt and chains updated.")

            elif choice == "5":
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid option. Choose 1–5.")

        except KeyboardInterrupt:
            print("\n👋 Exiting gracefully...")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()