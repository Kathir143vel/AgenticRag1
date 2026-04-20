"""Mining CapEx Extraction Agent."""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MARKDOWN_FOLDER = (
    "C:\\Users\\kathir.vel\\Desktop\\week5\\parsed_docs"
)
DB_DIR = "./chroma_db_mining"

# ---------------------------------------------------------
# PART 1: RESET & INDEX (FAST VERSION)
# ---------------------------------------------------------
def setup_database():
    """Initialize the vector database from markdown files."""
    print("=== PHASE 1: Setting up Database (Fast Mode) ===")  # noqa: T201

    # 1. Delete old database if it exists
    if os.path.exists(DB_DIR):
        print(f"Deleting existing database: {DB_DIR}")  # noqa: T201
        shutil.rmtree(DB_DIR)

    # 2. Load Markdown Files
    print(f"\nLoading markdown files from {MARKDOWN_FOLDER}...")  # noqa: T201
    docs = []
    folder = Path(MARKDOWN_FOLDER)

    if not folder.exists():
        print("❌ ERROR: Folder not found. Check the path.")  # noqa: T201
        return None

    for file in folder.glob("*.md"):
        print(f"  - Loading {file.name}...")  # noqa: T201
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            docs.append(Document(page_content=text,
                                 metadata={"source": file.name}))

    print(f"✅ Loaded {len(docs)} documents.")  # noqa: T201

    # 3. Chunking
    print("\nChunking documents...")  # noqa: T201
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks.")  # noqa: T201

    # 4. INDEXING
    print("\nIndexing with BAAI/bge-small-en-v1.5... "  # noqa: T201
          "(Fast Model, please wait ~3 mins)")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="mining_capex_reports"
    )
    print("✅ Database indexed successfully.")  # noqa: T201
    return vector_store


# ---------------------------------------------------------
# PART 2: THE AGENT
# ---------------------------------------------------------
def run_agent(vector_store):
    """Start the interactive agent loop."""
    print("\n=== PHASE 2: Starting Agent ===")  # noqa: T201

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
You are a Financial Data Extraction Agent.
Extract Capital Expenditure (CapEx) from the context.

Context:
{context}

Question:
{input}

Instructions:
1. Provide CapEx Total and Breakdown.
2. Cite the Source File.
""")

    def format_docs(docs):
        """Format retrieved documents into a string."""
        combined = ""
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            combined += f"Source: {source}\n{doc.page_content}\n\n"
        return combined

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
    )

    print("\n" + "=" * 50)  # noqa: T201
    print("AGENT IS READY.")  # noqa: T201
    print("=" * 50)  # noqa: T201

    while True:
        query = input("\nQuery: ")
        if query.lower() == "exit":
            break
        if not query.strip():
            continue

        print("Thinking...")  # noqa: T201
        result = rag_chain.invoke(query)
        print("\n--- ANSWER ---")  # noqa: T201
        print(result.content)  # noqa: T201
        print("-" * 30)  # noqa: T201


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    store = setup_database()
    if store:
        run_agent(store)