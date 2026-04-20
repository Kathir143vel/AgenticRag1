"""Chunking module for mining reports."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

# ---------------------------------------------------------
# STRATEGY:
# 1. Split by Headers (##, ###) to isolate CapEx sections.
# 2. This keeps tables (usually under headers) intact.
# 3. If a chunk is still too big, split by characters as a backup.
# ---------------------------------------------------------


def parse_markdown_files(folder_path: str) -> list[dict]:
    """
    Read markdown files from the folder.

    Returns a list of dictionaries containing text content and source file.
    """
    docs = []
    folder = Path(folder_path)

    for file in folder.glob("*.md"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            docs.append({"content": content, "source": file.name})

    return docs


def chunk_documents(all_docs: list[dict]) -> list:
    """
    Split documents into chunks based on Markdown headers.

    Designed to preserve table integrity.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    # Fallback splitter for very large sections (rare, but good safety)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        length_function=len,
    )

    final_chunks = []

    for doc in all_docs:
        # Step A: Split by Headers (Crucial for CapEx isolation)
        md_header_splits = markdown_splitter.split_text(doc["content"])

        for split in md_header_splits:
            # Add metadata about the source file
            split.metadata["source"] = doc["source"]

            # Step B: If a header section is huge (e.g., long narrative),
            # split it further by characters.
            splits = text_splitter.split_documents([split])

            final_chunks.extend(splits)

    return final_chunks


# --- EXECUTION ---

# 1. Load the Markdown files created in Step 1
markdown_docs = parse_markdown_files(
    r"C:\Users\kathir.vel\Desktop\week5\parsed_docs"
)

# 2. Perform Chunking
chunks = chunk_documents(markdown_docs)

# 3. PREVIEW: Check if a chunk looks like a table
for i, chunk in enumerate(chunks):
    # Look for chunks that likely contain tables (have pipes '|')
    if "|" in chunk.page_content[:100]:
        print("\n--- Sample Chunk (Likely Table) ---")  # noqa: T201
        print(f"Source: {chunk.metadata['source']}")  # noqa: T201
        print(chunk.page_content[:300])  # noqa: T201
        break

print("✅ Step 2 Complete.")  # noqa: T201
print(f"Total Chunks Created: {len(chunks)}")  # noqa: T201