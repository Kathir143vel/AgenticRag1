import pymupdf4llm
import os
from pathlib import Path

# 1. Define your folder containing the 5 PDFs
pdf_folder = r"C:\Users\kathir.vel\Desktop\week5\5 pdf"
output_folder = r"C:\Users\kathir.vel\Desktop\week5\parsed_docs"
os.makedirs(output_folder, exist_ok=True)

# 2. Loop through each PDF and convert to Markdown
all_docs_text = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(pdf_folder, filename)
        print(f"Processing {filename}...")
        
        # Pymupdf4llm converts the PDF to markdown text in one go
        # This preserves tables as Markdown tables!
        md_text = pymupdf4llm.to_markdown(filepath)
        
        # Save the markdown file for backup/inspection
        output_path = os.path.join(output_folder, f"{filename}.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_text)
            
        # Store in a list with metadata (Source filename)
        all_docs_text.append({
            "source": filename,
            "content": md_text
        })

print(f"\n✅ Step 1 Complete. Parsed {len(all_docs_text)} documents.")