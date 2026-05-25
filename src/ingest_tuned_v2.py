# Lab 7.3 — Ingest Tuned v2
# chunk_size: 512 → 400 (middle ground)

import os, re
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

load_dotenv()

loader = PyMuPDFReader()
documents = []
for file in os.listdir("data/pdfs"):
    if file.endswith(".pdf"):
        docs = loader.load(file_path=f"data/pdfs/{file}")
        documents.extend(docs)

splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)

cleaned_documents = []
for doc in documents:
    cleaned_text = re.sub(r"Menu|Top \^|Policies and Procedures.*", "", doc.text)
    cleaned_documents.append(Document(text=cleaned_text, metadata={"source": file, "type": "policy"}))

nodes = splitter.get_nodes_from_documents(cleaned_documents)
print(f"Created {len(nodes)} chunks")

index = VectorStoreIndex(nodes)
index.storage_context.persist(persist_dir="storage_tuned_v2")
print("Done!")