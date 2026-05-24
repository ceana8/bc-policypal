import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Document
import re


load_dotenv()

print("Loading documents...")

loader = PyMuPDFReader()



loader = PyMuPDFReader()

documents = []

pdf_folder = "data/pdfs"

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):

        path = os.path.join(
            pdf_folder,
            file
        )

        print(f"Loading {file}")

        docs = loader.load(
            file_path=path
        )

        documents.extend(docs)

print(f"Loaded {len(documents)} documents")


splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

cleaned_documents = []

for doc in documents:

    cleaned_text = re.sub(
        r"Menu|Top \^|Policies and Procedures.*",
        "",
        doc.text
    )

    cleaned_documents.append(
        Document(
            text=cleaned_text
        )
    )
    
nodes = splitter.get_nodes_from_documents(
    cleaned_documents
)

print(f"Created {len(nodes)} chunks")

index = VectorStoreIndex(nodes)

index.storage_context.persist(
    persist_dir="storage"
)

print("BCPolicyPal v2 index created successfully")