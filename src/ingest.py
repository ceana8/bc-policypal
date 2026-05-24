from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader
)

from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

print("Loading documents...")

documents = SimpleDirectoryReader(
    "data/pdfs"
).load_data()

print(f"Loaded {len(documents)} documents")

splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

nodes = splitter.get_nodes_from_documents(
    documents
)

print(f"Created {len(nodes)} chunks")

index = VectorStoreIndex(
    nodes
)

index.storage_context.persist(
    persist_dir="storage"
)

print("BCPolicyPal v2 index created successfully")