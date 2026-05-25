import os
import re

from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader


def infer_topic(text):
    lower = text.lower()
    if "ferpa" in lower or "directory information" in lower:
        return "ferpa"
    if "residency" in lower or "resident tuition" in lower or "visa" in lower:
        return "residency"
    if "register" in lower or "waitlist" in lower or "registration" in lower:
        return "registration"
    if "academic" in lower or "gpa" in lower or "dismissal" in lower:
        return "academic_standing"
    if "discrimin" in lower or "equal opportunity" in lower or "harassment" in lower:
        return "equity"
    return "general"


def main():
    load_dotenv()

    print("Loading documents for tuned index...")
    loader = PyMuPDFReader()
    documents = []
    pdf_folder = "data/pdfs"

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            print(f"Loading {filename}")
            docs = loader.load(file_path=path)

            for d in docs:
                cleaned_text = re.sub(r"Menu|Top \^|Policies and Procedures.*", "", d.text)
                documents.append(
                    Document(
                        text=cleaned_text,
                        metadata={
                            "source": filename,
                            "topic": infer_topic(cleaned_text),
                            "type": "policy",
                        },
                    )
                )

    print(f"Loaded {len(documents)} cleaned documents")

    splitter = SentenceSplitter(chunk_size=400, chunk_overlap=80)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks")

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir="storage_tuned")
    print("Tuned index created successfully in storage_tuned")


if __name__ == "__main__":
    main()
