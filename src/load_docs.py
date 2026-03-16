import os
from pypdf import PdfReader

DOCS_PATH = "docs"

def load_documents():
    documents = []

    for file in os.listdir(DOCS_PATH):
        if file.endswith(".pdf"):
            path = os.path.join(DOCS_PATH, file)

            reader = PdfReader(path)
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            documents.append({
                "file_name": file,
                "text": text
            })

    return documents


if __name__ == "__main__":
    docs = load_documents()

    for doc in docs:
        print("Loaded:", doc["file_name"])
        print("Characters:", len(doc["text"]))
        print("---------------------")