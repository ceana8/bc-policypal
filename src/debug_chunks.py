from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    "data/pdfs"
).load_data()

for i, doc in enumerate(documents):
    print("="*50)
    print(f"Document {i+1}")
    print(doc.text[:1000])