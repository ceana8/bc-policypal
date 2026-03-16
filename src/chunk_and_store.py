import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# This is the folder where your Bellevue College policy PDFs are stored. 
# Make sure to update this path if your documents are in a different location if interested on this project
DOCS_PATH = "docs"

# Initialize embedding model
# This model converts text into numerical vectors (embeddings)
# so the computer can compare meaning, not just exact words.
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize vector database
# Think of this as creating a small AI-ready database in memory
# where we will store chunks of policy text and their embeddings.

# Initialize persistent ChromaDB database
client = chromadb.PersistentClient(path="data/chroma_db")

# Try to load the collection if it exists

try:
    collection = client.get_collection("bc_policies")

# If it doesn't exist, create it
except Exception:
    collection = client.create_collection(name="bc_policies")


# A collection is like a table or bucket inside ChromaDB.
# It will hold:
# - the chunk text
# - the embedding vector
# - metadata like which PDF the chunk came from


def load_documents():
    documents = []
# Reads all PDF files inside the docs folder and extracts their text.
# Loop through every file in the docs folder
    for file in os.listdir(DOCS_PATH):
        if file.endswith(".pdf"):  # Only process PDF files
            path = os.path.join(DOCS_PATH, file)

            reader = PdfReader(path)
            text = ""  # Start with an empty string for all the text in this PDF

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            documents.append({
                "file_name": file,
                "text": text
            })

    return documents


def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

# Optional: clear old data so you don't duplicate chunks every time you rerun


existing = collection.count()
if existing > 0:
    print(f"Collection already has {existing} items.")
    print("Delete the collection manually if you want a clean rebuild.")
    
    
def store_chunks():
    docs = load_documents()
    """
    Full pipeline:
    1. Load PDFs
    2. Extract text
    3. Split text into chunks
    4. Convert each chunk into an embedding
    5. Store chunk + embedding + source metadata in ChromaDB
    """

    id_counter = 0

    for doc in docs:
        chunks = chunk_text(doc["text"])

        for chunk in chunks:
            embedding = model.encode(chunk).tolist()

            collection.add(  # Store this chunk in ChromaDB
                documents=[chunk],  # the actual chunk text
                embeddings=[embedding],  # the vector representation
                ids=[str(id_counter)],    # unique ID for this chunk
                metadatas=[{"source": doc["file_name"]}]   # track which PDF it came from
            )

            id_counter += 1


if __name__ == "__main__":
    store_chunks()
    print("Policy chunks stored in vector database.")