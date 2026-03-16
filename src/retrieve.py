from sentence_transformers import SentenceTransformer
import chromadb

# Load the same embedding model used when storing the chunks
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="data/chroma_db")


# Get the same collection name you used earlier
collection = client.get_collection("bc_policies")


def search_policies(query, top_k=3):
    """
    Takes a user question, converts it into an embedding,
    and returns the most relevant policy chunks.
    """

    # Convert the user question into an embedding vector
    query_embedding = model.encode(query).tolist()

    # Search ChromaDB for the most similar stored chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


if __name__ == "__main__":
    question = input("Ask a Bellevue College policy question: ")

    results = search_policies(question)

    print("\nTop matching policy chunks:\n")

    for i in range(len(results["documents"][0])):
        print(f"Result {i+1}")
        print("Source:", results["metadatas"][0][i]["source"])
        print("Text:", results["documents"][0][i][:700])  # show first 700 characters
        print("-" * 80)