from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage

load_dotenv()

print("Loading saved BCPolicyPal v2 index...")

storage_context = StorageContext.from_defaults(
    persist_dir="storage"
)

index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(
    similarity_top_k=4
)

print("Ask BCPolicyPal a question. Type 'exit' to stop.\n")

while True:
    question = input("Question: ")

    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response)

    print("\nSources:")
    for source in response.source_nodes:
        print("-" * 50)
        print(f"Score: {source.score}")
        print(source.node.text[:500])
        print()