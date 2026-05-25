from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage


def normalize_query(question):
    lower = question.lower()
    if "unfair" in lower or "discrimin" in lower or "harass" in lower:
        return question + " discrimination equal opportunity complaint process"
    if "homesick" in lower or "student club" in lower or "housing" in lower:
        return question + " campus service support"
    return question


def main():
    load_dotenv()

    print("Loading tuned BCPolicyPal index...")
    storage_context = StorageContext.from_defaults(persist_dir="storage_tuned")
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(similarity_top_k=4)

    print("Ask BCPolicyPal (Tuned) a question. Type 'exit' to stop.\n")

    while True:
        question = input("Question: ")
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        normalized = normalize_query(question)
        response = query_engine.query(normalized)

        print("\nAnswer:")
        print(response)

        print("\nSources:")
        for source in response.source_nodes:
            print("-" * 50)
            print(f"Score: {source.score}")
            print(source.node.text[:500])
            print()


if __name__ == "__main__":
    main()
