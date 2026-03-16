import os
from dotenv import load_dotenv
from openai import OpenAI

from src.retrieve import search_policies

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def format_retrieved_chunks(results):
    """
    Convert retrieved ChromaDB results into a readable context block
    for the OpenAI model.
    """
    context_parts = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source = meta.get("source", "Unknown source")
        context_parts.append(f"[Source {i}: {source}]\n{doc}")

    return "\n\n".join(context_parts)


def get_source_names(results):
    """
    Extract unique source file names from retrieval results.
    """
    metadatas = results["metadatas"][0]
    sources = []

    for meta in metadatas:
        source = meta.get("source", "Unknown source")
        if source not in sources:
            sources.append(source)

    return sources


def generate_policy_answer(question, top_k=3):
    """
    Run the full RAG pipeline:
    1. Retrieve the most relevant policy chunks
    2. Send them to OpenAI
    3. Return a grounded answer + source list
    """
    results = search_policies(question, top_k=top_k)
    context = format_retrieved_chunks(results)
    sources = get_source_names(results)

    prompt = f"""
You are a Bellevue College policy assistant helping students understand official policies.

Use ONLY the policy excerpts below to answer the student's question.
Be sure to be friendly and answer in a way that's easy for students to understand.

Rules:
- Give a clear and concise answer.
- Do not invent policies, deadlines, procedures, or exceptions.
- Do not explain your reasoning.
- Do not mention retrieval, excerpts, or internal notes.
- If the answer is not found in the policy documents, say that the available policy documents do not contain that information.
- If the answer is missing, recommend contacting Bellevue College directly or checking the official Bellevue College website.
- End naturally and helpfully.

Student question:
{question}

Policy excerpts:
{context}
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    answer_text = response.output_text

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources
    }


if __name__ == "__main__":
    question = input("Ask a Bellevue College policy question: ")
    output = generate_policy_answer(question)

    print("\nAnswer:\n")
    print(output["answer"])

    print("\nSources:")
    for source in output["sources"]:
        print("-", source)