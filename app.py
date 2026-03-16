import streamlit as st
from src.generate_answer import generate_policy_answer

st.set_page_config(page_title="BC PolicyPal", page_icon="🎓")

st.title("🎓 BC PolicyPal")
st.write("Ask questions about Bellevue College policies.")
with st.sidebar:
    st.header("About BC PolicyPal")

    st.write("""
    BC PolicyPal is an AI assistant that answers questions
    using Bellevue College policy documents.

    Technologies used:
    - Retrieval-Augmented Generation (RAG)
    - SentenceTransformers embeddings
    - ChromaDB vector search
    - OpenAI LLM reasoning
    """)
    st.write("---")

    st.write("""
⚠️ This assistant is not an official source.
For official guidance please contact Bellevue College.
""")


# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input("Ask a question about BC policies")

if question:

    # show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # generate answer
    with st.spinner("Searching policies..."):
        result = generate_policy_answer(question)

    answer = result["answer"]
    sources = result["sources"]

    response_text = answer + "\n\n**Sources:**\n"
    for source in sources:
        response_text += f"- {source}\n"

    # show assistant message
    with st.chat_message("assistant"):
        st.markdown(response_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )