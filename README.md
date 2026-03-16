# 🎓 BCPolicyPal

BCPolicyPal is an NLP assistant that helps students quickly understand Bellevue College policies without reading long documents.

The system uses Retrieval-Augmented Generation (RAG) to search policy documents and generate grounded answers.


## 🚀 Features

• Ask questions about Bellevue College policies  
• Semantic search across policy documents  
• Answers generated using an LLM with supporting context  
• Simple web interface built with Streamlit  

Example question:

> "How do parking permits work?"

The system retrieves the relevant policy text and generates a clear explanation.


## 🧠 How It Works

BCPolicyPal uses a **Retrieval-Augmented Generation (RAG) pipeline**:

1. Policy PDFs are loaded
2. Text is extracted and chunked
3. Chunks are converted into embeddings using **SentenceTransformers**
4. Embeddings are stored in **ChromaDB**
5. When a user asks a question:
   - the system searches for relevant policy chunks
   - sends them to an LLM
   - generates a grounded answer


## 🛠 Tech Stack

Python  
Streamlit  
ChromaDB (Vector Database)  
SentenceTransformers (Embeddings)  
OpenAI API (LLM)  
PyPDF (PDF text extraction)


## 📂 Project Structure
