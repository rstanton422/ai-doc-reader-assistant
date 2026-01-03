# ai-doc-reader-assistant
chat bot that reads a PDF and answers questions based solely on document.
This project defintiely pushed me and required tons of research to understand and apply concepts here within.
Definitely not a master and had to enlist online help... but one doesn't learn by reading alone. Applying and trying, stumbling and succeeding are all part of the journey to mastery.

# AI Document Assistant (RAG Pipeline)

## Project Overview
A retrieval-augmented generation (RAG) application that allows users to chat with their own PDF documents. Unlike standard ChatGPT, this tool sources answers *only* from the provided text, eliminating hallucinations and providing source citations.

## Tech Stack
* **Python 3.10**
* **LLM Engine:** OpenAI GPT-3.5 Turbo
* **Orchestration:** LangChain
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Frontend:** Streamlit

## How It Works
1.  **Ingestion:** PDF is loaded and split into 1000-character semantic chunks using `RecursiveCharacterTextSplitter`.
2.  **Embedding:** Text chunks are converted into vector embeddings using `text-embedding-3-small`.
3.  **Retrieval:** When a user asks a question, the app performs a similarity search in FAISS to find the top 3 relevant chunks.
4.  **Generation:** The relevant chunks + user question are fed into GPT-3.5 with a strict "answer only from context" system prompt.

## How to Run
1.  Clone the repo.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Set up `.env` file with `OPENAI_API_KEY`.
4.  Run app: `streamlit run app.py`
