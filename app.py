import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ═══════════════════════════════════════════════════════════════════════════
# INITIAL SETUP
# ═══════════════════════════════════════════════════════════════════════════
# Load environment variables (like API keys) from .env file
# This keeps sensitive info out of the code - never commit your keys to git!
load_dotenv()

# Configure the Streamlit page - this stuff shows up in the browser tab
st.set_page_config(page_title="AI Document Assistant", page_icon=":-)")
st.title("Ask Me Anything About the Document")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: FILE UPLOAD
# ═══════════════════════════════════════════════════════════════════════════
upload_pdf = st.file_uploader("Upload a PDF document", type='pdf')

# Only show the rest of the UI if user has uploaded something
if upload_pdf is not None:
    
    # ───────────────────────────────────────────────────────────────────────
    # THE "PROCESS DOCUMENT" BUTTON
    # ───────────────────────────────────────────────────────────────────────
    # This prevents processing on every page refresh - only runs when clicked
    if st.button("Press Me So I Can Absorb Your Document"):
        
        with st.spinner("Chewing on what was uploaded..."):
            
            # ═══════════════════════════════════════════════════════════════
            # File Handling between Streamlit and LangChain
            # ═══════════════════════════════════════════════════════════════
            # Streamlit holds uploaded files in RAM as a BytesIO object
            # LangChain's PyPDFLoader expects an actual file path on disk
            # Solution: Write it to a temporary file, then delete it later
            
            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as file:  # "wb" = write binary mode
                file.write(upload_pdf.getbuffer())  # getbuffer() grabs the raw bytes
            
            # ───────────────────────────────────────────────────────────────
            # STEP 2: Load the PDF
            # ───────────────────────────────────────────────────────────────
            # PyPDFLoader extracts text from each page
            loader = PyPDFLoader(temp_file)
            documents = loader.load()  # Returns a list of Document objects (one per page)
            
            # ───────────────────────────────────────────────────────────────
            # STEP 3: Chunk the Document
            # ───────────────────────────────────────────────────────────────
            # Why chunk? GPT has a limited context window - can't process 100+ page docs
            # We break it into bite-sized pieces (1000 chars each)
            # overlap=200 means chunks share some text so context doesn't get cut mid-sentence
            # Think of it like overlapping puzzle pieces - easier to see the full picture
            # below code is from LangChain docs in applying its use
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,      # Max characters per chunk
                chunk_overlap=200     # How much chunks share with neighbors
            )
            splits = text_splitter.split_documents(documents)
            
            # ───────────────────────────────────────────────────────────────
            # STEP 4: Create Embeddings + Store in Vector Database
            # ───────────────────────────────────────────────────────────────
            # This is THE MAGIC of RAG (Retrieval Augmented Generation):
            # 1. Each chunk gets converted to a vector (list of numbers) by OpenAI
            # 2. Vectors capture the *meaning* of text, not just keywords
            # 3. FAISS stores these vectors in a way that makes similarity search FAST
            #    (FAISS = Facebook AI Similarity Search, it's like a database for meanings)
            # 
            # Pro tip: This step costs money (OpenAI API calls), but we only do it once
            vector_storage = FAISS.from_documents(
                documents=splits,
                embedding=OpenAIEmbeddings()  # This calls OpenAI's embedding API
            )
            
            # ───────────────────────────────────────────────────────────────
            # STEP 5: Save the Vector Database Locally
            # ───────────────────────────────────────────────────────────────
            # Save to disk so we don't have to rebuild (and repay for) embeddings every time
            # This creates a folder with index files that FAISS can reload later
            # folder creates two files"
            #   index.faiss: The binary file containing the actual vector data and search structures
            #   index.pkl: A pickle file that stores the original text content (docstore) and 
            #              metadata associated with those vectors
            vector_storage.save_local("faiss_index_react")
            
            # Show success message with useful info
            st.success(f"Processed {len(splits)} chunks! Fire away on the questions")
            
            # Clean up - delete the temp PDF since we're done with it
            os.remove(temp_file)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Question and Answer Time
    # ═══════════════════════════════════════════════════════════════════════
    # Only show this section if we've already processed a document
    # (Checking if the FAISS index folder exists)
    if os.path.exists("faiss_index_react"):
        
        st.markdown("---")  # Visual separator - just a horizontal line
        st.subheader("Ask a Question")
        
        # Text input for user's question
        user_question = st.text_input("What do you want to know about this document?")
        
        # Only run the search/answer logic if user typed something
        if user_question:
            with st.spinner("Thinkin'..."):
                try:
                    # ───────────────────────────────────────────────────────
                    # STEP 7: Load the Saved Vector Database
                    # ───────────────────────────────────────────────────────
                    # Reload our previously saved FAISS index from disk
                    vector_storage = FAISS.load_local(
                        "faiss_index_react",
                        OpenAIEmbeddings(),
                        allow_dangerous_deserialization=True  # Needed for local pickle files
                    )
                    
                    # ───────────────────────────────────────────────────────
                    # STEP 8: Find Relevant Chunks
                    # ───────────────────────────────────────────────────────
                    # Convert user's question to a vector, then find the 3 most similar chunks
                    # "similarity" search = cosine similarity between vectors
                    # Think: "Which chunks talk about similar things to my question?"
                    retriever = vector_storage.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}  # Return top 3 matches
                    )
                    relevant_docs = retriever.invoke(user_question)
                    
                    # ═══════════════════════════════════════════════════════
                    # STEP 9: Build the LangChain Pipeline
                    # ═══════════════════════════════════════════════════════
                    # This is where we assemble the answer generation chain
                    
                    from langchain_openai import ChatOpenAI
                    from langchain_core.prompts import ChatPromptTemplate
                    from langchain_core.output_parsers import StrOutputParser
                    from langchain_core.runnables import RunnablePassthrough
                    
                    # Initialize GPT-3.5 with temperature=0 for consistent, factual answers
                    # (Higher temperature = more creative/random, lower = more deterministic)
                    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                    
                    # ───────────────────────────────────────────────────────
                    # STEP 10: Create the Prompt Template
                    # ───────────────────────────────────────────────────────
                    # This is our instructions to the AI
                    # We're forcing it to ONLY use the retrieved chunks (no hallucinations!)
                    template = """Answer the question based ONLY on the following context:
                    {context}

                    Question: {question}
                    """
                    prompt = ChatPromptTemplate.from_template(template)
                    
                    # ───────────────────────────────────────────────────────
                    # Helper Function: Format Retrieved Chunks
                    # ───────────────────────────────────────────────────────
                    # Combine the 3 chunks into one big string with double line breaks
                    # This becomes the "context" that gets inserted into the prompt
                    # Takes a list of document chunks and smushes them into one big string with 
                    # double line breaks between each chunk.
                    def format_docs(docs):
                        return "\n\n".join([d.page_content for d in docs])
                    
                    # ───────────────────────────────────────────────────────
                    # STEP 11: Assemble the Chain
                    # ───────────────────────────────────────────────────────
                    # This is LangChain's "pipe" syntax (|)
                    # Read it left to right: Input --> Transform --> LLM --> Parse Output
                    # 
                    # Flow:
                    # 1. {"context": format_docs(), "question": pass-through}
                    # 2. Inject those into the prompt template
                    # 3. Send completed prompt to GPT
                    # 4. Parse the response as a plain string
                    chain = (
                        {
                            "context": lambda x: format_docs(relevant_docs),  # Format chunks
                            "question": RunnablePassthrough()  # Pass question through unchanged
                        }
                        | prompt              # Fill in the template
                        | llm                 # Send to GPT
                        | StrOutputParser()   # Extract text from response
                    )
                    
                    # ───────────────────────────────────────────────────────
                    # STEP 12: Get the Answer
                    # ───────────────────────────────────────────────────────
                    # Run the whole chain with the user's question
                    response = chain.invoke(user_question)
                    
                    # Display the AI's answer
                    st.write(response)
                    
                    # ───────────────────────────────────────────────────────
                    # BONUS: Show Sources (The Pro Move™)
                    # ───────────────────────────────────────────────────────
                    # This is how you build trust - let users verify the AI's sources
                    # Hidden by default in an expandable section to avoid clutter
                    with st.expander("See relevant document chunks"):
                        for i, doc in enumerate(relevant_docs):
                            st.write(f"**Chunk {i+1}:**")
                            st.write(doc.page_content)
                    
                except Exception as e:
                    # If anything breaks, show a friendly error message
                    st.error(f"Error: {e}")
                    st.info("Tip: Make sure you processed a document first!")