import os
import streamlit as st
from src.config import settings
from src.rag_engine import RAGPipeline
from src.bot_agent import LocalAgent

st.set_page_config(page_title="Local RAG Agent", layout="wide")
st.title("🦙 Local AI Agent with RAG")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None

# Sidebar for File Uploads
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDFs, TXTs, or CSVs", 
        type=["pdf", "txt", "csv"], 
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing and vectorizing documents..."):
                file_paths = []
                # Save files to temp directory
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(settings.UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)

                # Initialize RAG and process
                rag = RAGPipeline()
                rag.ingest_files(file_paths)
                
                # Initialize Agent
                retriever = rag.get_retriever()
                st.session_state.agent = LocalAgent(retriever=retriever)
                
                st.success("Documents processed successfully! You can now chat.")
        else:
            st.warning("Please upload files first.")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if st.session_state.agent is None:
            response = "Please upload and process documents from the sidebar before chatting."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.chat(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")