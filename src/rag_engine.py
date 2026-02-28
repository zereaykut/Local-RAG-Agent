import os
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from src.config import settings

class RAGPipeline:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
        self.vector_store = None

    def load_specific_document(self, file_path: str):
        """Loads a specific file based on its extension."""
        ext = os.path.splitext(file_path)[-1].lower()
        
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.csv':
                loader = CSVLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path)
            else:
                print(f"Unsupported file type: {ext}")
                return []
            return loader.load()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def ingest_files(self, file_paths: list[str]):
        """Processes uploaded files and builds a new vector store."""
        documents = []
        for path in file_paths:
            docs = self.load_specific_document(path)
            documents.extend(docs)

        if not documents:
            raise ValueError("No valid documents found to index.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE, 
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)

        # Clear existing vector store to prevent cross-contamination between sessions
        if os.path.exists(settings.VECTOR_STORE_PATH):
            shutil.rmtree(settings.VECTOR_STORE_PATH)
        os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)

        self.vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings,
            persist_directory=settings.VECTOR_STORE_PATH
        )
        return self.vector_store

    def get_retriever(self):
        if not self.vector_store:
            # Try loading from disk if it exists
            if os.path.exists(settings.VECTOR_STORE_PATH) and os.listdir(settings.VECTOR_STORE_PATH):
                self.vector_store = Chroma(
                    persist_directory=settings.VECTOR_STORE_PATH,
                    embedding_function=self.embeddings
                )
            else:
                return None
        return self.vector_store.as_retriever(search_kwargs={"k": 5})